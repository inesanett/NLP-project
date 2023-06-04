import torch
import numpy as np
import pandas as pd
import datasets
from datasets import Dataset, load_from_disk
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score
import pickle
from torch import nn, Tensor
import torchtext
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator 
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def data_process(data, vocab, tokenizer):
    return [(vocab(tokenizer(item['sentence'])), item['label']) for item in data ]

def clean_dataset(dataset):
  #Removing empty lines
  n=len(dataset)
  for i in range(n-1, -1, -1):
    tuple=dataset[i]
    if (len(tuple[0]) == 0):
        dataset.pop(i)


def main():

    #Loading and preprocessing data
    print("Reading in data...")
    train_df=pd.read_csv("../data/train.csv")
    val_df=pd.read_csv("../data/valid.csv")
    test_df=pd.read_csv("../data/test.csv")

    train_df = train_df.dropna()
    val_df = val_df.dropna()
    test_df = test_df.dropna()

    train_df=train_df.loc[train_df['sentence'].str.len() >0]
    val_df=val_df.loc[val_df['sentence'].str.len() >0]
    test_df=test_df.loc[test_df['sentence'].str.len() >0]

    train_df["sentence"] = train_df["sentence"].astype(str)
    val_df["sentence"] = val_df["sentence"].astype(str)
    test_df["sentence"] = test_df["sentence"].astype(str)


    train_dataset = Dataset.from_pandas(train_df[["sentence", "label"]])
    val_dataset = Dataset.from_pandas(val_df[["sentence", "label"]])
    test_dataset = Dataset.from_pandas(test_df[["sentence", "label"]])

    dataset = datasets.DatasetDict({"train":train_dataset,
                                "val":val_dataset,
                                "test":test_dataset})
   

    print("Building vocabluary...")
    tokenizer = get_tokenizer("basic_english")
    vocab = build_vocab_from_iterator(map(tokenizer, dataset['train']['sentence']), specials=['<pad>', '<unk>'])
    vocab.set_default_index(vocab['<unk>']) 

    train_dataset = data_process(dataset["train"], vocab, tokenizer)
    valid_dataset = data_process(dataset["val"], vocab, tokenizer)
    test_dataset = data_process(dataset["test"], vocab, tokenizer)

    clean_dataset(train_dataset)
    clean_dataset(valid_dataset)
    clean_dataset(test_dataset)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _collate_fn(batch):
        text = [torch.tensor(item[0], dtype=torch.long) for item in batch]
        label_list = [item[1] for item in batch]

        padded_text = nn.utils.rnn.pad_sequence(text, batch_first=True)
        lengths = torch.tensor([len(item) for item in text], dtype=torch.long)
        label_list = torch.tensor(label_list, dtype=torch.float)

        return padded_text.to(device), label_list.to(device), lengths.to(device)

    batch_size = 32  

    print("Creating dataloaders...")
    train_dl = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True, collate_fn=_collate_fn)
    valid_dl = DataLoader(valid_dataset, batch_size=batch_size,
                        shuffle=False, collate_fn=_collate_fn)
    test_dl = DataLoader(test_dataset, batch_size=batch_size,
                        shuffle=False, collate_fn=_collate_fn)

    #Parameters
    vocab_size = len(vocab)
    embed_dim = 32
    lstm_hidden_size = 64
    fc_hidden_size = 64
    num_epochs = 10

    torch.manual_seed(1)
    model = LSTM(vocab_size, embed_dim, lstm_hidden_size, fc_hidden_size) 
    model = model.to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Model: ", model)

    def train(dataloader):
        # Set the model to the training mode
        model.train()
        total_acc, total_loss = 0, 0
        all_labels=[]
        all_probabilities=[]
        # Iterate through each batch in the training dataloader
        for text_batch, label_batch, lengths in dataloader:
            # Zero the gradients to prevent explosion
            optimizer.zero_grad()
            # Predict the output
            pred = model(text_batch, lengths)[:, 0]

            all_probabilities.extend(pred.detach().cpu().numpy())
            all_labels.extend(label_batch.detach().cpu().numpy())
            # Calculate the loss
            loss = loss_fn(pred, label_batch)
            # Backward pass on the loss
            loss.backward()
            # Update the model's weights
            optimizer.step()
            #calculate total accuracy
            total_acc += ((pred>=0.5).float() == label_batch).float().sum().item()
            #calculate total loss
            total_loss += loss.item()*label_batch.size(0)
        
        roc_auc=roc_auc_score(all_labels, all_probabilities)
        return total_acc/len(dataloader.dataset), total_loss/len(dataloader.dataset), roc_auc
 

    def evaluate(dataloader):
        # Set the model to the evaluation mode
        model.eval()
        total_acc, total_loss = 0, 0
        all_labels=[]
        all_probabilities=[]
        # Turn off the gradient recording 
        with torch.no_grad():
            for text_batch, label_batch, lengths in dataloader:
                #predict
                pred = model(text_batch, lengths)[:, 0]
                all_probabilities.extend(pred.detach().cpu().numpy())
                all_labels.extend(label_batch.detach().cpu().numpy())
                #calculate loss
                loss = loss_fn(pred, label_batch)
                #calculate totalaccuracy
                total_acc += ((pred>=0.5).float() == label_batch).float().sum().item()
                #calculate total loss
                total_loss += loss.item()*label_batch.size(0)

        roc_auc=roc_auc_score(all_labels, all_probabilities)
        return total_acc/len(dataloader.dataset), total_loss/len(dataloader.dataset), roc_auc


    torch.manual_seed(1)

    print("Training...")
    acc_train_list=[]
    loss_train_list=[]
    acc_valid_list=[]
    loss_valid_list=[]
    auc_train_list=[]
    auc_valid_list=[]

    for epoch in range(num_epochs):
        acc_train, loss_train, auc_train = train(train_dl)
        acc_valid, loss_valid, auc_valid = evaluate(valid_dl)

        acc_train_list.append(acc_train)
        loss_train_list.append(loss_train)
        acc_valid_list.append(acc_valid)
        loss_valid_list.append(loss_valid)
        auc_train_list.append(auc_train)
        auc_valid_list.append(auc_valid)
        
        print(f'Epoch {epoch} loss: {loss_train:.4f} val_loss: {loss_valid:.4f} accuracy: {acc_train:.4f} val_accuracy: {acc_valid:.4f} roc-auc: {auc_train:.4f} val_roc-auc: {auc_valid:.4f}')
        torch.save(model, f"model{epoch}")
    
    #All results
    zipped= list(zip(range(num_epochs), acc_train_list, loss_train_list, acc_valid_list, loss_valid_list, auc_train_list, auc_valid_list))
    results_df = pd.DataFrame(zipped, columns=["Epoch", "acc_train", "loss_train", "acc_valid", "loss_valid", "auc_train", "auc_valid"])
    results_df.to_csv("lstm_results.csv")

    #Evaluating on test set
    def evaluate_final(dataloader):

        model.eval()
        all_predictions=[]
        all_labels=[]
        all_probabilities=[]
        
        with torch.no_grad():
            for sentence_batch, label_batch, lengths in dataloader:

                probabilities = model(sentence_batch, lengths)[:, 0]
                predictions = (probabilities>=0.5).float()

                all_predictions.extend(predictions)
                all_probabilities.extend(probabilities)
                all_labels.extend(label_batch)
        

        print(classification_report(all_labels, all_predictions))
        print("ROC AUC SCORE:", roc_auc_score( all_labels, all_probabilities))

    
    #Choosing best model
    best_model=np.argmax(results_df.auc_valid)
    model=torch.load(f"model{best_model}")
    print(f"Best model is from epoch{best_model}")
    print(results_df.iloc[best_model])

    print("Validation:")
    evaluate_final(valid_dl)
    print("Test:")
    evaluate_final(test_dl)


class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, lstm_hidden_size, fc_hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 
                                      embed_dim, 
                                      padding_idx=0) 
        self.rnn = nn.LSTM(embed_dim, lstm_hidden_size, 
                           batch_first=True)
        self.fc1 = nn.Linear(lstm_hidden_size, fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, lengths):
        out = self.embedding(text)
        out = nn.utils.rnn.pack_padded_sequence(out, lengths.cpu().numpy(), enforce_sorted=False, batch_first=True)
        out, (hidden, cell) = self.rnn(out)
        out = hidden[-1, :, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out  

       

if __name__ == "__main__":
    main()
