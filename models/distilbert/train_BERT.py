import evaluate
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments
)
import numpy as np
import pandas as pd
import datasets
from datasets import Dataset, load_from_disk
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score
from scipy.special import softmax
import pickle


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    return metrics.compute(predictions=predictions, references=labels)

def tokenize_function(example, tokenizer):
    return tokenizer(example["sentence"], truncation=True, padding=True)

def main():

    checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
    #Loading and preprocessing data
    train_df=pd.read_csv("../data/train.csv")
    val_df=pd.read_csv("../data/valid.csv")
    test_df=pd.read_csv("../data/test.csv")

    train_df = train_df.dropna()
    val_df = val_df.dropna()
    test_df = test_df.dropna()

    train_df["sentence"] = train_df["sentence"].astype(str)
    val_df["sentence"] = val_df["sentence"].astype(str)
    test_df["sentence"] = test_df["sentence"].astype(str)


    train_dataset = Dataset.from_pandas(train_df[["sentence", "label"]])
    val_dataset = Dataset.from_pandas(val_df[["sentence", "label"]])
    test_dataset = Dataset.from_pandas(test_df[["sentence", "label"]])

    dataset = datasets.DatasetDict({"train":train_dataset,
                                "val":val_dataset,
                                "test":test_dataset})

    tokenized_datasets = dataset.map(tokenize_function, batched=True, fn_kwargs=dict(tokenizer=tokenizer))
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence", '__index_level_0__'])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format(type="torch")


    #Can be used to save the tokenized data to disk and then load it after running the code for the first time
    #Otherwise tokenization can take arounf half an hour
    # tokenized_datasets.save_to_disk("tokenized_datasets_distilbert")
    # tokenized_datasets = load_from_disk("tokenized_datasets_distilbert")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

    #TRAINING ARGUMENTS
    training_args = TrainingArguments(
    output_dir="distilbert",
    learning_rate=2e-5,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    #Training
    trainer.train()
    trainer.save_model()
   
    #Evaluation
    test_pred=trainer.predict(tokenized_datasets["test"], metric_key_prefix="test")
    val_pred=trainer.predict(tokenized_datasets["val"], metric_key_prefix="val")
    print(test_pred.metrics)
    print(val_pred.metrics)

    y_pred=np.argmax(test_pred.predictions, axis=1)
    print(classification_report(test_pred.label_ids, y_pred))
    probabilities = softmax(test_pred[0], axis=1)
    print(roc_auc_score("ROC AUC SCORE:", test_pred.label_ids, probabilities[:, 1]))

    #Saving the results
    with open('test_pred', 'wb') as f:
            pickle.dump(test_pred, f)

if __name__ == "__main__":
    main()
