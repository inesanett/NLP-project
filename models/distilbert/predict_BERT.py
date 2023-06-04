import argparse
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TextClassificationPipeline as pipeline
)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" #Otherwse will get errors

model = AutoModelForSequenceClassification.from_pretrained("distilbert")
tokenizer = AutoTokenizer.from_pretrained("distilbert")

if __name__ == "__main__":
    
    #Take user import and predict whether it's a spoiler or not (prints scores)
    parser = argparse.ArgumentParser()
    parser.add_argument("user_input")  # positional argument
    args = parser.parse_args()
    clf = pipeline(model=model, tokenizer=tokenizer, top_k=None)
    result = clf(args.user_input) 
    print(result)
