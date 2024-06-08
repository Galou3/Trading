# !pip install datasets transformers
import device as device
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf
import requests
from tqdm.auto import tqdm
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import datetime
import json
import pandas_datareader
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset, Value, ClassLabel, Features
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW, get_scheduler

# Path to your dataset
PATH = 'C:\\Users\\GaelT\\OneDrive\\Bureau\\TRADINGV2\\data\\sentiment_data'

# Load and preprocess the dataset
df = pd.read_csv(os.path.join(PATH, 'apr_22.csv'))
df.drop(labels=['type', 'id', 'subreddit.id', 'subreddit.name', 'subreddit.nsfw', 'created_utc', 'permalink', 'score'],
        axis=1, inplace=True)
df['body'] = df['body'].replace(r'\n', ' ', regex=True)
df.dropna(inplace=True)
df = df[df['body'].apply(lambda x: len(x) <= 512)]
df.rename(columns={"sentiment": "labels", "body": "text"}, inplace=True)
df.reset_index(drop=True, inplace=True)
df.drop_duplicates(subset="text", inplace=True)
df.to_csv(os.path.join(PATH, "preprocessed_df.csv"))

# Load preprocessed dataframe
df = pd.read_csv(os.path.join(PATH, "preprocessed_df.csv"), index_col=0)

# Sample the dataframe
n_sample = 300000
df_sample = df.sample(n=n_sample, random_state=42)
df = df_sample.copy()

# Convert labels to integers
df.loc[df['labels'] > 0, 'labels'] = 1  # POSITIVE
df.loc[df['labels'] == 0, 'labels'] = 2  # NEUTRAL
df.loc[df['labels'] < 0, 'labels'] = 0  # NEGATIVE

# Balance the dataset
df = (df.groupby('labels', as_index=False)
      .apply(lambda x: x.sample(n=30000, random_state=69))
      .reset_index(drop=True))

# Split the dataset
df_dev, df_test = train_test_split(df, test_size=0.3, random_state=69)
df_train, df_val = train_test_split(df_dev, test_size=0.2, random_state=69)
df_train.to_csv(os.path.join(PATH, "df_train.csv"), index=False)
df_val.to_csv(os.path.join(PATH, "df_val.csv"), index=False)
df_test.to_csv(os.path.join(PATH, "df_test.csv"), index=False)

# Prepare dataset for Huggingface
schema = Features({'text': Value(dtype='string', id=None),
                   'labels': ClassLabel(num_classes=3, id=None),
                   })
dataset_all = Dataset.from_pandas(df, schema)
dataset = dataset_all.train_test_split(test_size=0.3, seed=69)
dataset_train = dataset_all.train_test_split(test_size=0.2, seed=69)

# Tokenize dataset
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tok_ds_train = dataset_train["train"].map(tokenize_function, batched=True)
tok_ds_val = dataset_train["test"].map(tokenize_function, batched=True)
tok_ds_test = dataset["test"].map(tokenize_function, batched=True)

tok_ds_train = tok_ds_train.remove_columns(["text"])
tok_ds_val = tok_ds_val.remove_columns(["text"])
tok_ds_test = tok_ds_test.remove_columns(["text"])

# Fine-tune DistilBERT model
distilbert_model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=3)

distilbert_model.to(device)
train_dataloader = DataLoader(tok_ds_train, shuffle=True, batch_size=10)
val_dataloader = DataLoader(tok_ds_val, batch_size=10)
test_dataloader = DataLoader(tok_ds_test, batch_size=10)

optimizer = AdamW(distilbert_model.parameters(), lr=3e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
distilbert_model.to(device)
progress_bar = tqdm(range(num_training_steps))

distilbert_model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = distilbert_model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

torch.save(distilbert_model.state_dict(), PATH + "/distilbert_model.pth")

# Load and evaluate the model
distilbert_model.load_state_dict(torch.load(PATH + "/distilbert_model.pth"))
distilbert_model.eval()

from datasets import load_metric

metric1 = load_metric('accuracy')
metric2 = load_metric("precision")
metric3 = load_metric("recall")

progress_bar = tqdm(val_dataloader)
distilbert_model.eval()
preds = []

for batch in val_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = distilbert_model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    preds.append(predictions.cpu().detach().tolist())
    metric1.add_batch(predictions=predictions, references=batch["labels"])
    metric2.add_batch(predictions=predictions, references=batch["labels"])
    metric3.add_batch(predictions=predictions, references=batch["labels"])
    progress_bar.update(1)

accuracy = metric1.compute()
precision = metric2.compute(average='macro')
recall = metric3.compute(average='macro')

print(accuracy)
print(precision)
print(recall)
