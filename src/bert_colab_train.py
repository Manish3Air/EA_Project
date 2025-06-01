# Step 2: Your training code (after restarting runtime)
import os
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score

# Upload your files manually in Colab or mount Drive (change as needed)
from google.colab import files
uploaded = files.upload()

# Load and preprocess data
def load_data(filepath):
    df = pd.read_csv(filepath, sep=";", names=["text", "label"])
    df.dropna(inplace=True)
    df["text"] = df["text"].str.strip().str.lower()
    return df

train_df = load_data("train.txt")
test_df = load_data("test.txt")

# Label encoding
le = LabelEncoder()
train_df["label"] = le.fit_transform(train_df["label"])
test_df["label"] = le.transform(test_df["label"])

# Save label encoder
os.makedirs("saved_model", exist_ok=True)
with open("saved_model/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# Convert to HuggingFace Datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length")

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Compute class weights
labels = train_df["label"].values
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Custom Trainer with class weights
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(model.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Load model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=len(le.classes_)
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=10,
    report_to="none"
)

# Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

# Trainer
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Save model, tokenizer, and label encoder
model.save_pretrained("saved_model")
tokenizer.save_pretrained("saved_model")

# Zip and download model folder
import shutil
shutil.make_archive("bert_emotion_model", "zip", "saved_model")

files.download("bert_emotion_model.zip")
