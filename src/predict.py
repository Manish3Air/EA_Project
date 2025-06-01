# src/predict.py

import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import pickle
import os

# Load model & tokenizer
model_dir = "saved_model"
model = DistilBertForSequenceClassification.from_pretrained(model_dir)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)

# Load LabelEncoder
with open(os.path.join(model_dir, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)

# Function to predict emotion
def predict_emotion(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        emotion = label_encoder.inverse_transform([prediction])[0]
    return emotion

# Example
if __name__ == "__main__":
    while True:
        text = input("\nEnter a sentence (or 'exit'): ").strip()
        if text.lower() == "exit":
            break
        emotion = predict_emotion(text)
        print(f"Predicted Emotion: {emotion}")
