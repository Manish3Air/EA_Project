# src/bert_data.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import pickle
import os

def load_and_prepare_dataset(path, label_encoder_path="label_encoder.pkl", save_encoder=True):
    """
    Load dataset, preprocess, encode labels, return HF dataset and label encoder.
    """
    df = pd.read_csv(path, sep=";", names=["text", "label"])
    df = df.dropna()
    df["text"] = df["text"].str.lower().str.strip()

    # Encode labels
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["label"])

    if save_encoder:
        with open(label_encoder_path, "wb") as f:
            pickle.dump(le, f)

    # Convert to HuggingFace Dataset
    hf_dataset = Dataset.from_pandas(df)
    return hf_dataset, le


def load_label_encoder(label_encoder_path="label_encoder.pkl"):
    """
    Load label encoder from disk
    """
    if not os.path.exists(label_encoder_path):
        raise FileNotFoundError(f"LabelEncoder file not found at {label_encoder_path}")
    
    with open(label_encoder_path, "rb") as f:
        le = pickle.load(f)
    return le


def preprocess_text(text):
    """
    Basic text preprocessing
    """
    return text.lower().strip()
