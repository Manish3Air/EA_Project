# src/load_and_clean.py

import pandas as pd
import re

# Clean text function
def clean_text(text):
    text = re.sub(r"http\S+|@\S+|#[A-Za-z0-9_]+", "", text)  # remove links/mentions/hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # keep only letters and spaces
    text = text.lower().strip()
    return text

# Load dataset from txt file
def load_data(filepath):
    df = pd.read_csv(filepath, sep=';', header=None, names=['text', 'emotion'])
    df['text'] = df['text'].apply(clean_text)
    return df

# Example usage
if __name__ == "__main__":
    train_df = load_data("data/train.txt")
    test_df = load_data("data/test.txt")

    print("Sample data:\n", train_df.head())
    print("\nClass distribution:\n", train_df['emotion'].value_counts())
