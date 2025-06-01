# src/train_model.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from load_and_clean import load_data

# Load and prepare data
df = load_data("data/train.txt")
X = df["text"]
y = df["emotion"]

# Split into train and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Vectorize with TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# Train model
clf = LogisticRegression(max_iter=1000,class_weight='balanced')
clf.fit(X_train_vec, y_train)

# Evaluate
y_pred = clf.predict(X_val_vec)
print("Classification Report:\n", classification_report(y_val, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))

# Save model and vectorizer
joblib.dump(clf, "models/logistic_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")



# Get confusion matrix
cm = confusion_matrix(y_val, y_pred, labels=clf.classes_)

# Plot it
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Emotion Classifier')
plt.tight_layout()
plt.savefig("models/confusion_matrix.png")
plt.show()
