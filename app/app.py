# app.py

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
import pickle
import pandas as pd

# 👇 Replace with your model name on Hugging Face
MODEL_REPO = "Manish3Air/ai-sentiment-analyzer-model"

# Emoji mapping
EMOJI_MAP = {
    "joy": "😄",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😲"
}

# Load model and tokenizer from Hugging Face Hub
model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
model.eval()

# Load label encoder (make sure it's uploaded to the repo)
label_path = hf_hub_download(repo_id=MODEL_REPO, filename="label_encoder.pkl")
with open(label_path, "rb") as f:
    label_encoder = pickle.load(f)

# Streamlit UI
st.set_page_config(page_title="Emotion Analyzer", layout="centered")
st.title("🎭 Emotion Analyzer")

text = st.text_area("Enter a sentence to analyze its emotion:", height=150)

# After prediction (inside the 'Analyze' button block)
if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        pred_label = torch.argmax(probs, dim=1).item()
        confidence = torch.max(probs).item()
        emotion = label_encoder.inverse_transform([pred_label])[0]
        emoji = EMOJI_MAP.get(emotion, "")

        # Save to session state
        st.session_state.prediction = {
            "probs": probs,
            "emotion": emotion,
            "confidence": confidence
        }

# Show prediction if exists
if "prediction" in st.session_state:
    pred = st.session_state.prediction
    st.markdown(f"""
    ### 🎯 **Predicted Emotion:** `{pred['emotion'].capitalize()}` {EMOJI_MAP.get(pred['emotion'], "")}  
    #### 🔍 Confidence: `{pred['confidence'] * 100:.2f}%`
    """)

    if st.checkbox("Show probabilities for all emotions"):
        all_emotions = label_encoder.inverse_transform(list(range(len(pred['probs'][0]))))
        prob_dict = {label: float(prob) * 100 for label, prob in zip(all_emotions, pred['probs'][0])}
        df = pd.DataFrame({
            "Emotion": list(prob_dict.keys()),
            "Probability (%)": list(prob_dict.values()),
            "Emoji": [EMOJI_MAP.get(em, "") for em in prob_dict.keys()]
        }).sort_values("Probability (%)", ascending=False).reset_index(drop=True)
        st.dataframe(df, use_container_width=True)
