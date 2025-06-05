import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
import pickle
import pandas as pd
import os
import requests
from pathlib import Path

# Background image
bg_image_path = "background.jpg"

# Internet check
def is_internet_available(url="https://google.co.in", timeout=3):
    try:
        requests.get(url, timeout=timeout)
        return True
    except requests.RequestException:
        return False

# Streamlit config
st.set_page_config(page_title="Emotion Analyzer", layout="centered")

# ---------------- CSS Styling ------------------
st.markdown(f"""
    <style>
    .stApp {{
        background: url('{bg_image_path}');
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-position: center;
    }}
    .main-container {{
        background-color: rgba(255, 255, 255, 0.92);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem auto;
        max-width: 800px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }}
    .hero {{
        text-align: center;
        margin-bottom: 2.5rem;
    }}
    .hero h1 {{
        font-size: 3rem;
        color: #1a1a1a;
        margin-bottom: 0.3rem;
    }}
    .hero p {{
        font-size: 1.25rem;
        color: #444;
        margin: 0;
    }}
    textarea {{
        border-radius: 10px !important;
        border: 2px solid #3498db !important;
        font-size: 1.05rem !important;
    }}
    button[kind="primary"] {{
        background-color: #3498db !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 10px 22px !important;
    }}
    button[kind="primary"]:hover {{
        background-color: #2980b9 !important;
    }}
    .prediction {{
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-top: 20px;
    }}
    .emoji-large {{
        font-size: 2.3rem;
        vertical-align: middle;
        margin-left: 8px;
    }}
    </style>
""", unsafe_allow_html=True)

# ------------------ App Content -----------------

# Hero section
st.markdown("""
    <div class="hero">
        <h1>🎭 Emotion Analyzer</h1>
        <p>Understand your emotions through AI – Fast, Accurate & Fun!</p>
    </div>
""", unsafe_allow_html=True)

# Internet check
internet = is_internet_available()
source = "Hugging Face Hub" if internet else "Local"

if internet:
    st.success("🌐 Internet available — using Hugging Face model")
else:
    st.warning("📴 No internet — using local model")

# Load model
if source == "Hugging Face Hub":
    MODEL_REPO = "Manish3Air/ai-sentiment-analyzer-model"
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    label_path = hf_hub_download(repo_id=MODEL_REPO, filename="label_encoder.pkl")
else:
    MODEL_REPO = "./saved_model"
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    label_path = os.path.join(MODEL_REPO, "label_encoder.pkl")

with open(label_path, "rb") as f:
    label_encoder = pickle.load(f)

model.eval()

# Emoji map
EMOJI_MAP = {
    "joy": "😄", "sadness": "😢", "anger": "😠",
    "fear": "😨", "love": "❤️", "surprise": "😲"
}

# Text input
if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""

st.session_state["input_text"] = st.text_area(
    "✍️ Enter a sentence to analyze its emotion:",
    value=st.session_state["input_text"],
    height=150,
    placeholder="Type your sentence here..."
)

col1, col2 = st.columns([1, 1])
with col1:
    analyze_clicked = st.button("🔍 Analyze")
with col2:
    if st.button("🎯 Try Example"):
        st.session_state["input_text"] = "I love rainy season!"
        st.rerun()

text = st.session_state["input_text"]

# Analyze logic
if analyze_clicked:
    if not text.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_label = torch.argmax(probs, dim=1).item()
            confidence = torch.max(probs).item()
            emotion = label_encoder.inverse_transform([pred_label])[0]
            emoji = EMOJI_MAP.get(emotion, "")
            st.session_state.prediction = {
                "text": text, "probs": probs, "emotion": emotion, "confidence": confidence
            }
            if "history" not in st.session_state:
                st.session_state.history = []
            st.session_state.history.append({
                "Input": text, "Predicted Emotion": emotion,
                "Confidence (%)": f"{confidence * 100:.2f}%", "Emoji": emoji
            })

# Prediction Output
if "prediction" in st.session_state:
    pred = st.session_state.prediction
    st.markdown(f"""
    <div class="prediction">
        <h3>🎯 <b>Predicted Emotion:</b> {pred['emotion'].capitalize()} <span class="emoji-large">{EMOJI_MAP.get(pred['emotion'], "")}</span></h3>
        <h4>🔍 Confidence: {pred['confidence'] * 100:.2f}%</h4>
    </div>
    """, unsafe_allow_html=True)

    if st.checkbox("📊 Show probabilities for all emotions"):
        all_emotions = label_encoder.inverse_transform(list(range(len(pred['probs'][0]))))
        prob_dict = {label: float(prob) * 100 for label, prob in zip(all_emotions, pred['probs'][0])}
        df = pd.DataFrame({
            "Emotion": list(prob_dict.keys()),
            "Probability (%)": list(prob_dict.values()),
            "Emoji": [EMOJI_MAP.get(em, "") for em in prob_dict.keys()]
        }).sort_values("Probability (%)", ascending=False).reset_index(drop=True)
        st.bar_chart(df.set_index("Emotion")["Probability (%)"])
        st.download_button("📥 Download Predictions", df.to_csv(index=False).encode(), "emotion_probabilities.csv", "text/csv")

# History
if "history" in st.session_state and st.session_state.history:
    st.markdown("## 🕘 Prediction History (Recent)")
    history_df = pd.DataFrame(st.session_state.history[::-1])
    st.dataframe(history_df, use_container_width=True)

# Model Info
with st.expander("ℹ️ Model Info"):
    st.markdown("""
    - **Model Source:** Hugging Face 🤗  
    - **Architecture:** DistilBERT (fine-tuned)  
    - **Task:** Emotion Classification  
    - **Supported Emotions:** Joy, Sadness, Anger, Fear, Love, Surprise  
    - **Confidence Score:** Softmax probability of top class  
    - **Offline Mode:** Uses saved model in `./saved_model`  
    """)

# Close div
st.markdown("</div>", unsafe_allow_html=True)
