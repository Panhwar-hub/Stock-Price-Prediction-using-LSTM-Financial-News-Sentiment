"""Streamlit interface for LSTM stock-price prediction with FinBERT sentiment."""

import base64
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


PROJECT_ROOT = Path(__file__).resolve().parent

# ====================================
#        1) LOAD LSTM & SCALER
# ====================================

@st.cache_resource
def load_price_model():
    model = tf.keras.models.load_model(
        PROJECT_ROOT / "lstm_close_sentiment.h5",
        compile=False
    )
    with (PROJECT_ROOT / "scaler_close_sentiment_2f.pkl").open("rb") as file:
        scaler = pickle.load(file)
    return model, scaler

model, scaler = load_price_model()


# ====================================
#      2) LOAD SENTIMENT MODEL
# ====================================

MODEL_NAME = "ProsusAI/finbert"

@st.cache_resource
def load_sentiment_model():
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    return sentiment_pipe

sentiment_pipe = load_sentiment_model()


# ====================================
# 3) SENTIMENT SCORE FUNCTION (FAST)
# ====================================

def compute_sentiment_score(title, content):
    text = (title.strip() + ". " + content.strip()).strip()

    result = sentiment_pipe(text)[0]  # {'label': 'POSITIVE', 'score': 0.98}

    label = result["label"].lower()
    prob = float(result["score"])

    if "positive" in label:
        return prob
    if "negative" in label:
        return -prob
    return 0.0


# ====================================
#            4) STREAMLIT UI
# ====================================
st.markdown(
    """
    <style>
    /* Title */
    h1 {
        color: black !important;
    }

    /* Subheaders (Streamlit-safe targeting) */
    [data-testid="stSubheader"] {
        color: black !important;
    }

    /* Labels */
    label {
        color: black !important;
        font-weight: 600;
    }

    /* Inputs */
    input, textarea {
        color: black !important;
        background-color: rgba(255,255,255,0.85) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def set_bg_image(image_path):
    """Apply a local image as the Streamlit application background."""
    with image_path.open("rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ✅ Set your background image here
set_bg_image(PROJECT_ROOT / "4Tmn0.png")

st.title("📈 Stock Price Prediction (LSTM + News Sentiment)")

# --- Load Data ---
data = pd.read_csv(PROJECT_ROOT / "sentements_data_processed.csv")
data = data[['close', 'sentiment']]

# Last 10 historical points
window_size = 10
df_hist = data[['close', 'sentiment']].tail(window_size).reset_index(drop=True)

# --- User Inputs ---

# Make headings black
st.markdown("<h3 style='color:black;'>Enter new close price points (1–3):</h3>", unsafe_allow_html=True)

num_new = st.number_input(
    "Number of new points:", min_value=1, max_value=3, value=1, step=1
)

recent_close = []
for i in range(num_new):
    val = st.number_input(f"New close {i+1}", value=0.0, step=0.01, key=f"new_close_{i}")
    recent_close.append(val)

# Second heading (black)
st.markdown("<h3 style='color:black;'>Enter latest news</h3>", unsafe_allow_html=True)

title = st.text_input("News Title")
content = st.text_area("News Content")

# --- Prediction ---
if st.button("Predict Next Close Price"):
    
    # ---- Compute sentiment ----
    sentiment_score = compute_sentiment_score(title, content)
    
    # ---- Combine historical + new points to match window_size ----
    hist_needed = window_size - num_new
    hist_close = df_hist['close'].tolist()[-hist_needed:]
    hist_sentiment = df_hist['sentiment'].tolist()[-hist_needed:]
    
    full_close = hist_close + recent_close
    full_sentiment = hist_sentiment + [sentiment_score]*num_new
    
    # ---- Prepare LSTM input ----
    input_data = np.array([[c, s] for c, s in zip(full_close, full_sentiment)])
    
    # ---- Scale features ----
    input_scaled = scaler.fit_transform(input_data).reshape(1, window_size, 2)
    
    # ---- Predict ----
    pred_scaled = model.predict(input_scaled)
    
    # ---- Inverse transform ----
    pred_input = np.hstack([pred_scaled, np.zeros((1, 1))])
    predicted_price = scaler.inverse_transform(pred_input)[0, 0]
    
    st.success(f"Predicted Next Close Price: **{predicted_price:.2f}**")

