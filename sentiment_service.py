import torch
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

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
        device=device
    )

    return sentiment_pipe


def compute_sentiment_score(sentiment_pipe, title, content):
    text = (title.strip() + ". " + content.strip()).strip()

    result = sentiment_pipe(text)[0]

    label = result["label"].lower()
    prob = float(result["score"])

    if "positive" in label:
        return prob
    elif "negative" in label:
        return -prob
    return 0.0
