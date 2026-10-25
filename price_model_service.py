import tensorflow as tf
import pickle
import streamlit as st

@st.cache_resource
def load_price_model():
    model = tf.keras.models.load_model(
        "lstm_close_sentiment.h5",
        compile=False
    )

    scaler = pickle.load(open(
        "scaler_close_sentiment_2f.pkl", "rb"
    ))

    return model, scaler
