# Stock Price Prediction using LSTM & Financial News Sentiment

This project is a real-time stock price prediction web application built with Streamlit, combining:

- LSTM deep learning model for time-series price forecasting  
- FinBERT sentiment analysis for financial news  
- Multivariate prediction using price and sentiment  
- Interactive web-based UI  

The system takes the latest stock prices and news input, computes news sentiment, and predicts the next closing price using a trained LSTM model.

---

## Features

- Multivariate LSTM model (Close Price + Sentiment)
- Financial news sentiment using ProsusAI/FinBERT
- Real-time user input
- Scaled data processing using MinMaxScaler
- Streamlit web interfaces
- Automatic sentiment extraction from user-entered news
- Live prediction output

---

## Model Overview

| Component | Description |
|----------|-------------|
| LSTM Model | Predicts next stock closing price |
| Input Window | 10 historical time steps |
| Features | Close Price, Sentiment Score |
| Sentiment Model | ProsusAI/FinBERT |
| Scaling | MinMaxScaler (2 features) |

---

## Usage

1. Enter number of new closing prices (1–3)
2. Input the latest closing price values
3. Enter news title and content
4. Click "Predict Next Close Price"
5. View:
   - Sentiment Score
   - Raw Model Output
   - Final Predicted Close Price

---

## Setup and run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the FinBERT model from Hugging Face. The app expects the
trained LSTM model, scaler, and processed input CSV to be present locally.
Those large/generated assets are intentionally ignored by Git; document a
download location or provide a small public sample before sharing.

## Training

To reproduce the training workflow, place `sentements_data_processed.csv` in
the repository root and run:

```bash
python main.py
```

The workflow uses a 10-step window over close price and sentiment features.
