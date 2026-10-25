import numpy as np
from sklearn.preprocessing import MinMaxScaler

def scale_features(df):
    features = df[['close', 'news_sentiment_score']].values
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, scaler


def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size, 0])  # close price
    return np.array(X), np.array(y)
