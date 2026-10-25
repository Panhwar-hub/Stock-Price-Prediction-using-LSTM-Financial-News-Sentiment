from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def build_lstm(window_size, num_features=2):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(window_size, num_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model
