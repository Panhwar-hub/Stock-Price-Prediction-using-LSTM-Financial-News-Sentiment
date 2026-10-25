import numpy as np

def prepare_lstm_input_full_user(
    user_close_values,
    sentiment_score,
    scaler,
    window_size
):
    """
    Takes exactly `window_size` close values from user
    and applies the same sentiment score to all.
    """

    full_sentiment = [sentiment_score] * window_size

    input_data = np.array([
        [c, s] for c, s in zip(user_close_values, full_sentiment)
    ])

    input_scaled = scaler.transform(input_data).reshape(1, window_size, 2)

    return input_scaled


def predict_price(model, input_scaled, scaler):
    pred_scaled = model.predict(input_scaled)
    pred_input = np.hstack([pred_scaled, np.zeros((1, 1))])
    predicted_price = scaler.inverse_transform(pred_input)[0, 0]
    return predicted_price
