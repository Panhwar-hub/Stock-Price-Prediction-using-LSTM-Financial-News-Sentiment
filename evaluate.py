import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(actual_prices, predicted_prices):
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    mae = mean_absolute_error(actual_prices, predicted_prices)
    r2 = r2_score(actual_prices, predicted_prices)
    mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100

    print("\n📊 Model Evaluation Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")

    return rmse, mae, r2, mape


def plot_predictions(model, X_test, y_test, scaler, df, window_size, split):
    predicted = model.predict(X_test)

    predicted_prices = scaler.inverse_transform(
        np.concatenate([predicted, np.zeros((predicted.shape[0],1))], axis=1)
    )[:,0]

    actual_prices = scaler.inverse_transform(
        np.concatenate([y_test.reshape(-1,1), np.zeros((y_test.shape[0],1))], axis=1)
    )[:,0]

    compute_metrics(actual_prices, predicted_prices)

    plt.figure(figsize=(12,6))
    plt.plot(df['datetime'][window_size+split:], actual_prices, label='Actual')
    plt.plot(df['datetime'][window_size+split:], predicted_prices, label='Predicted')
    plt.xlabel('Datetime')
    plt.ylabel('Close Price')
    plt.title('Actual vs Predicted Close Prices with News Sentiment')
    plt.legend()
    plt.show()
