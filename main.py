"""Train and evaluate the LSTM model from the processed sentiment dataset."""

from dataloader import load_data
from preprocessing import create_sequences, scale_features
from model import build_lstm
from train import train_model
from evaluate import plot_predictions
from save_utils import save_model, save_scaler

# ==========================
# CONFIG
# ==========================
CSV_PATH = "sentements_data_processed.csv"
WINDOW_SIZE = 10
EPOCHS = 20
BATCH_SIZE = 32

# ==========================
# PIPELINE
# ==========================
def main():
    """Run the end-to-end training workflow."""
    dataframe = load_data(CSV_PATH)
    scaled_features, scaler = scale_features(dataframe)
    features, target = create_sequences(scaled_features, WINDOW_SIZE)
    model = build_lstm(WINDOW_SIZE)
    model, x_test, y_test, _, split = train_model(
        model, features, target, epochs=EPOCHS, batch_size=BATCH_SIZE
    )
    plot_predictions(model, x_test, y_test, scaler, dataframe, WINDOW_SIZE, split)
    save_model(model)
    save_scaler(scaler)


if __name__ == "__main__":
    main()
