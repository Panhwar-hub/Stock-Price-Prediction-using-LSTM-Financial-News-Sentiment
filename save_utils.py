import pickle

def save_model(model, model_path="lstm_close_sentiment.h5"):
    model.save(model_path)


def save_scaler(scaler, scaler_path="scaler_close_sentiment.pkl"):
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
