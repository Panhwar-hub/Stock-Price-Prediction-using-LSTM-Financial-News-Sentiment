def train_model(model, X, y, split_ratio=0.8, epochs=20, batch_size=32):
    split = int(len(X) * split_ratio)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test)
    )

    return model, X_test, y_test, history, split
