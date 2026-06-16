from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=20,
    batch_size=32
):

    checkpoint = ModelCheckpoint(
        filepath="models/best_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )

    return history