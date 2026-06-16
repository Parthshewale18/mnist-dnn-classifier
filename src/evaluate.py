import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

def evaluate_model(model, X_test, y_test):
    y_pred_proba = model.predict(X_test)
    y_pred = y_pred_proba.argmax(axis=1)

    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 50)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("=" * 50)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return y_pred

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

def plot_training_history(history):
    """
    Plot accuracy and loss curves.
    """

    # Accuracy Plot
    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()

    # Loss Plot
    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()
