"""
Main Pipeline for MNIST DNN Project
"""


from data_loader import load_mnist
from preprocessing import preprocess_images, validation_split
from model import build_model
from train import train_model
from evaluate import evaluate_model, plot_confusion_matrix, plot_training_history

def main():
    # Load Data
    print("\nLoading MNIST Dataset...")
    (X_train, y_train), (X_test, y_test) = load_mnist()

    # Preprocess Data
    print("\nPreprocessing Data...")
    X_train = preprocess_images(X_train)
    X_test = preprocess_images(X_test)
    X_train, X_val, y_train, y_val = validation_split(X_train, y_train)

    # Build Model
    print("\nBuilding DNN Model...")
    model = build_model()
    model.summary()

    # Training Model
    print("Training Model...")
    history = train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=20,
        batch_size=32
    )

    # Evaluate Model
    print("\nEvaluating Model...")
    y_pred = evaluate_model(model, X_test, y_test)

    # Visualization
    plot_confusion_matrix(y_test, y_pred)
    plot_training_history(history)

    # Save Final Model
    model.save("models/final_model.keras")

    print("\nModel saved successfully!")
    print("Location: models/final_model.keras")

if __name__ == "__main__":
    main()