from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

def load_mnist():
    """
    Load MNIST dataset from Keras.
    Returns:
        (X_train, y_train), (X_test, y_test)
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    return (X_train, y_train), (X_test, y_test)

def show_samples(X_train, y_train):
    plt.figure(figsize=(10,3))

    plt.imshow(X_train[0])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = load_mnist()

    print("\nFirst image shape:", X_train[0].shape)
    print("First label:", y_train[0])

    show_samples(X_train, y_train)