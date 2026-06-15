from sklearn.model_selection import train_test_split
import numpy as np

def preprocess_images(X):
    """
    Normalize and flatten images.

    Args:
        X (numpy.ndarray): Shape (n_samples, 28, 28)

    Returns:
        numpy.ndarray: Shape (n_samples, 784)
    """
    X = X.astype('float')/255.0
    X = X.reshape(X.shape[0], 28*28)

    return X

def validation_split(X_train, y_train):
    """
    Creates an Validation Set
    Returns:
         X_train, X_val, y_train, y_val
    """
    return train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=42,
        stratify=y_train
    )

if __name__ == "__main__":
    from tensorflow.keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = preprocess_images(X_train)
    X_test = preprocess_images(X_test)

    X_train, X_val, y_train, y_val = validatiion_split(X_train, y_train)

    print("Training Shape :", X_train.shape)
    print("Validation Shape :", X_val.shape)
    print("Test Shape :", X_test.shape)

    print("\nPixel Range:")
    print("Min:", np.min(X_train))
    print("Max:", np.max(X_train))