import tensorflow as tf
from tensorflow.keras.models import Sequential

def build_model():
    """ Creates and Complie the model"""
    model = Sequential([
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(128, activation = "relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer = 'Adam',
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
        )

    return model

if __name__ == "__main__" :
    model = build_model()
    model.summary()

  
