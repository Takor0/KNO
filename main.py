import argparse
from pathlib import Path

import tensorflow as tf


def get_or_create_mnist_model(x_train, y_train):
    model_path = Path("mnist_model.keras")
    if model_path.exists():
        return tf.keras.models.load_model(model_path)
    else:
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            x_train, y_train, epochs=5
        )  # użyj verbose=0 jeśli jest problem z konsolą
        model.save(model_path)
        return model


def main(path_to_image: str):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = get_or_create_mnist_model(x_train, y_train)

    model.evaluate(x_test, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Number detector",
        description="Detects handwritten numbers from 0 to 9",
    )

    parser.add_argument("filename")

    args = parser.parse_args()

    main(args.filename)
