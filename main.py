import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def plot_loss(history):
    plt.plot(history.history["loss"], label="loss")
    plt.ylim([0, 1])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("fir_curve.png")


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
        history = model.fit(x_train, y_train, epochs=5)
        plot_loss(history)
        model.save(model_path)
        return model


def get_image_as_tensor(path_to_image: str) -> np.ndarray:
    input_arr = tf.keras.utils.load_img(
        path_to_image, color_mode="grayscale", target_size=(28, 28)
    )
    input_arr = np.array([input_arr])
    return input_arr


def main(path_to_image: str):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = get_or_create_mnist_model(x_train, y_train)

    x = get_image_as_tensor(path_to_image)
    probs = model.predict(x, verbose=0)[0]
    print(
        "Twoja liczba to:",
        np.argmax(probs),
        "z prawdopodobieństwem:",
        probs[np.argmax(probs)],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Number detector",
        description="Detects handwritten numbers from 0 to 9",
    )

    parser.add_argument("filename")

    args = parser.parse_args()

    main(args.filename)
