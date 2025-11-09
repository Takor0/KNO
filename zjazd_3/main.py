from pathlib import Path

import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
from keras import Sequential

from keras.src import layers
from keras.src.optimizers import Adam

from sklearn.model_selection import train_test_split
import sklearn
import matplotlib.pyplot as plt


def plot_loss(history):
    plt.plot(history.history["loss"], label="loss")
    plt.ylim([0, 2])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("first_curve.png")


def get_data():
    df = pd.read_csv("data/wine.csv")
    class_hot_one = pd.get_dummies(df["class"]).astype(int)
    df = df.drop(columns=["class"])
    df = df.sample(frac=1, random_state=42)
    class_hot_one = class_hot_one.loc[df.index]
    X_train, X_test, y_train, y_test = train_test_split(
        df, class_hot_one, test_size=0.2, random_state=42
    )

    scaler = sklearn.preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def first_model(X_train, y_train, recalculate=False):
    model_path = Path("./first_model.keras")
    if not model_path.exists() or recalculate:
        model = Sequential(
            [
                layers.Input(shape=(13,), name="input"),
                layers.Dense(32, activation="relu", name="hidden_1"),
                layers.Dense(3, activation="softmax", name="output"),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        history = model.fit(X_train, y_train, epochs=10, batch_size=32)
        plot_loss(history)
        model.save(model_path)
    else:
        return tf.keras.models.load_model(model_path)

    return model


def second_model(X_train, y_train, recalculate=False):
    model_path = Path("./second_model.keras")
    if not model_path.exists() or recalculate:
        model = Sequential(
            [
                layers.Input(shape=(13,), name="input"),
                layers.Dense(
                    64,
                    activation="relu",
                    name="hidden_1",
                    kernel_initializer="HeNormal",
                ),
                layers.Dropout(0.2, name="hidden_3"),
                layers.Dense(3, activation="softmax", name="output"),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        history = model.fit(X_train, y_train, epochs=10, batch_size=32)
        plot_loss(history)
        model.save(model_path)
    else:
        return tf.keras.models.load_model(model_path)

    return model


def get_model(name, X_train, y_train, recalculate=False):
    if name == "first":
        return first_model(
            X_train=X_train, y_train=y_train, recalculate=recalculate
        )
    if name == "second":
        return second_model(
            X_train=X_train, y_train=y_train, recalculate=recalculate
        )
    raise Exception("Model not found")


def main(arguments):
    X_train, X_test, y_train, y_test = get_data()
    model = get_model(
        name="second", X_train=X_train, y_train=y_train, recalculate=False
    )

    x = np.array([list(arguments.__dict__.values())])
    correct_count = 0
    for x_test, (_, y) in zip(X_test, y_test.iterrows()):
        test_in = np.array([x_test])
        probs = model.predict(test_in, verbose=0)[0]
        pred_label = np.argmax(probs) + 1
        actual_label = np.argmax(list(y)) + 1
        correct = pred_label == actual_label
        correct_count += 1 if correct else 0

        print(
            f"probs: {probs}, predicted: {pred_label}, actual class: {actual_label}, correct: {pred_label == actual_label}"
        )
    print(f"Accuracy: {correct_count / len(X_test) * 100:.2f} %")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-alcohol", type=float, default=0.0)
    parser.add_argument("--alcohol", type=float, default=0.0)
    parser.add_argument("--ash", type=float, default=0.0)
    parser.add_argument("--alcalinity_of_ash", type=float, default=0.0)
    parser.add_argument("--magnesium", type=float, default=0.0)
    parser.add_argument("--total_phenols", type=float, default=0.0)
    parser.add_argument("--flavanoids", type=float, default=0.0)
    parser.add_argument("--nonflavanoid_phenols", type=float, default=0.0)
    parser.add_argument("--proanthocyanins", type=float, default=0.0)
    parser.add_argument("--color_intensity", type=float, default=0.0)
    parser.add_argument("--hue", type=float, default=0.0)
    parser.add_argument("--od_diluted_wines", type=float, default=0.0)
    parser.add_argument("--proline", type=float, default=0.0)

    main(parser.parse_args())
