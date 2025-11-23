from pathlib import Path

import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
from keras import Sequential

from keras.src import layers
from keras.src.optimizers import Adam

from sklearn.model_selection import train_test_split
from matplotlib.offsetbox import AnchoredText


def plot_loss(
        history, batch_size, epochs, learning_rate, loc="upper right"
):
    import matplotlib.pyplot as plt

    for key, label in [
        ("accuracy", "Accuracy"),
        ("loss", "Loss"),
    ]:
        savepath = f"{key}_curve.png"

        fig, ax = plt.subplots(constrained_layout=True)

        ax.plot(history[key], label=label)

        ax.set_ylim(0, 2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

        text = f"batch_size={batch_size}\nlr={learning_rate}\nepochs={epochs}"
        at = AnchoredText(
            text, loc=loc, pad=0.3, borderpad=0.4, prop={"family": "monospace"}
        )
        at.patch.set_facecolor("white")
        at.patch.set_alpha(0.85)
        at.patch.set_edgecolor("gray")
        ax.add_artist(at)

        fig.savefig(savepath, dpi=150)
        plt.close(fig)


def get_data():
    df = pd.read_csv("data/wine.csv")
    class_hot_one = pd.get_dummies(df["class"]).astype(int)
    df = df.drop(columns=["class"])
    df = df.sample(frac=1, random_state=42)
    class_hot_one = class_hot_one.loc[df.index]
    X_train, X_test, y_train, y_test = train_test_split(
        df, class_hot_one, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def second_model(norm_layer):
    model = Sequential(
        [
            norm_layer,
            layers.Dense(
                16,
                activation="relu",
                name="hidden_5",
                kernel_initializer="HeNormal",
            ),
            layers.Dropout(0.2, name="hidden_6"),
            layers.Dense(3, activation="softmax", name="output"),
        ]
    )

    return model


def get_model(
        X_train,
        y_train,
        recalculate=True,
        batch_size=32,
        epochs=10,
        learning_rate=0.001,
        norm_layer=None

):
    model_path = Path(f"OLD/model.keras")
    if not model_path.exists() or recalculate:
        model = second_model(
            norm_layer
        )

        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        fit_result = model.fit(
            X_train, y_train, epochs=epochs, batch_size=batch_size
        )

        plot_loss(
            fit_result.history,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
        )
        model.save(model_path)
    else:
        return tf.keras.models.load_model(model_path)

    return model


def main(arguments):
    X_train, X_test, y_train, y_test = get_data()

    adapt_data = np.array(X_train)
    norm_layer = layers.Normalization()
    norm_layer.adapt(adapt_data)



    model = get_model(
        X_train=X_train,
        y_train=y_train,
        recalculate=True,
        batch_size=32,
        epochs=30,
        learning_rate=0.001,
        norm_layer=norm_layer
    )

    x = np.array([list(arguments.__dict__.values())])
    probs = model.predict(x, verbose=0)[0]
    print("predicted class: ", np.argmax(probs) + 1)

    # exit()
    correct_count = 0

    for x_test, (_, y) in zip(X_test.itertuples(), y_test.iterrows()):
        x_test_list = list(x_test)
        x_test_list.pop(0)
        test_in = np.array([list(x_test_list)])
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
