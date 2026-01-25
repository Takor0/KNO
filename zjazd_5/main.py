import numpy as np
from keras.src import datasets
import keras_tuner as kt
from keras.callbacks import EarlyStopping
from keras.src import layers
from keras.src.optimizers import Adam
from keras.src.models import Sequential
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import keras
from matplotlib.offsetbox import AnchoredText


BATCH_SIZE = 2048
EPOCHS = 100
keras.mixed_precision.set_global_policy('mixed_float16')

def plot_loss(history, epochs, learning_rate, name, loc="upper right"):
    metrics_data = history.history

    for key, label in [
        ("accuracy", "Accuracy"),
        ("loss", "Loss"),
    ]:
        if key not in metrics_data:
            continue

        savepath = f"{name}_{key}_curve.png"
        fig, ax = plt.subplots(constrained_layout=True)

        ax.plot(metrics_data[key], label=f"Training {label}")

        val_key = f"val_{key}"
        if val_key in metrics_data:
            ax.plot(
                metrics_data[val_key],
                label=f"Validation {label}",
                linestyle="--",
            )

        if key == "accuracy":
            ax.set_ylim(0, 1.05)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

        text = f"batch_size={BATCH_SIZE}\nlr={learning_rate}\nepochs={epochs}"
        at = AnchoredText(
            text, loc=loc, pad=0.3, borderpad=0.4, prop={"family": "monospace"}
        )
        at.patch.set_facecolor("white")
        at.patch.set_alpha(0.85)
        at.patch.set_edgecolor("gray")
        ax.add_artist(at)

        fig.savefig(savepath, dpi=150)
        plt.close(fig)


def get_convolutional_model(hp):
    model = Sequential()

    hp_rotation = hp.Float("aug_rotation", min_value=0.0, max_value=0.1, step=0.01)
    model.add(layers.RandomRotation(factor=hp_rotation, input_shape=(28, 28, 1)))

    hp_zoom = hp.Float("aug_zoom", min_value=0.0, max_value=0.15, step=0.01)
    model.add(layers.RandomZoom(height_factor=hp_zoom, width_factor=hp_zoom))

    hp_trans = hp.Float("aug_translation", min_value=0.0, max_value=0.1, step=0.01)
    model.add(layers.RandomTranslation(height_factor=hp_trans, width_factor=hp_trans))

    #L1
    hp_conv_1 = hp.Int("conv_1", min_value=32, max_value=256, step=32)
    hp_filters_1 = hp.Choice("filter_size_1", values=[3, 5])
    model.add(
        layers.Conv2D(hp_conv_1, hp_filters_1, padding='same',name="input_conv_1",activation="relu"),
    )
    model.add(
        layers.MaxPooling2D(name="hidden_pool_1")
    )

    #L2
    hp_conv_2 = hp.Int("conv_2", min_value=32, max_value=256, step=32)
    hp_filters_2 = hp.Choice("filter_size_2", values=[3, 5])
    model.add(
        layers.Conv2D(hp_conv_2,hp_filters_2, padding='same',name="hidden_conv_2",activation="relu"),
    )
    model.add(
        layers.MaxPooling2D(name="hidden_pool_2")
    )

    #L3
    hp_conv_3 = hp.Int("conv_3", min_value=32, max_value=256, step=32)
    hp_filters_3 = hp.Choice("filter_size_3", values=[3, 5])
    model.add(
        layers.Conv2D(hp_conv_3,hp_filters_3, padding='same',name="hidden_conv_3", activation="relu"),
    )
    model.add(
        layers.MaxPooling2D(name="hidden_pool_3")
    )

    model.add(
        layers.Flatten()
    )
    model.add(
        layers.Dense(10, activation='softmax'),
    )

    hp_learning_rate = hp.Choice(
        "learning_rate", values=[1e-1, 1e-2, 1e-3, 1e-4]
    )

    model.compile(
        optimizer=Adam(learning_rate=hp_learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=True
    )

    return model



def get_dense_model(hp):
    model = Sequential()

    model.add(
        layers.Flatten(input_shape=(28, 28)),
    )

    hp_units_1 = hp.Choice("units_1", values=[64, 128, 256, 512])
    model.add(
        layers.Dense(units=hp_units_1, activation="relu", name="hidden_dense_1")
    )

    if hp.Boolean("is_layer_2"):
        hp_units_2 = hp.Choice("optional_units_2", values=[64, 128, 256])
        model.add(
            layers.Dense(
                units=hp_units_2,
                activation="relu",
                name="optional_hidden_dense_2",
            )
        )

    model.add(
        layers.Dense(10, activation="softmax", name="output"),
    )
    hp_learning_rate = hp.Choice(
        "learning_rate", values=[1e-1, 1e-2, 1e-3, 1e-4]
    )

    model.compile(
        optimizer=Adam(learning_rate=hp_learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_tuned_model(
    get_model_func,
    train_images,
    train_labels,
    name,
    project_name="wines",
):
    tuner = kt.Hyperband(
        get_model_func,
        objective="val_accuracy",
        max_epochs=EPOCHS,
        factor=3,
        directory="tuner_results",
        project_name=project_name,
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_accuracy", patience=2, verbose=1
    )

    tuner.search(
        train_images,
        train_labels,
        epochs=5,
        validation_split=0.2,
        callbacks=[early_stopping_callback],
        batch_size=BATCH_SIZE,
    )
    tuner.results_summary()
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)

    history = model.fit(
        train_images,
        train_labels,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[early_stopping_callback],
    )

    plot_loss(
        history=history,
        epochs=EPOCHS,
        learning_rate=best_hps.get("learning_rate"),
        name=name,
    )

    return model

def main():
    fashion_mnist = datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = (
        fashion_mnist.load_data()
    )
    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    #Dense
    # dense_model = get_tuned_model(
    #     get_model_func=get_dense_model,
    #     train_images=train_images,
    #     train_labels=train_labels,
    #     name="dense",
    # )
    # probs = dense_model.predict(test_images)
    # y_pred = np.argmax(probs, axis=1)
    # cm = confusion_matrix(test_labels, y_pred)
    # dense_model.save("dense_model.keras")

    #Conv
    conv_model = get_tuned_model(
        get_model_func=get_convolutional_model,
        train_images=train_images,
        train_labels=train_labels,
        name="conv",
        project_name="conv",
    )

    probs = conv_model.predict(test_images)
    y_pred = np.argmax(probs, axis=1)
    cm = confusion_matrix(test_labels, y_pred)
    conv_model.save("conv_model.keras")




if __name__ == "__main__":
    main()