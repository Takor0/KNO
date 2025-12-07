from keras.src import datasets
import keras_tuner as kt
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.src import layers
from keras.src.optimizers import Adam
from keras.src.models import Sequential

def train_model(X_train, y_train):
    pass


def get_conv_model():
    pass


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
        model.add(layers.Dense(units=hp_units_2, activation="relu", name="optional_hidden_dense_2"))

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

def main():
    fashion_mnist = datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    tuner = kt.Hyperband(
        get_dense_model,
        objective="val_accuracy",
        max_epochs=10,
        factor=3,
        directory="tuner_results",
        project_name="wines",
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_accuracy',
        mode='max',
        patience=3,
        verbose=1
    )

    tuner.search(
        train_images,
        train_labels,
        epochs=5,
        validation_split=0.2,
        callbacks=[early_stopping_callback],
        batch_size=512,
    )
    tuner.results_summary()
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)


if __name__ == '__main__':
    main()