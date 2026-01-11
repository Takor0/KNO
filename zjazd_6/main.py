from keras import layers, Input, Model
from keras import Sequential
from keras.preprocessing import image_dataset_from_directory
from keras.src.utils import save_img
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

def main():
    W = 120
    H = 120
    C = 3
    ENCODING_DIM = 64
    INPUT_SIZE = W * H * C
    BATCH_SIZE = 5
    EPOCHS = 50
    dataset = image_dataset_from_directory(
        "dataset",
        batch_size=BATCH_SIZE,
        image_size=(120, 120),
        labels=None,
        label_mode=None
    )

    norm_layer = layers.Rescaling(1./255)
    aug_layer = Sequential([
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ])

    mapped_ds = dataset.map(lambda x: (aug_layer(norm_layer(x)), norm_layer(x)))

    encoder = Sequential([
        layers.Input(shape=(W, H, C)),
        layers.Flatten(),
        layers.Dense(ENCODING_DIM, activation='relu')
    ], name="encoder")

    decoder = Sequential([
        layers.Input(shape=(ENCODING_DIM,)),
        layers.Dense(INPUT_SIZE, activation='sigmoid'),
        layers.Reshape((W, H, C))
    ], name="decoder")

    autoencoder_input = layers.Input(shape=(W, H, C))
    encoded_repr = encoder(autoencoder_input)
    decoded_repr = decoder(encoded_repr)
    autoencoder = Model(autoencoder_input, decoded_repr)

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    autoencoder.fit(
        mapped_ds,
        epochs=50
    )

    random_code = np.random.normal(size=(1, ENCODING_DIM))
    new_img = decoder.predict(random_code)
    save_img("generated/random_from_scratch.png", new_img[0])

main()