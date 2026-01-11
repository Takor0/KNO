from keras import layers, Input, Model
from keras import Sequential
from keras.preprocessing import image_dataset_from_directory
from keras.src.utils import save_img
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


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
    mapped_ds = dataset.map(lambda x: (norm_layer(x), norm_layer(x)))
    mapped_ds = mapped_ds.map(lambda x, y: (aug_layer(x), aug_layer(x)))

    encoder_model = Sequential([
        layers.Input(shape=(W, H, C)),
        layers.Flatten(),
        layers.Dense(ENCODING_DIM, activation='relu'),
        layers.Dense(INPUT_SIZE, activation='sigmoid'),
        layers.Reshape((W, H, C))
    ])

    encoder_model.compile(optimizer='adam', loss='binary_crossentropy')
    encoder_model.fit(
        mapped_ds,
        epochs=EPOCHS,
    )

    encoder = Model(encoder_model.input, encoder_model.layers[1].output)

    decoder_input = Input(shape=(ENCODING_DIM,))
    x = encoder_model.layers[2](decoder_input)
    decoder_output = encoder_model.layers[3](x)
    generator = Model(decoder_input, decoder_output)


    test_batch = next(iter(mapped_ds))[0]
    single_img = test_batch[0:1]

    actual_code = encoder.predict(single_img)
    reconstructed = generator.predict(actual_code)
    save_img("generated/overfitted_pokemon.png", reconstructed[0])


    import numpy as np
    random_variation = actual_code + 0.1 * np.random.normal(size=(1, ENCODING_DIM))
    new_img = generator.predict(random_variation)
    save_img("generated/random_new.png", new_img[0])

main()