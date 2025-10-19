import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(400, 400, 3)),

    # konwolucyjne
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

def load_images(image_path: str, subset: str):
    ds = tf.keras.utils.image_dataset_from_directory(
        image_path,
        validation_split=0.2,
        subset=subset,
        seed=123,
        image_size=(400, 400),
        batch_size=32
    )
    return ds

train_ds = load_images("PetImages/", "training")

val_ds = load_images("PetImages/", "validation")



history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

model.save("pet_classifier.keras")