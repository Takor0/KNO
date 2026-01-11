import argparse
import numpy as np
import keras
from PIL import Image, ImageOps

CLASS_NAMES = [
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def preprocess_image(image_path, debug_save_path="compressed.jpg"):
    img = Image.open(image_path)
    img = img.convert("L")
    img = ImageOps.invert(img)

    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    img.save(debug_save_path)
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    return img_array


def main():
    parser = argparse.ArgumentParser(description="klasyfikacja ubrania")
    parser.add_argument("image_path", type=str, help="ściezka do obrazka")
    parser.add_argument(
        "--model",
        type=str,
        default="conv_model.keras",
        help="ścieżka do modelu keras",
    )
    args = parser.parse_args()
    processed_image = preprocess_image(args.image_path)

    print(f"ładowanie modelu: {args.model}")
    model = keras.models.load_model(args.model)

    probs = model.predict(processed_image, verbose=0)
    predicted_idx = np.argmax(probs[0])
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = probs[0][predicted_idx]

    print("-" * 30)
    print(f"input: {args.image_path}")
    print(f"klasyfikacja: {predicted_class}")
    print(f"pewnośc:  {confidence:.2%}")
    print("-" * 30)


if __name__ == "__main__":
    main()
