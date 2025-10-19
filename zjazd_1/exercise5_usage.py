import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("pet_classifier.keras")


def pred(fname):
    some_animal = tf.keras.utils.load_img(fname, target_size=(400, 400))
    input_arr = np.array([some_animal])
    # input_arr = input_arr / 255.0
    probs = model.predict(input_arr, verbose=0)
    cls = ["Cat", "Dog"][np.argmax(probs)]
    print(probs, "->", cls)


pred("jh.jpg")
pred("dog.jpg")
pred("dog2.jpg")
pred("cat.jpg")
pred("human.jpg")
