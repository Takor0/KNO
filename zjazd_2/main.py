import argparse

import tensorflow as tf
import numpy as np


@tf.function
def rotate(x: float, y: float, degrees=90):
    tensor = tf.constant([[x, y]], dtype=tf.float32)
    d_tensor = tf.constant(
        [
            np.cos(np.radians(degrees)),
            -np.sin(np.radians(degrees)),
            np.sin(np.radians(degrees)),
            np.cos(np.radians(degrees)),
        ],
        shape=(2, 2),
        dtype=tf.float32,
    )

    return tf.matmul(tensor, d_tensor)


@tf.function
def calc_linear(tensor_x, tensor_y):
    def ret_ok():
        rev_x = tf.linalg.inv(tensor_x)
        return tf.matmul(rev_x, tensor_y)

    def ret_no():
        tf.print("Brak jednego rozwiązania")
        return tf.constant([0], dtype=tf.float32)

    return tf.cond(tf.equal(tf.linalg.det(tensor_x), 0), ret_no, ret_ok)


def main():
    x, y = 0, 2
    # print(rotate(x, y, 90))

    tx = lambda lst: tf.constant(lst, dtype=tf.float32)
    ty = lambda lst: tf.constant(lst, shape=(len(lst), 1), dtype=tf.float32)

    list_x = [[3, 2, 1], [-1, 0, 2], [1, -2, -1]]
    list_y = [4, 4, -4]
    print(calc_linear(tx(list_x), ty(list_y)))

    list_x = [[3, -1, 2], [4, 2, -5], [2, -7, 11]]
    list_y = [0, 0, 0]

    print(calc_linear(tx(list_x), ty(list_y)))

    N = 3
    list_x = []
    list_y = []

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n")
    parser.add_argument(
        "-x", "--xvalues", help="List of arguments for X and y, separated by ;"
    )

    args = parser.parse_args()
    x = args.xvalues
    xvalues = x.split(";")
    for xval in xvalues:
        parts = xval.split(" ")
        X = [float(part) for part in parts[:-1]]
        y = float(parts[-1])

        if len(X) != int(args.n):
            print("Błędna liczba elementów w argumencie")
            exit(1)
        list_x.append(X)
        list_y.append([y])

    print(calc_linear(list_x, list_y))


if __name__ == "__main__":
    main()
