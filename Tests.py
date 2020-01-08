from datetime import datetime

import numpy as np
import cv2
import Images
from glob import glob
import os
from fractions import Fraction
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def test():
    """arr = np.array([[1,2,3], [4,5,6]])
    arr = np.vstack([arr, [7,8,9]])
    print(arr)
    arr = np.concatenate((arr, np.array([[7,8,9]])), axis=0)
    print(arr)
    """

    arr = np.array([[1], [0]])
    arr = np.vstack([arr, [1]])
    print(arr)

    """
    # tf.compat.v1.disable_eager_execution()
    x = tf.placeholder(dtype="float", shape=None)  # tfp.placeholder(dtype="float", shape=None)
    y = x * 2

    with tf.Session() as session:  # tfp.Session() as session:
        result = session.run(y, feed_dict={x: [1, 2, 3]})
        print(result)
    """


def main():
    images = Images.ImagesController(vehicles_path="C:\\Users\\orik\\DLImages\\vehicles\\",
                                     non_vehicles_path="C:\\Users\\orik\\DLImages\\non-vehicles\\",
                                     batch_size=100)
    ff = images.get_next_batch()
    file = images._get_next_path(True)
    f = cv2.imread(file)
    cv2.imshow("image1", f)
    f2 = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    cv2.imshow("image2", f2)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    # arr = np.array(f)
    # i=1


def test2():
    arr = np.array([[1, 2], [3, 4]])
    var = np.var(arr)
    mean = np.mean(arr)
    # arr = arr * 2
    arr = (arr - mean) / var
    i = 1


def test3():
    img = np.array(cv2.cvtColor(cv2.imread("C:\\Users\\orik\\Tests\\1.jpg"), cv2.COLOR_BGR2GRAY)).astype(float)
    normalized_img = np.zeros((480, 852), dtype=float)
    normalized_img = cv2.normalize(img, normalized_img, 0.0, 1.0, cv2.NORM_MINMAX, dtype=-1)
    cv2.imshow("image1", img)
    # cv2.imshow("image2", normalized_img)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    cv2.destroyAllWindows()


def rename_images():
    for f in glob(pathname="C:\\Users\\orik\\DLImages\\vehicles\\Train\\image*.png", recursive=True):
        os.rename(src=f, dst=f.replace("image", "Right", 1))


def fract():
    fr = Fraction(2, 4)
    i = 1


if __name__ == '__main__':
    # fract()
    print(datetime.now())
