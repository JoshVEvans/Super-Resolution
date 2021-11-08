import tensorflow as tf


import numpy as np
import random
import cv2


# This method returns the batch of images using multiprocessing for faster interpolation
def load_multiprocessing(multiprocessing_pool, image_paths):
    batch = multiprocessing_pool.map(get_set, image_paths)

    X = []
    y = []

    for iteration in batch:
        X.append(iteration[0])
        y.append(iteration[1])

    # X = np.array(X) / 255
    # y = np.array(y) / 255

    X = np.array(X)
    y = np.array(y)

    X = (X - np.mean(X, axis=(0, 1, 2))) / np.std(X, axis=(0, 1, 2))
    y = (y - np.mean(y, axis=(0, 1, 2))) / np.std(y, axis=(0, 1, 2))

    return X, y


# This method returns the batch of images of length batch_size
def load(image_paths):
    X = []
    y = []

    for image_path in image_paths:
        a, b = get_set(image_path)

        X.append(a)
        y.append(b)

    # X = np.array(X) / 255
    # y = np.array(y) / 255

    X = np.array(X)
    y = np.array(y)

    X = (X - np.mean(X, axis=(0, 1, 2))) / np.std(X, axis=(0, 1, 2))
    y = (y - np.mean(y, axis=(0, 1, 2))) / np.std(y, axis=(0, 1, 2))

    return X, y


# This method returns a set of X, y images.
def get_set(image_path):
    dim = 64
    # Read Image from Path
    image = cv2.imread(image_path)

    # Crop / Resize Sample
    temp_dim = image.shape

    if temp_dim[0] >= dim and temp_dim[1] >= dim:
        # Crop Image
        image = np.array(tf.image.random_crop(image, size=(dim, dim, 3)))
    else:
        # Resize Image
        image = cv2.resize(image, (dim, dim))

    ### Data Augmentation ###
    # Horizontal Flipping
    if random.choice([True, False]):
        image = cv2.flip(image, 1)
    # Vertical Flipping
    if random.choice([True, False]):
        image = cv2.flip(image, 0)
    # Rotation
    if random.choice([True, False]):
        temp = random.choice([0, 1, 2])
        image = cv2.rotate(image, temp)

    # Store y
    y = image

    ### Image Interpolation ###
    # Random scaling with different scaling methods
    scale = random.choice([2, 3, 4])
    """
    0: Nearest-neighbor
    1: Bilinear
    2: Bicubic
    3: Area
    4: Lanczos
    """
    image = cv2.resize(
        image,
        (dim // scale, dim // scale),
        interpolation=cv2.INTER_AREA,
    )
    image = cv2.resize(image, (dim, dim), interpolation=random.choice([1, 2, 3, 4]))

    # Store X
    X = image

    return X, y
