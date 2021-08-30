import Utils
import evaluate
import GAN_Network as Network

from keras.utils import plot_model
from keras import backend as K
import tensorflow as tf

import multiprocessing as mp
from tqdm import trange
import importlib
import numpy as np
import random
import cv2

import os


def augment_image(image):
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

    return image


dim = 64


def get_LR(image):

    """
    # Read image from path
    image = cv2.imread(image_path)
    # Crop / Resize Sample
    temp_dim = image.shape

    if temp_dim[0] >= dim and temp_dim[1] >= dim:
        # Crop Image
        image = np.array(tf.image.random_crop(image, size=(dim, dim, 3)))
    else:
        # Resize Image
        image = cv2.resize(image, (dim, dim))

    # Augment Image
    image = augment_image(image)"""

    # Image Interpolation
    scale = 2
    image = cv2.resize(
        image,
        (dim // scale, dim // scale),
        interpolation=random.choice([0, 1, 2, 3, 4]),
    )

    return image


def get_HR(image_path):

    # Read image from path
    image = cv2.imread(image_path)
    # Crop / Resize Sample
    temp_dim = image.shape

    if temp_dim[0] >= dim and temp_dim[1] >= dim:
        # Crop Image
        image = np.array(tf.image.random_crop(image, size=(dim, dim, 3)))
    else:
        # Resize Image
        image = cv2.resize(image, (dim, dim))

    # Augment Image
    image = augment_image(image)

    return image


def training(dataset_path, epochs=1, batch_size=128):
    # Load Paths
    image_paths = list(os.listdir(dataset_path))
    image_paths = np.array(
        [f"{dataset_path}{image_path}" for image_path in image_paths]
    )

    # Create GAN
    generator = Network.create_generator()
    generator.summary()
    discriminator = Network.create_disciminator()
    discriminator.summary()
    gan = Network.create_gan(discriminator, generator)
    gan.summary()

    # Graph Model
    plot_model(generator, to_file="generator.png")
    plot_model(discriminator, to_file="discriminator.png")
    plot_model(gan, to_file="gan.png")

    # Parameters
    discrimator_train_steps = 1  # Number of times to train discriminator
    generator_train_steps = 1  # Number of times to train discriminator
    iteration = 0
    steps_per_epoch = 100

    # Metrics
    metrics = []

    # Multiprocessing
    workers = mp.cpu_count()
    if batch_size <= mp.cpu_count():
        workers = batch_size
    workers = 4
    print(f"Workers: {workers}")
    p = mp.Pool(workers)

    for e in range(1, epochs):
        print("-" * 25 + f"Epoch: {e}/{epochs}" + "-" * 25)
        print("-" * 10 + f"Iteration: {iteration}" + "-" * 10)
        # Parameters
        m_loss = 0
        generator_loss = 0
        discriminator_loss = 0

        tr = trange(steps_per_epoch, desc=f"Epoch: {e}", leave=True)
        for i in tr:

            ### Learning Rate Scheduler ###
            if iteration % 5e4 == 0 and iteration != 0:
                K.set_value(
                    gan.optimizer.learning_rate, gan.optimizer.learning_rate / 10
                )
                print(f"Learning Rate Updated to {gan.optimizer.learning_rate}")

            """TRAIN DISCRIMINIATOR"""

            ### Discriminator Output(y) ###
            y_real = np.full(shape=batch_size, fill_value=0.9)
            # y_real = np.ones(shape=batch_size)
            y_fake = np.zeros(shape=batch_size)

            for _ in range(discrimator_train_steps):
                ### Discriminator Input(X) ###
                # Get HR images
                batch_paths = image_paths[
                    np.random.randint(0, len(image_paths), size=(batch_size))
                ]
                X_real = np.array(list(p.map(get_HR, batch_paths)))

                # Get LR Images
                # batch_paths = image_paths[
                #    np.random.randint(0, len(image_paths), size=(batch_size))
                # ]
                X_fake = np.array(list(p.map(get_LR, X_real)))

                X_real = X_real / 255
                X_fake = X_fake / 255

                # Generate images
                X_fake = generator(X_fake)

                ### Train Discriminator ###
                discriminator.trainable = True

                discriminator_loss += discriminator.train_on_batch(X_real, y_real)
                discriminator_loss += discriminator.train_on_batch(X_fake, y_fake)

                discriminator.trainable = False  # Ensures that discriminator remains static during training of generator in GAN

            ### Train Generator ###
            # Tricking the noised input of the Generator as real data
            y_gen = np.ones(batch_size)

            # Multiple GAN passes
            for _ in range(generator_train_steps):
                # Get LR Images
                batch_paths = image_paths[
                    np.random.randint(0, len(image_paths), size=(batch_size))
                ]
                HR_images = np.array(list(p.map(get_HR, batch_paths)))
                LR_images = np.array(list(p.map(get_LR, HR_images)))
                HR_images = HR_images / 255
                LR_images = LR_images / 255

                loss = gan.train_on_batch(LR_images, [HR_images, y_gen])
                m_loss += loss[1]
                generator_loss += loss[2]

            # Update Progress Bar
            tr.set_postfix(
                d_loss=discriminator_loss / 2 / discrimator_train_steps / (i + 1),
                g_loss=generator_loss / generator_train_steps / (i + 1),
                m_loss=m_loss / generator_train_steps / (i + 1),
            )

            iteration += 1

        # Analysis
        discriminator_loss = (
            discriminator_loss / 2 / discrimator_train_steps / steps_per_epoch
        )
        generator_loss = generator_loss / generator_train_steps / steps_per_epoch

        metrics.append([e, discriminator_loss, generator_loss])
        # np.save("weights/Super Resolution", np.array(metrics))
        # generator.save(f"weights/GAN/{e}-{generator_loss}.hdf5")
        # generator.save(f"weights/Super Resolution/{e}.hdf5")
        generator.save_weights(f"weights/Super Resolution/weights.hdf5")

        # Reloads
        importlib.reload(evaluate)

        ### Test Model ###
        evaluate.main(generator, concat=False, summary=False)

        # Remaking of Processing Pool is required with 'WDSR'
        p = mp.Pool(workers)


if __name__ == "__main__":
    dataset_path = "data/train/DIV2K_FLICKR2K/"
    batch_size = 16
    training(dataset_path=dataset_path, epochs=10000, batch_size=batch_size)
