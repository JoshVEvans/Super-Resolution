import os
import tensorflow as tf

from keras.utils import plot_model

from keras.layers import (
    Input,
    Conv2D,
    Conv2DTranspose,
    Add,
    ReLU,
    PReLU,
    LeakyReLU,
    BatchNormalization,
    Dropout,
    MaxPool2D,
    Concatenate,
)
from keras.initializers import RandomNormal

from keras.models import Model
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from tensorflow.python.keras.layers.convolutional import UpSampling2D
from tensorflow.python.keras.layers.core import Dense, Flatten


# If uncommented, forces the use of the cpu or gpu
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Hides tensorflow outputs and warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Prevents complete memory allocation of gpu
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


def init():
    return RandomNormal(mean=0.0, stddev=0.02)


def optimizer():
    return Adam(lr=1e-4)  # standard lr: 0.0002


def create_generator():
    # Parameters
    filters = 64
    # Initialize Input
    inputX = Input(shape=(None, None, 3))

    # Initial Residual Layer
    x = Conv2D(
        filters=filters, kernel_size=3, padding="same", kernel_initializer=init()
    )(inputX)
    start = x

    # Residual Layers
    for _ in range(24):
        x = residual_block(x, filters=filters, kernel_size=3)

    # Add Residuals
    x = Add()([start, x])

    # Upscaling
    x = Conv2DTranspose(
        filters=filters,
        kernel_size=4,
        strides=2,
        padding="same",
        kernel_initializer=init(),
    )(x)

    # Output Layer
    x = Conv2D(
        filters=filters, kernel_size=3, padding="same", kernel_initializer=init()
    )(x)
    x = LeakyReLU()(x)

    x = Conv2D(
        filters=3,
        kernel_size=3,
        padding="same",
        activation="sigmoid",
        kernel_initializer=init(),
    )(x)

    # Create  model
    model = Model(inputs=inputX, outputs=x, name="Generator")

    return model


def residual_block(inputX, filters, kernel_size):
    x = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer=init(),
    )(inputX)
    x = LeakyReLU()(x)
    x = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer=init(),
    )(x)
    x *= 0.1
    x = Add()([inputX, x])

    return x


def create_disciminator():
    # Parameters
    filters = 64

    # Initialize Input
    inputX = Input(shape=(64, 64, 3))

    # 1st Group
    x = discriminator_block(inputX, filters)

    # 2nd Group
    x = discriminator_block(x, 2 * filters)

    # 3rd Group
    x = discriminator_block(x, 4 * filters)

    # 4th Group
    x = discriminator_block(x, 8 * filters)

    # Flatten
    x = Flatten()(x)
    x = Dense(1024, kernel_initializer=init())(x)
    x = LeakyReLU()(x)
    x = Dense(1, activation="sigmoid", kernel_initializer=init())(x)

    # Create and Compile model
    model = Model(inputs=inputX, outputs=x, name="Discriminator")
    model.compile(optimizer=optimizer(), loss="binary_crossentropy")

    return model


def discriminator_block(x, filters):
    x = Conv2D(
        filters=filters, kernel_size=3, padding="same", kernel_initializer=init()
    )(x)
    x = LeakyReLU()(x)
    x = Conv2D(
        filters=filters,
        kernel_size=4,
        strides=2,
        padding="same",
        kernel_initializer=init(),
    )(x)
    x = LeakyReLU()(x)

    return x


def create_gan(discriminator, generator):
    discriminator.trainable = False

    gan_input = Input(shape=(None, None, 3))
    g_output = generator(gan_input)

    d_output = discriminator(g_output)

    gan = Model(
        inputs=gan_input,
        outputs=[g_output, d_output],
        name="Generative_Adversarial_Network",
    )

    gan.compile(
        optimizer=optimizer(),
        loss=["mse", "binary_crossentropy"],
        loss_weights=[1, 1e-3],
    )

    return gan


if __name__ == "__main__":
    # Create GAN
    generator = create_generator()
    generator.summary()
    discriminator = create_disciminator()
    discriminator.summary()
    # gan = create_gan(discriminator, generator)
    # gan.summary()

    # Graph Model
    plot_model(generator, show_shapes=True, to_file="generator.png")
    plot_model(discriminator, show_shapes=True, to_file="discriminator.png")
    # plot_model(gan, show_shapes=True, to_file="gan.png")
