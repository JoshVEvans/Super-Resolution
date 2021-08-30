import os
import tensorflow as tf

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
from keras.activations import swish
from keras.initializers import HeUniform
from keras.models import Model
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import plot_model


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
    return HeUniform()


def ssim_loss(y_true, y_pred):
    loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1))
    return loss


def SRCNN():
    # Input Layer
    inputX = Input(shape=(None, None, 3))

    # Patch Extraction and Representation
    x = Conv2D(filters=64, kernel_size=9, padding="same", activation="relu")(inputX)

    # Non-Linear Mapping
    x = Conv2D(filters=32, kernel_size=1, padding="same", activation="relu")(x)

    # Reconstruction - Output Layer
    x = Conv2D(filters=3, kernel_size=5, padding="same")(x)

    ### Create Model ###
    # Optimizer
    adam = Adam(learning_rate=1e-4)
    # startrate: learning_rate = 1E-4

    # Create and Compile model
    model = Model(inputs=inputX, outputs=x, name="SRCNN")
    model.compile(optimizer=adam, loss=ssim_loss, metrics=["accuracy"])

    return model


def EDSR():
    # Parameters
    filters = 128

    # Initialize Input
    inputX = Input(shape=(None, None, 3))

    # Initial Residual Layer
    x = Conv2D(filters=filters, kernel_size=3, padding="same")(inputX)
    start = x

    # Residual Layers
    # Standard: 9, 24, 49
    # Paper Uses 16, 32 ResBlocks
    for i in range(24):
        x = residual_block(x, filters=filters, kernel_size=3)

    # Add Residuals
    x = Add()([start, x])

    # Reconstruction - Output Layer
    x = Conv2D(filters=3, kernel_size=3, padding="same")(x)

    # Optimizer
    adam = Adam(learning_rate=1e-4)
    # startrate 20 Conv Layers: learning_rate = 1e-4

    # Create and Compile model
    model = Model(inputs=inputX, outputs=x, name="EDSR")
    model.compile(optimizer=adam, loss="mae", metrics=["accuracy"])

    return model


def residual_block(inputX, filters, kernel_size):
    """
    x
		|\
		| \
		|  conv2d
		|  activation
		|  conv2d
        |  (multiply scaling)
		| /
		|/
		+ (addition here)
		|
		result
    """
    x = Conv2D(
        filters=filters, kernel_size=kernel_size, activation="relu", padding="same"
    )(inputX)
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding="same")(x)
    x *= 0.1
    x = Add()([inputX, x])

    return x


def DENSESR():
    # Parameters
    filters = 16
    layers = []

    # Initialize Input
    inputX = Input(shape=(None, None, 3))

    x = Conv2D(filters=filters, kernel_size=3, padding="same")(inputX)
    start = x

    for i in range(19):
        x = dense_block(x, filters=filters, kernel_size=3)

        if i % 1 == 0:
            layers.append(x)

    # Add Residuals
    x = Concatenate()([start, *layers])

    # Reconstruction - Output Layer
    x = Conv2D(filters=3, kernel_size=3, padding="same")(x)

    # Optimizer
    adam = Adam(learning_rate=1e-4)
    # startrate 20 Conv Layers: learning_rate = 1e-4

    # Create and Compile model
    model = Model(inputs=inputX, outputs=x, name="DENSESR")
    model.compile(optimizer=adam, loss=ssim_loss, metrics=["accuracy"])

    return model


def dense_block(inputX, filters, kernel_size):
    layers = [inputX]
    x = inputX

    for _ in range(4):
        x = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            activation="relu",
        )(x)
        layer = x
        x = Concatenate()([*layers, x])
        layers.append(layer)

    x = Conv2D(filters=filters, kernel_size=3, padding="same", activation="relu")(x)

    return x


def UNET():
    # Parameters
    filters = 32

    inputX = Input(shape=(None, None, 3))

    # Initial Convolution Layer
    x = Conv2D(
        filters=filters,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_initializer=init(),
    )(inputX)

    # Encoder
    e_1 = Conv2D(
        filters=filters * 2,
        kernel_size=4,
        strides=2,
        padding="same",
        activation="relu",
        kernel_initializer=init(),
    )(x)
    e_2 = Conv2D(
        filters=filters * 4,
        kernel_size=4,
        strides=2,
        padding="same",
        activation="relu",
        kernel_initializer=init(),
    )(e_1)
    e_3 = Conv2D(
        filters=filters * 8,
        kernel_size=4,
        strides=2,
        padding="same",
        activation="relu",
        kernel_initializer=init(),
    )(e_2)
    e_4 = Conv2D(
        filters=filters * 16,
        kernel_size=4,
        strides=2,
        padding="same",
        activation="relu",
    )(e_3)

    # Decoder
    d_1 = Conv2DTranspose(
        filters=filters * 8,
        kernel_size=4,
        strides=2,
        padding="same",
        activation="relu",
        kernel_initializer=init(),
    )(e_4)
    a_1 = Concatenate()([e_3, d_1])
    d_2 = Conv2DTranspose(
        filters=filters * 4,
        kernel_size=4,
        strides=2,
        padding="same",
        activation="relu",
        kernel_initializer=init(),
    )(a_1)
    a_2 = Concatenate()([e_2, d_2])
    d_3 = Conv2DTranspose(
        filters=filters * 2,
        kernel_size=4,
        strides=2,
        padding="same",
        activation="relu",
        kernel_initializer=init(),
    )(a_2)
    a_3 = Concatenate()([e_1, d_3])
    d_4 = Conv2DTranspose(
        filters=filters,
        kernel_size=4,
        strides=2,
        padding="same",
        activation="relu",
        kernel_initializer=init(),
    )(a_3)
    a_4 = Concatenate()([x, d_4])

    # Output Layer
    output = Conv2D(
        filters=3,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_initializer=init(),
    )(a_4)

    # Optimizer
    adam = Adam(learning_rate=1e-4)
    # startrate : learning_rate = 1e-4

    model = Model(inputs=inputX, outputs=output, name="UNET")
    model.compile(optimizer=adam, loss="mae", metrics=["accuracy"])

    return model


def Inception():
    inputX = Input(shape=(None, None, 3))
    x = inputX

    for _ in range(9):  # Standard: 9
        x = inception_block(x)

    x = Conv2D(
        filters=3,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_initializer=init(),
    )(x)

    model = Model(inputs=inputX, outputs=x, name="Inception")
    model.compile(optimizer=Adam(learning_rate=1e-4), loss="mse", metrics=["accuracy"])

    return model


def inception_block(inputX):
    branch_0 = Conv2D(
        filters=64,
        kernel_size=1,
        padding="same",
        activation="relu",
        kernel_initializer=init(),
    )(inputX)

    branch_1 = Conv2D(
        filters=96,
        kernel_size=1,
        padding="same",
        activation="relu",
        kernel_initializer=init(),
    )(inputX)
    branch_1 = Conv2D(
        filters=128,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_initializer=init(),
    )(branch_1)

    branch_2 = Conv2D(
        filters=16,
        kernel_size=1,
        padding="same",
        activation="relu",
        kernel_initializer=init(),
    )(inputX)
    branch_2 = Conv2D(
        filters=32,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_initializer=init(),
    )(branch_2)
    branch_2 = Conv2D(
        filters=32,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_initializer=init(),
    )(branch_2)

    branch_3 = MaxPool2D(pool_size=1, padding="same")(inputX)
    branch_3 = Conv2D(
        filters=32,
        kernel_size=1,
        padding="same",
        activation="relu",
        kernel_initializer=init(),
    )(branch_3)

    x = Concatenate()([inputX, branch_0, branch_1, branch_2, branch_3])
    x = ReLU()(x)

    return x


if __name__ == "__main__":
    print("=" * 98)
    print("-" * 98)
    print("=" * 98)

    model = EDSR()
    print(model.summary())
    plot_model(model, show_shapes=True, to_file="model.png")
