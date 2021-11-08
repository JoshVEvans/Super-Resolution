import os
import tensorflow as tf

from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.layers import Input, Conv2D, Concatenate, Add

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


def loss(y_true, y_pred):
    alpha = 0.84
    ssim = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1))
    l1 = abs(y_true - y_pred)

    loss = alpha * ssim + (1 - alpha) * l1

    return loss


def SRCNN():
    # Initialize Input
    inputX = Input(shape=(None, None, 3))

    # Hidden Layers
    x = Conv2D(filters=64, kernel_size=9, padding="same", activation="relu")(inputX)
    x = Conv2D(filters=32, kernel_size=1, padding="same", activation="relu")(x)

    # Reconstruction - Output Layer
    x = Conv2D(filters=3, kernel_size=5, padding="same")(x)

    # Create and Compile model
    model = Model(inputs=inputX, outputs=x, name="SRCNN")
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=loss, metrics=["accuracy"])

    return model


def VDSR():
    # Parameters
    filters = 64

    # Initialize Input
    inputX = Input(shape=(None, None, 3))

    x = Conv2D(filters=filters, kernel_size=3, padding="same", activation="relu")(
        inputX
    )
    start = x

    # Residual Layers
    for _ in range(18):
        x = Conv2D(filters=filters, kernel_size=3, padding="same", activation="relu")(x)

    # Add Residuals
    x = Add()([start, x])

    # Reconstruction - Output Layer
    x = Conv2D(filters=3, kernel_size=3, padding="same")(x)

    # Create and Compile model
    model = Model(inputs=inputX, outputs=x, name="VDSR")
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=loss, metrics=["accuracy"])

    return model


def MODEL_LARGE():
    # Parameters
    filters = 128

    # Initialize Input
    inputX = Input(shape=(None, None, 3))

    # Initial Hidden Layer
    x = Conv2D(filters=filters, kernel_size=3, padding="same")(inputX)
    start = x

    # Residual Layers
    # Large Model: 24 residuals 128 filters
    for i in range(24):
        x = residual_block_large(x, filters=filters, kernel_size=3)

    # Add Residuals
    x = Add()([start, x])

    # Reconstruction - Output Layer
    x = Conv2D(filters=3, kernel_size=3, padding="same")(x)

    # Create and Compile model
    model = Model(inputs=inputX, outputs=x, name="SR_LARGE")
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=loss, metrics=["accuracy"])

    return model


def MODEL_XL():
    # Parameters
    filters = 192

    # Initialize Input
    inputX = Input(shape=(None, None, 3))

    # Initial Hidden Layer
    x = Conv2D(filters=filters, kernel_size=3, padding="same")(inputX)
    start = x

    # Residual Layers
    # Large Model: 24 residuals 128 filters
    for i in range(24):
        x = residual_block_large(x, filters=filters, kernel_size=3)

    # Add Residuals
    x = Add()([start, x])

    # Reconstruction - Output Layer
    x = Conv2D(filters=3, kernel_size=3, padding="same")(x)

    # Create and Compile model
    model = Model(inputs=inputX, outputs=x, name="SR_LARGE")
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=loss, metrics=["accuracy"])

    return model


def residual_block_large(inputX, filters, kernel_size):
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


def MODEL_SMALL():
    # Parameters
    filters = 64

    # Initialize Input
    inputX = Input(shape=(None, None, 3))

    # Initial Hidden Layer
    x = Conv2D(filters=filters, kernel_size=3, padding="same")(inputX)
    start = x

    # Residual Layers
    for i in range(9):
        x = residual_block_small(x, filters=filters, kernel_size=3)

    # Add Residuals
    x = Concatenate()([start, x])

    # Reconstruction - Output Layer
    x = Conv2D(filters=3, kernel_size=3, padding="same")(x)

    # Create and Compile model
    model = Model(inputs=inputX, outputs=x, name="SR_SMALL")
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=loss, metrics=["accuracy"])

    return model


def residual_block_small(inputX, filters, kernel_size):
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
    x = Concatenate()([inputX, x])

    return x


if __name__ == "__main__":
    print("=" * 98)
    print("-" * 98)
    print("=" * 98)

    model = MODEL_XL()
    print(model.summary())
    plot_model(model, show_shapes=True, to_file="model.png")
