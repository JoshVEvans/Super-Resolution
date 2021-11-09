import os

from keras.models import load_model

import numpy as np
import cv2


def evaluate(model, scale=2, concat=True, summary=True):
    if summary:
        model.summary()

    dir_original = "evaluation/original/"
    dir_output = "evaluation/output/"

    image_names = os.listdir(dir_original)

    for image_name in image_names:
        # Read in and format original image
        image = cv2.imread(f"{dir_original}{image_name}")
        input = image

        # Interpolate image
        dim = image.shape
        image = cv2.resize(
            image, (dim[1] // scale, dim[0] // scale), interpolation=cv2.INTER_AREA
        )
        image = cv2.resize(image, (dim[1], dim[0]), interpolation=cv2.INTER_CUBIC)
        interpolated = image

        # Write Interpolated
        cv2.imwrite(f"evaluation/interpolated/{image_name}", interpolated)

        # Standardize images
        image = np.reshape(image, (1, *dim))
        means = np.mean(image, axis=(0, 1, 2))
        stds = np.std(image, axis=(0, 1, 2))
        image = (image - means) / stds

        # Get Output
        output = np.array(model(image)[0])
        output = output * stds + means

        # White Strip
        white = np.full(
            shape=(output.shape[0], output.shape[1] // 50, 3), fill_value=255
        )

        # Write Output
        cv2.imwrite(f"{dir_output}{image_name}", output)

        if concat:
            cv2.imwrite(
                f"evaluation/Combined/{image_name}",
                np.concatenate((input, white, interpolated, white, output), axis=1),
            )


def inference(model, scale=2, summary=True):
    if summary:
        model.summary()

    dir_original = "inference/original/"
    dir_output = "inference/upscaled/"

    image_names = os.listdir(dir_original)

    for image_name in image_names:
        # Read in and format original image
        image = cv2.imread(f"{dir_original}{image_name}")

        # Upscale image
        dim = image.shape
        image = cv2.resize(
            image, (dim[1] * scale, dim[0] * scale), interpolation=cv2.INTER_LANCZOS4
        )
        dim = image.shape

        # Standardize images
        image = np.reshape(image, (1, *dim))
        means = np.mean(image, axis=(0, 1, 2))
        stds = np.std(image, axis=(0, 1, 2))
        image = (image - means) / stds

        # Get Output
        output = np.array(model(image)[0])
        output = output * stds + means

        # Write Output
        cv2.imwrite(f"{dir_output}{image_name}", output)


def loss(y_true, y_pred):
    alpha = 0.84
    ssim = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1))
    l1 = abs(y_true - y_pred)

    loss = alpha * ssim + (1 - alpha) * l1

    return loss


if __name__ == "__main__":
    import tensorflow as tf

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

    model = load_model("weights/LARGE.h5", custom_objects={"loss": loss})
    evaluate(model, scale=2)
