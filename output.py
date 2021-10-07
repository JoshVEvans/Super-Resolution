import os

from keras.models import load_model

import numpy as np
import cv2


def evaluate(model, scale=2, concat=True, summary=True):
    if summary:
        model.summary()

    dir_original = "GAN/original/"
    dir_output = "GAN/output/"

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
        image = cv2.resize(image, (dim[1], dim[0]), interpolation=cv2.INTER_LANCZOS4)
        interpolated = image

        # Write Interpolated
        cv2.imwrite(f"GAN/interpolated/{image_name}", interpolated)

        image = np.reshape(image, (1, *dim)) / 255

        # Get Output
        output = np.array(model(image)[0])
        output = output * 255

        # Write Output
        cv2.imwrite(f"{dir_output}{image_name}", output)

        if concat:
            cv2.imwrite(
                f"GAN/Combined/{image_name}",
                np.concatenate((input, interpolated, output), axis=1),
            )


def inference(model, scale=2, summary=True):
    if summary:
        model.summary()

    dir_original = "inference/original/"
    dir_output = "inference/upscaled/"

    image_names = os.listdir(dir_original)

    for image_name in image_names:
        # Read in and format original image
        image = cv2.imread(f"{dir_original}{image_name}") / 255

        # Upscale image
        dim = image.shape
        image = cv2.resize(
            image, (dim[1] * scale, dim[0] * scale), interpolation=cv2.INTER_LANCZOS4
        )
        dim = image.shape
        image = np.reshape(image, (1, *dim))

        # Get Output
        output = np.array(model(image)[0]) * 255

        # Write Output
        cv2.imwrite(f"{dir_output}{image_name}", output)


if __name__ == "__main__":
    model = load_model("weights/LARGE_BEST.h5")
    evaluate(model, scale=2)
