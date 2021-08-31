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
        image = cv2.imread(f"{dir_original}{image_name}") / 255
        input = image * 255

        # Interpolate image
        dim = image.shape
        image = cv2.resize(image, (dim[1] // scale, dim[0] // scale))
        image = cv2.resize(image, (dim[1], dim[0]))
        interpolated = image * 255

        # Get Output
        image = np.reshape(image, (1, *dim))
        output = np.array(model(image)[0])
        output = output * 255

        # Write Output
        cv2.imwrite(f"{dir_output}{image_name}", output)

        if concat:
            cv2.imwrite(
                f"evaluation/Combined/{image_name}",
                np.concatenate((input, interpolated, output), axis=1),
            )


def predict(model, summary=True):

    if summary:
        model.summary()

    dir_original = "prediction/original/"
    dir_output = "prediction/upscaled/"

    image_names = os.listdir(dir_original)

    for image_name in image_names:
        # Read in and format original image
        image = cv2.imread(f"{dir_original}{image_name}") / 255

        # Upscale image
        scale = 2
        dim = image.shape
        image = cv2.resize(
            image, (dim[1] * scale, dim[0] * scale), interpolation=cv2.INTER_LANCZOS4
        )
        dim = image.shape
        image = np.reshape(image, (1, *dim))

        # Get Output
        output = np.array(model(image)[0]) * 255
        print(output.shape)

        # Write Output
        cv2.imwrite(f"{dir_output}{image_name}", output)


if __name__ == "__main__":

    model = load_model("weights/BEST_WEIGHTS.h5")
    evaluate(model)
