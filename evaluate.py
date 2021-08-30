import os

import numpy as np
import cv2


def main(model, concat=True, summary=True):
    # model.load_weights("weights/Super Resolution/weights.hdf5")
    model.load_weights(
        "weights\Super Resolution\BEST - 141-0.03283986663352698-0.8788642883300781.hdf5"
    )

    if summary:
        model.summary()

    dir_original = "Super Resolution/evaluation/original/"
    dir_interpolated = "Super Resolution/evaluation/interpolated/"
    dir_output = "Super Resolution/evaluation/output/"

    image_names = os.listdir(dir_original)

    for image_name in image_names:
        # Read in and format original image
        image = cv2.imread(f"{dir_original}{image_name}") / 255
        input = image * 255

        # Interpolate image
        scale = 2
        dim = image.shape
        image = cv2.resize(
            image, (dim[1] // scale, dim[0] // scale), interpolation=cv2.INTER_AREA
        )  # , interpolation=cv2.INTER_AREA
        image = cv2.resize(
            image, (dim[1], dim[0]), interpolation=cv2.INTER_LANCZOS4
        )  # , interpolation=cv2.INTER_LANCZOS4

        # Write interpolated image
        cv2.imwrite(f"{dir_interpolated}{image_name}", image * 255)
        interpolated = image * 255
        image = np.reshape(image, (1, *dim))

        # Get Output
        output = np.array(model(image)[0])
        output = output * 255

        # Write Output
        cv2.imwrite(f"{dir_output}{image_name}", output)

        if concat:
            cv2.imwrite(
                f"Super Resolution/evaluation/Combined/{image_name}",
                np.concatenate((input, output, interpolated), axis=1),
            )


if __name__ == "__main__":
    import Network

    model = Network.EDSR()
    main(model)
