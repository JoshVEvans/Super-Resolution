import Network

import tensorflow as tf
import keras

import numpy as np
import cv2

import random
import os


# Directories
data_path = "data/train/COCO/"
save_path = "data/outputs/"

# Load Directory
image_list = os.listdir(data_path)
image_num = 4

# Load Image
original = cv2.imread(data_path + image_list[image_num]) / 255
dim = original.shape


# Interpolate Image
scale = 2
interpolated = cv2.resize(
    original, (dim[1] // scale, dim[0] // scale), interpolation=cv2.INTER_LANCZOS4
)
interpolated = cv2.resize(interpolated, (dim[1], dim[0]))

interpolated = cv2.resize(original, (dim[1] * scale, dim[0] * scale))


# Format Interpolated Image
X = np.reshape(interpolated, (1, dim[0] * 2, dim[1] * 2, 3))

# Load Network
model = Network.VDSR()
model.load_weights("weights\Super Resolution\VDSR\weights.67-0.15879.hdf5")
print(model.summary())


output = model.predict(X)[0]

print(output.shape)

# Write Images
cv2.imwrite(save_path + "interpolated.png", interpolated * 255)
cv2.imwrite(save_path + "CNN.png", output * 255)
cv2.imwrite(save_path + "original.png", original * 255)

show = False

if show:
    # Show Images
    cv2.imshow("original", original)
    cv2.imshow("output", output)
    cv2.imshow("interpolated", interpolated)
    cv2.waitKey()
