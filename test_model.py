import os

from keras.models import load_model
import tensorflow as tf
import numpy as np


def ssis_value(y_true, y_pred):
    ssim = tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1))
    return ssim


dataset_path = "data/Set5/"
images = np.array(
    [f"{dataset_path}{image_path}" for image_path in os.listdir(dataset_path)]
)

weights_path = ''