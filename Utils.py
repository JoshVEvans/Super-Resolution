from keras import Sequential
from keras.layers import Conv2D, ReLU, LeakyReLU, BatchNormalization
from keras.layers.experimental.preprocessing import Resizing
from keras.optimizers import Adam
from skimage.metrics import structural_similarity as ssim
import tensorflow as tf
import numpy as np
import random
from tqdm import tqdm
import cv2
import os


# Hides tensorflow outputs and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Forces the use of the cpu or gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Prevents memory overflow errors
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.95))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

# Loss
def mssim_loss(y_true, y_pred):
    loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1))
    return loss

def modelSRCNN():
    
    # define model type
    SRCNN = Sequential()

    # add model layers
    SRCNN.add(Conv2D(filters=128, kernel_size=(9,9), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True, input_shape=(None,None,3)))
    SRCNN.add(Conv2D(filters=64, kernel_size=(5,5), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True))
    SRCNN.add(Conv2D(filters=3, kernel_size=(5,5), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True))

    # define optimizer
    adam = Adam(learning_rate = 0.001)

    # compile model
    SRCNN.compile(optimizer = adam, loss = mssim_loss, metrics = [mssim_loss, 'mean_squared_error', 'accuracy'])

    return SRCNN

def model_large_SRCNN():
    
    # define model type
    SRCNN = Sequential()

    # add model layers
    SRCNN.add(Conv2D(filters=64, kernel_size=(9,9), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True, input_shape=(None,None,3)))
    SRCNN.add(Conv2D(filters=64, kernel_size=(7,7), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True, input_shape=(None,None,3)))
    SRCNN.add(Conv2D(filters=64, kernel_size=(5,5), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True))
    SRCNN.add(Conv2D(filters=3, kernel_size=(3,3), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True))

    # define optimizer
    adam = Adam(learning_rate = 0.001)

    # compile model
    SRCNN.compile(optimizer = adam, loss = mssim_loss, metrics = [mssim_loss, 'mean_squared_error', 'accuracy'])

    return SRCNN


def modelVDSR():

    # define model type
    VDSR = Sequential()

    # add model layers
    VDSR.add(Conv2D(filters=64, kernel_size=(3,3), kernel_initializer='glorot_uniform', padding='same', activation='relu', use_bias=True, input_shape=(None,None,3)))
    VDSR.add(BatchNormalization())
    for _ in range(18):
        VDSR.add(Conv2D(filters=64, kernel_size=(3,3), kernel_initializer='glorot_uniform', padding='same', activation='relu', use_bias=True))
        VDSR.add(BatchNormalization())
    VDSR.add(Conv2D(filters=3, kernel_size=(3,3), kernel_initializer='glorot_uniform', padding='same', activation='relu', use_bias=True))

    # define optimizer
    adam = Adam(learning_rate = 6.25E-05)
    # startrate: learning_rate = 0.001

    # compile model
    VDSR.compile(optimizer = adam, loss = mssim_loss, metrics = [mssim_loss, 'mean_squared_error', 'accuracy'])

    return VDSR

def modelVDSR_smaller():

    # define model type
    VDSR = Sequential()

    # add model layers
    VDSR.add(Conv2D(filters=128, kernel_size=(9,9), kernel_initializer='glorot_uniform', padding='same', activation='relu', use_bias=True, input_shape=(None,None,3)))
    VDSR.add(BatchNormalization())
    for _ in range(9):
        VDSR.add(Conv2D(filters=64, kernel_size=(5,5), kernel_initializer='glorot_uniform', padding='same', activation='relu', use_bias=True))
        VDSR.add(BatchNormalization())
    VDSR.add(Conv2D(filters=3, kernel_size=(3,3), kernel_initializer='glorot_uniform', padding='same', activation='relu', use_bias=True))

    # define optimizer
    adam = Adam(learning_rate = 0.001)
    # startrate: learning_rate = 0.001

    # compile model
    VDSR.compile(optimizer = adam, loss = mssim_loss, metrics = [mssim_loss, 'mean_squared_error', 'accuracy'])

    return VDSR

if __name__ == '__main__':
    model = modelVDSR()
    print(model.summary())
