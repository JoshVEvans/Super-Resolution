from tensorflow.keras.utils import Sequence
import tensorflow as tf
import numpy as np
import random
import cv2
import os


class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, file_names, image_path, to_fit=True,
                 batch_size=32, dim=(128, 128), n_channels = 3,
                 n_classes=10, shuffle=True, scale=[2]):
        """Initialization
        :param file_names: list of image file names
        :param image_path: path to images location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of channels for the image
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        :param scale: image resize factor
        """
        self.fileNames = file_names
        self.image_path = image_path
        self.image_path2 = ['data/interpolated/net_3_data/2x/','data/interpolated/net_3_data/3x/','data/interpolated/net_3_data/4x/']
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.scale = scale
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.list_IDs = [i for i in range(len(file_names))]
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        self.lr_dim = tuple([x // random.choice(self.scale) for x in self.dim])
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]


        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        y = self._generate_y(list_IDs_temp)

        if self.to_fit:
            X = self._generate_X(list_IDs_temp)

            # Required fix due to bug
            X = np.array(X)
            y = np.array(y)
            #i = 0
            for i in range(len(X)):
                # Random Flipping
                if random.choice([True, False]):
                    X[i] = cv2.flip(X[i], 1)
                    y[i] = cv2.flip(y[i], 1)
                if random.choice([True, False]):
                    X[i] = cv2.flip(X[i], 0)
                    y[i] = cv2.flip(y[i], 0)
                if random.choice([True, False]):
                    temp = random.choice([0, 1, 2])
                    X[i] = cv2.rotate(X[i], temp)
                    y[i] = cv2.rotate(y[i], temp)
                #i += 1

            return X, y
        else:
            return X
        

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _generate_X(self, list_IDs_temp):
        #def _generate_X(self, y):
        """Generates data containing batch_size images
        :param y: high resolution images from the training set
        :return: batch of images
        """
        # Initialization
        X = np.empty((self.batch_size, *self.dim, 3))

        '''
        # Required due to bug in cv2
        y = np.array(y)

        
        # Generate data
        for i, img in enumerate(y):
            # Store sample
            img = cv2.resize(img, self.lr_dim)
            img = cv2.resize(img, self.dim)
            X[i,] = img

        '''

        # Generate data 2nd NN:
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            img = cv2.imread(random.choice(self.image_path2) + self.fileNames[ID])
            
            # Image pre processing
            img = cv2.resize(img, self.dim, interpolation=cv2.INTER_AREA)
            img = img / 255.0

            X[i,] = img
        X = tf.clip_by_value(X, 0, 1)

        return X

    def _generate_y(self, list_IDs_temp):
        """Generates data containing batch_size masks
        :param list_IDs_temp: list of label ids to load
        :return: batch
        """
        y = np.empty((self.batch_size, *self.dim, 3))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            img = self._load_image(self.image_path + self.fileNames[ID])
            y[i,] = img

        y = tf.clip_by_value(y, 0, 1)

        return y

    def _load_image(self, image_path):
        """Load image
        :param image_path: path to image to load
        :return: loaded image
        """
        img = cv2.imread(image_path)
        # Change image color profile

        # Force image into batch dimensions
        img = cv2.resize(img, self.dim, interpolation=cv2.INTER_AREA)
        
        img = img / 255.0

        return img
