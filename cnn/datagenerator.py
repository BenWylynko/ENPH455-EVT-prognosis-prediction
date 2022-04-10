
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import numpy as np
import keras
import tensorflow as tf
import pickle
from load_utils import load_niftii
from pathlib import Path
import random
import scipy.ndimage as ndimage

BASE = r'Z:\EVT Project Data\NiftII\Complete'
SAVE_BASE = r'C:\Users\Ben\Downloads\data'

def path_gen_saved(p, vol=256):
    # Define path in saved data from ID (path to original data file)
    return Path(SAVE_BASE, f"{vol}_{vol}", f"{p.parents[0]}_{p.stem}.nii")

#augmentation
@tf.function
def rotate(volume, min=-1000, max=500):
    """Rotate the volume. Random angles generated separately for each rotation plane"""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-20, -10, -5, 0, 5, 10, 20]
        # 1st plane
        volume = ndimage.rotate(volume, random.choice(angles), axes=(0, 1), reshape=False)
        #2nd
        volume = ndimage.rotate(volume, random.choice(angles), axes=(0, 2), reshape=False)
        #3rd
        volume = ndimage.rotate(volume, random.choice(angles), axes=(1, 2), reshape=False)

        volume[volume < min] = min
        volume[volume > max] = max
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.double)
    return augmented_volume

class DataGenerator(tf.keras.utils.Sequence): #for multiprocessing
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=None, dim=(32, 32, 32), n_channels=1,
                 n_classes=2, shuffle=True, training=True, vol=256):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels #id -> label dict
        self.list_IDs = list_IDs #list of IDs we wish to generate at each pass
        self.n_channels = n_channels
        self.n_classes = n_classes #what are these?
        self.shuffle = shuffle
        self.training = training
        self.vol = vol

        self.on_epoch_end() #run once in init
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size)) #common value so the model sees each training sample at most once per epoch

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes) #shuffle order that samples are fed to the classifier
        if self.training:
            print(f"{self.batch_size} samples per training batch")
        if not self.training:
            print(f"{self.batch_size} samples per validation batch")

    def __getitem__(self, index):
        'Generate one batch of data by batch index'
        # Generate indexes of samples for the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs (map sample indices to IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int) #change to bool?

        # Generate data
        for i, ID  in enumerate(list_IDs_temp):
            # Store sample (with filepath)
            X[i,] = load_niftii(path_gen_saved(ID, self.vol), shape_std=False)[..., np.newaxis]
            #augment if this generator is used for training
            if self.training:
                X[i,] = rotate(X[i,])

            # Store class (from dict)
            y[i] = self.labels[ID.parents[0].stem]

        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes) #onehot
