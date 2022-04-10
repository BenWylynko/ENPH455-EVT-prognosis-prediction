import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
os.environ["PYTHONPATH"] = r"C:\Users\Ben\OneDrive-yahoo\OneDrive\Documents\ENPH455\ENPH455-EVT-prognosis-prediction"

from load_utils import load_nrrd, pad_nrrd, resize_volume, load_all_segmentations, load_masked_imgs
import matplotlib.pyplot as plt
from pathlib import Path
from keras import models, layers, Input, Model, losses, callbacks, optimizers, regularizers
from tensorflow.keras.utils import to_categorical
import numpy as np
import argparse
import pandas as pd
import cv2
import scipy.ndimage as ndimage
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from augmented import generator
import keras.backend as K
from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import Callback, EarlyStopping
from keras.models import model_from_json
from sklearn import metrics
K.clear_session()
from utils.split import load_split_2, gen_split_2, gen_split_3, load_ids_labels, load_split_3
from utils.metrics import get_metrics
from sklearn.model_selection import StratifiedKFold
import json
from math import pi

BASE = r'Z:\EVT Project Data\NiftII\Complete'
SAVE_PATH = r'C:\Users\Ben\Downloads\data\segmented'
JSON_BASE = r'C:\Users\Ben\OneDrive-yahoo\OneDrive\Documents\ENPH455\ENPH455-EVT-prognosis-prediction'
N_LAYERS = None
IMG_SHAPE = (39, 162, 55)
MASKED = None

def inject_noise(img):
    """
    Args:
    img: Rank 3 image 
    """
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, img.shape)
    gauss = gauss.reshape(img.shape)
    noisy = img + gauss
    return noisy

rot_range = 30 #degrees
shear_range = pi / 4 #radians
train_aug = generator.customImageDataGenerator(
    rotation_range = rot_range, 
    shear_range = shear_range, 
    preprocessing_function = inject_noise
)
val_aug = generator.customImageDataGenerator(
    rotation_range = rot_range, 
    shear_range = shear_range, 
    preprocessing_function = inject_noise
)

def get_model_2layers(shape):

    inputs = Input((*shape, 1))

    x = layers.Conv3D(filters=64, kernel_size=2, activation="relu", 
        kernel_regularizer = regularizers.l1(1e-4), bias_regularizer = regularizers.l1(1e-4), activity_regularizer = regularizers.l1(1e-4))(inputs)
    x = layers.MaxPool3D(pool_size=3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu", 
        kernel_regularizer = regularizers.l1(1e-4), bias_regularizer = regularizers.l1(1e-4), activity_regularizer = regularizers.l1(1e-4))(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=2, activation="sigmoid")(x)

    # Define the model.
    model = Model(inputs, outputs, name="3dcnn_2")
    return model

def get_model_3layers(shape):
    inputs = Input((*shape, 1))

    x = layers.Conv3D(filters=64, kernel_size=2, activation="relu", 
        kernel_regularizer = regularizers.l1(1e-4), bias_regularizer = regularizers.l1(1e-4), activity_regularizer = regularizers.l1(1e-4))(inputs)
    x = layers.MaxPool3D(pool_size=3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=2, activation="relu", 
        kernel_regularizer = regularizers.l1(1e-4), bias_regularizer = regularizers.l1(1e-4), activity_regularizer = regularizers.l1(1e-4))(x)
    x = layers.MaxPool3D(pool_size=3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu", 
        kernel_regularizer = regularizers.l1(1e-4), bias_regularizer = regularizers.l1(1e-4), activity_regularizer = regularizers.l1(1e-4))(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=2, activation="sigmoid")(x)

    # Define the model.
    model = Model(inputs, outputs, name="3dcnn_3")
    return model


"""
Plotting
"""

def plot_seg2d(ax, img, dim=0):
    """2d plot grid along the specified dimension"""
    f, axs = plt.subplots(5, 2)
    for i in range(2):
        for j in range(5):
            ind = i * 50 + j * 10
            if dim == 0:
                arr = img[ind, :, :]
            elif dim == 1:
                arr = img[:, ind, :]
            else:
                arr = img[:, :, ind]
            axs[j, i].imshow(arr, aspect='auto')
            axs[j, i].set_title(f"{ind}")
    plt.show()

def plot_seg1d(axs, img, N=6, dim=0):
    """1d plot grid along the specified dimension
        Args:
            axs: list of axes to use for plotting
            img: np array
            N: num rows
            dim: dim along img to show
    """
    #get spacing for N images
    spacing = np.linspace(0, img.shape[dim], num=N, dtype=int)
    gap = spacing[1] - spacing[0]
    for i in range(N):
        ind = i * gap
        if ind == img.shape[dim]:
            ind -= 1
        if dim == 0:
            arr = img[ind, :, :]
        elif dim == 1:
            arr = img[:, ind, :]
        else:
            arr = img[:, :, ind]
        axs[i].imshow(arr, aspect='auto')
        axs[i].set_title(f"{ind}")


def describe_seg(fpath=None, id=None, N=6, img=None):
    """Plot the segmentation along all 3 dimensions"""
    if fpath is not None:
        nrrddata = load_nrrd(fpath)  
    elif img is not None:
        nrrddata = img
    sh = nrrddata.shape

    #try all dimensions
    f, (axs1, axs2, axs3) = plt.subplots(3, N)
    f.suptitle(f"{id}, {sh}")
    plot_seg1d(axs1, nrrddata)
    plot_seg1d(axs2, nrrddata, dim=1)
    plot_seg1d(axs3, nrrddata, dim=2)  
    
    plt.show()

def describe_seg_all(img1, img2, img3, id, N=6):
    #sh = img.shape

    fig = plt.figure(figsize=(20, 10),constrained_layout=True)
    fig.tight_layout()
    gs = fig.add_gridspec(1, 3)

    gs00 = gs[0].subgridspec(3, N, hspace=1)
    gs01 = gs[1].subgridspec(3, N, hspace=2)
    gs02 = gs[2].subgridspec(3, N, hspace=3)

    
    for i in range(3):
        #gridspec 1
        ax1 = [fig.add_subplot(gs00[i, j]) for j in range(N)]
        plot_seg1d(ax1, img1, dim=i)
        #gridspec 2
        ax2 = [fig.add_subplot(gs01[i, j]) for j in range(N)]
        plot_seg1d(ax2, img2, dim=i)
        #gridspec 2
        ax3 = [fig.add_subplot(gs02[i, j]) for j in range(N)]
        plot_seg1d(ax3, img3, dim=i) 
    
    plt.show()

"""
Reshaping / playing with dimensions
"""

def get_size_data(ims, ids):
    size_data = pd.DataFrame()
    for i, imgg in enumerate(ims):
        w,h,d = imgg.shape
        size_data=size_data.append([[w,h,d,imgg.size, ids[i]]])
    return size_data

def remove_extent_along_axis(img, ax=0, masked=False):
    """Remove padding along specified axis"""

    dim = img.shape[ax]
    sh = list(img.shape)
    sh.pop(ax)

    #specify condition for valid pixels
    if masked:
        cond = lambda x: np.where(x != -1000.0)
    else:
        cond = lambda x: np.where(x == 1)
    #loop all slices in this dim
    min1, min2 = sh[0] // 2, sh[1] // 2
    max1, max2 = sh[0] // 2, sh[1] // 2
    for i in range(dim):
        if ax == 0:
            #(r, c) = np.where(img[i, :, :] == 1)
            (r, c) = cond(img[i, :, :])
        elif ax == 1:
            #(r, c) = np.where(img[:, i, :] == 1)
            (r, c) = cond(img[:, i, :])
        else:
            #(r, c) = np.where(img[:, :, i] == 1)
            (r, c) = cond(img[:, :, i])

        if len(r) > 0:
            if np.min(r) < min1:
                min1 = np.min(r)
            if np.max(r) > max1:
                max1 = np.max(r)
        if len(c) > 0:
            if np.min(c) < min2:
                min2 = np.min(c)
            if np.max(c) > max2:
                max2 = np.max(c)

    if ax == 0:
        im_crop = img[:, min1:max1+1, min2:max2+1]
    elif ax == 1:
        im_crop = img[min1:max1+1, :, min2:max2+1]
    else:
        im_crop = img[min1:max1+1, min2:max2+1, :]
    return im_crop


def remove_extent(img, masked=False, plot=False):
    """
    Remove padding around the segmentation 
    Args:
        img: 3d segmenation volume
        pad: Padding to keep around segmentation on each dimension
    """

    sh = img.shape

    if plot:
        describe_seg(img=img)

    #do just 1, see what happens
    img = remove_extent_along_axis(img, ax=0, masked=masked)
    img = remove_extent_along_axis(img, ax=1, masked=masked)
    img = remove_extent_along_axis(img, ax=2, masked=masked)

    if plot:
        describe_seg(img=img)

    return img

def reshape(img, size):
    """Resize along all axes"""
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / size[-1]
    width = current_width / size[0]
    height = current_height / size[1]
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height

    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

"""
Modifying pixel values
"""
def normalize(image, MIN_BOUND=-1000.0, MAX_BOUND=600):
    """Normalize masked image"""
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

def zero_center(image, pix_mean):
    image = image - pix_mean
    return image


"""
Metrics
"""

def get_auc(y, pred):
    return metrics.roc_auc_score(y, pred)

"""
Related to model
"""
def save_model(model, name):
    model.save_weights(f"models/{name}.h5")
    model_json = model.to_json()
    with open(f'models/{name}.json', "w") as json_file:
        json_file.write(model_json)

def load_model(name):
    json_file = open(f"models/{name}.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(f"models/{name}.h5")
    return model

def ims_from_ids(ims, ids):
    train_inds = [ids.index(id_) for id_ in ids]
    x = np.array([ims[i] for i in train_inds])
    return x

def k_folds_train(x_train, y_train, x_test, y_test, n_folds, test_size, epochs=10, batch_size=10):
    
    for i, (train_index, val_index) in enumerate(StratifiedKFold(n_folds, shuffle=True).split(x_train, y_train)):
        print(f"Training wtih {len(train_index)} train samples, {len(val_index)} val samples, {len(x_test)} test samples")

        fold_x_train, fold_x_val = x_train[train_index], x_train[val_index]
        fold_y_train, fold_y_val = [y_train[ind] for ind in train_index], [y_train[ind] for ind in val_index]

        #to categorical
        fold_y_train = to_categorical(fold_y_train, num_classes=2)
        fold_y_val = to_categorical(fold_y_val, num_classes=2)

        seed = 42
        train_generator = train_aug.flow(fold_x_train,fold_y_train, batch_size=batch_size)
        val_generator = val_aug.flow(fold_x_val,fold_y_val, batch_size=batch_size)

        fit_and_eval(train_generator, val_generator, x_test, y_test, epochs=epochs, ind=i, test_size=test_size)

        
def fit_and_eval(train_generator, val_generator, x_test, y_test, epochs=10, ind=None, test_size=None):
    """
    Do fitting and evaluation, to be used for training
    """
    if N_LAYERS == 2:
        model = get_model_2layers((39, 162, 55))
    elif N_LAYERS == 3:
        model = get_model_3layers((39, 162, 55))
    else:
        raise ValueError("N_LAYERS shouldn't be None here...")

    model.summary()
    model.compile(optimizer='adam',
                        loss='binary_crossentropy', 
                        metrics=[tf.keras.metrics.AUC()])
    checkpoint_cb = callbacks.ModelCheckpoint(
        "3d_segmentation_classification.h5", save_best_only=True
        )

    early_stopping = EarlyStopping(monitor='auc', patience=3, verbose=1)

    
    # Fit the model
    history = model.fit(train_generator,
        steps_per_epoch = 30, #N samples / batch size is 10 / 10 = 1
        epochs = epochs, 
        validation_data = val_generator,
        validation_steps = 10, #N val samples // batch size: 5 // 10 = 0
        verbose=1, 
        batch_size=64, 
        #callbacks=[checkpoint_cb]
    )
    

    if MASKED == True:
        sname = f'cnn_seg_masked__{ind}_{test_size}_{N_LAYERS}_{IMG_SHAPE}'
    elif MASKED == False:
        sname = f'cnn_seg__{ind}_{test_size}_{N_LAYERS}_{IMG_SHAPE}'
    else:
        raise ValueError(f"Masked: {MASKED}")

    save_model(model, sname)

    ###test
    predictions = model.predict(x_test)
    predicted_classes = np.argmax(predictions, axis=1)
    
    #get metric values
    y_pred = list(map(lambda x: x == 1, predicted_classes))
    get_metrics(y_pred, y_test, sname)

def preprocess(ims, ids):

    ### Preprocessing
    #remove extra extent beyond segmentations (do this first so I can use -1000 as fill value)
    ims_noextent = [remove_extent(im, masked=MASKED, plot=False) for im in ims]

    #reshape all to mean along each axis (before normalizing to save memory)
    ims_reshaped = np.empty((len(ids), *IMG_SHAPE))
    dims = np.array([im.shape for im in ims_noextent])
    m1, m2, m3 = np.mean(dims[:, -1], axis=0), np.mean(dims[:, 0], axis=0), np.mean(dims[:, 1], axis=0)
    for i, im in enumerate(ims_noextent):
        ims_reshaped[i] = reshape(im, IMG_SHAPE) 

    if MASKED:
        #if masked images, normalize
        ims_reshaped = [normalize(im) for im in ims_reshaped]

        #zero center with global mean
        m = np.mean([np.mean(im) for im in ims_reshaped]) #might break
        ims_reshaped = [zero_center(im, m) for im in ims_reshaped]
        
    return ims_reshaped

def main(args):

    global N_LAYERS
    N_LAYERS = args.n_layers #modifies global scope var
    global MASKED
    MASKED = args.masked

    if args.train:
        args.prep_train = True

    #characterize volume sizes
    ids = ["0157754", "0337411", "0357680", "0444750", "0513431", "0673127", "0702581", "0732825", 
        "0783417", "0840732", "0863628", "0869517", "1069211", "1175412", "1302751", "1305527", "1310557"]

    if args.masked:
        ims = load_masked_imgs(ids)
    else:
        ims = load_all_segmentations(ids)

    size_data = get_size_data(ims, ids)

    ims_reshaped = preprocess(ims, ids)
    
    #check sizes again
    print("new sizes:")
    size_data = get_size_data(ims_reshaped, ids)

    #plot all preprocessed images
    N = 3
    if args.plot:
        for i, im in enumerate(ims_reshaped):
            if i < N:
                describe_seg(img=im, id=ids[i])

    ### prep datasets for training
    if args.prep_train:

        #load labels from pickle
        ids_labels = load_ids_labels()

        #format labels
        labels = [ids_labels[id_] for id_ in ids]
        print(f"Number of usable segmentations: {len(labels)}")

        if not args.k_fold: #generate train / val / test splits
            ### split into train / test
            if args.gen_split:
                sdir = Path(Path(__file__).parents[1], 'utils')
                x_train, y_train, x_val, y_val, x_test, y_test = gen_split_3(sdir, ims_reshaped, ids, labels, ids_labels, args.test_size)

            else:
                #load splitted IDs+
                y_train, y_val, y_test = load_split_3(args.test_size)

                #split images correctly (map indices to images)
                x_train = ims_from_ids(ims_reshaped, y_train)
                x_val = ims_from_ids(ims_reshaped, y_val)
                x_test = ims_from_ids(ims_reshaped, y_test)

                #check class split in datasets
                print(f"Train set N: {len(y_train)}, 1st class: {(len(y_train) - sum(y_train))/ len(y_train)} %, 2nd class: {sum(y_train) / len(y_train)} %")
                print(f"Validation set N: {len(y_val)}, 1st class: {(len(y_val) - sum(y_val))/ len(y_val)} %, 2nd class: {sum(y_val) / len(y_val)} %")
                print(f"Test set N: {len(y_test)}, 1st class: {(len(y_test) - sum(y_test))/ len(y_test)} %, 2nd class: {sum(y_test) / len(y_test)} %")


            #map to bool labels
            y_train = [ids_labels[id_] for id_ in y_train]
            y_val = [ids_labels[id_] for id_ in y_val]
            y_test = [ids_labels[id_] for id_ in y_test]

            
            #to categorical
            y_train = to_categorical(y_train, num_classes=2)
            y_val = to_categorical(y_val, num_classes=2)

            # Train model on dataset
            epochs=10
            batch_size = 10

            #need to reshape for this
            x_train = np.expand_dims(x_train, axis=-1)
            x_val = np.expand_dims(x_val, axis=-1)
            x_test = np.expand_dims(x_test, axis=-1)

            #same seed so generated data matches generated labels
            seed = 42
            train_generator = train_aug.flow(x_train,y_train, batch_size=batch_size)
            val_generator = val_aug.flow(x_val,y_val, batch_size=batch_size)


        else: #K fold training

            if args.test_size == 0.25:
                n_folds = 3 #50 / 25 / 25 split
            elif args.test_size == 0.5:
                n_folds = 2 #25% / 25% / 50% split
            else:
                n_folds = 2 #faster

            if args.gen_split:
                #make split into train / test
                sdir = Path(Path(__file__).parents[1], 'utils')
                x_train, y_train, x_test, y_test = gen_split_2(sdir, ims_reshaped, ids, labels, ids_labels, args.test_size)
            else:
                y_train, y_test = load_split_2(args.test_size)

            if args.train:
                
                #split images correctly (map indices to images)
                x_train = ims_from_ids(ims_reshaped, y_train)
                x_test = ims_from_ids(ims_reshaped, y_test)

                #map to bool labels
                y_train = [ids_labels[id_] for id_ in y_train]
                y_test = [ids_labels[id_] for id_ in y_test]

                #check class split in datasets
                print(f"Train set N: {len(y_train)}, 1st class: {(len(y_train) - sum(y_train))/ len(y_train)} %, 2nd class: {sum(y_train) / len(y_train)} %")
                print(f"Test set N: {len(y_test)}, 1st class: {(len(y_test) - sum(y_test))/ len(y_test)} %, 2nd class: {sum(y_test) / len(y_test)} %")

                # Train model on dataset
                epochs=10
                batch_size = 10

                #need to reshape for this
                x_train = np.expand_dims(x_train, axis=-1)
                x_test = np.expand_dims(x_test, axis=-1)

                print(f"Training with {n_folds} folds")
                k_folds_train(x_train, y_train, x_test, y_test, n_folds, args.test_size, batch_size=30)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--prep-train", dest='prep_train', action='store_true')
    parser.add_argument("--no-prep-train", dest='prep_train', action='store_false')
    parser.set_defaults(prep_train=True)
    parser.add_argument("--train", dest='train', action='store_true')
    parser.add_argument("--no-train", dest='train', action='store_false')
    parser.set_defaults(train=False)
    parser.add_argument("--plot", dest='plot', action='store_true')
    parser.add_argument("--no-plot", dest='plot', action='store_false')
    parser.set_defaults(plot=False)
    parser.add_argument("--gen-split", dest='gen_split', action='store_true')
    parser.add_argument("--no-gen-split", dest='gen_split', action='store_false')
    parser.set_defaults(gen_split=False)
    parser.add_argument("--k-fold", dest='k_fold', action='store_true')
    parser.add_argument("--no-k-fold", dest='k_fold', action='store_false')
    parser.set_defaults(k_fold=False)
    parser.add_argument("--masked", dest='masked', action='store_true')
    parser.add_argument("--no-masked", dest='masked', action='store_false')
    parser.set_defaults(masked=True)
    parser.add_argument("-t", "--test-size", default=0.25, type=float) #rest is used for k-fold training
    parser.add_argument("-n", "--n-layers", default=2, type=int) 
    args = parser.parse_args()


    if args.n_layers != 2 and args.n_layers != 3:
        raise ValueError(f"Invalid value for n_layers: {args.n_layers}")


    main(args)
    
    