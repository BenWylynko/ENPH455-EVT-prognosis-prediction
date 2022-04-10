import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from pathlib import Path
from load_utils import load_all_nii
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from datagenerator import DataGenerator
import pickle
import numpy as np
#from tensorflow import keras
from keras import models, layers, Input, Model, losses, callbacks, optimizers, regularizers
import matplotlib.pyplot as plt
import argparse
from utils.split import load_split_2, gen_split_2, gen_split_3, load_ids_labels, load_split_3
from load_utils import load_nrrd, pad_nrrd, resize_volume, load_all_segmentations
from utils.metrics import get_metrics
import scipy.ndimage as ndimage
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.utils import to_categorical
from augmented import generator
from tensorflow.keras.callbacks import Callback, EarlyStopping
from cnn_seg import save_model, get_auc

CHECKPOINT_PATH = Path(r'C:\Users\Ben\OneDrive-yahoo\OneDrive\Documents\ENPH455\ENPH455-EVT-prognosis-prediction\python3.9\cnn\3d_basic_classification.h5')
SAVE_PATH = r'C:\Users\Ben\Downloads\data\whole'

def get_model_128(n_conv_layers=1):
    inputs = Input((128, 128, 128, 1))

    x = layers.Conv3D(filters=64, kernel_size=2, activation="relu", 
        kernel_regularizer = regularizers.l1(1e-4), bias_regularizer = regularizers.l1(1e-4), activity_regularizer = regularizers.l1(1e-4))(inputs)
    x = layers.MaxPool3D(pool_size=3)(x)
    x = layers.BatchNormalization()(x)

    if n_conv_layers==2:
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
    model = Model(inputs, outputs, name="3dcnn")
    return model

def get_model_256():
    inputs = Input((256, 256, 256, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu", 
        kernel_regularizer = regularizers.l1(1e-4), bias_regularizer = regularizers.l1(1e-4), activity_regularizer = regularizers.l1(1e-4))(inputs)
    x = layers.MaxPool3D(pool_size=3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu", 
        kernel_regularizer = regularizers.l1(1e-4), bias_regularizer = regularizers.l1(1e-4), activity_regularizer = regularizers.l1(1e-4))(x)
    x = layers.MaxPool3D(pool_size=3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu", 
        kernel_regularizer = regularizers.l1(1e-4), bias_regularizer = regularizers.l1(1e-4), activity_regularizer = regularizers.l1(1e-4))(x)
    x = layers.MaxPool3D(pool_size=3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu", 
        kernel_regularizer = regularizers.l1(1e-4), bias_regularizer = regularizers.l1(1e-4), activity_regularizer = regularizers.l1(1e-4))(x)
    x = layers.MaxPool3D(pool_size=3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu", 
        kernel_regularizer = regularizers.l1(1e-4), bias_regularizer = regularizers.l1(1e-4), activity_regularizer = regularizers.l1(1e-4))(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=2, activation="sigmoid")(x)

    # Define the model.
    model = Model(inputs, outputs, name="3dcnn")
    return model

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

def ims_from_ids(ims, ids):
    train_inds = [ids.index(id_) for id_ in ids]
    x = np.array([ims[i] for i in train_inds])
    return x

# Parameters
params = {'dim': (128, 128, 128),
          'n_classes': 2,
          'n_channels': 1,
          'shuffle': True, 
          'vol': 128}

def k_folds_train(x_train, y_train, x_test, y_test, n_folds, test_size, epochs=10, batch_size=10):
    for i, (train_index, val_index) in enumerate(StratifiedKFold(n_folds, shuffle=True).split(x_train, y_train)):
        print(f"Training wtih {len(train_index)} train samples, {len(val_index)} val samples, {len(x_test)} test samples")
        fold_x_train ,fold_x_val=x_train[train_index], x_train[val_index]
        fold_y_train, fold_y_val=[y_train[ind] for ind in train_index], [y_train[ind] for ind in val_index]

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

    model = get_model_128()
    model.summary()
    model.compile(optimizer='adam',
                        loss='binary_crossentropy', 
                        metrics=[tf.keras.metrics.AUC()])
    checkpoint_cb = callbacks.ModelCheckpoint(
        "3d_segmentation_classification.h5", save_best_only=True
        )

    #early_stopping = EarlyStopping(monitor=tf.keras.metrics.AUC(), patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='auc', patience=3, verbose=1)

    # Fit the model
    history = model.fit(train_generator,
        steps_per_epoch = 20, #N samples / batch size is 10 / 10 = 1
        epochs = epochs, 
        validation_data = val_generator,
        validation_steps = 10, #N val samples // batch size: 5 // 10 = 0
        verbose=1, 
    )

    if ind is None:
        save_model(model, 'ct_model')
    else:
        save_model(model, f'ct_model_{ind}')

    ###test
    predictions = model.predict(x_test)
    predicted_classes = np.argmax(predictions, axis=1)
    
    #get metric values
    y_pred = list(map(lambda x: x == 1, predicted_classes))
    get_metrics(y_pred, y_test)

#list of patientid/filename (all cases I have)
ids = [Path('0157754', '7 Post Contrast_2.nii'), 
    Path('0174960', '11 Post Contrast.nii'), 
    Path('0337411', '601 Axial125x06 - imageOrientationPatient 1_11.nii'),
    Path('0357680', '4 Axial 1.25 x 1.25_1.nii'), 
    #Path('0497694', '602 Sagittal AR20 2x2 - imageOrientationPatient 1_4.nii'), 
    Path('0513431', '8 Post Contrast.nii'),  
    Path('0702581', '12 Axial 125 Std.nii'),
    Path('0732825', '602 Coronal Head - imageType DERIVED-SECONDARY-REFORMATTED-AVERAGE.nii'), 
    Path('0778928', '602 Sagittal AR20 2x2 - imageOrientationPatient 1_8.nii'), 
    Path('0783417', '10 Post Contrast_1.nii'), 
    Path('0786080', '604 Sagittal MIP 5x2 - imageType DERIVED-SECONDARY-REFORMATTED-MIP_14.nii'), 
    Path('0840732', '5 Post Contrast.nii'), 
    Path('0863628', '10 Post Contrast_3.nii'), 
    Path('0869517', '7 Post Contrast_1.nii'), 
    Path('1069211', '7 Post Contrast_4.nii'), 
    Path('1175412', '10 Post Contrast_2.nii'), 
    Path('1302751', '7 Post Contrast_1.nii'), 
]

test_ids = [
    Path('0444750', '604 Sagittal MIP 5x2 - imageType DERIVED-SECONDARY-REFORMATTED-MIP_15.nii'), 
    Path('0673127', '604 Sagittal MIP 5x2 - imageType DERIVED-SECONDARY-REFORMATTED-MIP_11.nii'),  
]

im_ids = ["0157754", "0337411", "0357680", "0444750", "0513431", "0673127", "0702581", "0732825", "0778928", 
    "0783417", "0786080", "0840732", "0863628", "0869517", "1069211", "1175412", "1302751", "1305527", "1310557"]

train_aug = generator.customImageDataGenerator(
    rotation_range = 20
)
val_aug = generator.customImageDataGenerator(
    rotation_range = 20
)
test_aug = generator.customImageDataGenerator(
    rotation_range = 20
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest='train', action='store_true')
    parser.add_argument("--no-train", dest='train', action='store_false')
    parser.set_defaults(train=True)
    parser.add_argument("--k-fold", dest='k_fold', action='store_true')
    parser.add_argument("--no-k-fold", dest='k_fold', action='store_false')
    parser.set_defaults(k_fold=False)
    parser.add_argument("-s", "--split", default='even')
    parser.add_argument("--gen-split", dest='gen_split', action='store_true')
    parser.add_argument("--no-gen-split", dest='gen_split', action='store_false')
    parser.set_defaults(gen_split=False)
    args = parser.parse_args()

    #load labels from pickle
    ids_labels = load_ids_labels()

    #split into train / test
    ids_train, ids_val, _, _ = train_test_split(
        ids, np.arange(len(ids)), test_size=0.33, random_state=42) #ys are dummy var


    # Datasets
    partition = {} # IDs
    partition['train'] = ids_train
    partition['validation'] = ids_val
    labels = ids_labels # Labels

    # Generators
    training_generator = DataGenerator(partition['train'], labels, training=True, batch_size=8, **params)
    validation_generator = DataGenerator(partition['validation'], labels, training=False, batch_size=4, **params)
    test_generator = DataGenerator(test_ids, labels, training=False, batch_size=1, **params)
    
    #load labels from pickle
    ids_labels = load_ids_labels()

    #format labels
    labels = [ids_labels[id_] for id_ in im_ids]
    print(f"Number of usable segmentations: {len(labels)}")

    if args.train:
        if not args.k_fold:
            #basic CNN
            model = get_model_128()
            model.summary()

            model.compile(optimizer='adam',
                        #loss=losses.BinaryCrossentropy(from_logits=False), #return a probability
                        loss='binary_crossentropy', 
                        metrics=['accuracy'])
            #callbacks (change learning rate, checkpoints)
            checkpoint_cb = callbacks.ModelCheckpoint(
            "3d_basic_classification.h5", save_best_only=True
            )
            early_stopping_cb = callbacks.EarlyStopping(monitor="val_loss", patience=3)

            # Train model on dataset
            epochs=10
            history = model.fit_generator(training_generator,
                                validation_data=validation_generator,
                                epochs=epochs, 
                                verbose=2, 
                                #callbacks=[checkpoint_cb, early_stopping_cb],
                                validation_steps = 1, #validating 5 times per epoch 
            )
            #plots
            # summarize history for accuracy
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'validate'], loc='upper left')
            plt.savefig('acc.png')
            plt.show()
            # summarize history for loss
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'validate'], loc='upper left')
            plt.savefig('loss.png')
            plt.show()

        else: #do K folds
            #load all images
            ims = load_all_segmentations(im_ids)
            #reshape
            ims_reshaped = np.array([reshape(im, (128, 128, 128)) for im in ims])

            #save preprocessed images
            for i, im in enumerate(ims_reshaped):
                savepath = Path(SAVE_PATH, im_ids[i])
                savepath.mkdir(parents=True, exist_ok=True)
                np.save(Path(savepath, 'im.npy'), ims_reshaped[i])

            #80%, 10%, 10% or 50%, 25%, 25%
            if args.split == 'even':
                val_test_size = 0.25
            else:
                val_test_size = 0.1

            #80%, 10%, 10% or 50%, 25%, 25%
            if args.split == 'even':
                val_test_size = 0.25
            else:
                val_test_size = 0.1

            if args.train:
                if val_test_size == 0.25:
                    n_folds = 3 #50 / 25 / 25 split
                elif val_test_size == 0.1:
                    n_folds = 9 #80 / 10 / 10 split

                if args.gen_split:
                    #make split into train / test
                    sdir = Path(Path(__file__).parents[1], 'utils')
                    x_train, y_train, x_test, y_test = gen_split_2(sdir, ims_reshaped, im_ids, labels, ids_labels, val_test_size)
                else:
                    y_train, y_test = load_split_2(val_test_size)

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

                k_folds_train(x_train, y_train, x_test, y_test, n_folds, 0.25)

    else:
        model = models.load_model(CHECKPOINT_PATH)

    #did predictiions for different sets
    #2 cases + validation set
    test_val_generator = DataGenerator(partition['validation'] + test_ids, labels, training=False, batch_size=1, **params)
    predictions = model.predict_generator(test_val_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    #true labels
    true_classes = [ids_labels[str(id_.parents[0])] for id_ in test_ids + ids_val]
    #metrics
    print(f"AUC with images {test_ids + ids_val}: {roc_auc_score(true_classes, predicted_classes)}")

    #all images
    all_generator = DataGenerator(partition['train'] + partition['validation'] + test_ids, labels, training=False, batch_size=1, **params)
    predictions = model.predict_generator(all_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    #true labels
    true_classes = [ids_labels[str(id_.parents[0])] for id_ in test_ids + ids_val + ids_train]
    #metrics
    print(f"AUC with images {test_ids + ids_val + ids_train}: {roc_auc_score(true_classes, predicted_classes)}")

