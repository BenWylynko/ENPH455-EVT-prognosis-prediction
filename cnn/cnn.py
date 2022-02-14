import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from pathlib import Path
from load_utils import load_all_nii, load_nrrd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from datagenerator import DataGenerator
import pickle
import numpy as np
#from tensorflow import keras
from keras import models, layers, Input, Model, losses, callbacks, optimizers, regularizers
import matplotlib.pyplot as plt
import argparse

CHECKPOINT_PATH = Path(r'C:\Users\Ben\OneDrive-yahoo\OneDrive\Documents\ENPH455\ENPH455-EVT-prognosis-prediction\python3.9\cnn\3d_basic_classification.h5')

def get_model_128():
    inputs = Input((128, 128, 128, 1))

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


# Parameters
params = {'dim': (128, 128, 128),
          'n_classes': 2,
          'n_channels': 1,
          'shuffle': True, 
          'vol': 128}


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

if __name__ == "__main__":

    #load from checkpoint file
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest='train', action='store_true')
    parser.add_argument("--no-train", dest='train', action='store_false')
    parser.set_defaults(train=True)
    args = parser.parse_args()

    #load labels from pickle
    a_file = open("patient_id_label_dict.pkl", "rb")
    ids_labels = pickle.load(a_file)
    print(ids_labels)

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
    

    if args.train:

        #basic CNN
        model = get_model_128()
        model.summary()

        model.compile(optimizer='adam',
                    #loss=losses.BinaryCrossentropy(from_logits=False), #return a probability
                    loss='binary_crossentropy', 
                    metrics=['accuracy'])
        #callbacks (change learning rate, checkpoints)
        #took it out for now since I might need tensorflow
        """
        initial_learning_rate = 0.0001
        lr_schedule = optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
        )
        """

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

    else:
        model = models.load_model(CHECKPOINT_PATH)

    

    #get AUC (conf mat with final predictions from the validation set)
    #0444750, 0673127...
    #is it ok to generate AUC with the entire dataset for preliminary results?

    """
    #try the 2 cases
    predictions = model.predict_generator(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    #true labels
    true_classes = [ids_labels[str(id_.parents[0])] for id_ in test_ids]
    #metrics
    print(predicted_classes, true_classes)
    print(f"AUC with images {test_ids}: {roc_auc_score(true_classes, predicted_classes)}")
    """

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

