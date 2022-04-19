# ENPH455-EVT-prognosis-prediction
Classification models to predict the outcome from EVT treatment for ischemic strokes, for the ENPH455 thesis course.
The code requires the following python packages: `matplotlib`, `pandas`, `sklearn`, `numpy`, `opencv`, `tensorflow Keras`.

## Directory structure
```
├───cnn
│   ├───augmented
    A open source image augmenting tool for 3D volumes: https://github.com/dhuy228/augmented-volumetric-image-generator?fbclid=IwAR233EWFalmNOdZ8oAMvGt9w3M8zhiFQApMdi8f2ls40Xq0GKrGGLYpjsEo 

│   ├───models
    Contains trained CNN models in h5 and json files

├───models
Contains trained baseline models in pickle files 

├───results
Contains json files with result metrics on test sets for each model

├───spreadsheets
Contains the spreadsheets used to train / evaluate the imaging feature models.

├───utils
Contains utils to load NIFTII and NRRD files, pickle files with patients split into train / test or train / val / test splits, <br/>
utils to get performance metrics.
```

## Spreasheets
The spreadsheets dir contains the imaging features, at different levels of processing. <br/>
**FAILED EVT** - the raw data, containing additional patient data <br/>
**encoded** - All features, encoded categorically <br/>
**encoded_usable** - TICI encoded as binary True / False <br/>
**encoded_usable_corr** - After feature removal with correlation (threshold 0.3 correlation coefficient) <br/>
**encoded_usable_final** - After feature removal with correlation and p values (0.2 threshold p value)

The CR column represents the ID of each patient. 

## Script usage
To see the available parameters for each script, run:
```
python {SCRIPT_NAME} -h
```

Brief descriptions of relevant scripts are shown below: 

`baseline.py` - trains and evaluates a baseline model <br/>
`baseline_wrapper.py` - runs `baseline.py` for all models and data splits <br/> 
`feature_removal.py` - generates `encoded_usable_corr` and `encoded_usable_final` spreadsheets <br/>
`filter_usable.py` - Converts TICI from categorically encoded to binary labels <br/>
`format_spreadsheet.py` - Generates `encoded` from `FAILED EVT` <br/>
`load_utils.py` - utils to load NRRD and NIFTII files <br/>
`regression_modelling.py` - Does logistic regression modelling of all categorical features <br/>
`univariate.py` - univariate analysis of spreadsheet data

**cnn** <br/>
`cnn.py` - train a CNN on raw images <br/>
`cnn_seg.py` - train a CNN on the segmented / masked images <br/>
`pred_all.py` - get predictions with all segmented and masked models <br/>
`pred_raw.py` - get predictions with raw image CNN


