import argparse
from pathlib import Path
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB, ComplementNB
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from utils.split import load_split_2, gen_split_2, load_ids_labels, gen_split_2_stratified
from utils.metrics import get_metrics
import json
from sklearn import svm
import pickle
import joblib

BASE = 'Z:\EVT Project Data/de-identified patient data'
ids = ["0157754", "0337411", "0357680", "0444750", "0513431", "0673127", "0702581",
        "0783417",  "0840732", "0863628", "0869517", "1069211", "1175412", "1302751", "1305527", "1310557"]


def load(fpath):
    #load data
    print(f"Loading spreadsheet {fpath}")
    df = pd.read_excel( #use TICI
        fpath)
    return df

def encode_nb_kn(x_train, x_test, model, mode, with_test=True):
    """Use ordinal for categorical naive bayes, onehot for everything else"""
    if model == 'nb' and mode == 'cat':
        enc = OrdinalEncoder(handle_unknown='error')
    else:
        enc = OneHotEncoder(handle_unknown='ignore')
    if with_test:
        enc.fit(pd.concat([x_train, x_test], axis=0))
        return enc.transform(x_train),  enc.transform(x_test)
    else:
        enc.fit(x_train)
        return enc.transform(x_train)

def encode_svm(x_train, x_test):
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(pd.concat([x_train, x_test], axis=0))
    return enc.transform(x_train).toarray(),  enc.transform(x_test).toarray()

def save_model(gs, name):
    joblib.dump(gs.best_estimator_, f'models/{name}.pickle')

def load_model(name):
    #return model of correct type
    with open(f'models/{name}.pickle', 'rb') as handle:
        modelInstance = joblib.load(handle)
    return modelInstance

def gridsearch(x_train, y_train, x_test, model, mode, n_folds, sname):
    """Trying different naive bayes models, each with different hyperparameters"""

    if model == 'nb':
        if mode == 'cat':
            param_grid = {'class_prior': [
                        [100.0, 1.0], 
                        [10.0, 1.0], 
                        [1.0, 1.0], 
                        [1.0, 10.0],  
                        [1.0, 100.0]],  
                    }  
            grid = GridSearchCV(CategoricalNB(), param_grid, cv=n_folds)

        elif mode == 'comp':
            param_grid = {'class_prior': [
                        [100.0, 1.0], 
                        [10.0, 1.0], 
                        [1.0, 1.0], 
                        [1.0, 10.0],  
                        [1.0, 100.0]],  
                    } 
            grid = GridSearchCV(ComplementNB(), param_grid, cv=n_folds)

    elif model == 'kn':
        param_grid = {
        'weight':
            ['uniform', 'distance']
        }
        grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=n_folds)

    else: #svm
        param_grid = {'C': [0.1, 1, 10, 100],  
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
                'gamma':['scale', 'auto'],
                'kernel': ['linear', 'rbf'],
                'class_weight': ['balanced', #inverse of class dist
                    {True: 100.0, False: 1.0}, 
                    {True: 10.0, False: 1.0}, 
                    {True: 1.0, False: 1.0}, 
                    {True: 1.0, False: 10.0}, 
                    {True: 1.0, False: 100.0}, 
                    ], 
                } 

        grid = GridSearchCV(svm.SVC(), param_grid, cv=n_folds)

    # fitting the model for grid search 
    grid.fit(x_train, y_train)

    #save
    save_model(grid, sname)

    # return predictions for best parameter set and best parameters
    return grid.best_params_, grid.predict(x_test)

def main(args):
    fpath = Path(BASE, args.infile)
    assert fpath.exists()

     #load data
    df = load(fpath)
 
    #load labels from pickle
    ids_labels = load_ids_labels()

    #format labels
    labels = [ids_labels[id_] for id_ in ids]
    print(f"Number of usable segmentations: {len(labels)}")

    #we don't need to map to binary again if we're passing the feature removed sheet
    if "encoded.xlsx" in str(fpath):
        TICI_bin = df["TICI"] >= 3
    else: #using the feature removed sheet
        TICI_bin = df["TICI"]

    if "Unnamed: 0" in df.columns and "TICI" in df.columns:
        df = df.drop(columns=["Unnamed: 0", "TICI"])

    #format CR so we can split
    try:
        df['CR'] = df['CR'].astype(str)
        df['CR'] = df['CR'].apply(lambda id_: '0' + id_ if len(id_) == 6 else id_)
    except KeyError:
        print("Need key 'CR' for patient IDs")

    if args.gen_split:
        sdir = Path(Path(__file__).parents[1], 'utils')
        y_train, y_test = gen_split_2(sdir, LOVs, ids, labels, ids_labels, args.test_size)
    else:
        #load splitted IDs
        y_train, y_test = load_split_2(args.test_size)

        #split rows into datasets by CR
        print("separating dataset rows")
        
        x_train = df.loc[df['CR'].isin(y_train)]
        x_test = df.loc[df['CR'].isin(y_test)]

    #remove CR from df
    x_train = x_train.drop(labels='CR', axis=1)
    x_test = x_test.drop(labels='CR', axis=1)

    #map to bool labels
    y_train = [ids_labels[id_] for id_ in y_train]
    y_test = [ids_labels[id_] for id_ in y_test]

    #Onehot feature encoding
    x_train, x_test = encode_nb_kn(x_train, x_test, args.model, args.mode)

    if args.test_size == 0.25:
        n_folds = 3 #50 / 25 / 25 split
    elif args.test_size == 0.5:
        n_folds = 2 #25% / 25% / 50% split
    else:
        n_folds = 2

    print(f"Building {args.model} with features {df.columns}")
    sname = f'{args.model}_{args.test_size}'
    params, preds = gridsearch(x_train, y_train, x_test, args.model, args.mode, n_folds, sname)

    #get metric vlaues
    get_metrics(preds, y_test, args.model, args.test_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-in", "--infile", default="failed_evt\encoded_usable_corr.xlsx", 
        help="Name of file to load for modelling")
    parser.add_argument("-m", "--model", default="nb", 
        help="Model to use for classification, options 'nb', 'kn', 'svm'") 
    parser.add_argument("-o", "--mode", default="cat", 
        help="Mode of model to use for classification, options 'cat', 'comp'") 
    parser.add_argument("--gen-split", dest='gen_split', action='store_true')
    parser.add_argument("--no-gen-split", dest='gen_split', action='store_false')
    parser.set_defaults(gen_split=False)
    parser.add_argument("-t", "--test-size", default=0.25) #rest is used for k-fold
    args = parser.parse_args()

    print(f"running script")
    main(args)