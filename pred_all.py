
from baseline import load_model, encode_nb_kn, load, get_metrics
from utils.split import load_ids_labels
import numpy as np
from itertools import product
from pathlib import Path
import pandas as pd
import json

BASE = 'Z:\EVT Project Data/de-identified patient data'
JSON_BASE = r'C:\Users\Ben\OneDrive-yahoo\OneDrive\Documents\ENPH455\ENPH455-EVT-prognosis-prediction'
ids = ["0157754", "0337411", "0357680", "0444750", "0513431", "0673127", "0702581", "0732825", 
        "0783417", "0840732", "0863628", "0869517", "1069211", "1175412", "1302751", "1305527", "1310557"]

def pred_all(x, y_true, model_name, test_size):

    #N folds from test size 
    if test_size == 0.25:
        n_folds = 3 #50 / 25 / 25 split
    else:
        n_folds = 2

    aucs = [None] * n_folds
    ms = [None] * n_folds

    for i in range(n_folds):
        #load model
        sname = f'{model_name}_{test_size}'
        model = load_model(sname)

        #get predictions
        y_pred = model.predict(x)

        #get metircs
        aucs[i], ms[i] = get_metrics(y_pred, y_true, model, test_size, sname=sname)

    return np.mean(aucs), ms[0]

def save_all(mean_auc, conf_mat, model_name, test_size):
    sname = f'{model_name}_{test_size}'
    with open(Path(JSON_BASE, 'results', f'{sname}_all.json'), 'w') as fp:
        json.dump({'mean_auc': mean_auc, 'conf_mat': conf_mat.tolist()}, fp)

if __name__ == "__main__":
    #preprocessing, get predictions for ALL data for each parameter combination
    model_vals = ['nb', 'kn', 'svm'] 
    test_size_vals = [0.25, 0.2]
    infile = "failed_evt\encoded_usable_final.xlsx"
    mode = 'cat'

    for (model, test_size) in product(model_vals, test_size_vals):
        fpath = Path(BASE, infile)

        #load data
        df = load(fpath)

        #load labels
        ids_labels = load_ids_labels()
        #format labels
        labels = [ids_labels[id_] for id_ in ids]

        #prep to get rows
        if "Unnamed: 0" in df.columns and "TICI" in df.columns:
            df = df.drop(columns=["Unnamed: 0", "TICI"])
        df['CR'] = df['CR'].astype(str)
        df['CR'] = df['CR'].apply(lambda id_: '0' + id_ if len(id_) == 6 else id_)
        #get rows
        x_all = df.loc[df['CR'].isin(ids)]
        x_all = x_all.drop(labels='CR', axis=1)

        #encoding
        x = encode_nb_kn(x_all, pd.Series([]), model, mode, 
            with_test=False)

        #get avg AUC, conf matrix for 0th model
        mean_auc, conf_mat = pred_all(x, labels, model, test_size)

        print(f"Mean AUC: {mean_auc}, last model conf mat: {conf_mat}")
        #save all results
        save_all(mean_auc, conf_mat, model, test_size)
