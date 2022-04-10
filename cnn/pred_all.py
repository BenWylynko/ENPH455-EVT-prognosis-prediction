#do prediction with all patient data
from cnn_seg import load_model, preprocess, get_metrics
from load_utils import load_all_segmentations, load_masked_imgs
from utils.split import load_ids_labels
import numpy as np
from itertools import product
from pathlib import Path
import json

JSON_BASE = r'C:\Users\Ben\OneDrive-yahoo\OneDrive\Documents\ENPH455\ENPH455-EVT-prognosis-prediction'
ids = ["0157754", "0337411", "0357680", "0444750", "0513431", "0673127", "0702581", "0732825", 
        "0783417", "0840732", "0863628", "0869517", "1069211", "1175412", "1302751", "1305527", "1310557"]

def pred_all(ims, y_true, masked, test_size, n_layers, img_shape):

    #N folds from test size 
    if test_size == 0.25:
        n_folds = 3 #50 / 25 / 25 split
    else:
        n_folds = 2

    aucs = [None] * n_folds
    ms = [None] * n_folds

    for i in range(n_folds):

        #load model
        if masked:
            sname = f'cnn_seg_masked__{i}_{test_size}_{n_layers}_{img_shape}'
        else:
            sname = f'cnn_seg__{i}_{test_size}_{n_layers}_{img_shape}'
        model = load_model(sname)

        #get predictions
        preds = model.predict(ims)
        pred_classes = np.argmax(preds, axis=1)
        
        #get metric values
        y_pred = list(map(lambda x: x == 1, pred_classes))

        #get metircs
        aucs[i], ms[i] = get_metrics(y_pred, y_true, sname, save=False)

    return np.mean(aucs), ms[0] #just give conf mat of 1st model, not for all folds

def save_all(mean_auc, conf_mat, masked, test_size, n_layers, img_shape):
    if masked:
        sname = f'cnn_seg_masked_{test_size}_{n_layers}_{img_shape}'
    else:
        sname = f'cnn_seg_{test_size}_{n_layers}_{img_shape}'
    with open(Path(JSON_BASE, 'results', f'{sname}_all.json'), 'w') as fp:
        json.dump({'mean_auc': mean_auc, 'conf_mat': conf_mat.tolist()}, fp)

if __name__ == "__main__":
    #preprocessing, get predictions for ALL data for each parameter combination
    masked_vals = [True, False]
    test_size_vals = [0.25, 0.2]
    n_layers_vals = [2, 3]

    for (masked, test_size, n_layers) in product(masked_vals, test_size_vals, n_layers_vals):
        #set img shape manually
        img_shape = (39, 162, 55)

        #load ims for all IDS
        if masked:
            ims = load_masked_imgs(ids)
        else:
            ims = load_all_segmentations(ids)

        #load labels
        ids_labels = load_ids_labels()
        #format labels
        labels = [ids_labels[id_] for id_ in ids]
        #categorical
        #labels = to_categorical(labels, num_classes=2)

        ims_reshaped = preprocess(ims, ids)

        #get avg AUC, conf matrix for 0th model
        mean_auc, conf_mat = pred_all(ims_reshaped, labels, masked, test_size, n_layers, 
            img_shape)

        print(f"Mean AUC: {mean_auc}, last model conf mat: {conf_mat}")
        #save all results
        save_all(mean_auc, conf_mat, masked, test_size, n_layers, img_shape)