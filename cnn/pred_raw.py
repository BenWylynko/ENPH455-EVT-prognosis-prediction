#try to get trained model from h5 file
from pred_all import save_all
from cnn_seg import load_model, get_metrics, ims_from_ids
from utils.split import load_ids_labels, load_split_2
import numpy as np
from pathlib import Path
import json
from load_utils import load_all_segmentations
from cnn import reshape

JSON_BASE = r'C:\Users\Ben\OneDrive-yahoo\OneDrive\Documents\ENPH455\ENPH455-EVT-prognosis-prediction'
ids = ["0157754", "0337411", "0357680", "0444750", "0513431", "0673127", "0702581", "0732825", 
        "0783417", "0840732", "0863628", "0869517", "1069211", "1175412", "1302751", "1305527", "1310557"]

def pred_all(ims, y_true, sname):
    model = load_model(sname)
    #get predictions
    preds = model.predict(ims)
    pred_classes = np.argmax(preds, axis=1)
    
    #get metric values
    y_pred = list(map(lambda x: x == 1, pred_classes))

    #get metircs
    sname += '_all'
    auc, m = get_metrics(y_pred, y_true, sname)
    print(f"All samples AUC: {auc}, conf mat: {m}")

def pred_test_set(x_test, y_true, sname):
    model = load_model(sname)
    #get predictions
    preds = model.predict(x_test)
    pred_classes = np.argmax(preds, axis=1)
    
    #get metric values
    y_pred = list(map(lambda x: x == 1, pred_classes))

    #get metircs
    auc, m = get_metrics(y_pred, y_true, sname)
    print(f"Test samples AUC: {auc}, conf mat: {m}")


if __name__ == "__main__":
    #load all images
    ims = load_all_segmentations(ids)
    #reshape
    ims_reshaped = np.array([reshape(im, (128, 128, 128)) for im in ims])

    #load model file
    sname = 'ct_model_0'

    #load labels
    ids_labels = load_ids_labels()
    #format labels
    labels = [ids_labels[id_] for id_ in ids]

    #split test set
    y_train, y_test = load_split_2(0.25)
    #split images correctly (map indices to images)
    x_train = ims_from_ids(ims_reshaped, y_train)
    x_test = ims_from_ids(ims_reshaped, y_test)
    #map to bool labels
    y_train = [ids_labels[id_] for id_ in y_train]
    y_test = [ids_labels[id_] for id_ in y_test]

    #do predictions
    pred_all(ims_reshaped, labels, sname)
    pred_test_set(x_test, y_test, sname)
    