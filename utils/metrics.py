from sklearn import metrics
import numpy as np
import json
from pathlib import Path

JSON_BASE = r'C:\Users\Ben\OneDrive-yahoo\OneDrive\Documents\ENPH455\ENPH455-EVT-prognosis-prediction'

def get_auc(y, pred):
    return metrics.roc_auc_score(y, pred)

def get_metrics(y_pred, y_true, model, test_size, sname=None):

    #evaluation scores 
    #TP, FP
    TN, FP, FN, TP = metrics.confusion_matrix(y_true, y_pred).ravel()

    #AUC
    auc = get_auc(y_true, y_pred)

    #specificity (how many bad outcomes correctly identified) and sensitivity (how many good outcomes correctly identified)
    sens = TP / (TP + FN)
    spec = TN / (FN + FP)

    #precision (fraction of true positive over all positive identified) / PPV
    prec = TP / (TP + FP)

    #NPV (true negatives over all negatives identified)
    npv = TN / (TN + FN)

    #F1
    F1 = 2*prec*sens / (prec + sens)

    m = np.array([[TP, FP], [FN, TN]])
    print(f"Confusion matrix: {m}")
    print(f"AUC: {auc}")
    print(f"specificity: {spec}")
    print(f"sensitivity: {sens}")
    print(f"precision: {prec}")
    print(f"NPV: {npv}")
    print(f"F1: {F1}")

    #save
    d = {
        'auc': auc, 
        'sens': sens, 
        'spec': spec, 
        'prec': prec, 
        'npv': npv, 
        'F1': F1, 
        'conf_mat': m.tolist()
    }
    if sname is None:
        sname = f'{model}_{test_size}'

    print(f'save name: {sname}')
    with open(Path(JSON_BASE, 'results', f'{sname}.json'), 'w') as fp:
        json.dump(d, fp)
    return auc, m