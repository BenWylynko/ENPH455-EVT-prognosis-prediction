import argparse
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from pathlib import Path
import numpy as np
from sklearn import metrics

BASE = 'Z:\EVT Project Data/de-identified patient data'


#def svmWrapper()

def conf_mat(preds, ys):
    """
    Basic confusion matrix
    """

    TP = len(np.where(np.logical_and(preds == 1, ys == 1))[0])
    TN = len(np.where(np.logical_and(preds == 0, ys == 0))[0])
    FP = len(np.where(np.logical_and(preds == 1, ys == 0))[0])
    FN = len(np.where(np.logical_and(preds == 0, ys == 1))[0])
    m = np.array([[TP, FP], [FN, TN]])
    print(f"Confusion matrix: {m}")

def load(fpath):
    #load data
    print(f"Loading spreadsheet {fpath}")
    df_all = pd.read_excel( #use "Reperfusion score" instead of TICI
        fpath,  
        sheet_name='All observations')

    df_acc = pd.read_excel( #use TICI
        fpath, 
        sheet_name='ACC cases')
    return df_all, df_acc

def load_one(fpath):
    #load data
    print(f"Loading spreadsheet {fpath}")
    df = pd.read_excel( #use TICI
        fpath)
    return df

def get_auc(y, pred):
    return metrics.roc_auc_score(y, pred)

def svm_gridsearch(x_train, y_train, x_test):
    param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'gamma':['scale', 'auto'],
              'kernel': ['linear', 'rbf']} 

    grid = GridSearchCV(svm.SVC(), param_grid)

    # fitting the model for grid search 
    grid.fit(x_train, y_train)

    # return predictions for best parameter set and best parameters
    return grid.best_params_, grid.predict(x_test)


def main(fpath):
    #soure https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python

    #load data
    df_all, df_acc = load(fpath)
    print(df_all.shape)
    print(df_acc.shape)
 
    #we don't need to map to binary again if we're passing the feature removed sheet
    if "encoded.xlsx" in str(fpath):
        TICI_bin = df_acc["TICI"] >= 3
        Reperf_bin = df_all["Reperfusion_score"] >= 3
    else: #using the feature removed sheet
        TICI_bin = df_acc["TICI"]
        Reperf_bin = df_all["Reperfusion_score"]

        #also change the columns to use (only want columns in the sheet)
        global AllColNames, ACCColNames
        AllColNames = list(df_all.columns)
        ACCColNames = list(df_acc.columns)
        AllColNames.remove('Unnamed: 0')
        AllColNames.remove('Reperfusion_score')
        ACCColNames.remove('Unnamed: 0')
        ACCColNames.remove('TICI')

    df_all = df_all.drop(columns=["Unnamed: 0", "Reperfusion_score"])
    df_acc = df_acc.drop(columns=["Unnamed: 0", "TICI"])


    #split both datasets into train / test
    x_train_all, x_test_all, y_train_all, y_test_all = train_test_split(df_all, Reperf_bin, 
        test_size=0.3,random_state=109, stratify=Reperf_bin) 
    print(df_acc.shape, TICI_bin.shape)
    x_train_acc, x_test_acc, y_train_acc, y_test_acc = train_test_split(df_acc, TICI_bin, 
        test_size=0.3,random_state=110, stratify=TICI_bin) 


    #prediction with gridsearch (to also get best parameters)
    params_all, preds_all = svm_gridsearch(x_train_all, y_train_all, x_test_all)
    print(params_all)
    params_acc, preds_acc = svm_gridsearch(x_train_acc, y_train_acc, x_test_acc)
    print(params_acc)

    #generate confusion matrices
    conf_mat(preds_all, y_test_all)
    conf_mat(preds_acc, y_test_acc)

    #evaluation scores 
    #AUC
    print(f"Model AUC for all: {get_auc(y_test_all, preds_all)}")
    print(f"Model AUC for acc: {get_auc(y_test_acc, preds_acc)}")

    #specificity (how many bad outcomes correctly identified) and sensitivity (how many good outcomes correctly identified)
    TN_all, FP_all, FN_all, TP_all = metrics.confusion_matrix(y_test_all, preds_all).ravel()
    sens_all = TP_all / (TP_all + FN_all)
    spec_all = TN_all / (FN_all + FP_all)
    TN_acc, FP_acc, FN_acc, TP_acc = metrics.confusion_matrix(y_test_acc, preds_acc).ravel()
    sens_acc = TP_acc / (TP_acc + FN_acc)
    spec_acc = TN_acc / (FN_acc + FP_acc)
    print(f"For all, specificity = {spec_all}, sensitivity = {sens_all}")
    print(f"For acc, specificity = {spec_acc}, sensitivity = {sens_acc}")

def main_one(fpath):
     #load data
    df = load_one(fpath)
    print(df.shape)
 
    #we don't need to map to binary again if we're passing the feature removed sheet
    if "encoded.xlsx" in str(fpath):
        TICI_bin = df["TICI"] >= 3
    else: #using the feature removed sheet
        TICI_bin = df["TICI"]

        #also change the columns to use (only want columns in the sheet)
        global colNames
        colNames = list(df.columns)
        colNames.remove('Unnamed: 0')
        colNames.remove('TICI')

    df = df.drop(columns=["Unnamed: 0", "TICI"])

    print(f"Building SVM with features {df.columns}")
    #split both datasets into train / test
    x_train, x_test, y_train, y_test = train_test_split(df, TICI_bin, 
        test_size=0.3,random_state=109, stratify=TICI_bin) 
    print(df.shape, TICI_bin.shape)

    #prediction with gridsearch (to also get best parameters)
    params, preds = svm_gridsearch(x_train, y_train, x_test)
    print(params)

    #generate confusion matrices
    conf_mat(preds, y_test)

    #evaluation scores 
    #AUC
    print(f"Model AUC for all: {get_auc(y_test, preds)}")

    #specificity (how many bad outcomes correctly identified) and sensitivity (how many good outcomes correctly identified)
    TN, FP, FN, TP = metrics.confusion_matrix(y_test, preds).ravel()
    sens = TP / (TP + FN)
    spec = TN / (FN + FP)
    print(f"For all, specificity = {spec}, sensitivity = {sens}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-in", "--infile", default="feature_removed_corr.xlsx", 
        help="Name of file to load for modelling")
    parser.add_argument("-s", default="both", 
        help="2 sheets or no")
    args = parser.parse_args()

    fpath = Path(BASE, args.infile)
    assert fpath.exists()

    print(f"running script")
    if args.s == 'both':
        main(fpath)
    else:
        main_one(fpath)