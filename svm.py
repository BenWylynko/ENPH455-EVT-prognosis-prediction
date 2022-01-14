import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from pathlib import Path
import numpy as np

BASE = 'Z:\EVT Project Data/de-identified patient data'

#wrapper class for sklearn SVM
class SVM():
    def __init__(self, kern = 'linear'):
        #initialize SVM model, specifying kernel
        self.clf = svm.SVC(kernel = kern)

    def train(self, X_train, y_train):
        #train with training set
        self.clf.fit(X_train, y_train)

    def predict(self, X_test):
        #get predicted class for a row / set of rows
        y_pred = self.clf.predict(X_test)
        return y_pred

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

def main(fpath):
    #soure https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python

    #load data
    print(f"Loading spreadsheet {fpath}")
    df_all = pd.read_excel( #use "Reperfusion score" instead of TICI
        fpath, 
        nrows=82, 
        sheet_name='All observations')

    df_acc = pd.read_excel( #use TICI
        fpath, 
        nrows=43, 
        sheet_name='ACC cases')

    #we don't need to map to binary again if we're passing the feature removed sheet
    if "feature_removed" not in str(fpath):
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
        test_size=0.3,random_state=109) 
    x_train_acc, x_test_acc, y_train_acc, y_test_acc = train_test_split(df_acc, TICI_bin, 
        test_size=0.3,random_state=110) 

    
    #SVM models for both datasets 
    kern = 'rbf'
    svm_all = SVM(kern)
    svm_acc = SVM(kern)
    #train
    svm_all.train(x_train_all, y_train_all)
    svm_acc.train(x_train_acc, y_train_acc)
    #test
    preds_all = svm_all.predict(x_test_all)
    preds_acc = svm_acc.predict(x_test_acc)

    #generate confusion matrices
    conf_mat(preds_all, y_test_all)
    conf_mat(preds_acc, y_test_acc)

    #evaluation scores (AUC)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-in", "--infile", default="feature_removed.xlsx", 
        help="Name of file to load for modelling")
    args = parser.parse_args()

    fpath = Path(BASE, args.infile)
    assert fpath.exists()

    main(fpath)