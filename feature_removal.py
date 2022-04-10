"""
This script uses 2 methods to remove dependent / irrelevant features from the dataset
-remove highly correlated features by considering feature pairs
-remove feature pairs which have a large associated p value in a logistic regression model

Source: https://towardsdatascience.com/feature-selection-correlation-and-p-value-da8921bfb3cf
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm
from sklearn.feature_selection import chi2
np.random.seed(123)

BASE = 'Z:\EVT Project Data/de-identified patient data'

OneColNames = [ 
    'CR', 'LOV', 'SIDE', 'HU', 'Hyperdense Thro', 
    'Ca at LVO', 'Ca Number', 'ICAS Proximal', 'Ca PA/Supra ICA',
    'Comparison CTRL', 'Tortuos parent art', 'Kink Parent']


def feature_removal_corr(df, corr, thres = 0.6):
    """
    Remove features based on correlations
    Args:
        df (data)
        corr (matrix of correlation coeffs)
        Optional: thres (threshold correlation for removal)
    Returns:
        df with features removed (1 from each pair with high correlation)
    """
    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i+1, corr.shape[0]):
            if abs(corr.iloc[i,j]) >= thres:
                if columns[j]:
                    columns[j] = False
    print(f"Correlation removing columns {list(df.columns[columns == False])}")
    selected_columns = df.columns[columns]
    return df[selected_columns], selected_columns

def feature_removal_p_value(df, cols, y, sig_p = 0.05):
    """
    Remove features based on p values in logistic regression model
    Args:
        df (data)
        cols (columns in the df)
        y (array of lablels)
        Optional: sig_p (threshold p value for removal)
    Returns:
        df with features removed (1 from each pair with high p)
    """

    def backwardElimination(x, columns):
        """Remove features with p value above the threshold in a multivariate model"""
        numVars = len(x.columns)
        print(f"Number of initial columns: {numVars}")
        for i in range(0, numVars):
            #use log reg
            _, ps = chi2(x, y)
            maxVar = max(ps).astype(float)
            if maxVar > sig_p:
                for j in range(0, numVars - i - 1):
                    if (ps[j].astype(float) == maxVar):
                        print(f"Removing col {j} from dataset with {len(columns)} columns")
                        x = x.drop(columns = x.columns[j])
                        columns = np.delete(columns, j)
        return x, columns

    def remove_with_p(x, columns):
        """Remove features with p value above the threshold in a univariate model"""
        numVars = len(x.columns)
        
        #get p values in univariate models
        ps = np.zeros((numVars))
        for i in range(numVars):
            mod = sm.Logit(y, x.iloc[:, i])
            fii = mod.fit()
            summary = fii.summary2()
            ps[i] = summary.tables[1]['P>|z|'][0]

        cols_to_remove = [columns[i] for i in range(numVars) if ps[i] > sig_p]

        print(f"Removing {cols_to_remove} from dataset with {len(columns)} columns")

        x = x.drop(columns = cols_to_remove)
        columns = np.delete(columns, [columns.get_loc(col) for col in cols_to_remove])

        return x, columns

    #use univariate method
    data_filtered, selected_cols = remove_with_p(df, 
        cols)

    return data_filtered, selected_cols

def save(df_all, df_acc, cols_all, cols_acc, Reperf_bin, TICI_bin, fpath):

    #back to dataframes
    df_all = pd.DataFrame(data=df_all, columns=cols_all)
    df_all.insert(0, 'Reperfusion_score', Reperf_bin)

    df_acc = pd.DataFrame(data=df_acc, columns=cols_acc)
    df_acc.insert(0, 'TICI', TICI_bin)

    #save
    with pd.ExcelWriter(fpath) as writer:  
        df_all.to_excel(writer, sheet_name='All observations')
        df_acc.to_excel(writer, sheet_name='ACC cases')

def save_one(df, cols, TICI_bin, fpath):
    df = pd.DataFrame(data=df, columns=cols)
    df.insert(0, 'TICI', TICI_bin)

    with pd.ExcelWriter(fpath) as writer:  
        df.to_excel(writer)


def main(fpath):
    
    df = pd.read_excel( #use "Reperfusion score" instead of TICI
        fpath, 
        nrows=67)

    #remove unnamed col
    try:
        df.pop("Unnamed: 0")
    except KeyError:
        print("Unnamed col not in all df")
 
    #binarized labels
    print(df.columns)
    if df.TICI.dtypes != 'bool':
        TICI_bin = df["TICI"] >= 3
    else:
        TICI_bin = df["TICI"]

    #for both, only keep relevant columns
    df = df[OneColNames]

    """
    Show correlation coefficients
    """
    fig, ax = plt.subplots(ncols=1)

    #for all cols
    corr = df.corr()
    sns.heatmap(corr, ax=ax, xticklabels=True, yticklabels=True)

    plt.show()

    print(f"All initial cols: {OneColNames}")

    #corr removal
    corr_thresh = 0.3
    df, cols = feature_removal_corr(df, corr)

    #save after corr removal
    results_path = Path(fpath.parents[0], f"{fpath.stem}_corr.xlsx")
    save_one(df, cols, TICI_bin, results_path)

    #p value removal
    p_thresh = 0.2
    df, cols = feature_removal_p_value(df, cols, TICI_bin, sig_p = p_thresh)

    #save after both methods
    results_path = Path(fpath.parents[0], f"{fpath.stem}_final.xlsx")
    save_one(df, cols, TICI_bin, results_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-in", "--infile", default="failed+evt\encoded_usable.xlsx", 
        help="Name of file to load for modelling")
    args = parser.parse_args()

    fpath = Path(BASE, args.infile)
    print(f"Using {fpath}")
    assert fpath.exists()

    print(f"running feature removal")
    main(fpath)