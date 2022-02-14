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

AllColNames = ['Date', 'Age', 'Gender', 'Center',
    'MRP', 'Assistant', 'Post_6_hrs', 'After_Hours', 'Time', 
    'NIHSS_Arrival', 'NIHSS_day_1', 'tPA_given', 'CT_APECTS_arrival',
    'Core(mL)', 'Mismatch_Volume(mL)', 'collateral score',
    'Clot_arrival', 'KGH_LOS', 'MRSS_90days']

ACCColNames = [
    'LOV', 'SIDE', 'HU', 'Hyperdense Thro', 
    'Ca at LVO', 'Ca Number', 'ICAS Proximal', 'Ca PA/Supra ICA',
    'Comparison CTRL', 'Tortuos parent art', 'Kink Parent',
    'Device', 'ICA OCCL on CTA']

OneColNames = [ 
    'LOV', 'SIDE', 'HU', 'Hyperdense Thro', 
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

            print(ps.shape)
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

    #use multivariate method         
    """
    data_filtered, selected_cols = backwardElimination(df, 
        cols)
    """
    #use univariate method
    data_filtered, selected_cols = remove_with_p(df, 
        cols)

    print(f"p value removing columns {list(set(cols).difference(set(selected_cols)))}")
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

"""
Loading data
"""
def main_both(fpath):
    fname = fpath.stem

    df_all = pd.read_excel( #use "Reperfusion score" instead of TICI
        fpath, 
        nrows=82, 
        sheet_name='All observations')
    
    df_acc = pd.read_excel( #use TICI
        fpath, 
        nrows=43, 
        sheet_name='ACC cases')

    #remove unnamed col from All
    try:
        df_all.pop("Unnamed: 0")
    except KeyError:
        print("Unnamed col not in all df")
    #remove unnamed col from Acc
    try:
        df_acc.pop("Unnamed: 0")
    except KeyError:
        print("Unnamed col not in acc df")
    #remove patient ID from ACC
    try:
        df_acc.pop("patient ID")
    except KeyError:
        print("patient ID not in acc df")   
    #remove case number from ACC
    try:
        df_acc.pop("Case Number")
    except KeyError:
        print("case number not in ACC df")   


    """
    if 'Unnamed: 0' in list(df_acc.columns):
        #remove unnamed, patient ID, case number cols
        df_acc = df_acc.iloc[:,3:-1]
    """

    #binarized labels
    if df_all.Reperfusion_score.dtypes != 'bool':
        Reperf_bin = df_all["Reperfusion_score"] >= 3
        TICI_bin = df_acc["TICI"] >= 3
    else:
        Reperf_bin = df_all["Reperfusion_score"]
        TICI_bin = df_acc["TICI"]

    #for both, only keep relevant columns
    df_all = df_all[AllColNames]
    df_acc = df_acc[ACCColNames]

    """
    Show correlation coefficients
    """
    fig, (ax_all,ax_acc) = plt.subplots(ncols=2)

    #for all cols
    corr_all = df_all.corr()
    sns.heatmap(corr_all, ax=ax_all, xticklabels=True, yticklabels=True)

    #ACC
    corr_acc = df_acc.corr()
    sns.heatmap(corr_acc, ax=ax_acc, xticklabels=True, yticklabels=True)

    #plt.show()

    print(f"All initial cols: {AllColNames}")
    print(f"Acc initial cols: {ACCColNames}")

    #corr removal
    corr_thresh = 0.5
    df_all, cols_all = feature_removal_corr(df_all, corr_all)
    df_acc, cols_acc = feature_removal_corr(df_acc, corr_acc)

    #save after corr removal
    results_path = f"Z:\EVT Project Data\de-identified patient data\{fname}_corr.xlsx"
    save(df_all, df_acc, cols_all, cols_acc, Reperf_bin, TICI_bin, results_path)

    #p value removal
    p_thresh = 0.05
    df_all, cols_all = feature_removal_p_value(df_all, cols_all, Reperf_bin, sig_p = p_thresh)
    df_acc, cols_acc = feature_removal_p_value(df_acc, cols_acc, TICI_bin, sig_p = p_thresh)

    #save after both methods
    results_path = f"Z:\EVT Project Data\de-identified patient data\{fname}_final.xlsx"
    save(df_all, df_acc, cols_all, cols_acc, Reperf_bin, TICI_bin, results_path)

def main_one(fpath):
    
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
    parser.add_argument("-in", "--infile", default="encoded.xlsx", 
        help="Name of file to load for modelling")
    parser.add_argument("-s", default="both", 
        help="2 sheets or no")
    args = parser.parse_args()

    fpath = Path(BASE, args.infile)
    print(f"Using {fpath}")
    assert fpath.exists()

    print(f"running feature removal")
    if args.s == 'both':
        main_both(fpath)
    else:
        main_one(fpath)