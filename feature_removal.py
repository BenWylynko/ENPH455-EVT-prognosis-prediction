"""
This script uses 2 methods to remove dependent / irrelevant features from the dataset
-remove highly correlated features by considering feature pairs
-remove feature pairs which have a large associated p value in a logistic regression model

Source: https://towardsdatascience.com/feature-selection-correlation-and-p-value-da8921bfb3cf
"""

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
            if corr.iloc[i,j] >= thres:
                if columns[j]:
                    columns[j] = False
    print(f"Removing columns {list(df.columns[columns == False])}")
    selected_columns = df.columns[columns]
    return df[selected_columns], selected_columns

def feature_removal_p_value(df, cols, y, sheet_name, sig_p = 0.05):
    """
    Remove features based on p values in logistic regression model
    Args:
        df (data)
        cols (columns in the df)
        y (array of lablels)
        sheet_name (all / acc)
        Optional: sig_p (threshold p value for removal)
    Returns:
        df with features removed (1 from each pair with high p)
    """

    def backwardElimination(x, columns):
        numVars = len(x.columns)
        for i in range(0, numVars):
            #use log reg
            #regressor_log = sm.Logit(y, x).fit() #gives singular matrix error
            #regressor_log = LogisticRegression(random_state=0, fit_intercept=True, penalty='none').fit(X, Y) #changed to log reg
            _, ps = chi2(x, y)
            #print(ps)
            maxVar = max(ps).astype(float)
            if maxVar > sig_p:
                for j in range(0, numVars - i):
                    if (ps[j].astype(float) == maxVar):
                        x = x.drop(columns = x.columns[j])
                        columns = np.delete(columns, j)

                        
        #regressor_log.summary()
        return x, columns
    
    data_filtered, selected_cols = backwardElimination(df, 
        cols)
    print(f"Removing columns {list(set(cols).difference(set(selected_cols)))}")
    return data_filtered, selected_cols

"""
Loading data
"""
fpath = 'Z:\EVT Project Data/de-identified patient data/encoded.xlsx'

df_all = pd.read_excel( #use "Reperfusion score" instead of TICI
    fpath, 
    nrows=82, 
    sheet_name='All observations')
#remove Unnamed column
df_all = df_all.iloc[:,1:-1]

df_acc = pd.read_excel( #use TICI
    fpath, 
    nrows=43, 
    sheet_name='ACC cases')
#remove unnamed, patient ID, case number cols
df_acc = df_acc.iloc[:,3:-1]

#binarized labels
Reperf_bin = df_all["Reperfusion_score"] >= 3
TICI_bin = df_acc["TICI"] >= 3

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
df_all, cols_all = feature_removal_corr(df_all, corr_all)
df_acc, cols_acc = feature_removal_corr(df_acc, corr_acc)
print(df_all.shape, len(cols_all))

#p value removal
p_thresh = 0.2
df_all, cols_all = feature_removal_p_value(df_all, cols_all, Reperf_bin, 'all', sig_p = p_thresh)
df_acc, cols_acc = feature_removal_p_value(df_acc, cols_acc, TICI_bin, 'acc', sig_p = p_thresh)
print(df_all.shape, len(cols_all))

#back to dataframes
df_all = pd.DataFrame(data=df_all, columns=cols_all)
df_all.insert(0, 'Reperfusion_score', Reperf_bin)

df_acc = pd.DataFrame(data=df_acc, columns=cols_acc)
df_acc.insert(0, 'TICI', TICI_bin)

#save
results_path = r"Z:\EVT Project Data\de-identified patient data\feature_removed.xlsx"
with pd.ExcelWriter(results_path) as writer:  
    df_all.to_excel(writer, sheet_name='All observations')
    df_acc.to_excel(writer, sheet_name='ACC cases')