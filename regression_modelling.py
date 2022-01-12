import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import statsmodels.api as sm
import datetime
from sklearn.feature_selection import chi2
import argparse
from pathlib import Path

BASE = 'Z:\EVT Project Data/de-identified patient data'

#add Date back later
AllColNames = ['Age', 'Gender', 'Center',
    'MRP', 'Assistant', 'Post_6_hrs', 'After_Hours', 'Time', 
    'NIHSS_Arrival', 'NIHSS_day_1', 'tPA_given', 'CT_APECTS_arrival',
    'Core(mL)', 'Mismatch_Volume(mL)', 'collateral score',
    'Clot_arrival', 'KGH_LOS', 'MRSS_90days']

ACCColNames = [
    'LOV', 'SIDE', 'HU', 'Hyperdense Thro', 
    'Ca at LVO', 'Ca Number', 'ICAS Proximal', 'Ca PA/Supra ICA',
    'Comparison CTRL', 'Tortuos parent art', 'Kink Parent',
    'Device', 'ICA OCCL on CTA']


def filter_nans(X, y):
    """
    Filter out NaN rows from X, removing corresponding rows from y
    Returns X without NaN rows, Y without corresponding rows
    """
    nan_inds = np.where(X.isnull())[0]
    if len(nan_inds) > 0:
        for i in sorted(nan_inds, reverse=True):
            X.pop(i)
            y.pop(i)

    return X, y

def conf_mat(preds, ys):
    """
    Basic confusion matrix, to check logistic regression performance
    """

    #(N, 2) to (N) 
    preds_bin = np.array([0 if p[0] > p[1] else 1 for p in preds])
    TP = len(np.where(np.logical_and(preds_bin == 1, ys == 1))[0])
    TN = len(np.where(np.logical_and(preds_bin == 0, ys == 0))[0])
    FP = len(np.where(np.logical_and(preds_bin == 1, ys == 0))[0])
    FN = len(np.where(np.logical_and(preds_bin == 0, ys == 1))[0])
    m = np.array([[TP, FP], [FN, TN]])
    print(f"Confusion matrix: {m}")

def log_std_err(X, Y, log):
    """
    Get standard error for sklearn logistic regression
    """

    #TODO: Why are some of the variances negative? This shouldn't be possible

    #source: https://stats.stackexchange.com/questions/89484/how-to-compute-the-standard-errors-of-a-logistic-regressions-coefficients
    predProbs = log.predict_proba(X) #(43, 2)

    #for cov matrix computation, put column of 1s in front
    X_m = np.hstack([np.ones((X.shape[0], 1)), X])
    #diagonals filled with each predicted obs variance
    V = np.diagflat(np.product(predProbs, axis=1))
    #mat mul to get cov matrix
    M = np.linalg.inv(X_m.T @ V @ X_m)
    #TODO: take abs value, this is bad but doing for now
    M = np.absolute(M)
    #sqrts of diagonals for standard errors
    std_errs = np.sqrt(np.diag(M))

    #need to pop a value to get correct num (1st I think)
    std_errs = np.delete(std_errs, 0)

    return std_errs

def fit_log(X, Y, single=True):
    """
    Fit a logistic regression with each variable in X
    Y is TICI / Reperfusion score
    Single: True if running with a single predictor
    Return estimate, standard error, p value

    If single is False, return arrays of beta, serr, p for each independent feature
    """

    if len(X.shape) == 1:
        X = np.array(X).reshape(-1, 1)
    
    if single:
        mod = sm.Logit(Y,X)
        fii = mod.fit()
        summary = fii.summary2()

        beta = summary.tables[1]['Coef.'][0]
        serr = summary.tables[1]['Std.Err.'][0]
        p = summary.tables[1]['P>|z|'][0]
        return beta, serr, p
    else:
        log = LogisticRegression(random_state=0, fit_intercept=True, penalty='none').fit(X, Y)
        betas = log.coef_
        betas = np.reshape(betas, (betas.shape[1])) #estimate

        #get standard error
        serrs = log_std_err(X, Y, log)

        #get p values
        scores, ps = chi2(X, Y)

        return betas, serrs, ps


def remove_samples(X, Y):
    """
    Remove all samples which have NaN in any of the features
    """
    #get indices of NaNs
    for i, (name, row) in enumerate(X.iterrows()):
        #if any value is NaN, remove the row from X
        if any(row.isnull()):
            X = X.drop([i])
            Y = Y.drop([i])

    return X, Y

def table2_model(df, labels, sheet_name):
    """
    Generate stats for table 2 (simple log reg models)
    """

    if sheet_name == 'all':
        cols = AllColNames
    else:
        cols = ACCColNames

    print(f"\nTable 2 modelling for {sheet_name} data\n")
    betas = np.empty((len(cols)))
    serrs = np.empty((len(cols)))
    ps = np.empty((len(cols)))
    for i, colName in enumerate(cols):
        print(f"Performing logistic regression with column {colName}")
        #generate coeffs with single predictor models
        betas[i], serrs[i], ps[i] = fit_log(df[colName], labels.copy())
        print(f"Beta: {betas[i]}, standard error: {serrs[i]}, p value: {ps[i]}")

    data = np.array([cols, betas, serrs, ps]).T
    return pd.DataFrame(data = data,
        columns=['Feature name', 'Estimate', 'Standard Error', 'P value'])

def table3_model(df, labels, sheet_name):
    """
    Generate stats for table 3 (multiple log reg models)
    """

    if sheet_name == 'all':
        cols = AllColNames
    else:
        cols = ACCColNames

    #create model for All cases
    data = df[cols]

    #generate coeffs with multiple predictors model
    betas, serrs, ps = fit_log(data, labels.copy(), single=False)
    
    #return dataframe
    data = np.column_stack((cols, betas, serrs, ps))
    return pd.DataFrame(data = data,
        columns=['Feature name', 'Estimate', 'Standard Error', 'P value'])


def main(fpath):

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

    
    ### Rove features with low likelihood of being relevant, or redundant
    headings_remove = ['ACC', 'ratio', 'DC_Disposition', 'Comments']
    for h in headings_remove:
        if h in df_all.columns:
            del df_all[h]
        if h in df_acc.columns:
            del df_acc[h]


    ### standardize the data
    # all
    all_cols = df_all.columns
    scaler_all = MinMaxScaler().fit(df_all)
    df_all = scaler_all.transform(df_all)

    # acc
    acc_cols = df_acc.columns
    scaler_acc = MinMaxScaler().fit(df_acc)
    df_acc = scaler_acc.transform(df_acc)

    #cast back to dataframes
    df_all = pd.DataFrame(data=df_all, columns=all_cols)
    df_acc = pd.DataFrame(data=df_acc, columns=acc_cols)



    """
    Table 2 (modelling each variable individually against
    TICI / reperfusion score)
    """
    t2_all = table2_model(df_all, Reperf_bin, 'all')
    t2_acc = table2_model(df_acc, TICI_bin, 'acc')


    """
    Table 3
    Modelling the effect of each feature in 2 logistic regression models (1 for All cases, 1 for ACC)
    Trying this first since it makes the most of the available data, and I could create 2 separate SVMs
    which use the 2 sets of features independently (ensemble)
    """
    print("\nPerforming Table 3 regression modelling\n")

    t3_all = table3_model(df_all, Reperf_bin, 'all')
    t3_acc = table3_model(df_acc, TICI_bin, 'acc')

    #save analysis values in spreadsheet
    results_path = Path(BASE, fpath.stem + '_log_results.xlsx')
    print(results_path)

    #save
    with pd.ExcelWriter(results_path) as writer:  
        t2_all.to_excel(writer, sheet_name='Table 2 All')
        t2_acc.to_excel(writer, sheet_name='Table 2 ACC')
        t3_all.to_excel(writer, sheet_name='Table 3 All')
        t3_acc.to_excel(writer, sheet_name='Table 3 ACC')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-in", "--infile", default="encoded.xlsx", 
        help="Name of file to load for modelling")
    args = parser.parse_args()

    fpath = Path(BASE, args.infile)
    assert fpath.exists()

    main(fpath)


