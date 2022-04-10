import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import datetime
from sklearn.feature_selection import chi2
import argparse
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

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

feature_mapping = {
    'LOV': ['ICA', 'Terminus', 'M1', 'M2'], 
    'SIDE': ['Rt', 'Lt'], 
    'HU': ['<50', '50-99'], 
    'Hyperdense Thro': ['No', 'Yes'], 
    'Hyperdense': ['No', 'Yes'], 
    'Ca at LVO': ['No', 'Yes'], 
    'Ca': ['No', 'Yes'],
    'Ca number': ['0', '1'], 
    'ICAS Proximal': ['No', 'Yes'], 
    'ICAS': ['No', 'Yes'], 
    'Ca PA/Supra ICA': ['No', 'Yes'],
    'Ca_PA': ['No', 'Yes'],
    'Comparison CTRL': ['less severe', 'same', 'more severe'],
    'Comparison': ['less severe', 'same', 'more severe'],
    'Tortuos parent art': ['No', 'Yes'],  
    'Tortuos': ['No', 'Yes'],  
    'Kink Parent': ['Kink', 'Coil', 'None'], 
    'TICI': ['0', '1', '2A', '2B', '2C', '3'], 
}

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

def fit_log(X, Y, single=True, name='corr'):
    """
    Fit a logistic regression with each variable in X
    Y is TICI / Reperfusion score
    Single: True if running with a single predictor
    Return estimate, standard error, p value

    If single is False, return arrays of beta, serr, p for each independent feature
    """
    if isinstance(X, pd.DataFrame):
        name = X.columns[0]
    else: #Series
        name = X.name
    if single: #return at least 2 sets of values
        first_val = X.unique()[0]
        #X = pd.get_dummies(X, columns=[name])
        X_in = pd.DataFrame([X, Y], index=[name, 'TICI']).T

        #with name col as categorical type
        X_in[name] = X_in[name].astype('category')
        X_in['TICI'] = X_in['TICI'].astype(int)
 
        mod = smf.logit(f"TICI ~ C({name}, Treatment('{first_val}'))", data=X_in)
        fii = mod.fit()
        summary = fii.summary2()
        betas = summary.tables[1]['Coef.']
        serrs = summary.tables[1]['Std.Err.']
        ps = summary.tables[1]['P>|z|']
        return betas, serrs, ps
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, 
            random_state=71, stratify=Y)

        #should be able to do multivariate with statsmodels (R backend)
        #X_in = pd.concat([X_train, y_train], axis=1)
        X_in = pd.concat([X, Y], axis=1)
        X_in['TICI'] = X_in['TICI'].astype(int)
        
        if name == 'corr':
            mod = smf.logit(f"TICI ~ LOV + SIDE + HU + Hyperdense + Ca + ICAS + Ca_PA + Comparison + Tortuos", 
                data=X_in, family='binomial')
        else:
            mod = smf.logit(f"TICI ~ LOV + SIDE + HU + Hyperdense + Ca + Ca_num + ICAS + Ca_PA + Comparison + Tortuos + Kink", 
                data=X_in, family='binomial')
        fii = mod.fit()
        summary = fii.summary2()
        betas = summary.tables[1]['Coef.']
        serrs = summary.tables[1]['Std.Err.']
        ps = summary.tables[1]['P>|z|']

        
        #get AUC
        yhat = mod.predict(X)
        preds = list(map(round,yhat))      
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
    elif sheet_name == 'acc':
        cols = ACCColNames
    else:
        cols = df.columns
        #dropping features due to singular matrix error
        #drop kink parent for now
        #cols = cols.drop("Kink Parent")
        #drop Comparison CTRL too\
        if "Comparison CTRL" in cols:
            cols = cols.drop("Comparison CTRL")
        elif "Comparison" in cols:
            cols = cols.drop("Comparison")

    print(f"\nTable 2 modelling for {sheet_name} data\n")
    betas = []
    serrs = []
    ps = []
    for i, colName in enumerate(cols):
        print(f"Performing logistic regression with column {colName}")
        #generate coeffs with single predictor models
        b, s, p = fit_log(df[colName], labels.copy())
        betas.append(np.array(b))
        serrs.append(np.array(s))
        ps.append(np.array(p))

    references = [feature_mapping[cat][0] for cat in cols]
    cat_vals = [feature_mapping[cat][1:] for cat in cols]
    data = np.array([cols, betas, serrs, ps, references, cat_vals]).T
    return pd.DataFrame(data = data,
        columns=['Feature name', 'Estimate', 'Standard Error', 'P value', 'reference', 'categorical values'])

def table3_model(df, labels, sheet_name, fname):
    """
    Generate stats for table 3 (multiple log reg models)
    """

    if sheet_name == 'all':
        cols = AllColNames
    elif sheet_name == 'acc':
        cols = ACCColNames
    else:
        cols = df.columns

    #create model for All cases
    data = df[cols]

    #generate coeffs with multiple predictors model
    betas, serrs, ps = fit_log(data, labels.copy(), single=False, name=fname)

    #return dataframe
    data = np.column_stack((cols, betas, serrs, ps))
    return pd.DataFrame(data = data,
        columns=['Feature name', 'Estimate', 'Standard Error', 'P value'])

def main(fpath, scale, do_t2=True, do_t3=True):
    df = pd.read_excel( 
        fpath, 
        nrows=82)

    #we don't need to map to binary again if we're passing the feature removed sheet
    if "usable" not in str(fpath):
        TICI_bin = df["TICI"] >= 3
    else: #using the feature removed sheet
        TICI_bin = df["TICI"]

    if 'Device' in df.columns:
        df = df.drop(columns=["Device"])
    if 'ICA OCCL on CTA' in df.columns:
        df = df.drop(columns=["ICA OCCL on CTA"])

    df = df.drop(columns=["Unnamed: 0", "TICI"])

    ### standardize the data
    if scale:
        # all
        cols = df.columns
        scaler = MinMaxScaler().fit(df)
        df = scaler.transform(df)

        #cast back to dataframe
        df = pd.DataFrame(data=df, columns=cols)


    """
    Table 2 (modelling each variable individually against
    TICI / reperfusion score)
    """
    if do_t2:
        t2 = table2_model(df, TICI_bin, 'one')


    """
    Table 3
    Modelling the effect of each feature in 2 logistic regression models (1 for All cases, 1 for ACC)
    Trying this first since it makes the most of the available data, and I could create 2 separate SVMs
    which use the 2 sets of features independently (ensemble)
    """
    print("\nPerforming Table 3 regression modelling\n")
    if 'corr' in str(fpath):
        fname = 'corr'
    else:
        fname= 'aa'
    if do_t3:
        t3 = table3_model(df, TICI_bin, 'one', fname=fname)

    #save analysis values in spreadsheet
    results_path = Path(fpath.parents[0], fpath.stem + '_log_results.xlsx')

    #save
    with pd.ExcelWriter(results_path) as writer:  
        t2.to_excel(writer, sheet_name='Table 2')
        if do_t3:
            t3.to_excel(writer, sheet_name='Table 3')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-in", "--infile", default="encoded.xlsx", 
        help="Name of file to load for modelling")
    parser.add_argument("-s", default="both", 
        help="2 sheets or no")
    parser.add_argument("--scale", dest='scale', action='store_true')
    parser.add_argument("--no-scale", dest='scale', action='store_false')
    parser.set_defaults(scale=True)
    parser.add_argument("--no-t2", dest='t2', action='store_false')
    parser.set_defaults(t2=True)
    parser.add_argument("--no-t3", dest='t3', action='store_false')
    parser.set_defaults(t3=True)
    args = parser.parse_args()

    fpath = Path(BASE, args.infile)
    assert fpath.exists()

    main(fpath, args.scale, args.t2, args.t3)


