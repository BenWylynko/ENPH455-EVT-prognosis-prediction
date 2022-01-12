import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np
import csv
import datetime
from scipy.stats import spearmanr
import random
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression

os.chdir('Z:\EVT Project Data/de-identified patient data')

headings = ['patient ID', 'Case Number', 'ACC', 'Date', 'Age', 'Gender', 'Center',
     'MRP', 'Assistant', 'Post_6_hrs', 'After_Hours', 'Time', 
      'NIHSS_Arrival', 'NIHSS_day_1', 'tPA_given', 'CT_APECTS_arrival',
      'Core(mL)', 'Mismatch_Volume(mL)', 'ratio', 'collateral score',
      'Clot_arrival', 'KGH_LOS', 'DC_Disposition', 'MRSS_90days', 
       'LOV', 'SIDE', 'HU', 'Hyperdense Thro', 
       'Ca at LVO', 'Ca Number', 'ICAS Proximal', 'Ca PA/Supra ICA',
       'Comparison CTRL', 'Tortuos parent art', 'Kink Parent', 'TICI',
        'Device', 'Passes', 'Complication', 'ICA OCCL on CTA', 
        'Comments']

headingsAll = ['Age', 'Gender', 'Center',
            'MRP', 'Assistant', 'Post_6_hrs', 'After_Hours', 'Time', 
            'NIHSS_Arrival', 'NIHSS_day_1', 'tPA_given', 'CT_APECTS_arrival',
            'Core(mL)', 'Mismatch_Volume(mL)', 'collateral score',
            'Clot_arrival', 'KGH_LOS', 'MRSS_90days']

headingsAcc = [
            'LOV', 'SIDE', 'HU', 'Hyperdense Thro', 
            'Ca at LVO', 'Ca Number', 'ICAS Proximal', 'Ca PA/Supra ICA',
            'Comparison CTRL', 'Tortuos parent art', 'Kink Parent',
            'Device', 'Passes', 'Complication', 'ICA OCCL on CTA']
headingsAcc.append('TICI') #there's 1 missing TICI value so I'll impute it. This isn't ideal...

#define headings which should be categorical
catHeadingsAll = ['NIHSS_Arrival', 'NIHSS_day_1', 'tPA_given', 'collateral score']
catHeadingsAcc = ['Ca at LVO', 'Ca Number', 'ICAS Proximal', 'Ca PA/Supra ICA', 
    'Comparison CTRL', 'Tortuos parent art', 'Kink Parent', 'Device', 'Passes', 
    'Complication', 'ICA OCCL on CTA']

def mice_inpute_feature(X, sheet):
    """
    Use MICE to impute the missing values. 
    Should only be used after all NaNs which shouldn't be imputed with MICE have been filled since this method
    will impute all NaNs. 
    """

    #specify headings to use to build the linear regression models
    if sheet == "all":
        headings = headingsAll
        headingsCat = catHeadingsAll
    else:
        headings = headingsAcc
        headingsCat = catHeadingsAcc

    #get min and max bounds of each column which will be imputed
    min_maxes = np.empty((len(headingsCat), 2))
    for i, h in enumerate(headingsCat):
        min_maxes[i] = (X[h].min(), X[h].max())


    #source: https://www.numpyninja.com/post/mice-and-knn-missing-value-imputations-through-python
    #only use specified headings for lin reg models and only impute values in these headings
    lr = LinearRegression()
    imp = IterativeImputer(estimator=lr,missing_values=np.nan, max_iter=10, 
        verbose=2, imputation_order='roman',random_state=0)

    #need to track indices of each heading
    locs = np.empty((len(headings)), dtype=str)
    for i, h in enumerate(headings):
        locs[i] = h

    #perform imputations
    X_headings = X[headings].copy()
    X_headings=imp.fit_transform(X_headings)

    #replace the corresponding columns in X, bounding to > 0
    for i, h in enumerate(headings):
        X[h] = np.where(X_headings[:, i] < 0, 0, X_headings[:, i])
    

    #for categorical headings, discretize imputed values
    for i, h in enumerate(headingsCat):
        #apply floor, ceiling based on max column value
        X[h] = X[h].clip(lower = min_maxes[i, 0]).clip(upper = min_maxes[i, 1])
        #use round to discretize these columns
        X[h] = X[h].astype(float).round(decimals = 0)

    return X


def Encoder(df):
    columnsToEncode = list(df.select_dtypes(include=['object']))
    #don't want to convert certain columns which aren't likely to correlate with TICI
    try:
        columnsToEncode.remove('ACC')
        columnsToEncode.remove('ratio')
        columnsToEncode.remove('DC_Disposition')
        columnsToEncode.remove('Comments')
    except ValueError as e:
        print(e)


    le = LabelEncoder()
    for featureind, feature in enumerate(columnsToEncode):
        print(f"\nEncoding feature {feature}")
        #get indices of NaNs
        nan_inds = df[feature].isnull()

        #get unique non-NaN values
        if feature == "TICI" or feature == "Reperfusion_score":
            uniques = ["0", "1", "2A", "2B", "2C", "3"]
        elif feature == "Comparison CTRL":
            uniques = ["less severe", "same", "more severe"]
        else:
            uniques = list(set(df[feature]))
            if df[feature].isnull().sum() != 0:
                uniques.pop(uniques.index(np.nan))

        #map values to categories 
        categories = np.arange(len(uniques))

        #check the mapping
        for u, cat in zip(uniques, categories):
            print(f"{u} -> {cat}")
            
        for i, v in enumerate(df[feature]):
            if not nan_inds[i]:
                if feature == "TICI" or feature == "Reperfusion_score":
                    df.loc[i, feature] = categories[uniques.index(str(v))]
                else:
                    df.loc[i, feature] = categories[uniques.index(v)]
        
        #map NaN to 6 (additional category) for Assistant
        if feature == "Assistant":
            for i, v in enumerate(df[feature]):
                if nan_inds[i]:
                    df.loc[i, feature] = 6

    return columnsToEncode
            
def format_pipeline(df, sheet):

    #remove leading, trailing whitespace
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    #do encoding
    encodedCols = Encoder(df)
    print(f"Encoded columns: {encodedCols}")

    #Date in days
    df["Date"] = df["Date"].map(lambda x: x.to_pydatetime())
    df["Date"] = df["Date"].map(lambda x: 365*x.year + 30*x.month + x.day)

    #Time
    df["Time"] = df["Time"].map(lambda x: x.hour + (x.minute / 60.0) if isinstance(x, datetime.time) else random.random()*24)

    #MICE imputing (only use valid numerical columns)
    df_out = mice_inpute_feature(df, sheet)

    return df_out
    

fpath = 'Z:\EVT Project Data/de-identified patient data/Patient_Data.xlsx'

df1 = pd.read_excel(
    fpath, 
    nrows=43, 
    sheet_name='All observations')

df2 = pd.read_excel(
    fpath, 
    nrows=43, 
    sheet_name='ACC cases')

df1_encoded = format_pipeline(df1, "all")
df2_encoded = format_pipeline(df2, "acc")

#save both dataframes into 1 csv
with pd.ExcelWriter('encoded.xlsx') as writer:  
    df1_encoded.to_excel(writer, sheet_name='All observations')
    df2_encoded.to_excel(writer, sheet_name='ACC cases')




        
