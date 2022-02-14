from imblearn.over_sampling import SMOTE
import pandas as pd
import argparse
from pathlib import Path

BASE =  r'Z:\EVT Project Data/de-identified patient data'

def main_both(fpath):
    df_all = pd.read_excel( #use "Reperfusion score" instead of TICI
        fpath, 
        nrows=82, 
        sheet_name='All observations')

    df_acc = pd.read_excel( #use TICI
        fpath, 
        nrows=43, 
        sheet_name='ACC cases')

    #we don't need to map to binary again if we're passing the feature removed sheet
    if "encoded.xlsx" in str(fpath):
        TICI_bin = df_acc["TICI"] >= 3
        Reperf_bin = df_all["Reperfusion_score"] >= 3
    else: #using the feature removed sheet
        TICI_bin = df_acc["TICI"]
        Reperf_bin = df_all["Reperfusion_score"]

    df_all = df_all.drop(columns=["Unnamed: 0", "Reperfusion_score"])
    df_acc = df_acc.drop(columns=["Unnamed: 0", "TICI"])

    # Apply the random over-sampling
    smote_all = SMOTE()
    smote_acc = SMOTE()
    df_all_resampled, reperf_resampled = smote_all.fit_resample(df_all, Reperf_bin)
    df_acc_resampled, TICI_resampled = smote_acc.fit_resample(df_acc, TICI_bin)

    #check how class balanced things are
    print(f"All: {len(reperf_resampled == 0)} class 1 instances, {len(reperf_resampled == 1)} class 2 instances")
    print(f"Acc: {len(TICI_resampled == 0)} class 1 instances, {len(TICI_resampled == 1)} class 2 instances")

    #save the balanced dataset
    #back to dataframes 
    df_all_resampled = pd.DataFrame(data=df_all_resampled, columns=df_all.columns)
    df_all_resampled.insert(0, 'Reperfusion_score', reperf_resampled)

    df_acc_resampled = pd.DataFrame(data=df_acc_resampled, columns=df_acc.columns)
    df_acc_resampled.insert(0, 'TICI', TICI_resampled)

    #save
    results_path = r"Z:\EVT Project Data\de-identified patient data\encoded_usable_res_smote.xlsx"
    with pd.ExcelWriter(results_path) as writer:  
        df_all_resampled.to_excel(writer, sheet_name='All observations')
        df_acc_resampled.to_excel(writer, sheet_name='ACC cases')

def main_one(fpath):
    df = pd.read_excel( #use "Reperfusion score" instead of TICI
        fpath, 
        nrows=55)


    #we don't need to map to binary again if we're passing the feature removed sheet
    if "encoded.xlsx" in str(fpath):
        TICI_bin = df["TICI"] >= 3
    else: #using the feature removed sheet
        TICI_bin = df["TICI"]

    df = df.drop(columns=["Unnamed: 0", 'AGE', 'Device', 'Passes', 'Complication', 'ICA OCCL on CTA', "TICI"])

    # Apply the random over-sampling
    smote = SMOTE()
    df_resampled, TICI_resampled = smote.fit_resample(df, TICI_bin)

    #check how class balanced things are
    print(f"Acc: {len(TICI_resampled == 0)} class 1 instances, {len(TICI_resampled == 1)} class 2 instances")

    #save the balanced dataset
    #back to dataframes 
    df_resampled = pd.DataFrame(data=df_resampled, columns=df.columns)
    df_resampled.insert(0, 'TICI', TICI_resampled)

    #save
    results_path = Path(fpath.parents[0], "encoded_usable_res_smote.xlsx")
    with pd.ExcelWriter(results_path) as writer:  
        df_resampled.to_excel(writer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-in", "--infile", default="encoded_usable.xlsx", 
        help="Name of file to load for modelling")
    parser.add_argument("-s", default="both", 
        help="2 sheets or no")
    args = parser.parse_args()

    fpath = Path(BASE, args.infile)
    print(f"Using filepath {fpath}")
    assert fpath.exists()

    if args.s == 'both':
        main_both(fpath)
    else:
        main_one(fpath)