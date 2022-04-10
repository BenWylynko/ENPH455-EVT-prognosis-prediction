from imblearn.over_sampling import SMOTE
import pandas as pd
import argparse
from pathlib import Path

BASE =  r'Z:\EVT Project Data/de-identified patient data'

def main(fpath):
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
    args = parser.parse_args()

    fpath = Path(BASE, args.infile)
    print(f"Using filepath {fpath}")
    assert fpath.exists()

    main(fpath)