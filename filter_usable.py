import pandas as pd
import argparse
from pathlib import Path

BASE =  r'Z:\EVT Project Data/de-identified patient data'

def main(fpath):

    print(f"Loading spreadsheet {fpath}")

    df = pd.read_excel( #use "Reperfusion score" instead of TICI
        fpath, 
        nrows=55)

    if "feature_removed" not in str(fpath):
        TICI_bin = df["TICI"] >= 3
    else:
        TICI_bin = df["TICI"]

    df = df.drop(columns=["Unnamed: 0", 'TICI'])

    #back to dataframes 
    df = pd.DataFrame(data=df, columns=df.columns)
    df.insert(0, 'TICI', TICI_bin)

    #save
    results_path = Path(fpath.parents[0], 'encoded_usable.xlsx')
    with pd.ExcelWriter(results_path) as writer:  
        df.to_excel(writer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-in", "--infile", default="encoded.xlsx", 
        help="Name of file to load for modelling")
    args = parser.parse_args()

    fpath = Path(BASE, args.infile)
    print(f"Using filepath {fpath}")
    assert fpath.exists()

    main(fpath)