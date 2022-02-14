import pandas as pd
import argparse
from pathlib import Path

BASE =  r'Z:\EVT Project Data/de-identified patient data'
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


def main(fpath):

    print(f"Loading spreadsheet {fpath}")

    """
    df_all = pd.read_excel( #use "Reperfusion score" instead of TICI
        fpath, 
        nrows=82, 
        sheet_name='All observations')

    df_acc = pd.read_excel( #use TICI
        fpath, 
        nrows=43, 
        sheet_name='ACC cases')
    """

    df = pd.read_excel( #use "Reperfusion score" instead of TICI
        fpath, 
        nrows=55)

    """
    #we don't need to map to binary again if we're passing the feature removed sheet
    if "feature_removed" not in str(fpath):
        TICI_bin = df_acc["TICI"] >= 3
        Reperf_bin = df_all["Reperfusion_score"] >= 3
    else: #using the feature removed sheet
        TICI_bin = df_acc["TICI"]
        Reperf_bin = df_all["Reperfusion_score"]
    """

    if "feature_removed" not in str(fpath):
        TICI_bin = df["TICI"] >= 3
    else:
        TICI_bin = df["TICI"]

    """
    df_all = df_all.drop(columns=["Unnamed: 0", "Reperfusion_score"])
    df_acc = df_acc.drop(columns=["Unnamed: 0", "TICI"])
    """
    df = df.drop(columns=["Unnamed: 0", 'TICI'])

    #just select features of interest
    #df_all = df_all[AllColNames]
    #df_acc = df_acc[ACCColNames]


    #back to dataframes 
    """
    df_all = pd.DataFrame(data=df_all, columns=df_all.columns)
    df_all.insert(0, 'Reperfusion_score', Reperf_bin)

    df_acc = pd.DataFrame(data=df_acc, columns=df_acc.columns)
    df_acc.insert(0, 'TICI', TICI_bin)
    """

    df = pd.DataFrame(data=df, columns=df.columns)
    df.insert(0, 'TICI', TICI_bin)

    #save
    results_path = Path(fpath.parents[0], 'encoded_usable.xlsx')
    with pd.ExcelWriter(results_path) as writer:  
        #df_all.to_excel(writer, sheet_name='All observations')
        #df_acc.to_excel(writer, sheet_name='ACC cases')
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