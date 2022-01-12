import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

# Swap function
def swapPositions(list, pos1, pos2):
     
    list[pos1], list[pos2] = list[pos2], list[pos1]
    return list
 
def binColAnalysis(col):
    #Univariate analysis for a binary column
    return col.value_counts()
        

def numColAnalysis(col):
    #Univariate analysis for a numerical column
    
    #ignore NaNs, strings

    #NaN inds
    nan_inds = col.isnull()

    #string inds (for ratio)
    str_inds = np.zeros(len(col))
    for i in col.index:
        if type(col[i]) == str:
            str_inds[i] = 1

    #apply mask
    mask = np.logical_or(nan_inds, str_inds)
    masked = np.ma.masked_array(col, mask=mask)

    #calc mean, medium with mask
    mean = masked.mean()
    median = np.ma.median(masked)

    #gen histogram
    nbins = int(len(masked) / 5)

    return mean, median, nbins

fpath_orig = 'Z:\EVT Project Data/de-identified patient data/Patient_Data.xlsx'
fpath = 'Z:\EVT Project Data/de-identified patient data/encoded.xlsx'

df1_orig = pd.read_excel(
    fpath_orig, 
    nrows=43, 
    sheet_name='All observations')

df2_orig = pd.read_excel(
    fpath_orig, 
    nrows=43, 
    sheet_name='ACC cases')

df1 = pd.read_excel(
    fpath, 
    nrows=43, 
    sheet_name='All observations')

df2 = pd.read_excel(
    fpath, 
    nrows=43, 
    sheet_name='ACC cases')

#remove leading, trailing whitespace
df1_orig = df1_orig.applymap(lambda x: x.strip() if isinstance(x, str) else x)
df2_orig = df2_orig.applymap(lambda x: x.strip() if isinstance(x, str) else x)
df1 = df1.applymap(lambda x: x.strip() if isinstance(x, str) else x)
df2 = df2.applymap(lambda x: x.strip() if isinstance(x, str) else x)

#load TICI as numerical (so encoded)

#for string and binary cols, count value occurences
nonNumCols = [
    "Gender", "Center", "MRP", "Assistant", "Post_6_hrs", "After_Hours", "tPA_given", "LOV", 
    "SIDE", "HU", "Hyperdense Thro", "Ca at LVO", "Ca Number", "ICAS Proximal", "Ca PA/Supra ICA", 
    "Comparison CTRL", "Tortuos parent art", "Kink Parent", "TICI", "Device", "Passes", 
    "Complication", "ICA OCCL on CTA"
]


#for numerical columns generate histograms, mean and median
#need to ignore "infinite" in ratio (add ratio later)
numCols = [
    "Age", "Time", "NIHSS_Arrival", "NIHSS_day_1", "CT_APECTS_arrival", "Core(mL)", 
    "Mismatch_Volume(mL)", "KGH_LOS", "MRSS_90days", "TICI"
]


#need to handle logic to load column from All observations if possible, else ACC cases
allCols = ["Date", "Age", "Gender", "Center", "MRP", "Assistant", "Post_6_hrs", "After_Hours", 
    "Time", "NIHSS_Arrival", "NIHSS_day_1", "tPA_given", "CT_APECTS_arrival", "Core(mL)", 
    "Mismatch_Volume(mL)", "ratio", "collateral score", "Clot_arrival", 
    "Reperfusion_score", "KGH_LOS", "DC_Disposition", "MRSS_90days"]

#check ACC cases columns with TICI heading
ACCCols = ["LOV", "SIDE", "HU", "Hyperdense Thro", 
    "Ca at LVO", "Ca Number", "ICAS Proximal", "Ca PA/Supra ICA", "Comparison CTRL", 
    "Tortuos parent art", "Kink Parent", "TICI", "Device", "Passes", "Complication", 
    "ICA OCCL on CTA"]


binColAnalysis(df1["Gender"])

#get values for numerical columns
numVals = {}

for col in numCols:
    if col in allCols: #account for the sheet I should use
        numVals[col] = numColAnalysis(df1[col])
    else:
        numVals[col] = numColAnalysis(df2[col])

#get values for categorical columns
catVals = {}

for col in nonNumCols:
    if col in allCols: #account for the sheet I should use
        catVals[col] = binColAnalysis(df1_orig[col])
    else:
        catVals[col] = binColAnalysis(df2_orig[col])

#make plots for numerical columns
fig, ax = plt.subplots(nrows=5, ncols=2)

plt.rcParams["axes.titlesize"] = 8

tici_map = {"0": "1", "1": "0", "2": "3", "2A": "3", "2B": "4", "2C": "5"}

for i, j in np.ndindex((5, 2)):
    colInd = i + 5*j
    
    if colInd == 9:
        break
    
    if numCols[colInd] in allCols:
        data = df1[numCols[colInd]]
        colName = df1[numCols[colInd]].name
    else:
        data = df2[numCols[colInd]]
        colName = df2[numCols[colInd]].name
    print(f"Encoding column {colName}")

    mean, median, nbins= numVals[colName]

    ax[i, j].set_title(f"{colName}    mean: {round(mean, 2)}     median: {round(median, 2)}", fontdict={"fontsize": 8})
    ax[i, j].hist(data, bins=nbins)

    ax[i, j].tick_params(axis='both', labelsize=8)

#remove extra plot
fig.delaxes(ax[4, 1])

fig.tight_layout()
plt.show()

#make plots for categorical columns
fig, ax = plt.subplots(nrows=4, ncols=5)

plt.rcParams["axes.titlesize"] = 8

for i, j in np.ndindex((4, 5)):
    colInd = i + 5*j
    
    if colInd == 23:
        break
    
    if nonNumCols[colInd] in allCols:
        colName = df1[nonNumCols[colInd]].name
    else:
        colName = df2[nonNumCols[colInd]].name
    print(f"Encoding column {colName}")

    items = list(catVals[colName].items())
    bins = [i[0] for i in items]
    counts = [i[1] for i in items]

    #for TICI, [3, '2B', '2A', '2C', 1, 0] -> [3, '2C', '2B', '2A', 1, 0]
    if colName == "Reperfusion_score" or colName == "TICI":
        
        #swap 1, 3 then swap 2, 3
        bins = swapPositions(bins, 1, 3)
        bins = swapPositions(bins, 2, 3)

        counts = swapPositions(counts, 1, 3)
        counts = swapPositions(counts, 2, 3)

    

    ax[i, j].set_ylabel("count")
    ax[i, j].set_title(f"{colName}", fontdict={"fontsize": 8})
    x = np.arange(len(bins)) #correct length
    ax[i, j].bar(x, counts) #x coords, heights (counts)
    ax[i, j].set_xticks(x, bins) 


    if colName == "MRP" or colName == "Center" or colName == "Assistant":
        
        ax[i, j].tick_params(axis='x', labelsize=6)
        ax[i, j].tick_params(axis='y', labelsize=8)

        for tick in ax[i, j].xaxis.get_major_ticks()[1::2]:
            tick.set_pad(15)

    elif colName == "LOV":
        bins = ["M1", "M2", "ICA terminus", "ICA non-terminus"]
        ax[i, j].set_xticks(x, bins) 

        ax[i, j].tick_params(axis='x', labelsize=6)
        ax[i, j].tick_params(axis='y', labelsize=8)

        for tick in ax[i, j].xaxis.get_major_ticks()[1::2]:
            tick.set_pad(15)

    else:
        ax[i, j].tick_params(axis='both', labelsize=8)

#remove extra plot
fig.delaxes(ax[3, 4])

fig.tight_layout()
plt.show()
