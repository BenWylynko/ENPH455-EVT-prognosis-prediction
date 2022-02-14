# Want to visualize the dataset when oversampling is done
# Apparently I can reduce the data to 2 PCAs to plot in 2D space, this is interesting
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from imblearn.over_sampling import ADASYN

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

#load data
fpath = r'Z:\EVT Project Data/de-identified patient data\encoded_usable.xlsx'
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
if "encoded.xlsx" in str(fpath):
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


# Instanciate a PCA object for the sake of easy visualisation
pca_all = PCA(n_components=2)
pca_acc = PCA(n_components=2)
# Fit and transform x to visualise inside a 2D feature space
df_all_vis = pca_all.fit_transform(df_all)
df_acc_vis = pca_acc.fit_transform(df_acc)

# Apply the random over-sampling
ada_all = ADASYN()
ada_acc = ADASYN()
df_all_resampled, reperf_resampled = ada_all.fit_resample(df_all, Reperf_bin)
df_all_res_vis = pca_all.transform(df_all_resampled)
df_acc_resampled, TICI_resampled = ada_acc.fit_resample(df_acc, TICI_bin)
df_acc_res_vis = pca_acc.transform(df_acc_resampled)

#plotting
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

#all
ax1.scatter(df_all_vis[Reperf_bin == 0, 0], df_all_vis[Reperf_bin == 0, 1], label="Class #0",
                 alpha=0.5)
ax1.scatter(df_all_vis[Reperf_bin == 1, 0], df_all_vis[Reperf_bin == 1, 1], label="Class #1",
                 alpha=0.5)
ax1.set_title('All data original')

ax2.scatter(df_all_res_vis[reperf_resampled == 0, 0], df_all_res_vis[reperf_resampled == 0, 1],
            label="Class #0", alpha=.5)
ax2.scatter(df_all_res_vis[reperf_resampled == 1, 0], df_all_res_vis[reperf_resampled == 1, 1],
            label="Class #1", alpha=.5)
ax2.set_title('All data oversampled ADASYN')

#acc
ax3.scatter(df_acc_vis[TICI_bin == 0, 0], df_acc_vis[TICI_bin == 0, 1], label="Class #0",
                 alpha=0.5)
ax3.scatter(df_acc_vis[TICI_bin == 1, 0], df_acc_vis[TICI_bin == 1, 1], label="Class #1",
                 alpha=0.5)
ax3.set_title('Acc data original')

ax4.scatter(df_acc_res_vis[TICI_resampled == 0, 0], df_acc_res_vis[TICI_resampled == 0, 1],
            label="Class #0", alpha=.5)
ax4.scatter(df_acc_res_vis[TICI_resampled == 1, 0], df_acc_res_vis[TICI_resampled == 1, 1],
            label="Class #1", alpha=.5)
ax4.set_title('Acc data oversampled ADASYN')

plt.show()

#check how class balanced things are
print(f"All: {len(reperf_resampled == 0)} class 1 instances, {len(reperf_resampled == 1)} class 2 instances")
print(f"Acc: {len(TICI_resampled == 0)} class 1 instances, {len(TICI_resampled == 1)} class 2 instances")

"""
#save the balanced dataset
#back to dataframes 
df_all_resampled = pd.DataFrame(data=df_all_resampled, columns=df_all.columns)
df_all_resampled.insert(0, 'Reperfusion_score', reperf_resampled)

df_acc_resampled = pd.DataFrame(data=df_acc_resampled, columns=df_acc.columns)
df_acc_resampled.insert(0, 'TICI', TICI_resampled)
"""

#save
results_path = r"Z:\EVT Project Data\de-identified patient data\encoded_usable_res.xlsx"
"""
with pd.ExcelWriter(results_path) as writer:  
    df_all_resampled.to_excel(writer, sheet_name='All observations')
    df_acc_resampled.to_excel(writer, sheet_name='ACC cases')
"""

save(df_all_resampled, df_acc_resampled, AllColNames, ACCColNames, reperf_resampled, TICI_resampled, results_path)




