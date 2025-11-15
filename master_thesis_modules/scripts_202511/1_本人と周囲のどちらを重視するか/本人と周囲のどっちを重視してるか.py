import pandas as pd
import matplotlib.pyplot as plt

csv_path="/home/hayashide/kazu_ws/master_thesis/master_thesis_modules/scripts_202511/questionaire_1b.csv"
csv_data = pd.read_csv(csv_path,index_col="1b",header=0)
print(csv_data)

analysis_data = pd.DataFrame(columns=csv_data.columns)
analysis_data.loc["本人に関する差",:]=abs((csv_data.loc[9,:]+csv_data.loc[10,:])/2-(csv_data.loc[11,:]+csv_data.loc[12,:])/2)
analysis_data.loc["周囲に関する差",:]=abs((csv_data.loc[9,:]+csv_data.loc[11,:])/2-(csv_data.loc[10,:]+csv_data.loc[12,:])/2)
print(analysis_data)

plt.scatter(analysis_data.loc["本人に関する差",:],analysis_data.loc["周囲に関する差",:])
plt.plot([0,3.5],[0,3.5])
# plt.xlim([0,3.5])
# plt.ylim([0,3.5])
plt.xlabel("Sensitivity towards risk of the patient")
plt.ylabel("Sensitivity towards risk of the environment")
plt.savefig("/home/hayashide/kazu_ws/master_thesis/master_thesis_modules/scripts_202511/本人のリスクより周囲のリスクを気にしている人の方が多い.png")