import pandas as pd
import os
import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append(os.path.expanduser("~")+"/kazu_ws/master_thesis/master_thesis_modules")
from scripts.AHP.get_comparison_mtx_v3 import *

csv_path = "/home/hayashide/kazu_ws/master_thesis/master_thesis_modules/scripts_202511/2_中村さんと百武さんのAHPとFuzzy/questionaire_1b.csv"
data = pd.read_csv(csv_path,index_col="1b",header=0)
print(data)

convert_dict={
    5:1,
    4:0.75,
    3:0.5,
    2:0.25,
    1:0,
}

for staff_name in data.columns:
    for i in data.index:
        print(i,staff_name)
        data.loc[i,staff_name]=convert_dict[data.loc[i,staff_name]]

print(data)

TFN_WIDTH = 0.5
for staff_name in data.columns:
    TFN_data=pd.DataFrame(columns=["l","c","r"])
    for i in data.index:
        TFN_data.loc[i,:]=[data.loc[i,staff_name]-TFN_WIDTH,data.loc[i,staff_name],data.loc[i,staff_name]+TFN_WIDTH]
    TFN_csv_path=f"/media/hayashide/MasterThesis/common/TFN_{staff_name}.csv"
    TFN_data.to_csv(TFN_csv_path,index=False,header=False)
    print(TFN_data)