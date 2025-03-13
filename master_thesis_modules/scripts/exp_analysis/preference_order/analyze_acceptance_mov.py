import os
import sys
from pprint import pprint

import numpy as np
import pandas as pd
import scipy.stats as stats


sys.path.append(".")
sys.path.append("..")
if "catkin_ws" in os.getcwd():
    sys.path.append("/catkin_ws/src/master_thesis_modules")
else:
    sys.path.append(os.path.expanduser("~")+"/kazu_ws/master_thesis/master_thesis_modules")
from scripts.management.manager import Manager


csv_path="C:/Users/hyper/kazu_ws/master_thesis/master_thesis_modules/scripts/exp_analysis/preference_order/preference.csv"
data=pd.read_csv(csv_path,header=0,index_col=0).T
# print(data)
# print(data["経験年数"].astype(float).mean())
# data=data[data["経験年数"].astype(float)>data["経験年数"].astype(float).mean()/2]
# print(data)
# raise NotImplementedError
# data=data[data["経験年数"]<10]

result_dict={}
result_table_dict={}

case_list=sorted(list(set([k.split("_")[0] for k in data.keys() if "事例" in k])))
condition_list=sorted(list(set([k.split("_")[1] for k in data.keys() if "事例" in k])))