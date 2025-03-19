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


csv_path="C:/Users/hyper/kazu_ws/master_thesis/master_thesis_modules/scripts/exp_analysis/notification_necessity/notification_necessity.csv"
data=pd.read_csv(csv_path,header=0,index_col=0).T
print(data)
case_list=[k for k in data.keys() if "通知" in k]
print(case_list)

# 全体での，各事例の受容率
for case_name in case_list:
    print(case_name,data[case_name].astype(int).sum(),"/",len(data))

# 職歴で分けた時の，各事例の受容率
experience_year=15
for case_name in case_list:
    df=data[data["経験年数"].astype(float)<experience_year]
    print(case_name,f"{experience_year}年未満",df[case_name].astype(int).sum(),"/",len(df))
for case_name in case_list:
    df=data[data["経験年数"].astype(float)>=experience_year]
    print(case_name,f"{experience_year}年以上",df[case_name].astype(int).sum(),"/",len(df))
