import os
import sys
sys.path.append(".")
sys.path.append("..")
if "catkin_ws" in os.getcwd():
    sys.path.append("/catkin_ws/src/master_thesis_modules")
else:
    sys.path.append(os.path.expanduser("~")+"/kazu_ws/master_thesis/master_thesis_modules")

from icecream import ic
from pprint import pprint
from glob import glob

import numpy as np
import pandas as pd
import plotly.subplots as sp
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


from scripts.management.manager import Manager

class Visualizer(Manager):
    def __init__(self,trial_name,strage):
        super().__init__()
        self.trial_name=trial_name
        self.strage=strage
        self.data_dir_dict=self.get_database_dir(trial_name=trial_name,strage=strage)

    def delay_by_fps_control(self,trial_name_false,trial_name_true):
        self.data_dir_dict_false=self.get_database_dir(trial_name=trial_name_false,strage=strage)
        self.data_dir_dict_true=self.get_database_dir(trial_name=trial_name_true,strage=strage)
        merge_data=pd.DataFrame()
        delay_data=pd.DataFrame(columns=["timestamp_false","priority_order",])
        
        # Falseのときのデータ収集
        csv_paths=sorted(glob(self.data_dir_dict_false["trial_dir_path"]+"/data_*_eval.csv"))
        patients=[]
        for csv_path in csv_paths:
            patient=os.path.basename(csv_path)[len("data_"):-len("_eval.csv")]
            data=pd.read_csv(csv_path,header=0)
            merge_data["timestamp"]=data["timestamp"]
            merge_data["false_"+patient+"_10000000"]=data["10000000"].values
            patients.append(patient)
        merge_data["false_top"]=merge_data[[f"false_{p}_10000000" for p in patients]].idxmax(axis=1)
        print(merge_data)
        # Trueのときのデータ収集
        csv_paths=sorted(glob(self.data_dir_dict_true["trial_dir_path"]+"/data_*_eval.csv"))
        patients=[]
        for csv_path in csv_paths:
            patient=os.path.basename(csv_path)[len("data_"):-len("_eval.csv")]
            data=pd.read_csv(csv_path,header=0)
            merge_data["true_"+patient+"_10000000"]=data["10000000"].values
            patients.append(patient)
        merge_data["true_top"]=merge_data[[f"true_{p}_10000000" for p in patients]].idxmax(axis=1)
        print(merge_data)

        # Falseのときの切り替わりタイミング
        change_idx_false=merge_data["false_top"]!=merge_data["false_top"].shift()
        print(merge_data.loc[change_idx_false,["timestamp","false_top"]])
        # Trueのときの切り替わりタイミング
        change_idx_true=merge_data["true_top"]!=merge_data["true_top"].shift()
        print(merge_data.loc[change_idx_true,["timestamp","true_top"]])
        # 遅延を計算
        delay_data=pd.concat([merge_data.loc[change_idx_false,["timestamp","false_top"]].reset_index(drop=True),merge_data.loc[change_idx_true,["timestamp","true_top"]].reset_index(drop=True)],axis=1)
        delay_data.columns=["false_timestamp","false_top","true_timestamp","true_top"]
        delay_data["false_top"]=[v[len("false_"):-len("_10000000")] for v in delay_data["false_top"]]
        delay_data["true_top"]=[v[len("true_"):-len("_10000000")] for v in delay_data["true_top"]]
        delay_data["delay"]=delay_data["true_timestamp"]-delay_data["false_timestamp"]
        print(delay_data)
        # latex形式でprint
        print(delay_data.to_latex(index=False))

        pass

if __name__=="__main__":
    # trial_name="20250113NormalSimulation"
    # trial_name="20250110SimulationMultipleRisks/no_00005"
    # trial_name="20250120FPScontrolFalse"
    trial_name="20250120FPScontrolTrue"
    # trial_name="20250108DevMewThrottlingExp"
    # trial_name="20250115PullWheelchairObaachan2"
    # trial_name="20250121ChangeCriteriaBefore"
    strage="NASK"
    cls=Visualizer(trial_name=trial_name,strage=strage)
    cls.delay_by_fps_control(trial_name_false="20250120FPScontrolFalse",trial_name_true="20250120FPScontrolTrue")
    pass