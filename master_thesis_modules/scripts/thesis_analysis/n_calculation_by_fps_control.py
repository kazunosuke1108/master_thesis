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

    def count_n_feature_calculation(self,trial_name_false,trial_name_true):
        self.data_dir_dict_false=self.get_database_dir(trial_name=trial_name_false,strage=self.strage)
        self.data_dir_dict_true=self.get_database_dir(trial_name=trial_name_true,strage=self.strage)

        n_calc_df=pd.DataFrame()

        node_blip=["50000000","50000010","50001000","50001010"]
        node_yolo=["50000100"]
        node_pos=["50001020","50001100","50001110"]

        # FPS無の場合
        n_patient=len(sorted(glob(self.data_dir_dict_false["trial_dir_path"]+"/data_*_eval.csv")))
        n_calc_df.loc["Average frequency of calculation using BLIP","Without FPS control"]=20*10*len(node_blip)*n_patient/10
        n_calc_df.loc["Average frequency of calculation using YOLO","Without FPS control"]=20*10*len(node_yolo)*n_patient/10
        n_calc_df.loc["Average frequency of calculation using position","Without FPS control"]=20*10*len(node_pos)*n_patient/10

        # FPS有の場合
        n_blip=0
        n_yolo=0
        n_pos=0
        for csv_path in sorted(glob(self.data_dir_dict_true["trial_dir_path"]+"/data_*_beforeIn.csv")):
            data=pd.read_csv(csv_path,header=0)
            blip_data=data[node_blip]
            yolo_data=data[node_yolo]
            pos_data=data[node_pos]
            # print(blip_data)
            print((~np.isnan(blip_data)).sum())
            print((~np.isnan(blip_data)).sum().sum())
            if "C" in csv_path:
                raise NotImplementedError
            n_blip+=(~np.isnan(blip_data)).sum().sum()
            n_yolo+=(~np.isnan(yolo_data)).sum().sum()
            n_pos+=(~np.isnan(pos_data)).sum().sum()
        n_calc_df.loc["Average frequency of calculation using BLIP","With FPS control"]=n_blip/10
        n_calc_df.loc["Average frequency of calculation using YOLO","With FPS control"]=n_yolo/10
        n_calc_df.loc["Average frequency of calculation using position","With FPS control"]=n_pos/10
        n_calc_df["Reduce ratio"]=np.round(100*(1-n_calc_df["With FPS control"]/n_calc_df["Without FPS control"]),1)
        print(n_calc_df)
        

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
    cls.count_n_feature_calculation(trial_name_false="20250120FPScontrolFalse",trial_name_true="20250120FPScontrolTrue")
    pass