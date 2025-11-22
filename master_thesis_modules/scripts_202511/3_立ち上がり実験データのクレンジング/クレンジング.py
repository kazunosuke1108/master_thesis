import os
import sys
import copy
import time
from glob import glob
import json
import cv2
import atexit
from pprint import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# pip install watchdog

sys.path.append(".")
sys.path.append("..")
if "catkin_ws" in os.getcwd():
    sys.path.append("/catkin_ws/src/master_thesis_modules")
else:
    sys.path.append(os.path.expanduser("~")+"/kazu_ws/master_thesis/master_thesis_modules")
from scripts.management.manager import Manager

class Visualizer(Manager):
    def __init__(self,trial_name,strage):
        super().__init__()
        self.data_dir_dict=self.get_database_dir(trial_name=trial_name,strage=strage)
        # 位置・bboxデータ
        self.df_after_reid_csv_path=self.data_dir_dict["mobilesensing_dir_path"]+"/csv/df_after_reid.csv"
        self.df_after_reid=pd.read_csv(self.df_after_reid_csv_path,header=0).dropna(how="all",axis=1)
        # 特徴量から評価までいろいろ入ってる
        self.df_eval_csv_path=self.data_dir_dict["mobilesensing_dir_path"]+"/csv/df_eval.csv"
        self.df_eval=pd.read_csv(self.df_eval_csv_path,header=0).dropna(how="all",axis=1)        
        self.patients=sorted(list(set([k.split("_")[0] for k in self.df_after_reid.keys() if (("timestamp" not in k) and ("activeBinary" not in k))])))

        self.spatial_normalization_param=np.sqrt(2)*6

        # self.df_eval

    def repair_data(self):
        for patient in self.patients:
            print(patient)
            # 車椅子の位置
            for idx,row in self.df_eval.iterrows():
                min_candidates=[
                    np.sqrt((self.df_eval.loc[idx,patient+"_60010000"]-self.df_eval.loc[idx,"00001_60010000"])**2+(self.df_eval.loc[idx,patient+"_60010001"]-self.df_eval.loc[idx,"00001_60010001"])**2),
                    np.sqrt((self.df_eval.loc[idx,patient+"_60010000"]-self.df_eval.loc[idx,"00002_60010000"])**2+(self.df_eval.loc[idx,patient+"_60010001"]-self.df_eval.loc[idx,"00002_60010001"])**2),
                    np.sqrt((self.df_eval.loc[idx,patient+"_60010000"]-self.df_eval.loc[idx,"00006_60010000"])**2+(self.df_eval.loc[idx,patient+"_60010001"]-self.df_eval.loc[idx,"00006_60010001"])**2),
                ]

                

            # 点滴の位置
            # スタッフの位置
            # スタッフの向き
        pass

if __name__=="__main__":
    cls=Visualizer(trial_name="20251122_postAnalysis3",strage="NASK")
    cls.repair_data()