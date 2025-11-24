import os
import sys
import copy
import time
from glob import glob
import pickle
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
        self.df_eval.ffill(inplace=True)
        self.df_eval.bfill(inplace=True)
        self.df_eval["timestamp"]=self.df_eval[[k for k in list(self.df_eval.keys()) if "timestamp" in k]].mean(axis=1)
        self.patients=sorted(list(set([k.split("_")[0] for k in self.df_after_reid.keys() if (("timestamp" not in k) and ("activeBinary" not in k))])))

        self.data_dicts={}
        for patient in self.patients:
            cols=[k.split("_")[1] for k in list(self.df_eval.keys()) if "timestamp" not in k]
            self.data_dicts[patient]=pd.DataFrame()
            self.data_dicts[patient]["timestamp"]=self.df_eval["timestamp"]
            for c in cols:
                try:
                    self.data_dicts[patient][int(c)] = self.df_eval[patient+"_"+c]
                except KeyError:
                    pass
        self.spatial_normalization_param=np.sqrt(2)*6

        # self.df_eval

    def repair_data(self):
        # nanの補間？
        structure_dict={
            "ivPole":[
                ],
            "wheelchair":[
                np.array([self.df_eval["00001_60010000"],self.df_eval["00001_60010001"]],),
                np.array([self.df_eval["00002_60010000"],self.df_eval["00002_60010001"]],),
                np.array([self.df_eval["00006_60010000"],self.df_eval["00006_60010001"]],),
                ],
            "handrail":{
                "xrange":[-10,-4],
                "yrange":[9,15]
                },
            "staff_station":{
                "pos":[-8,7],
                "direction":[0,0.1]
                }
        }
        for patient in self.patients:
            # 上位層はいったん消去
            keys=[k for k in list(self.data_dicts[patient].keys())]
            keys.remove("timestamp")
            self.data_dicts[patient][[k for k in keys if int(k)<50000000]]=np.nan
            # 属性
            # 患者か
            if patient in ["00042","00043"]:
                self.data_dicts[patient][50000000]="no"
            else:
                self.data_dicts[patient][50000000]="yes"
            self.data_dicts[patient][50000001]=1
            # 年齢
            if patient in ["00042","00043"]:
                self.data_dicts[patient][50000010]="young"
            else:
                self.data_dicts[patient][50000010]="old"
            self.data_dicts[patient][50000011]=1
            # 動作
            pass

            # 物体
            self.data_dicts[patient]=self.object_snapshot(data_dict=self.data_dicts[patient],structure_dict=structure_dict)
            # スタッフの位置
            # スタッフの向き
            closest_staff="00042"
            self.data_dicts[patient][50001100]=self.data_dicts[closest_staff][60010000]
            self.data_dicts[patient][50001101]=self.data_dicts[closest_staff][60010001]
            self.data_dicts[patient][50001110]=self.data_dicts[closest_staff][60010000].diff()
            self.data_dicts[patient][50001111]=self.data_dicts[closest_staff][60010001].diff()
            self.data_dicts[patient].to_csv(f"/home/hayashide/kazu_ws/master_thesis/master_thesis_modules/scripts_202511/3_立ち上がり実験データのクレンジング/data_{patient}_eval.csv",index=False)
        picklepath="/home/hayashide/kazu_ws/master_thesis/master_thesis_modules/scripts_202511/3_立ち上がり実験データのクレンジング/data_dicts.pickle"
        with open(picklepath, mode='wb') as f:
            pickle.dump(self.data_dicts,f)

    def object_snapshot(self,data_dict,structure_dict):
        """
        structure_dict={
            "ivPole":[
                np.array([0,0]), # muに相当
                np.array([0,0]),
                ],
            "wheelchair":[
                np.array([0,0]),
                np.array([0,0]),
                ],
            "handrail":{
                "xrange":[6,15],
                "yrange":[-11,-4]
                }
        }
        """
        x=np.array([data_dict[60010000],data_dict[60010001]])
        # 点滴
        potential_ivPole=0
        for mu in structure_dict["ivPole"]:
            potential_ivPole+=self.gauss_func(x=x,mu=mu,r=3)
        potential_ivPole=np.clip(potential_ivPole,0,1)
        data_dict[50001000]=potential_ivPole
        data_dict[50001001]=potential_ivPole
        data_dict[50001002]=1
        data_dict[50001003]=1
        # 車椅子
        potential_wheelchair=0
        for mu in structure_dict["wheelchair"]:
            potential_wheelchair+=self.gauss_func(x=x,mu=mu,r=3)
            print(potential_wheelchair)
        print(x,mu)
        potential_wheelchair=np.clip(potential_wheelchair,0,1)
        data_dict[50001010]=potential_wheelchair
        data_dict[50001011]=potential_wheelchair
        data_dict[50001012]=1
        data_dict[50001013]=1
        # 手すり
        closest_wall=np.argmin([
            abs(structure_dict["handrail"]["xrange"][0]-data_dict[60010000]),
            abs(structure_dict["handrail"]["yrange"][0]-data_dict[60010001]),
            abs(structure_dict["handrail"]["xrange"][1]-data_dict[60010000]),
            abs(structure_dict["handrail"]["yrange"][1]-data_dict[60010001]),
            ])
        if closest_wall==0:                
            data_dict[50001020]=structure_dict["handrail"]["xrange"][0]
            data_dict[50001021]=data_dict[60010001]
        elif closest_wall==1:
            data_dict[50001020]=data_dict[60010000]
            data_dict[50001021]=structure_dict["handrail"]["yrange"][0]
        elif closest_wall==2:
            data_dict[50001020]=structure_dict["handrail"]["xrange"][1]
            data_dict[50001021]=data_dict[60010001]
        elif closest_wall==3:
            data_dict[50001020]=data_dict[60010000]
            data_dict[50001021]=structure_dict["handrail"]["yrange"][1]
        return data_dict
    
    def gauss_func(self,x,mu,r):
        sigma=r/3
        x=np.array(x)
        mu=np.array(mu)
        norm=np.linalg.norm(x-mu)
        # 正規化ver.
        # val=1/(np.sqrt(2*np.pi)*sigma)*np.exp(-norm**2/(2*sigma**2))
        # 最大値が1
        val=np.exp(-norm**2/(2*sigma**2))
        return val
    
if __name__=="__main__":
    cls=Visualizer(trial_name="20251122_postAnalysis3",strage="NASK")
    cls.repair_data()

# 最短の物体を見つける系
            # for idx,row in self.df_eval.iterrows():
            #     min_candidates={
            #         "00001":np.sqrt((self.df_eval.loc[idx,patient+"_60010000"]-self.df_eval.loc[idx,"00001_60010000"])**2+(self.df_eval.loc[idx,patient+"_60010001"]-self.df_eval.loc[idx,"00001_60010001"])**2),
            #         "00002":np.sqrt((self.df_eval.loc[idx,patient+"_60010000"]-self.df_eval.loc[idx,"00002_60010000"])**2+(self.df_eval.loc[idx,patient+"_60010001"]-self.df_eval.loc[idx,"00002_60010001"])**2),
            #         "00006":np.sqrt((self.df_eval.loc[idx,patient+"_60010000"]-self.df_eval.loc[idx,"00006_60010000"])**2+(self.df_eval.loc[idx,patient+"_60010001"]-self.df_eval.loc[idx,"00006_60010001"])**2),
            #     }
            #     min_id=min(min_candidates,key=min_candidates.get)
            #     self.df_eval[patient+"_50001100"]=self.df_eval[min_id+"_60010000"]
            #     self.df_eval[patient+"_50001101"]=self.df_eval[min_id+"_60010001"]
