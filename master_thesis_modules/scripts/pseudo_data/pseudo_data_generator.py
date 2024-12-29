import os
import sys

import numpy as np
import pandas as pd

sys.path.append(".")
sys.path.append("..")
sys.path.append(os.path.expanduser("~")+"/kazu_ws/master_thesis/master_thesis_modules")
from scripts.management.manager import Manager

class PseudoDataGenerator(Manager):
    def __init__(self,trial_name,strage):
        super().__init__()
        self.trial_name=trial_name
        self.strage=strage
        self.data_dir_dict=self.get_database_dir(trial_name=self.trial_name,strage=self.strage)

    def action_to_pose(self,df,action_dict):# 患者ごとに切り出したデータを食わせる
        
        pass

    def get_pseudo_data(self,graph_dicts,general_dict,zokusei_dict,position_dict,action_dict,surrounding_objects):
        people=list(graph_dicts.keys())
        patients=[p for p in people if zokusei_dict[p]["patient"]=="yes"]
        
        # DataFrameの作成
        data_dicts={}
        t=np.arange(general_dict["start_timestamp"],general_dict["end_timestamp"]+1e-4,step=1/general_dict["fps"])
        for person in patients:
            df=pd.DataFrame(t,columns=["timestamp"])
            for node in graph_dicts[person]["node_dict"].keys():
                df[node]=np.nan
            data_dicts[person]=df

        # 属性情報の入力
        for patient in patients:
            data_dicts[patient][50000000]=zokusei_dict[patient]["patient"]
            data_dicts[patient][50000001]=1
            data_dicts[patient][50000010]=zokusei_dict[patient]["age"]
            data_dicts[patient][50000011]=1

        # 患者の軌道を生成
        for patient in patients:
            data_dicts[patient][60010000]=position_dict[patient][0]
            data_dicts[patient][60010001]=position_dict[patient][1]

        # 患者の姿勢情報の生成
        for patient in patients:
            data_dicts[patient]=self.action_to_pose(data_dicts[patient],action_dict[patient])


        # 看護師の軌道情報の生成
        return data_dicts
    

if __name__=="__main__":
    trial_name="BuildSimulator20241229"
    strage="NASK"
    cls=PseudoDataGenerator(trial_name=trial_name,strage=strage)