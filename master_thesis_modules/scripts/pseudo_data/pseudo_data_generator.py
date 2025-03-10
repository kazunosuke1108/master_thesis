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

    def action_to_pose(self,df,action_dict):
        for a_dict in action_dict.values():
            start=a_dict["start_timestamp"]
            end=a_dict["end_timestamp"]
            label=a_dict["label"]
            if label=="sit":
                df[50000100][(df["timestamp"]>=start) & (df["timestamp"]<=end)]=0
                df[50000101][(df["timestamp"]>=start) & (df["timestamp"]<=end)]=0
                df[50000102][(df["timestamp"]>=start) & (df["timestamp"]<=end)]=0
                df[50000103][(df["timestamp"]>=start) & (df["timestamp"]<=end)]=0.5
            elif label=="stand":
                df[50000100][(df["timestamp"]>=start) & (df["timestamp"]<=end)]=1
                df[50000101][(df["timestamp"]>=start) & (df["timestamp"]<=end)]=0
                df[50000102][(df["timestamp"]>=start) & (df["timestamp"]<=end)]=0
                df[50000103][(df["timestamp"]>=start) & (df["timestamp"]<=end)]=1
            elif label=="standup":
                df[50000100][(df["timestamp"]>=start) & (df["timestamp"]<=end)]=\
                np.interp(df["timestamp"][(df["timestamp"]>=start) & (df["timestamp"]<=end)],[start,end],[0,1])
                df[50000101][(df["timestamp"]>=start) & (df["timestamp"]<=end)]=\
                np.interp(df["timestamp"][(df["timestamp"]>=start) & (df["timestamp"]<=end)],[start,(start+end)/2,end],[0,0.3,0])
                df[50000102][(df["timestamp"]>=start) & (df["timestamp"]<=end)]=0
                df[50000103][(df["timestamp"]>=start) & (df["timestamp"]<=end)]=\
                np.interp(df["timestamp"][(df["timestamp"]>=start) & (df["timestamp"]<=end)],[start,end],[0.5,1])
            elif label=="sitdown":
                df[50000100][(df["timestamp"]>=start) & (df["timestamp"]<=end)]=\
                np.interp(df["timestamp"][(df["timestamp"]>=start) & (df["timestamp"]<=end)],[start,end],[1,0])
                df[50000101][(df["timestamp"]>=start) & (df["timestamp"]<=end)]=\
                np.interp(df["timestamp"][(df["timestamp"]>=start) & (df["timestamp"]<=end)],[start,(start+end)/2,end],[0,0.3,0])
                df[50000102][(df["timestamp"]>=start) & (df["timestamp"]<=end)]=0
                df[50000103][(df["timestamp"]>=start) & (df["timestamp"]<=end)]=\
                np.interp(df["timestamp"][(df["timestamp"]>=start) & (df["timestamp"]<=end)],[start,end],[1,0.5])
                                
        return df

    def get_pseudo_data(self,graph_dicts,general_dict,zokusei_dict,position_dict,action_dict,surrounding_objects):
        people=list(zokusei_dict.keys())
        patients=[p for p in people if zokusei_dict[p]["patient"]=="yes"]
        non_patients=[p for p in people if zokusei_dict[p]["patient"]=="no"]
        
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

        # 物体の位置情報
        object_dict={}
        # 物体の列記
        for patient in patients:
            for obj in surrounding_objects[patient]:
                object_dict[obj+f"_{patient}"]=np.array([position_dict[patient][0],position_dict[patient][1]])

        # 各物体カテゴリ（車椅子とか）のうち，患者にとって最近傍のインスタンスについて距離を計算
        for patient in patients:
            # 最短の点滴
            best_d=np.inf
            for obj in [o for o in object_dict.keys() if "ivPole" in o]:
                d=np.linalg.norm(object_dict[obj]-position_dict[patient])
                if d<best_d:
                    closest_ivPole=obj
                    best_d=d
            data_dicts[patient][50001000]=object_dict[closest_ivPole][0]
            data_dicts[patient][50001001]=object_dict[closest_ivPole][1]
            # 最短の車椅子
            best_d=np.inf
            for obj in [o for o in object_dict.keys() if "wheelchair" in o]:
                d=np.linalg.norm(object_dict[obj]-position_dict[patient])
                if d<best_d:
                    closest_wheelchair=obj
                    best_d=d
            data_dicts[patient][50001010]=object_dict[closest_wheelchair][0]
            data_dicts[patient][50001011]=object_dict[closest_wheelchair][1]
        
        # 手すり？
        for patient in patients:
            closest_wall=np.argmin([
                abs(general_dict["xrange"][0]-position_dict[patient][0]),
                abs(general_dict["yrange"][0]-position_dict[patient][1]),
                abs(general_dict["xrange"][1]-position_dict[patient][0]),
                abs(general_dict["yrange"][1]-position_dict[patient][1]),
                ])
            if closest_wall==0:                
                data_dicts[patient][50001020]=general_dict["xrange"][0]
                data_dicts[patient][50001021]=position_dict[patient][1]
            elif closest_wall==1:
                data_dicts[patient][50001020]=position_dict[patient][0]
                data_dicts[patient][50001021]=general_dict["yrange"][0]
            elif closest_wall==2:
                data_dicts[patient][50001020]=general_dict["xrange"][1]
                data_dicts[patient][50001021]=position_dict[patient][1]
            elif closest_wall==3:
                data_dicts[patient][50001020]=position_dict[patient][0]
                data_dicts[patient][50001021]=general_dict["yrange"][1]
        # 看護師の軌道情報の生成
        if len(non_patients)>1:
            raise NotImplementedError("現在対応可能な非患者数は1です")
        non_patient=non_patients[0]
        for a_dict in action_dict[non_patient].values():
            start=a_dict["start_timestamp"]
            end=a_dict["end_timestamp"]
            label=a_dict["label"]
            extract_index=(data_dicts[patient]["timestamp"]>=start) & (data_dicts[patient]["timestamp"]<=end)
            if "_" in label:
                l,p=label.split("_")
                print(l,p)
                if l=="approach":
                    traj_x=np.interp(data_dicts[patient]["timestamp"][extract_index],[start,end],[position_dict[non_patient][0],position_dict[p][0]])
                    traj_y=np.interp(data_dicts[patient]["timestamp"][extract_index],[start,end],[position_dict[non_patient][1],position_dict[p][1]])
                elif l=="work":
                    traj_x=np.interp(data_dicts[patient]["timestamp"][extract_index],[start,end],[position_dict[p][0],position_dict[p][0]])
                    traj_y=np.interp(data_dicts[patient]["timestamp"][extract_index],[start,end],[position_dict[p][1],position_dict[p][1]])
                elif l=="leave":
                    traj_x=np.interp(data_dicts[patient]["timestamp"][extract_index],[start,end],[position_dict[p][0],position_dict[non_patient][0]])
                    traj_y=np.interp(data_dicts[patient]["timestamp"][extract_index],[start,end],[position_dict[p][1],position_dict[non_patient][1]])
                for patient in patients:
                    data_dicts[patient][50001100][extract_index]=traj_x
                    data_dicts[patient][50001101][extract_index]=traj_y
            else:
                for patient in patients:
                    data_dicts[patient][50001100][extract_index]=position_dict[non_patient][0]
                    data_dicts[patient][50001101][extract_index]=position_dict[non_patient][1]
            pass
        for patient in patients:
            data_dicts[patient][50001110]=data_dicts[patient][50001100].diff().values
            data_dicts[patient][50001111]=data_dicts[patient][50001101].diff().values
            # data_dicts[patient][50001110].fillna(method="ffill",inplace=True)
            # data_dicts[patient][50001110].fillna(method="bfill",inplace=True)
            # data_dicts[patient][50001111].fillna(method="ffill",inplace=True)
            # data_dicts[patient][50001111].fillna(method="bfill",inplace=True)
        
        # 背景差分の擬似データを生成
        bg_differencing_pseudo_data_rules={
            "sit":0.1,
            "stand":0.4,
            "standup":0.7,
            "sitdown":0.7,
        }
        for patient in patients:
            for _,a_dict in action_dict[patient].items():
                data_dicts[patient][70000000][(data_dicts[patient]["timestamp"]>=a_dict["start_timestamp"]) & ((data_dicts[patient]["timestamp"]<=a_dict["end_timestamp"]))]=\
                    bg_differencing_pseudo_data_rules[a_dict["label"]]

        return data_dicts
    
    def get_basic_check_data(self,graph_dicts,patients):
        fill_value=0.1
        nodes_4000=[k for k in graph_dicts["node_dict"].keys() if ((int(k)>=40000000) and (int(k)<50000000))]
        array=np.full((len(nodes_4000)+1,len(nodes_4000)),fill_value)
        for i in range(len(nodes_4000)):
            array[i+1,i]=1
        data=pd.DataFrame(data=array,columns=nodes_4000)
        data.loc[:,40000000]=str((0,0.3,0.6))
        data.loc[1,40000000]=str((0.4,0.7,1.0))
        data.loc[:,40000001]=str((0,0.25,0.5))
        data.loc[2,40000001]=str((0.5,0.75,1.0))
        data["timestamp"]=np.arange(len(nodes_4000)+1)
        print(data)
        data_dicts={patients[0]:data}
        return data_dicts

if __name__=="__main__":
    trial_name="20250129DevBasicCheck"
    strage="NASK"
    cls=PseudoDataGenerator(trial_name=trial_name,strage=strage)
    cls.get_basic_check_data()