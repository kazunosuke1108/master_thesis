import os
import sys
from glob import glob
from icecream import ic
import copy

import pandas as pd
import numpy as np



sys.path.append(".")
sys.path.append("..")
if "catkin_ws" in os.getcwd():
    sys.path.append("/catkin_ws/src/master_thesis_modules")
else:
    sys.path.append(os.path.expanduser("~")+"/kazu_ws/master_thesis/master_thesis_modules")

from scripts.management.manager import Manager
from scripts.preprocess.blipTools import blipTools

class PreprocessMaster(Manager,blipTools):
    def __init__(self,trial_name,strage):
        super().__init__()
        self.trial_name=trial_name
        self.strage=strage
        self.data_dir_dict=self.get_database_dir(trial_name=trial_name,strage=strage)

        # 看護師IDの登録
        self.nurse_id="ID_00003"
        # 壁
        if self.trial_name=="20250115PullWheelchairObaachan2":
            self.xrange=[6,15]
            self.yrange=[-11,-4]
            # staff station
            self.staff_station=[6,-7.5]
            self.staff_direction=[0,0.1]
        else:
            self.xrange=[-4,6]
            self.yrange=[5,10]
            # staff station
            self.staff_station=[-4,7.5]
            self.staff_direction=[0.1,0]

        # Annotation csvの読み込み
        self.annotation_dir_path=self.data_dir_dict["mobilesensing_dir_path"]+"/PullWheelchairObaachan"
        # annotation_csv_path=self.annotation_dir_path+"/csv/annotation/PullWheelchairObaachan_annotation_ytnpc2021h_20240827_192540New_fullimagePath.csv"
        annotation_csv_path=self.annotation_dir_path+"/csv/annotation/PullWheelchairObaachan_annotation_ytnpc2021h_20240827_192540New_fixposition.csv"
        self.annotation_data=pd.read_csv(annotation_csv_path,header=0)
        ic(self.annotation_data)

        # BLIP記入済みのデータをload
        feature_csv_paths=sorted(glob(self.data_dir_dict["trial_dir_path"]+"/data_*_raw.csv"))
        self.id_names=["ID_"+os.path.basename(p)[len("data_"):-len("_raw.csv")] for p in feature_csv_paths]
        self.feature_dict={}
        for id_name,feature_csv_path in zip(self.id_names,feature_csv_paths):
            self.feature_dict[id_name]=pd.read_csv(feature_csv_path,header=0)

    def main(self):
        # 毎行読み込む
        id_names=[k[:-len("_activeBinary")] for k in self.annotation_data.keys() if "activeBinary" in k]
        print(id_names)
        # 一括で記録できるものを片付ける
        for id_name in id_names:
            # 患者の位置
            self.feature_dict[id_name]["60010000"]=self.annotation_data[id_name+"_x"]
            self.feature_dict[id_name]["60010001"]=self.annotation_data[id_name+"_y"]
            # 看護師の位置
            self.feature_dict[id_name]["50001100"]=self.annotation_data[self.nurse_id+"_x"].values
            self.feature_dict[id_name]["50001101"]=self.annotation_data[self.nurse_id+"_y"].values
            self.feature_dict[id_name]["50001110"]=self.feature_dict[id_name]["50001100"].diff()
            self.feature_dict[id_name]["50001111"]=self.feature_dict[id_name]["50001101"].diff()
            # 看護師不在箇所の補填（Staff stationの位置と方角を記入）
            self.feature_dict[id_name]["50001100"].fillna(self.staff_station[0],inplace=True)
            self.feature_dict[id_name]["50001101"].fillna(self.staff_station[1],inplace=True)
            self.feature_dict[id_name]["50001110"].fillna(self.staff_direction[0],inplace=True)
            self.feature_dict[id_name]["50001111"].fillna(self.staff_direction[1],inplace=True)
            
        for i,row in self.annotation_data.iterrows():
            print("now processing...",i,"/",len(self.annotation_data))
            for id_name in id_names:
                closest_wall=np.argmin([
                    abs(self.xrange[0]-self.annotation_data.loc[i,id_name+"_x"]),
                    abs(self.yrange[0]-self.annotation_data.loc[i,id_name+"_y"]),
                    abs(self.xrange[1]-self.annotation_data.loc[i,id_name+"_x"]),
                    abs(self.yrange[1]-self.annotation_data.loc[i,id_name+"_y"]),
                    ])
                if closest_wall==0:                
                    self.feature_dict[id_name].loc[i,"50001020"]=self.xrange[0]
                    self.feature_dict[id_name].loc[i,"50001021"]=self.annotation_data.loc[i,id_name+"_y"]
                elif closest_wall==1:
                    self.feature_dict[id_name].loc[i,"50001020"]=self.annotation_data.loc[i,id_name+"_x"]
                    self.feature_dict[id_name].loc[i,"50001021"]=self.yrange[0]
                elif closest_wall==2:
                    self.feature_dict[id_name].loc[i,"50001020"]=self.xrange[1]
                    self.feature_dict[id_name].loc[i,"50001021"]=self.annotation_data.loc[i,id_name+"_y"]
                elif closest_wall==3:
                    self.feature_dict[id_name].loc[i,"50001020"]=self.annotation_data.loc[i,id_name+"_x"]
                    self.feature_dict[id_name].loc[i,"50001021"]=self.yrange[1]                
                self.feature_dict[id_name]["50001020"]
                
        for id_name in id_names:
            self.feature_dict[id_name].to_csv(self.data_dir_dict["trial_dir_path"]+f"/data_{id_name[len('ID_'):]}_raw.csv",index=False)
        # 身体特徴量の抽出
        
        # 位置情報
        # 点滴の紐付け
        # 車椅子の紐付け
        # 最寄り壁の算出

        # 看護師

        pass

if __name__=="__main__":
    trial_name="20250115PullWheelchairObaachan2"
    strage="NASK"
    cls=PreprocessMaster(trial_name=trial_name,strage=strage)
    cls.main()