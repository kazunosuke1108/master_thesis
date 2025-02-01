import os
import sys
from glob import glob
from pprint import pprint
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.append(".")
sys.path.append("..")
sys.path.append(os.path.expanduser("~")+"/kazu_ws/master_thesis/master_thesis_modules")
from scripts.management.manager import Manager

import numpy as np
import pandas as pd

import cv2

from multiprocessing import cpu_count,Process


class VideoVisualizer(Manager):
    def __init__(self,sensing_trial_name,evaluation_trial_name,notification_trial_name,visualize_trial_name,):
        super().__init__()
        self.sensing_trial_name=sensing_trial_name
        self.evaluation_trial_name=evaluation_trial_name
        self.notification_trial_name=notification_trial_name
        self.visualize_trial_name=visualize_trial_name
        self.sensing_dir_dict=self.get_database_dir(trial_name=sensing_trial_name,strage="NASK")
        self.evaluation_dir_dict=self.get_database_dir(trial_name=evaluation_trial_name,strage="NASK")
        self.notification_dir_dict=self.get_database_dir(trial_name=notification_trial_name,strage="NASK")
        self.visualize_dir_dict=self.get_database_dir(trial_name=visualize_trial_name,strage="NASK")
        
        # parameters
        smoothing_window=40
        
        # MobileSensing系のデータ
        sensing_csv_path="//NASK/common/FY2024/09_MobileSensing/Nagasaki20241205193158/csv/annotation/Nagasaki20241205193158_annotation_ytpc2024j_20241205_193158_fixposition.csv"
        self.sensing_data=pd.read_csv(sensing_csv_path,header=0)

        # リスク評価のデータ
        evaluation_csv_paths=sorted(glob("//192.168.1.5/common/FY2024/01_M2/05_hayashide/MasterThesis_database/20250108DevMewThrottlingExp/data_*_eval.csv"))
        self.patients=[os.path.basename(k)[len("data_"):-len("_eval.csv")] for k in evaluation_csv_paths]
        self.evaluation_data_dict={k:pd.read_csv(path) for k,path in zip(self.patients,evaluation_csv_paths)}
        ## 平滑化処理を入れておく（window幅は通知側と揃えないとまずい）
        smoothing_cols=[]
        for k in self.evaluation_data_dict[self.patients[0]].keys():
            try:
                int(k)
            except ValueError:
                continue
            if int(k)<40000000 or ((int(k)>=40000010) and int(k)<50000000):
                smoothing_cols.append(k)
        for patient in self.patients:
            self.evaluation_data_dict[patient]=self.evaluation_data_dict[patient].rolling(smoothing_window).mean()

        # 通知のデータ
        notification_csv_path="//192.168.1.5/common/FY2024/01_M2/05_hayashide/MasterThesis_database/20250124NotifyForNagasakiStaff1/20250124NotifyForNagasakiStaff1_20250108DevMewThrottlingExp_notify_history.csv"
        self.notification_data=pd.read_csv(notification_csv_path,header=0)

        if len(self.sensing_data)!=len(self.evaluation_data_dict[self.patients[0]]):
            raise Exception("MobileSensingとリスク評価でデータの長さが一致しない")
        
        # リスク評価の順位情報を用意しておく
        self.rank_data=self.get_rank_data()
        # rank_00: 危険度第1位患者の名称, 00000_rank: 患者ID00000の全体順位

        # 患者のイメージカラー
        self.patient_color_dict={p:c for p,c in zip(self.patients,self.get_colors())}


        
    def get_rank_data(self):
        rank_data=pd.DataFrame(self.sensing_data["timestamp"].values,columns=["timestamp"])
        # 全評価結果から10000000を取ってくる
        for patient in self.patients:
            rank_data[patient+"_risk"]=self.evaluation_data_dict[patient]["10000000"]
        # ランキングを作る
        # 各行ごとに順位付け

        for i,row in rank_data.iterrows():
            risks=np.array(row.values[1:])
            if np.isnan(risks[0]):
                continue
            rank_list=np.array(self.patients)[np.argsort(-risks)]
            rank_data.loc[i,[f"rank_{str(n).zfill(2)}" for n in range(len(self.patients))]]=rank_list

        for patient in self.patients:
            rank_data[patient+"_rank"]=np.nan
        for i,row in rank_data.iterrows():
            rank_list=rank_data.loc[i,[f"rank_{str(n).zfill(2)}" for n in range(len(self.patients))]]
            for rank,patient_name in enumerate(rank_list):
                try:
                    int(patient_name)
                except ValueError:
                    continue
                rank_data.loc[i,[f"{patient_name}_rank"]]=rank

        return rank_data

    def get_colors(self):
        # tab10カラーマップの上位10色を取得
        colors = plt.get_cmap("tab10").colors
        colors = [(int(b*255), int(g*255), int(r*255)) for r, g, b in colors]
        return colors


    def draw_bbox(self,i,img):
        for patient in self.patients:
            # patient_rank
            bbox_info=[
                (int(self.sensing_data.loc[i,f"ID_{patient}_bbox_higherX"]),int(self.sensing_data.loc[i,f"ID_{patient}_bbox_higherY"])),
                (int(self.sensing_data.loc[i,f"ID_{patient}_bbox_lowerX"]),int(self.sensing_data.loc[i,f"ID_{patient}_bbox_lowerY"])),
            ]
            if not np.isnan(self.rank_data.loc[i,patient+"_rank"]):
                thickness=len(self.patients)-int(self.rank_data.loc[i,patient+"_rank"])
            else:
                thickness=1
            cv2.rectangle(img,bbox_info[0],bbox_info[1],self.patient_color_dict[patient], thickness=thickness)
        return img
        
        

    def main(self):
        # 画像を1枚ずつ取り出す
        for i,row in self.sensing_data.iterrows():
            # print(self.sensing_data.loc[i,"fullrgb_imagePath"])
            elp_img_path="//NASK/common/FY2024/09_MobileSensing"+"/"+self.sensing_data.loc[i,"fullrgb_imagePath"]
            elp_img=cv2.imread(elp_img_path)
            # bounding boxを描く
            elp_img_with_box=self.draw_bbox(i,elp_img)
            cv2.imshow("ELP with BBOX",elp_img_with_box)
            cv2.waitKey(1)

            # raise NotImplementedError

        pass

if __name__=="__main__":
    sensing_trial_name="Nagasaki20241205193158"
    evaluation_trial_name="20250108DevMewThrottlingExp"
    notification_trial_name="20250124NotifyForNagasakiStaff1"
    visualize_trial_name="20250201VisualizeVideo"
    cls=VideoVisualizer(
        sensing_trial_name=sensing_trial_name,
        evaluation_trial_name=evaluation_trial_name,
        notification_trial_name=notification_trial_name,
        visualize_trial_name=visualize_trial_name,
    )
    cls.main()

"""
'ID_00003_x', 'ID_00003_y',
'ID_00003_bbox_lowerX', 'ID_00003_bbox_lowerY', 'ID_00003_bbox_higherX',
'ID_00003_bbox_higherY', 'ID_00003_object_id', 'ID_00003_imagePath',
'ID_00003_activeBinary',
"""