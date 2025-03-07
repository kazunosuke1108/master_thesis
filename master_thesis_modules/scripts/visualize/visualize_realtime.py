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
    
    def export_movies(self,map=False,bbox=True):
        # bbox
        if bbox:
            jpg_bbox_paths=sorted(glob(self.data_dir_dict["mobilesensing_dir_path"]+"/jpg/bbox/*.jpg"))
            mp4_bbox_path=self.data_dir_dict["mobilesensing_dir_path"]+"/mp4/bbox.mp4"
            self.jpg2mp4(image_paths=jpg_bbox_paths,mp4_path=mp4_bbox_path,fps=5)
        # map
        if map:
            jpg_map_paths=sorted(glob(self.data_dir_dict["mobilesensing_dir_path"]+"/jpg/map/*.jpg"))
            mp4_map_path=self.data_dir_dict["mobilesensing_dir_path"]+"/mp4/map.mp4"
            self.jpg2mp4(image_paths=jpg_map_paths,mp4_path=mp4_map_path,fps=5)

    def export_characters(self):
        plt.rcParams["figure.figsize"] = (20,5)
        def extract_bbox_img(elp_img_path,idx):
            t,b,l,r=df_after_reid.loc[idx,patient+"_bboxLowerY"],df_after_reid.loc[idx,patient+"_bboxHigherY"],df_after_reid.loc[idx,patient+"_bboxLowerX"],df_after_reid.loc[idx,patient+"_bboxHigherX"]
            t,b,l,r=int(t),int(b),int(l),int(r)
            bbox_elp_img=cv2.imread(elp_img_path)[t:b,l:r]
            bbox_elp_img = cv2.cvtColor(bbox_elp_img, cv2.COLOR_BGR2RGB) # imreadはBGRなのでRGBに変換
            
            return bbox_elp_img
        # load
        elp_img_paths=sorted(glob(self.data_dir_dict["mobilesensing_dir_path"]+"/jpg/elp/left/*.jpg"))
        elp_img_stamps=np.array([float(os.path.basename(p).split("_")[1][:-len(".jpg")]) for p in elp_img_paths])
        df_after_reid_csv_path=self.data_dir_dict["mobilesensing_dir_path"]+"/csv/df_after_reid.csv"
        df_after_reid=pd.read_csv(df_after_reid_csv_path,header=0).dropna(how="all",axis=1)
        df_eval_csv_path=self.data_dir_dict["mobilesensing_dir_path"]+"/csv/df_eval.csv"
        df_eval=pd.read_csv(df_eval_csv_path,header=0).dropna(how="all",axis=1)
        patients=sorted(list(set([k.split("_")[0] for k in df_after_reid.keys() if (("timestamp" not in k) and ("activeBinary" not in k))])))

        # 患者ごとに
        gs=GridSpec(1,len(patients))

        for i,patient in enumerate(patients):
            # 代表画像の時刻の決定
            idx=list(df_after_reid[patient+"_bboxHigherX"].dropna().index)[0]
            timestamp=df_after_reid.loc[idx,patient+"_timestamp"]
            closest_elp_img_path=elp_img_paths[np.argmin(abs(elp_img_stamps-timestamp))]

            # 画像を切り出し
            bbox_elp_img=extract_bbox_img(elp_img_path=closest_elp_img_path,idx=idx)
            
            # 職種判定状況のカウント
            try:
                possibility_of_staff=df_eval[patient+"_50000001"].mean()
            except KeyError:
                possibility_of_staff=np.nan
            print(patient,possibility_of_staff)

            # 貼り付け
            plt.subplot(gs[i])
            plt.imshow(bbox_elp_img)
            plt.title(f"{patient}\nStaff: {np.round(100*possibility_of_staff,1)}%")

            # 保存
        plt.savefig(self.data_dir_dict["mobilesensing_dir_path"]+"/graph/characters.jpg")
    
    def export_timeseries(self):
        # 観測値
        # 評価結果
        csv_eval_path=self.data_dir_dict["mobilesensing_dir_path"]+"/csv/df_eval.csv"
        eval_data=pd.read_csv(csv_eval_path,header=0)
        patients=sorted(list(set([k.split("_")[0] for k in eval_data.keys() if "timestamp" not in k])))
        nodes=sorted(list(set([k.split("_")[1] for k in eval_data.keys() if "timestamp" not in k])))

        # t_range=[1740479370,1740479450]


        for node in nodes:
            print(node)
            for patient in patients:
                if node in ["40000000","40000001",]: #TFN
                    try:
                        plt.plot(eval_data["timestamp"],[eval(d)[1] for d in eval_data[patient+"_"+node].fillna(method="ffill")],label=patient)
                    except TypeError:
                        pass
                elif node in ["50000000","50000010"]: #文字
                    # plt.plot(eval_data["timestamp"],eval_data[patient+"_"+node])
                    pass
                else:
                    # plt.plot(eval_data["timestamp"],eval_data[patient+"_"+node],label=patient)
                    plt.plot(eval_data["timestamp"].rolling(5).mean(),eval_data[patient+"_"+node].rolling(5).mean(),label=patient)
                    # plt.xlim(t_range)
            plt.legend()
            plt.grid()
            plt.title(node)
            plt.savefig(self.data_dir_dict["mobilesensing_dir_path"]+f"/graph/{node}.jpg")
            plt.close()
        # 通知結果
        pass

if __name__=="__main__":
    trial_name="20250307postAnalysis"
    strage="NASK"
    cls=Visualizer(trial_name,strage)
    cls.export_movies(map=False,bbox=True)
    cls.export_timeseries()
    # cls.export_characters()