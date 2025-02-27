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

class VisualizeQuestionare(Manager):
    def __init__(self):
        super().__init__()
        self.trial_dir_path="/home/hayashide/kazu_ws/master_thesis/master_thesis_modules/database/20250227Visualize2"
        os.makedirs(self.trial_dir_path+"/jpg",exist_ok=True)
        # ELP images
        self.elp_image_paths=sorted(glob("/home/hayashide/ytlab_ros_ws/ytlab_handheld_sensoring_system/ytlab_handheld_sensoring_system_modules/database/20250227Both3/jpg/elp/right/*.jpg"))
        self.elp_image_stamps=[float(os.path.basename(p).split("_")[1][:-len(".jpg")]) for p in self.elp_image_paths]

        # BBOXのデータ
        self.df_after_reid_csv_path="/home/hayashide/ytlab_ros_ws/ytlab_handheld_sensoring_system/ytlab_handheld_sensoring_system_modules/database/20250227Both3/csv/df_after_reid.csv"
        self.df_after_reid=pd.read_csv(self.df_after_reid_csv_path,header=0)#.dropna(how="all",axis=1)
        # ランキングのデータ（左上に出すなら）
        self.post_csv_path="/home/hayashide/ytlab_ros_ws/ytlab_handheld_sensoring_system/ytlab_handheld_sensoring_system_modules/database/20250227Both3/csv/df_post.csv"
        self.df_post=pd.read_csv(self.post_csv_path,header=0)
        # 通知のデータ
        self.notify_history_csv_path="/home/hayashide/ytlab_ros_ws/ytlab_handheld_sensoring_system/ytlab_handheld_sensoring_system_modules/database/20250227Both3/csv/notify_history_modifyName2.csv"
        self.notify_history=pd.read_csv(self.notify_history_csv_path,header=0)
        print(self.notify_history)

        self.colors_01 = plt.get_cmap("tab10").colors
        self.colors = [(int(b*255), int(g*255), int(r*255)) for r, g, b in self.colors_01]
        self.colors=self.colors+self.colors
        self.colors=self.colors+self.colors
        self.colors=self.colors+self.colors
        self.colors=self.colors+self.colors

        self.num2alpha = lambda c: chr(c+65)

        self.manage_name_dict()



    def manage_name_dict(self,patients=[]):
        if len(patients)==0:
            self.name_dict={}
        else:
            for patient in patients:
                if patient not in list(self.name_dict.keys()):
                    self.name_dict[patient]=self.num2alpha(len(self.name_dict))
        return self.name_dict


    def draw_bbox(self,elp_img,json_latest_data,patients):
        for patient in patients:
            if np.isnan(json_latest_data[f"{patient}_bboxHigherX"]):
                continue
            # patient_rank
            bbox_info=[
                (int(json_latest_data[f"{patient}_bboxHigherX"]),int(json_latest_data[f"{patient}_bboxHigherY"])),
                (int(json_latest_data[f"{patient}_bboxLowerX"]),int(json_latest_data[f"{patient}_bboxLowerY"])),
            ]
            # if not np.isnan(self.rank_data.loc[i,patient+"_rank"]):
            #     thickness=len(self.patients)-int(self.rank_data.loc[i,patient+"_rank"])
            # else:
            thickness=4
            cv2.rectangle(elp_img,bbox_info[0],bbox_info[1],self.colors[int(patient)], thickness=thickness)

            # 氏名を書き足す
            # if self.tf_draw_name_on_bbox:
            elp_img=self.draw_japanese_text(img=elp_img,text=self.name_dict[patient],position=bbox_info[0],bg_color=self.color_converter(self.colors[int(patient)]),font_size=45)
        return elp_img
    
    def draw_notification(self,img,notify_dict):
        img=self.draw_japanese_text(img=img,text=f"通知 {notify_dict['notificationId']+1}",position=(0,50),text_color=(255,255,255),bg_color=self.color_converter(self.colors[int(notify_dict["patient"])]),font_size=45)
        anchor=(50,600)
        text=notify_dict["sentence"].replace("バランス","姿勢")
        img=self.draw_japanese_text(img=img,text=text,position=anchor,text_color=(255,255,255),bg_color=self.color_converter(self.colors[int(notify_dict["patient"])]),font_size=45)
        return img        
    def main(self):
        for elp_image_path,timestamp in zip(self.elp_image_paths,self.elp_image_stamps):
            elp_img=cv2.imread(elp_image_path)
            self.timestamp=timestamp
            print(self.timestamp)
            elp_img=self.draw_japanese_text(img=elp_img,text=f"時刻 {np.round(self.timestamp-self.elp_image_stamps[0],1)}秒",position=(0,0),text_color=(0,0,0),bg_color=(255,255,255),font_size=45)

            idx=np.argmin(abs(self.df_after_reid["00000_timestamp"].values-timestamp))
            # print(idx)
            active_patients=sorted(list(set([k.split("_")[0] for k in self.df_after_reid.keys() if ~np.isnan(self.df_after_reid.loc[idx,k.split("_")[0]+"_x"])])))
            self.manage_name_dict(active_patients)
            elp_img=self.draw_bbox(elp_img=elp_img,json_latest_data=self.df_after_reid.loc[idx,:],patients=active_patients)
            
            notify_idxes = self.notify_history[(self.notify_history["timestamp"] < timestamp) & (self.notify_history["timestamp"] >= timestamp - 10)].index
            if len(notify_idxes)>0:
                notify_idx=notify_idxes[0]
                elp_img=self.draw_notification(img=elp_img,notify_dict=self.notify_history.loc[notify_idx,:])
            
            cv2.imwrite(self.trial_dir_path+"/jpg/"+f"{os.path.basename(elp_image_path)}",elp_img)
        
        pass
    def export(self):
        self.jpg2mp4(sorted(glob(self.trial_dir_path+"/jpg/*.jpg")),self.trial_dir_path+"/GUI.mp4",fps=3)

if __name__=="__main__":
    cls=VisualizeQuestionare()
    cls.main()
    cls.export()