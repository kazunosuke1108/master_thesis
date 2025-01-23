import os
import sys
from glob import glob
from icecream import ic
import copy
import time

import pandas as pd
import numpy as np

import cv2
from ultralytics import YOLO
import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)


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

        # 姿勢推定結果のプロット有無
        self.debug=False
        if self.debug:
            self.data_dir_dict["pose_dir_path"]=self.data_dir_dict["trial_dir_path"]+"/pose"
            os.makedirs(self.data_dir_dict["pose_dir_path"],exist_ok=True)

        # YOLO
        self.model=YOLO("yolo11x-pose.pt")

        # Annotation csvの読み込み
        self.annotation_dir_path=self.data_dir_dict["mobilesensing_dir_path"]+"/PullWheelchairObaachan"
        annotation_csv_path=self.annotation_dir_path+"/csv/annotation/PullWheelchairObaachan_annotation_ytnpc2021h_20240827_192540New_fixposition.csv"
        self.annotation_data=pd.read_csv(annotation_csv_path,header=0)
        ic(self.annotation_data)

        # BLIP記入済みのデータをload
        feature_csv_paths=sorted(glob(self.data_dir_dict["trial_dir_path"]+"/data_*_raw.csv"))
        self.id_names=["ID_"+os.path.basename(p)[len("data_"):-len("_raw.csv")] for p in feature_csv_paths]
        self.feature_dict={}
        for id_name,feature_csv_path in zip(self.id_names,feature_csv_paths):
            self.feature_dict[id_name]=pd.read_csv(feature_csv_path,header=0)
        self.KEYPOINTS_NAMES = [
            "nose",  # 0
            "l_eye",  # 1
            "r_eye",  # 2
            "l_ear",  # 3
            "r_ear",  # 4
            "l_shoulder",  # 5
            "r_shoulder",  # 6
            "l_elbow",  # 7
            "r_elbow",  # 8
            "l_wrist",  # 9
            "r_wrist",  # 10
            "l_hip",  # 11
            "r_hip",  # 12
            "l_knee",  # 13
            "r_knee",  # 14
            "l_ankle",  # 15
            "r_ankle",  # 16
        ]

        self.risky_motion_dict={
            40000010:np.array([1,         0,      0,      1]),
            40000011:np.array([0,         0.5,    0.5,      0.5]),
            40000012:np.array([0,         0,      0.5,      0.5]),
            40000013:np.array([0,         1,      np.nan, np.nan]),
            40000014:np.array([np.nan,    0,      1,      np.nan]),
            40000015:np.array([np.nan,    0.5,      0.5,      np.nan]),
            40000016:np.array([np.nan,    0,      0.5,      np.nan]),
        }

        # self.risky_landmark_dict={
        #     1:  np.array([1,         1,      np.nan, 1]),
        #     3:  np.array([0,         1,      1,      0]),
        #     4:  np.array([0,         0,      1,      0]),
        #     5:  np.array([0,         1,      np.nan, np.nan]),
        #     6:  np.array([np.nan,    np.nan, 1,      np.nan]),
        #     7:  np.array([np.nan,    1,      0,      0]),
        #     9:  np.array([np.nan,    0,      0,      np.nan]),
        # }

    def feature_stand(self,kp_dict):
        l_hip_length=np.sqrt(
            # (kp_dict["l_hip"]["x"]-kp_dict["l_knee"]["x"])**2+\
            (kp_dict["l_hip"]["y"]-kp_dict["l_knee"]["y"])**2
        )
        r_hip_length=np.sqrt(
            # (kp_dict["r_hip"]["x"]-kp_dict["r_knee"]["x"])**2+\
            (kp_dict["r_hip"]["y"]-kp_dict["r_knee"]["y"])**2
        )
        hip_length=np.nanmean([l_hip_length,r_hip_length])

        l_ankle_length=np.sqrt(
            # (kp_dict["l_ankle"]["x"]-kp_dict["l_knee"]["x"])**2+\
            (kp_dict["l_ankle"]["y"]-kp_dict["l_knee"]["y"])**2
        )
        r_ankle_length=np.sqrt(
            # (kp_dict["r_ankle"]["x"]-kp_dict["r_knee"]["x"])**2+\
            (kp_dict["r_ankle"]["y"]-kp_dict["r_knee"]["y"])**2
        )
        ankle_length=np.nanmean([l_ankle_length,r_ankle_length])
        # ankle_length=(l_ankle_length+r_ankle_length)/2
        # print(ankle_length,hip_length)

        stand_ratio=np.nanmax([0,np.nanmin([1,hip_length/ankle_length])])
        return stand_ratio
    
    def feature_stand_v2(self,kp_dict,t,b,l,r):
        # bboxの縦横比で考える。
        # 縦が横の2倍以下のとき座る。3倍以上で立つ。
        ratio=(b-t)/(r-l)
        sit_ratio=2
        stand_ratio=3.5
        ratio=(np.clip(ratio,sit_ratio,stand_ratio)-sit_ratio)/(stand_ratio-sit_ratio)
        ratio=np.clip(ratio,0,1)# 念の為
        return ratio

    def feature_lean(self,kp_dict):
        shoulder=np.array([np.nanmean([kp_dict["l_shoulder"]["x"],kp_dict["r_shoulder"]["x"]]),np.nanmean([kp_dict["l_shoulder"]["y"],kp_dict["r_shoulder"]["y"]])])
        hip=np.array([np.nanmean([kp_dict["l_hip"]["x"],kp_dict["r_hip"]["x"]]),np.nanmean([kp_dict["l_hip"]["y"],kp_dict["r_hip"]["y"]])])
        theta=abs(np.arctan2(abs(shoulder[0]-hip[0]),abs(shoulder[1]-hip[1])))
        # stand_ratio=np.nanmax([0,np.nanmin([1,theta/(np.pi/2)])])
        stand_ratio=theta/(np.pi/4)
        stand_ratio=np.clip(stand_ratio,0,1)
        return stand_ratio
    
    def feature_wrist(self,kp_dict):
        l_wrist_distance=np.sqrt(
            (kp_dict["l_hip"]["x"]-kp_dict["l_wrist"]["x"])**2+\
            (kp_dict["l_hip"]["y"]-kp_dict["l_wrist"]["y"])**2
        )
        r_wrist_distance=np.sqrt(
            (kp_dict["r_hip"]["x"]-kp_dict["r_wrist"]["x"])**2+\
            (kp_dict["r_hip"]["y"]-kp_dict["r_wrist"]["y"])**2
        )
        wrist_distance=np.nanmax([l_wrist_distance,r_wrist_distance])


        shoulder=np.array([np.nanmean([kp_dict["l_shoulder"]["x"],kp_dict["r_shoulder"]["x"]]),np.nanmean([kp_dict["l_shoulder"]["y"],kp_dict["r_shoulder"]["y"]])])
        hip=np.array([np.nanmean([kp_dict["l_hip"]["x"],kp_dict["r_hip"]["x"]]),np.nanmean([kp_dict["l_hip"]["y"],kp_dict["r_hip"]["y"]])])
        sebone=np.sqrt((shoulder-hip)[0]**2+(shoulder-hip)[1]**2)
        # wrist_ratio=np.nanmax([0,np.nanmin([1,wrist_distance/sebone])])
        wrist_ratio=wrist_distance/(sebone*2)
        wrist_ratio=np.clip(wrist_ratio,0,1)

        return wrist_ratio
    
    def feature_ankle(self,kp_dict):
        l_ankle_distance=np.sqrt(
            (kp_dict["l_hip"]["x"]-kp_dict["l_ankle"]["x"])**2+\
            (kp_dict["l_hip"]["y"]-kp_dict["l_ankle"]["y"])**2
        )
        r_ankle_distance=np.sqrt(
            (kp_dict["r_hip"]["x"]-kp_dict["r_ankle"]["x"])**2+\
            (kp_dict["r_hip"]["y"]-kp_dict["r_ankle"]["y"])**2
        )
        ankle_distance=np.nanmax([l_ankle_distance,r_ankle_distance])


        nose=np.array([np.nanmean([kp_dict["nose"]["x"],kp_dict["nose"]["x"]]),np.nanmean([kp_dict["nose"]["y"],kp_dict["nose"]["y"]])])
        hip=np.array([np.nanmean([kp_dict["l_hip"]["x"],kp_dict["r_hip"]["x"]]),np.nanmean([kp_dict["l_hip"]["y"],kp_dict["r_hip"]["y"]])])
        sebone=np.sqrt((nose-hip)[0]**2+(nose-hip)[1]**2)
        # print("l_ankle_distance",l_ankle_distance)
        # print("r_ankle_distance",r_ankle_distance)
        # print("ankle_distance",ankle_distance)
        # print("sebone",sebone)
        # ankle_ratio=np.nanmax([0,np.nanmin([1,ankle_distance/sebone])])
        ankle_ratio=ankle_distance/sebone
        ankle_ratio=np.clip(ankle_ratio,0,1)
        return ankle_ratio

    def main(self):
        id_names=[k[:-len("_activeBinary")] for k in self.annotation_data.keys() if "activeBinary" in k]
        print(id_names)
        # 毎行読み込む
        for i,row in self.annotation_data.iterrows():
            # bboxの内包関係チェッカー
            self.occlusion_dict={id_name:{"bbox_t":np.nan,"bbox_b":np.nan,"bbox_l":np.nan,"bbox_r":np.nan} for id_name in id_names}
            print("now processing...",i,"/",len(self.annotation_data))
            # 高画質jpgのpath取得
            rgb_image_path=self.annotation_data.loc[i,"fullrgb_imagePath"] # self.data_dir_dict["mobilesensing_dir_path"]+"/"+
            # try:
            diff_image_path=self.annotation_data.loc[i,"fulldiff_imagePath"] # self.data_dir_dict["mobilesensing_dir_path"]+"/"+
            if type(diff_image_path)!=str:
                continue
            # except TypeError:
            #     continue
                # pass # nanだった場合、直前の背景差分画像を使用
            rgb_img=cv2.imread(rgb_image_path)
            diff_img=cv2.imread(diff_image_path)

            for id_name in id_names:
                # bounding boxの切り出し
                t,b,l,r=row[id_name+"_bbox_lowerY"],row[id_name+"_bbox_higherY"],row[id_name+"_bbox_lowerX"],row[id_name+"_bbox_higherX"],
                if np.isnan(t) or np.isnan(b) or np.isnan(l) or np.isnan(r):
                    continue
                t,b,l,r=int(t),int(b),int(l),int(r)
                bbox_rgb_img=rgb_img[t:b,l:r]
                # 拡張bounding boxの切り出し
                extend_ratio=0.025
                t_e,b_e,l_e,r_e,=np.max([t-extend_ratio*rgb_img.shape[0],0]),np.min([b+extend_ratio*rgb_img.shape[0],rgb_img.shape[0]]),np.max([l-extend_ratio*rgb_img.shape[1],0]),np.min([r+extend_ratio*rgb_img.shape[1],rgb_img.shape[0]])
                t_e,b_e,l_e,r_e=int(t_e),int(b_e),int(l_e),int(r_e)
                self.occlusion_dict[id_name]["bbox_t"],self.occlusion_dict[id_name]["bbox_b"],self.occlusion_dict[id_name]["bbox_l"],self.occlusion_dict[id_name]["bbox_r"]=t_e,b_e,l_e,r_e
                extended_bbox_rgb_img=rgb_img[t_e:b_e,l_e:r_e]
                # 背景差分値の取得
                bbox_diff_img=diff_img[t:b,l:r]
                self.feature_dict[id_name].loc[i,"70000000"]=bbox_diff_img.mean()/255
                if id_name=="ID_00000":
                    # cv2.imshow("diff",bbox_diff_img)
                    cv2.imshow("rgb",extended_bbox_rgb_img)
                    cv2.waitKey(1)
                    print(bbox_diff_img.max()/255,bbox_diff_img.mean()/255)

                # 姿勢推定
                try:
                    results=self.model(extended_bbox_rgb_img)
                    if self.debug:
                        plotted_image=results[0].plot()
                        cv2.imwrite(self.data_dir_dict["pose_dir_path"]+f"/{trial_name}_{id_name}_{str(i).zfill(5)}.jpg",plotted_image)
                except ZeroDivisionError:
                    continue
                kp=results[0].keypoints
                try:
                    kp_xy=np.array(kp.xy[0].tolist())
                    kp_conf=np.array(kp.conf[0].tolist())
                except TypeError:
                    continue
                kp_dict={}
                for j,k in enumerate(self.KEYPOINTS_NAMES):
                    kp_dict[k]={
                        "x":int(kp_xy[j,0]) if int(kp_xy[j,0])!=0 else np.nan,
                        "y":int(kp_xy[j,1]) if int(kp_xy[j,1])!=0 else np.nan,
                        "conf":kp_conf[j],
                    }
                # 特徴量算出
                features=np.array([
                    self.feature_stand_v2(kp_dict,t,b,l,r),
                    # self.feature_stand(kp_dict),
                    self.feature_lean(kp_dict),
                    self.feature_wrist(kp_dict),
                    self.feature_ankle(kp_dict),
                    ]
                )
                self.feature_dict[id_name].loc[i,"50000100"]=features[0]
                self.feature_dict[id_name].loc[i,"50000101"]=features[1]
                self.feature_dict[id_name].loc[i,"50000102"]=features[2]
                self.feature_dict[id_name].loc[i,"50000103"]=features[3]
                # ic(kp_dict)
                # ic(features)
                # raise NotImplementedError
                # bounding boxの重複チェック・削除
                for opponent_id_name in id_names:
                    if opponent_id_name==id_name:
                        continue
                    if ((l_e<self.occlusion_dict[opponent_id_name]["bbox_l"]) & (self.occlusion_dict[opponent_id_name]["bbox_l"]<r_e)) and \
                        ((l_e<self.occlusion_dict[opponent_id_name]["bbox_r"]) & (self.occlusion_dict[opponent_id_name]["bbox_r"]<r_e)):
                        if (opponent_id_name!="ID_00004") and (opponent_id_name!="ID_00007") and (opponent_id_name!="ID_00008") and (opponent_id_name!="ID_00009"):# 一番手前になるのがほぼ明らかなのでID_00004は除外する。それ以外に関しては、IDが若い番号の方を消す                    
                            print(f"{opponent_id_name} is occluded by {id_name}. Remove {opponent_id_name}")
                            print(l_e,self.occlusion_dict[opponent_id_name]["bbox_l"],self.occlusion_dict[opponent_id_name]["bbox_r"],r_e)
                            self.feature_dict[opponent_id_name].loc[i,["50000100","50000101","50000102","50000103","70000000"]]=np.nan

            # raise NotImplementedError

        for id_name in id_names:
            self.feature_dict[id_name].to_csv(self.data_dir_dict["trial_dir_path"]+f"/data_{id_name[len('ID_'):]}_raw.csv",index=False)

        pass

if __name__=="__main__":
    trial_name="20250115PullWheelchairObaachan2"
    strage="NASK"
    cls=PreprocessMaster(trial_name=trial_name,strage=strage)
    cls.main()