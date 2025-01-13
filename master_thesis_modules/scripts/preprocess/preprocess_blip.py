import os
import sys
from icecream import ic
import copy

import pandas as pd
import numpy as np

import cv2

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

        # BLIPの起動
        self.blip_processor,self.blip_model,self.device=self.activate_blip()        


        # Annotation csvの読み込み
        self.annotation_dir_path=self.data_dir_dict["mobilesensing_dir_path"]+"/Nagasaki20241205193158"
        annotation_csv_path=self.annotation_dir_path+"/csv/annotation/Nagasaki20241205193158_annotation_ytpc2024j_20241205_193158_fixposition.csv"
        self.annotation_data=pd.read_csv(annotation_csv_path,header=0)
        ic(self.annotation_data)

        self.feature_data=pd.DataFrame(data=self.annotation_data["timestamp"].values,columns=["timestamp"])
        columns=[
            50000000,
            50000001,
            50000010,
            50000011,
            50000100,
            50000101,
            50000102,
            50000103,
            50001000,
            50001001,
            50001002,
            50001002,
            50001003,
            50001010,
            50001011,
            50001012,
            50001013,
            50001020,
            50001021,
            50001022,
            50001023,
            50001100,
            50001101,
            50001110,
            50001111,
            60010000,
            60010001,
        ]
        for col in columns:
            self.feature_data[col]=np.nan

        # questions
        self.questions={
            "patient":{"query":"Question: Is this person wearing a red shirt? Answer:","node_code":50000000},
            "age":{"query":"Question: Is this person old, middle, or young? Answer:","node_code":50000010},
            "ivPole":{"query":"Question: Is there an IV pole in this image? Answer:","node_code":50001000},
            "wheelchair":{"query":"Question: Are there any wheelchair in this picture? Answer:","node_code":50001010},
        }

    def main(self):
        # 毎行読み込む
        id_names=[k[:-len("_activeBinary")] for k in self.annotation_data.keys() if "activeBinary" in k]
        print(id_names)
        self.feature_dict={id_name:copy.deepcopy(self.feature_data) for id_name in id_names}
        for i,row in self.annotation_data.iterrows():
            print("now processing...",i,"/",len(self.annotation_data))
            # 高画質jpgのpath取得
            rgb_image_path=self.data_dir_dict["mobilesensing_dir_path"]+"/"+self.annotation_data.loc[i,"fullrgb_imagePath"]
            rgb_img=cv2.imread(rgb_image_path)
            for id_name in id_names:
                # bounding boxの切り出し
                t,b,l,r=row[id_name+"_bbox_lowerY"],row[id_name+"_bbox_higherY"],row[id_name+"_bbox_lowerX"],row[id_name+"_bbox_higherX"],
                if np.isnan(t) or np.isnan(b) or np.isnan(l) or np.isnan(r):
                    continue
                t,b,l,r=int(t),int(b),int(l),int(r)
                bbox_rgb_img=rgb_img[t:b,l:r]
                # 拡張bounding boxの切り出し
                extend_ratio_tb=0.2
                extend_ratio_lr=0.2
                t_e,b_e,l_e,r_e,=np.max([t-extend_ratio_tb*rgb_img.shape[0],0]),np.min([b+extend_ratio_tb*rgb_img.shape[0],rgb_img.shape[0]]),np.max([l-extend_ratio_lr*rgb_img.shape[1],0]),np.min([r+extend_ratio_lr*rgb_img.shape[1],rgb_img.shape[0]])
                t_e,b_e,l_e,r_e=int(t_e),int(b_e),int(l_e),int(r_e)
                extended_bbox_rgb_img=rgb_img[t_e:b_e,l_e:r_e]
                if id_name in ["ID_00000","ID_00001"]:
                    cv2.imshow(id_name,extended_bbox_rgb_img)
                    cv2.waitKey(1)

                # BLIP
                for q_title,q_info in self.questions.items():
                    if q_title=="patient":
                        # BLIP 患者？
                        answer,confidence=self.get_vqa(blip_processor=self.blip_processor,blip_model=self.blip_model,device=self.device,image=bbox_rgb_img,question=q_info["query"],confidence=True)
                        # 答えを加工
                        if (answer=="yes") or (answer=="Yes"):
                            answer="no"
                        else:
                            answer="yes"
                        # 答えを記録
                        self.feature_dict[id_name].loc[i,self.questions[q_title]["node_code"]]=answer
                        self.feature_dict[id_name].loc[i,50000001]=confidence
                    elif q_title=="age":
                        # BLIP 年齢？
                        answer,confidence=self.get_vqa(blip_processor=self.blip_processor,blip_model=self.blip_model,device=self.device,image=bbox_rgb_img,question=q_info["query"],confidence=True)
                        self.feature_dict[id_name].loc[i,self.questions[q_title]["node_code"]]=answer
                        self.feature_dict[id_name].loc[i,50000011]=confidence
                    elif q_title=="ivPole":
                        # BLIP 点滴？
                        answer,confidence=self.get_vqa(blip_processor=self.blip_processor,blip_model=self.blip_model,device=self.device,image=extended_bbox_rgb_img,question=q_info["query"],confidence=True)
                        print(id_name,answer,confidence,extended_bbox_rgb_img.shape)
                        if (answer=="yes") or (answer=="Yes"):
                            answer=(self.annotation_data.loc[i,id_name+"_x"],self.annotation_data.loc[i,id_name+"_y"])
                        else:
                            answer=(np.nan,np.nan)
                        self.feature_dict[id_name].loc[i,50001000]=answer[0]
                        self.feature_dict[id_name].loc[i,50001001]=answer[1]
                        self.feature_dict[id_name].loc[i,50001002]=confidence
                        self.feature_dict[id_name].loc[i,50001003]=confidence
                        
                    elif q_title=="wheelchair":
                        # BLIP 車椅子？
                        answer,confidence=self.get_vqa(blip_processor=self.blip_processor,blip_model=self.blip_model,device=self.device,image=extended_bbox_rgb_img,question=q_info["query"],confidence=True)
                        if (answer=="yes") or (answer=="Yes"):
                            answer=(self.annotation_data.loc[i,id_name+"_x"],self.annotation_data.loc[i,id_name+"_y"])
                        else:
                            answer=(np.nan,np.nan)
                        self.feature_dict[id_name].loc[i,50001010]=answer[0]
                        self.feature_dict[id_name].loc[i,50001011]=answer[1]    
                        self.feature_dict[id_name].loc[i,50001012]=confidence
                        self.feature_dict[id_name].loc[i,50001013]=confidence
                
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
    trial_name="20250107BlipRenewal"
    strage="NASK"
    cls=PreprocessMaster(trial_name=trial_name,strage=strage)
    cls.main()