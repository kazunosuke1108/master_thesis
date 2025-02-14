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

class PreprocessBlip(Manager,blipTools):
    def __init__(self):

        # BLIPの起動
        self.blip_processor,self.blip_model,self.device=self.activate_blip()

        # questions
        self.questions={
            "patient":{"query":"Question: Is this person wearing a red shirt? Answer:","node_code":"50000000"},
            "age":{"query":"Question: Is this person old, middle, or young? Answer:","node_code":"50000010"},
            "ivPole":{"query":"Question: Is there an IV pole in this image? Answer:","node_code":"50001000"},
            "wheelchair":{"query":"Question: Are there any wheelchair in this picture? Answer:","node_code":"50001010"},
        }

    def blip_snapshot(self,data_dict,rgb_img,t,b,l,r,extend_ratio_tb=0.2,extend_ratio_lr=0.2):
        # bounding boxの切り出し
        t,b,l,r=int(t),int(b),int(l),int(r)
        bbox_rgb_img=rgb_img[t:b,l:r]
        # 拡張bounding boxの切り出し
        t_e,b_e,l_e,r_e,=np.max([t-extend_ratio_tb*rgb_img.shape[0],0]),np.min([b+extend_ratio_tb*rgb_img.shape[0],rgb_img.shape[0]]),np.max([l-extend_ratio_lr*rgb_img.shape[1],0]),np.min([r+extend_ratio_lr*rgb_img.shape[1],rgb_img.shape[0]])
        t_e,b_e,l_e,r_e=int(t_e),int(b_e),int(l_e),int(r_e)
        extended_bbox_rgb_img=rgb_img[t_e:b_e,l_e:r_e]

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
                data_dict[self.questions[q_title]["node_code"]]=answer
                data_dict["50000001"]=confidence
            elif q_title=="age":
                # BLIP 年齢？
                answer,confidence=self.get_vqa(blip_processor=self.blip_processor,blip_model=self.blip_model,device=self.device,image=bbox_rgb_img,question=q_info["query"],confidence=True)
                data_dict[self.questions[q_title]["node_code"]]=answer
                data_dict["50000011"]=confidence
            elif q_title=="ivPole":
                # BLIP 点滴？
                answer,confidence=self.get_vqa(blip_processor=self.blip_processor,blip_model=self.blip_model,device=self.device,image=extended_bbox_rgb_img,question=q_info["query"],confidence=True)
                if (answer=="yes") or (answer=="Yes"):
                    answer=1
                else:
                    answer=0
                data_dict["50001000"]=answer
                data_dict["50001001"]=answer
                data_dict["50001002"]=confidence
                data_dict["50001003"]=confidence
                
            elif q_title=="wheelchair":
                # BLIP 車椅子？
                answer,confidence=self.get_vqa(blip_processor=self.blip_processor,blip_model=self.blip_model,device=self.device,image=extended_bbox_rgb_img,question=q_info["query"],confidence=True)
                if (answer=="yes") or (answer=="Yes"):
                    answer=1
                else:
                    answer=0
                data_dict["50001010"]=answer
                data_dict["50001011"]=answer
                data_dict["50001012"]=confidence
                data_dict["50001013"]=confidence
        return data_dict

if __name__=="__main__":
    trial_name="20250115PullWheelchairObaachan"
    strage="NASK"
    cls=PreprocessMaster(trial_name=trial_name,strage=strage)
    cls.main()