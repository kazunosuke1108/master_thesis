import os
import sys
from glob import glob
from icecream import ic
import copy
import time
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

class PreprocessObject(Manager):
    def __init__(self):
        super().__init__()

        # 施設構造の定義
        # ## 11月データ用
        # self.xrange=[6,15]
        # self.yrange=[-11,-4]
        # ## 8月データ用        
        # # self.xrange=[-4,6]
        # # self.yrange=[5,10]
        # # # staff station

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
        x=np.array([data_dict["60010000"],data_dict["60010001"]])
        # 点滴
        potential_ivPole=0
        for mu in structure_dict["ivPole"]:
            potential_ivPole+=self.gauss_func(x=x,mu=mu,r=3)
        potential_ivPole=np.clip(potential_ivPole,0,1)
        data_dict["50001000"]=potential_ivPole
        data_dict["50001001"]=potential_ivPole
        data_dict["50001002"]=1
        data_dict["50001003"]=1
        # 車椅子
        potential_wheelchair=0
        for mu in structure_dict["wheelchair"]:
            potential_wheelchair+=self.gauss_func(x=x,mu=mu,r=3)
        potential_wheelchair=np.clip(potential_wheelchair,0,1)
        data_dict["50001010"]=potential_wheelchair
        data_dict["50001011"]=potential_wheelchair
        data_dict["50001012"]=1
        data_dict["50001013"]=1
        # 手すり
        closest_wall=np.argmin([
            abs(structure_dict["handrail"]["xrange"][0]-data_dict["60010000"]),
            abs(structure_dict["handrail"]["yrange"][0]-data_dict["60010001"]),
            abs(structure_dict["handrail"]["xrange"][1]-data_dict["60010000"]),
            abs(structure_dict["handrail"]["yrange"][1]-data_dict["60010001"]),
            ])
        if closest_wall==0:                
            data_dict["50001020"]=structure_dict["handrail"]["xrange"][0]
            data_dict["50001021"]=data_dict["60010001"]
        elif closest_wall==1:
            data_dict["50001020"]=data_dict["60010000"]
            data_dict["50001021"]=structure_dict["handrail"]["yrange"][0]
        elif closest_wall==2:
            data_dict["50001020"]=structure_dict["handrail"]["xrange"][1]
            data_dict["50001021"]=data_dict["60010001"]
        elif closest_wall==3:
            data_dict["50001020"]=data_dict["60010000"]
            data_dict["50001021"]=structure_dict["handrail"]["yrange"][1]
        return data_dict
    
    def handrail_snapshot(self,data_dict,dayroom_structure_dict={"xrange":[6,15],"yrange":[-11,-4]}):
        closest_wall=np.argmin([
            abs(dayroom_structure_dict["xrange"][0]-data_dict["60010000"]),
            abs(dayroom_structure_dict["yrange"][0]-data_dict["60010001"]),
            abs(dayroom_structure_dict["xrange"][1]-data_dict["60010000"]),
            abs(dayroom_structure_dict["yrange"][1]-data_dict["60010001"]),
            ])
        if closest_wall==0:                
            data_dict["50001020"]=dayroom_structure_dict["xrange"][0]
            data_dict["50001021"]=data_dict["60010001"]
        elif closest_wall==1:
            data_dict["50001020"]=data_dict["60010000"]
            data_dict["50001021"]=dayroom_structure_dict["yrange"][0]
        elif closest_wall==2:
            data_dict["50001020"]=dayroom_structure_dict["xrange"][1]
            data_dict["50001021"]=data_dict["60010001"]
        elif closest_wall==3:
            data_dict["50001020"]=data_dict["60010000"]
            data_dict["50001021"]=dayroom_structure_dict["yrange"][1]
        return data_dict