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

class PreprocessHandrail(Manager):
    def __init__(self):
        super().__init__()

        # 施設構造の定義
        ## 11月データ用
        self.xrange=[6,15]
        self.yrange=[-11,-4]
        ## 8月データ用        
        # self.xrange=[-4,6]
        # self.yrange=[5,10]
        # # staff station

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