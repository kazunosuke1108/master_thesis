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

class PreprocessHandrail(Manager):
    def __init__(self):
        super().__init__()

        # 施設構造の定義
        ## 11月データ用
        self.xrange=[6,15]
        self.yrange=[-11,-4]
        # staff station
        self.staff_station=[6,-7.5]
        self.staff_direction=[0,0.1]
        ## 8月データ用        
        # self.xrange=[-4,6]
        # self.yrange=[5,10]
        # # staff station
        # self.staff_station=[-4,7.5]
        # self.staff_direction=[0.1,0]        

    def handrail_snapshot(self,data_dict):
        closest_wall=np.argmin([
            abs(self.xrange[0]-data_dict["60010000"]),
            abs(self.yrange[0]-data_dict["60010001"]),
            abs(self.xrange[1]-data_dict["60010000"]),
            abs(self.yrange[1]-data_dict["60010001"]),
            ])
        if closest_wall==0:                
            data_dict["50001020"]=self.xrange[0]
            data_dict["50001021"]=data_dict["60010001"]
        elif closest_wall==1:
            data_dict["50001020"]=data_dict["60010000"]
            data_dict["50001021"]=self.yrange[0]
        elif closest_wall==2:
            data_dict["50001020"]=self.xrange[1]
            data_dict["50001021"]=data_dict["60010001"]
        elif closest_wall==3:
            data_dict["50001020"]=data_dict["60010000"]
            data_dict["50001021"]=self.yrange[1]
        return data_dict