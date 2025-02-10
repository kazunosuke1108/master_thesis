import os
import sys
import copy
import dill
from glob import glob
import json
import cv2

from icecream import ic
from pprint import pprint

import numpy as np
import pandas as pd

sys.path.append(".")
sys.path.append("..")
sys.path.append(os.path.expanduser("~")+"/kazu_ws/master_thesis/master_thesis_modules")
from scripts.management.manager import Manager


json_path="/media/hayashide/MobileSensing/20250207Dev/json/dict_after_reid.json"

json_data=Manager().load_json(json_path)
patients=sorted(list(set([k.split("_")[0] for k in json_data.keys()])))

elp_img_path=sorted(glob("/media/hayashide/MobileSensing/20250207Dev/jpg/elp/left/*.jpg"))[-1]
elp_img=cv2.imread(elp_img_path)

data_dicts={}
for patient in patients:
    data_dicts[patient]={}
    # bbox img
    t,b,l,r=json_data[patient+"_bboxLowerY"],json_data[patient+"_bboxHigherY"],json_data[patient+"_bboxLowerX"],json_data[patient+"_bboxHigherX"]
    t,b,l,r=int(t),int(b),int(l),int(r)
    bbox_img=elp_img[t:b,l:r]

    # luminance (7)

    # position (6)
    data_dicts[patient]["60010000"]=json_data[patient+"_x"]
    data_dicts[patient]["60010001"]=json_data[patient+"_y"]

    # feature (5)
    ## 属性
    ## 動作
    ## 物体
    ## 見守り状況
    
# 5000/0000 ~ 7000/0000 の特徴量が入ったdictを作っていく．