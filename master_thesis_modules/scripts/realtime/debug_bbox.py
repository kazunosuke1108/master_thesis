import cv2
import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt
import os
import sys

sys.path.append(".")
sys.path.append("..")
if "catkin_ws" in os.getcwd():
    sys.path.append("/catkin_ws/src/master_thesis_modules")
else:
    sys.path.append(os.path.expanduser("~")+"/kazu_ws/master_thesis/master_thesis_modules")
from scripts.management.manager import Manager

# elp_img_path="/home/hayashide/ytlab_ros_ws/ytlab_handheld_sensoring_system/ytlab_handheld_sensoring_system_modules/database/20250225elpdsr/jpg/elp/right/r_1740471053.439751.jpg"
# csv_path="/home/hayashide/ytlab_ros_ws/ytlab_handheld_sensoring_system/ytlab_handheld_sensoring_system_modules/database/20250225elpdsr/csv/df_before_reid.csv"
# elp_img_path="/home/hayashide/ytlab_ros_ws/ytlab_handheld_sensoring_system/ytlab_handheld_sensoring_system_modules/database/20250225Ball/jpg/elp/right/r_1740479635.3428032.jpg"
# csv_path="/home/hayashide/ytlab_ros_ws/ytlab_handheld_sensoring_system/ytlab_handheld_sensoring_system_modules/database/20250225Ball/csv/df_before_reid.csv"

# elp_img_path="/home/hayashide/ytlab_ros_ws/ytlab_handheld_sensoring_system/ytlab_handheld_sensoring_system_modules/database/20250225A/jpg/elp/right/r_1740471053.103919.jpg"
# csv_path="/home/hayashide/ytlab_ros_ws/ytlab_handheld_sensoring_system/ytlab_handheld_sensoring_system_modules/database/20250225A/csv/df_before_reid.csv"

elp_img_path="/home/hayashide/ytlab_ros_ws/ytlab_handheld_sensoring_system/ytlab_handheld_sensoring_system_modules/database/20250226B28/jpg/elp/right/r_1740479482.8749435.jpg"
csv_path="/home/hayashide/ytlab_ros_ws/ytlab_handheld_sensoring_system/ytlab_handheld_sensoring_system_modules/database/20250226B28/csv/df_after_reid.csv"


elp_img=cv2.imread(elp_img_path)
# elp_img=cv2.cvtColor(elp_img, cv2.COLOR_BGR2RGB)

data=pd.read_csv(csv_path)
patients=sorted(list(set([k.split("_")[0] for k in data.keys()])))
print(patients)
for idx in [417]:
    for patient in patients:
        try:
            bbox_info=[
                (int(data.loc[idx,f"{patient}_bboxHigherX"]),int(data.loc[idx,f"{patient}_bboxHigherY"])),
                (int(data.loc[idx,f"{patient}_bboxLowerX"]),int(data.loc[idx,f"{patient}_bboxLowerY"])),
            ]
            # bbox_info=[
            #     (int(data.loc[1,f"{patient}_bboxHigherX"]),int(data.loc[1,f"{patient}_bboxHigherY"])),
            #     (int(data.loc[1,f"{patient}_bboxLowerX"]),int(data.loc[1,f"{patient}_bboxLowerY"])),
            # ]
        except ValueError:
            print("ERROR")
            continue
        thickness=4
        cv2.rectangle(elp_img,bbox_info[0],bbox_info[1],(255,0,0), thickness=thickness)
        print(f"draw {patient}")
    cv2.imwrite(f"{os.path.basename(csv_path[:-4])}.jpg",elp_img)
    # cv2.imshow("test",elp_img)
    # cv2.waitKey(1)
    # time.sleep(0.01)

# colors = plt.get_cmap("tab10").colors
# colors = [(int(b*255), int(g*255), int(r*255)) for r, g, b in colors]

# def draw_bbox(elp_img,json_latest_data,patients):
#     for patient in patients:
#         if np.isnan(json_latest_data[f"{patient}_bboxHigherX"]):
#             continue
#         # patient_rank
#         bbox_info=[
#             (int(json_latest_data[f"{patient}_bboxHigherX"]),int(json_latest_data[f"{patient}_bboxHigherY"])),
#             (int(json_latest_data[f"{patient}_bboxLowerX"]),int(json_latest_data[f"{patient}_bboxLowerY"])),
#         ]
#         # if not np.isnan(self.rank_data.loc[i,patient+"_rank"]):
#         #     thickness=len(self.patients)-int(self.rank_data.loc[i,patient+"_rank"])
#         # else:
#         thickness=4
#         cv2.rectangle(elp_img,bbox_info[0],bbox_info[1],colors[int(patient)], thickness=thickness)
#     return elp_img

# elp_img_paths=[]