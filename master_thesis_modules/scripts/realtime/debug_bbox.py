import cv2
import numpy as np

import matplotlib.pyplot as plt
colors = plt.get_cmap("tab10").colors
colors = [(int(b*255), int(g*255), int(r*255)) for r, g, b in colors]

def draw_bbox(elp_img,json_latest_data,patients):
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
        cv2.rectangle(elp_img,bbox_info[0],bbox_info[1],colors[int(patient)], thickness=thickness)
    return elp_img

elp_img_paths=[]