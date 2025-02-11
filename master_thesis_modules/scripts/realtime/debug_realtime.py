import os
import sys
import copy
from glob import glob
import json
import cv2

from icecream import ic
from pprint import pprint
import numpy as np
import pandas as pd

sys.path.append(".")
sys.path.append("..")
if "catkin_ws" in os.getcwd():
    sys.path.append("/catkin_ws/src/master_thesis_modules")
else:
    sys.path.append(os.path.expanduser("~")+"/kazu_ws/master_thesis/master_thesis_modules")
from scripts.management.manager import Manager
from scripts.master_v3 import Master
from scripts.preprocess.preprocess_blip_snapshot import PreprocessBlip
from scripts.preprocess.preprocess_yolo_snapshot import PreprocessYolo
from scripts.preprocess.preprocess_handrail_snapshot import PreprocessHandrail

# preprocessorのインスタンス
cls_blip=PreprocessBlip()
cls_yolo=PreprocessYolo()
cls_handrail=PreprocessHandrail()

# 施設構造の事前情報
dayroom_structure_dict={"xrange":[6,15],"yrange":[-11,-4]} # 11月
# dayroom_structure_dict={"xrange":[-4,6],"yrange":[5,10]} # 08月
ss_structure_dict={"pos":[6,-7.5],"direction":[0,0.1]} # 11月
# ss_structure_dict={"pos":[-4,7.5],"direction":[0.1,0]} # 08月

# 計測結果の取得
json_latest_path="/media/hayashide/MobileSensing/20250207Dev/json/dict_after_reid.json"
json_previous_path="/media/hayashide/MobileSensing/20250207Dev/json/dict_after_reid_old.json"
json_latest_data=Manager().load_json(json_latest_path)
json_previous_data=Manager().load_json(json_previous_path)

# 基本情報の取得
patients=sorted(list(set([k.split("_")[0] for k in json_latest_data.keys()])))
elp_img_path=sorted(glob("/media/hayashide/MobileSensing/20250207Dev/jpg/elp/left/*.jpg"))[-1]
elp_img=cv2.imread(elp_img_path)

# 1人ずつ評価
data_dicts={}
print(f"評価開始前のpatients: {patients}")
for patient in patients:
    # bbox img
    t,b,l,r=json_latest_data[patient+"_bboxLowerY"],json_latest_data[patient+"_bboxHigherY"],json_latest_data[patient+"_bboxLowerX"],json_latest_data[patient+"_bboxHigherX"]
    try:
        t,b,l,r=int(t),int(b),int(l),int(r)
    except ValueError: # NaNが入っていた場合
        # patients.remove(patient)
        print(f"bbox情報が不正のため削除 {patient}")
        continue
    bbox_img=elp_img[t:b,l:r]

    data_dicts[patient]={}
    # luminance (7)

    # position (6)
    data_dicts[patient]["60010000"]=json_latest_data[patient+"_x"]
    data_dicts[patient]["60010001"]=json_latest_data[patient+"_y"]

    # feature (5)
    ## BLIP系 属性・物体(手すり以外)
    data_dicts[patient]=cls_blip.blip_snapshot(data_dicts[patient],elp_img,t,b,l,r,)
    ## 動作
    data_dicts[patient],success_flag=cls_yolo.yolo_snapshot(data_dicts[patient],elp_img,t,b,l,r,)
    if not success_flag:
        print(f"姿勢推定が不正のため削除 {patient}")
        # patients.remove(patient)
        del data_dicts[patient]
        continue
    ## 物体(手すり)
    data_dicts[patient]=cls_handrail.handrail_snapshot(data_dicts[patient],dayroom_structure_dict)

patients=list(data_dicts.keys())

## 見守り状況
# ========= debug用のデータ =========
# data_dicts["00000"]["50000000"]="no"
# # data_dicts["00001"]["50000000"]="yes"
# data_dicts["00003"]["50000000"]="yes"
# data_dicts["00006"]["50000000"]="yes"
# data_dicts["00007"]["50000000"]="yes"
# # data_dicts["00009"]["50000000"]="yes"

def get_relative_distance(data_dicts,p,s,):
    d=np.linalg.norm(np.array([data_dicts[p]["60010000"],data_dicts[p]["60010001"]])-np.array([data_dicts[s]["60010000"],data_dicts[s]["60010001"]]))
    return d

# スタッフがいるかどうかを判定
print(patients)
pprint(data_dicts)
staff=[patient for patient in patients if data_dicts[patient]["50000000"]=="no"]
if len(staff)>0:
    print(staff)
    # いる
    for patient in patients:
        distances=[get_relative_distance(data_dicts,patient,s) for s in staff]
        closest_staff=staff[np.array(distances).argmin()]
        data_dicts[patient]["50001100"]=data_dicts[closest_staff]["60010000"]
        data_dicts[patient]["50001101"]=data_dicts[closest_staff]["60010001"]
        data_dicts[patient]["50001110"]=data_dicts[closest_staff]["60010000"]-json_previous_data[closest_staff+"_x"]
        data_dicts[patient]["50001111"]=data_dicts[closest_staff]["60010001"]-json_previous_data[closest_staff+"_y"]

else:
    # いない
    for patient in patients:
        data_dicts[patient]["50001100"]=ss_structure_dict["pos"][0]
        data_dicts[patient]["50001101"]=ss_structure_dict["pos"][1]
        data_dicts[patient]["50001110"]=ss_structure_dict["direction"][0]
        data_dicts[patient]["50001111"]=ss_structure_dict["direction"][1]
    pass

# data_dictsに対してリスクを計算
cls_master=Master(data_dicts)
data_dicts=cls_master.evaluate()
print(data_dicts)

# データの吐き出し
## json_latest_data (特徴量) と評価結果をまとめてjson
export_data={
    "sources":json_latest_data,
    "results":data_dicts,
}
Manager().write_json(export_data,json_path=os.path.split(json_latest_path)[0]+"/data_dicts_eval.json")
print(export_data)

"""
            # 背景差分値の取得
            bbox_diff_img=diff_img[t:b,l:r]
            self.feature_dict[id_name].loc[i,"70000000"]=bbox_diff_img.mean()/255
            if id_name=="ID_00000":
                # cv2.imshow("diff",bbox_diff_img)
                cv2.imshow("rgb",extended_bbox_rgb_img)
                cv2.waitKey(1)
                print(bbox_diff_img.max()/255,bbox_diff_img.mean()/255)

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
"""