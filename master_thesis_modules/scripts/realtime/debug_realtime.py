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

import watchdog

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
from scripts.network.graph_manager_v3 import GraphManager
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
pprint(export_data)

# ランク評価

def guess_static_factor(data_dicts,focus_patient,additional_data_dicts):
    additional_data_dicts["static_factor"]={}
    focus_keys=[]
    # 4000番台だけを採用
    for k in data_dicts[list(data_dicts.keys())[0]].keys():
        if k=="timestamp":
            continue
        elif (int(k[0])>=5) or (int(k[0])<=3):
            continue
        else:
            focus_keys.append(k)
    focus_keys_static=[]
    for k in focus_keys:
        if int(k[-2])==0: # static node
            focus_keys_static.append(k)

    additional_data_dicts["static_factor"]["significance"]={}
    # 顕著なノードの検出
    patients=list(data_dicts.keys())
    
    for patient in patients:
        # if patient==focus_patient:
        #     continue
        additional_data_dicts["static_factor"]["significance"][patient]={}
        for k in focus_keys_static:
            try:
                additional_data_dicts["static_factor"]["significance"][patient][k]=data_dicts[focus_patient][k]-data_dicts[patient][k]
            except TypeError:
                additional_data_dicts["static_factor"]["significance"][patient][k]=data_dicts[focus_patient][k][1]-data_dicts[patient][k][1]
    
    additional_data_dicts["static_factor"]["significance"]["max"]={}
    for k in focus_keys_static:
        additional_data_dicts["static_factor"]["significance"]["max"][k]=np.array([additional_data_dicts["static_factor"]["significance"][p][k] for p in patients]).max()
    most_significant_node=focus_keys_static[np.array([additional_data_dicts["static_factor"]["significance"]["max"][k] for k in focus_keys_static]).argmax()]
    additional_data_dicts["static_factor"]["most_significant_node"]=most_significant_node
    return additional_data_dicts

def guess_dynamic_factor(data_dicts,focus_patient,additional_data_dicts):
    print("!!!!!!!!!!!! dynamic factorの算出方法は仮 !!!!!!!!!!!!")
    additional_data_dicts["dynamic_factor"]={}
    focus_keys=[]
    # 4000番台だけを採用
    for k in data_dicts[list(data_dicts.keys())[0]].keys():
        if k=="timestamp":
            continue
        elif (int(k[0])>=5) or (int(k[0])<=3):
            continue
        else:
            focus_keys.append(k)
    focus_keys_dynamic=[]
    for k in focus_keys:
        if int(k[-2])==1: # dynamic node 【ここが仮】
            focus_keys_dynamic.append(k)

    additional_data_dicts["dynamic_factor"]["significance"]={}
    # 顕著なノードの検出
    patients=list(data_dicts.keys())
    
    for patient in patients:
        # if patient==focus_patient:
        #     continue
        additional_data_dicts["dynamic_factor"]["significance"][patient]={}
        for k in focus_keys_dynamic:
            try:
                additional_data_dicts["dynamic_factor"]["significance"][patient][k]=data_dicts[focus_patient][k]-data_dicts[patient][k]
            except TypeError:
                additional_data_dicts["dynamic_factor"]["significance"][patient][k]=data_dicts[focus_patient][k][1]-data_dicts[patient][k][1]
    
    additional_data_dicts["dynamic_factor"]["significance"]["max"]={}
    for k in focus_keys_dynamic:
        additional_data_dicts["dynamic_factor"]["significance"]["max"][k]=np.array([additional_data_dicts["dynamic_factor"]["significance"][p][k] for p in patients]).max()
    most_significant_node=focus_keys_dynamic[np.array([additional_data_dicts["dynamic_factor"]["significance"]["max"][k] for k in focus_keys_dynamic]).argmax()]
    additional_data_dicts["dynamic_factor"]["most_significant_node"]=most_significant_node
    return additional_data_dicts

    # # average_df=pd.DataFrame(index=focus_keys)
    # # 4000番台の各項目について，患者間比較用の代表値を算出
    # for patient in data_dicts.keys():
    #     for node_code in focus_keys:
    #         if node_code in ["40000000","40000001"]:
    #             average_df.loc[node_code,patient]=np.mean([eval(v)[1] if not type(v)==float else np.nan for v in data_dicts[patient][node_code].values])

    #         else:
    #             average_df.loc[node_code,patient]=data_dicts[patient][node_code].mean()
    # average_df["risky"]=average_df.idxmax(axis=1)
    # average_df["significance"]=np.nan
    # average_df["node_type"]=[self.default_graph["node_dict"][int(idx)]["node_type"] for idx in list(average_df.index)]
    # for i,row in average_df.iterrows():
    #     patients=list(data_dicts.keys())
    #     total=row[patients].sum()
    #     others=total-row[row["risky"]]
    #     significance=abs(row[row["risky"]]-others/(len(patients)-1))
    #     average_df.loc[i,"significance"]=significance

    # factor_df=average_df[(average_df["risky"]==most_risky_patient)].sort_values("significance")
    # static_factor_df=factor_df[factor_df["node_type"]=="static"]
    # static_factor_nodes=static_factor_df.index[static_factor_df["significance"]==static_factor_df["significance"].max()].tolist()
    # if len(static_factor_nodes)>0:
    #     static_factor_node=static_factor_nodes[0]
    # else:
    #     static_factor_node=""
    # print(average_df)
    # print(factor_df)
    # return static_factor_node,factor_df

default_graph=GraphManager().get_default_graph()

def get_alert_sentence(most_risky_patient,static_factor_node,dynamic_factor_node):
    if static_factor_node=="":
        text_dynamic=default_graph["node_dict"][dynamic_factor_node]["description_ja"]
        alert_text=f"{most_risky_patient}さんが，{text_dynamic}ので，危険です．"
    else:
        text_static=default_graph["node_dict"][static_factor_node]["description_ja"]
        text_dynamic=default_graph["node_dict"][dynamic_factor_node]["description_ja"]
        alert_text=f"{most_risky_patient}さんが，元々{text_static}のに，{text_dynamic}ので，危険です．"
    return alert_text

def evaluate_rank(data_dicts,additional_data_dicts):
    additional_data_dicts["rank"]={}
    patients=list(data_dicts.keys())
    total_risks=[]
    for patient in patients:
        total_risks.append(data_dicts[patient]["10000000"])
    patients_rank=(-np.array(total_risks)).argsort()

    most_risky_patient=patients[np.array(total_risks).argmax()]

    additional_data_dicts=guess_static_factor(data_dicts,most_risky_patient,additional_data_dicts)
    additional_data_dicts=guess_dynamic_factor(data_dicts,most_risky_patient,additional_data_dicts)
    text=get_alert_sentence(most_risky_patient,
                            static_factor_node=additional_data_dicts["static_factor"]["most_significant_node"],
                            dynamic_factor_node=additional_data_dicts["dynamic_factor"]["most_significant_node"],
                            )
    print(text)
    additional_data_dicts["alert"]=text

    for patient,rank in zip(patients,patients_rank):
        additional_data_dicts["rank"][patient]={}
        additional_data_dicts["rank"][patient]["10000000"]=rank
    return additional_data_dicts

additional_data_dicts={}
additional_data_dicts=evaluate_rank(data_dicts,additional_data_dicts)
pprint(additional_data_dicts)
# Manager().write_json()



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