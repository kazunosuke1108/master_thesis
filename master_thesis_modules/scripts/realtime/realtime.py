import os
import sys
import copy
import time
from glob import glob
import json
import cv2
import atexit
from icecream import ic
from pprint import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# pip install watchdog
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

sys.path.append(".")
sys.path.append("..")
if "catkin_ws" in os.getcwd():
    sys.path.append("/catkin_ws/src/master_thesis_modules")
else:
    sys.path.append(os.path.expanduser("~")+"/kazu_ws/master_thesis/master_thesis_modules")
from scripts.management.manager import Manager
from scripts.master_v3 import Master
from scripts.network.graph_manager_v3 import GraphManager
# from scripts.preprocess.preprocess_blip_snapshot import PreprocessBlip
from scripts.preprocess.preprocess_zokusei_snapshot import PreprocessZokusei
from scripts.preprocess.preprocess_yolo_snapshot import PreprocessYolo
from scripts.preprocess.preprocess_objects_snapshot import PreprocessObject


# 定数
WATCHED_FILES = ["dict_after_reid.json","dict_after_reid_old.json"]
dayroom_structure_dict={"xrange":[6,15],"yrange":[-11,-4]} # 11月
# dayroom_structure_dict={"xrange":[-4,6],"yrange":[5,10]} # 08月
structure_dict={
    "ivPole":[
        np.array([-4,5]), # muに相当
        np.array([6,5]),
        ],
    "wheelchair":[
        np.array([0,6]),
        np.array([0,8]),
        ],
    "handrail":{
        "xrange":[6,15],
        "yrange":[-11,-4]
        }
}
ss_structure_dict={"pos":[6,-7.5],"direction":[0,0.1]} # 11月
# ss_structure_dict={"pos":[-4,7.5],"direction":[0.1,0]} # 08月
colors_01 = plt.get_cmap("tab10").colors
colors = [(int(b*255), int(g*255), int(r*255)) for r, g, b in colors_01]

w=5

# switch
draw_map=False
draw_bbox=True

class JSONFileChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.is_directory:
            return
        if os.path.basename(event.src_path) in WATCHED_FILES:
            print(f"File changed: {event.src_path}")
            cls.evaluate_main()

class RealtimeEvaluator(Manager,GraphManager):
    def __init__(self,trial_name,strage):
        super().__init__()
        self.trial_name=trial_name
        self.strage=strage
        self.data_dir_dict=self.get_database_dir(trial_name=self.trial_name,strage=self.strage)
        self.default_graph=self.get_default_graph()
        self.df_eval=pd.DataFrame()
        self.df_post=pd.DataFrame()
        self.notify_history=pd.DataFrame(columns=["notificationId","timestamp","relativeTimestamp","patient","sentence","type","10000000"])

        self.logger=self.prepare_log(self.data_dir_dict["mobilesensing_dir_path"]+"/log")
        self.map_fig,self.map_ax=self.plot_map_matplotlib()

        self.notify_interval_dict={"notice":5,"help":10,"notice2help":5}
        self.notify_threshold_by_dinamic_factor={
            "40000000":0.3,
            "40000001":0.4,
            "40000010":0.2,
            "40000011":0.33,
            "40000012":0.4,
            "40000013":0.6,
            "40000014":0.6,
            "40000015":0.5,
            "40000016":0.5,
            "40000100":0.6,
            "40000101":0.6,
            "40000102":0.6,
            "40000110":0.5,
            "40000111":0.5,
        }
        self.notification_id=0
        self.previous_risky_patient=""

        self.luminance_threshold=0.5
        self.logger.info(f"RealtimeEvaluator has woken up")

    def load_data(self):
        all_files_found=False
        self.logger.info(f"jsonのload開始")
        while not all_files_found:
            try:
                # json
                json_latest_path=self.data_dir_dict["mobilesensing_dir_path"]+"/json/dict_after_reid.json"
                json_previous_path=self.data_dir_dict["mobilesensing_dir_path"]+"/json/dict_after_reid_old.json"
                self.json_latest_data=Manager().load_json(json_latest_path)
                self.json_previous_data=Manager().load_json(json_previous_path)
                # 患者情報
                self.patients=sorted(list(set([k.split("_")[0] for k in self.json_latest_data.keys()])))
                # 時刻
                self.timestamp=self.json_latest_data[self.patients[0]+"_timestamp"]
                self.logger.info(f"解析時刻: {self.timestamp}")
                # ELP
                elp_img_paths=sorted(glob(f"/catkin_ws/src/database/{self.trial_name}/jpg/elp/left/*.jpg"))
                self.elp_img_path=elp_img_paths[abs(np.array([float(os.path.basename(p).split("_")[1][:-len(".jpg")])-self.timestamp for p in elp_img_paths])).argmin()]
                self.logger.info(f"解析ELP: {self.elp_img_path}")
                self.elp_img=cv2.imread(self.elp_img_path)
                # self.elp_img=cv2.cvtColor(self.elp_img, cv2.COLOR_BGR2RGB)
                # 背景差分画像
                diff_img_paths=sorted(glob(f"/catkin_ws/src/database/{self.trial_name}/jpg/diff/left/*.jpg"))
                self.diff_img_path=diff_img_paths[abs(np.array([float(os.path.basename(p).split("_")[1][:-len(".jpg")])-self.timestamp for p in diff_img_paths])).argmin()]
                self.logger.info(f"解析diff: {self.diff_img_path}")
                self.diff_img=cv2.imread(self.diff_img_path)
                all_files_found=True
            except FileNotFoundError:
                print(self.get_timestamp(),"json not found")
                time.sleep(0.1)
                continue
            except IndexError:
                print(self.get_timestamp(),"ELP not found")
                time.sleep(0.1)
                continue
        return json_latest_path,self.json_latest_data

    def get_features(self):
        def extract_luminance(diff_img,patient):
            t,b,l,r=self.json_latest_data[patient+"_bboxLowerY"],self.json_latest_data[patient+"_bboxHigherY"],self.json_latest_data[patient+"_bboxLowerX"],self.json_latest_data[patient+"_bboxHigherX"]
            try:
                t,b,l,r=int(t),int(b),int(l),int(r)
                bbox_diff_img=diff_img[t:b,l:r]
                luminance=bbox_diff_img.mean()/255
            except ValueError: # NaNが入っていた場合
                # self.patients.remove(patient)
                luminance=0
            return luminance
        self.logger.info(f"特徴量の算出開始 Time: {np.round(time.time()-self.start,4)}")
        data_dicts={}
        def fps_judger(active=False):
            fps_control_dicts={}

            lateset_patients=sorted(list(set([k.split("_")[0] for k in self.json_latest_data.keys()])))
            previous_patients=sorted(list(set([k.split("_")[0] for k in self.df_eval.keys()])))
            for patient in lateset_patients:
                fps_control_dicts[patient]={k:True for k in self.default_graph["node_dict"].keys() if k[0]=="5"}
                # 属性 (初回のみ)
                if active:
                    if patient in previous_patients: # 以前観測されていた患者なら，推論をOFFにする
                        for k in fps_control_dicts[patient].keys():
                            if (("500000" in k)):# or ("5000100" in k) or ("5000101" in k)): 
                                fps_control_dicts[patient][k]=False
                    # 物体 (初回または動作を検知したときのみ)
                    if patient in previous_patients: # 以前観測されていた患者なら，推論をOFFにする
                        # 動作量が一定以下ならば，推論をOFFにする
                        luminance=extract_luminance(diff_img=self.diff_img,patient=patient)
                        if luminance<self.luminance_threshold:
                            for k in fps_control_dicts[patient].keys():
                                if (("5000100" in k) or ("5000101" in k)): 
                                    fps_control_dicts[patient][k]=False

            # 動作（FPSが上限を上回らない見込みのため割愛）
            # 位置（FPSが上限を上回らない見込みのため割愛）

            return fps_control_dicts
        # FPS制御
        self.fps_control_dicts=fps_judger()
        self.logger.info(f"fps_control_dicts:{self.fps_control_dicts}")
        self.logger.info(f"評価開始前のpatients: {self.patients}")
        # 1人ずつ評価
        for patient in self.patients:
            # bbox img
            t,b,l,r=self.json_latest_data[patient+"_bboxLowerY"],self.json_latest_data[patient+"_bboxHigherY"],self.json_latest_data[patient+"_bboxLowerX"],self.json_latest_data[patient+"_bboxHigherX"]
            try:
                t,b,l,r=int(t),int(b),int(l),int(r)
            except ValueError: # NaNが入っていた場合
                # self.patients.remove(patient)
                self.logger.info(f"bbox情報が不正のため削除 {patient}")
                continue
            # bbox_img=self.elp_img[t:b,l:r]

            data_dicts[patient]={}
            # luminance (7)

            # position (6)
            data_dicts[patient]["60010000"]=self.json_latest_data[patient+"_x"]
            data_dicts[patient]["60010001"]=self.json_latest_data[patient+"_y"]

            # feature (5)
            ## BLIP系 属性・物体(手すり以外)
            self.logger.info(f"BLIP開始 Time: {np.round(time.time()-self.start,4)}")
            data_dicts[patient]=cls_zokusei.zokusei_snapshot(data_dict=data_dicts[patient],rgb_img=self.elp_img,t=t,b=b,l=l,r=r,)#(data_dicts[patient],self.elp_img,t,b,l,r,self.fps_control_dicts[patient])
            ## 動作
            self.logger.info(f"YOLO開始 Time: {np.round(time.time()-self.start,4)}")
            data_dicts[patient],success_flag=cls_yolo.yolo_snapshot(data_dicts[patient],self.elp_img,t,b,l,r,)
            if not success_flag:
                self.logger.info(f"姿勢推定が不正のため削除 {patient}")
                # self.patients.remove(patient)
                del data_dicts[patient]
                continue
            self.logger.info(f"手すり開始 Time: {np.round(time.time()-self.start,4)}")
            ## 物体(手すり)
            data_dicts[patient]=cls_object.object_snapshot(data_dicts[patient],structure_dict)
            
            # FPS制御で省略されたデータを補う
            for k in self.fps_control_dicts[patient].keys():
                if self.fps_control_dicts[patient][k]==False:
                    data_dicts[patient][k]=self.df_eval.loc[self.df_eval.index[-1], f"{patient}_{k}"]



        self.patients=list(data_dicts.keys())

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

        self.logger.info(f"見守り関連開始 Time: {np.round(time.time()-self.start,4)}")
        # スタッフがいるかどうかを判定
        staff=[patient for patient in self.patients if data_dicts[patient]["50000000"]=="no"]
        if len(staff)>0:
            # print(staff)
            # いる
            for patient in self.patients:
                distances=[get_relative_distance(data_dicts,patient,s) for s in staff]
                closest_staff=staff[np.array(distances).argmin()]
                data_dicts[patient]["50001100"]=data_dicts[closest_staff]["60010000"]
                data_dicts[patient]["50001101"]=data_dicts[closest_staff]["60010001"]
                data_dicts[patient]["50001110"]=data_dicts[closest_staff]["60010000"]-self.json_previous_data[closest_staff+"_x"]
                data_dicts[patient]["50001111"]=data_dicts[closest_staff]["60010001"]-self.json_previous_data[closest_staff+"_y"]

        else:
            # いない
            for patient in self.patients:
                data_dicts[patient]["50001100"]=ss_structure_dict["pos"][0]
                data_dicts[patient]["50001101"]=ss_structure_dict["pos"][1]
                data_dicts[patient]["50001110"]=ss_structure_dict["direction"][0]
                data_dicts[patient]["50001111"]=ss_structure_dict["direction"][1]
        
        return data_dicts
    
    def get_rank_and_text(self,data_dicts,additional_data_dicts):
        def guess_static_factor(data_dicts,focus_patient,additional_data_dicts):
            try:
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
                        if k in ["40000000","40000001"]: # TFNが格納されている場合
                            val_focus_patient=np.array([eval(v)[1] for v in self.df_eval[focus_patient+f"_{k}"].dropna().tail(5)]).mean()
                            val_patient=np.array([eval(v)[1] for v in self.df_eval[patient+f"_{k}"].dropna().tail(5)]).mean()
                            additional_data_dicts["static_factor"]["significance"][patient][k]=val_focus_patient-val_patient
                        else:
                            additional_data_dicts["static_factor"]["significance"][patient][k]=self.df_eval[focus_patient+f"_{k}"].tail(5).mean()-self.df_eval[patient+f"_{k}"].tail(5).mean()
                        # except TypeError:
                        #     if k in ["4000"]:
                        #         additional_data_dicts["static_factor"]["significance"][patient][k]=np.nan
                        #     else:
                        #         additional_data_dicts["static_factor"]["significance"][patient][k]=self.df_eval[focus_patient+f"_{k}"].tail(5).mean()[1]-self.df_eval[patient+f"_{k}"].tail(5).mean()[1]
                
                additional_data_dicts["static_factor"]["significance"]["max"]={}
                for k in focus_keys_static:
                    additional_data_dicts["static_factor"]["significance"]["max"][k]=np.array([additional_data_dicts["static_factor"]["significance"][p][k] for p in patients]).max()
                most_significant_node=focus_keys_static[np.array([additional_data_dicts["static_factor"]["significance"]["max"][k] for k in focus_keys_static]).argmax()]
                additional_data_dicts["static_factor"]["most_significant_node"]=most_significant_node
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                self.logger.error(f"line {exc_tb.tb_lineno}: {e}")
                
            return additional_data_dicts,most_significant_node

        def guess_dynamic_factor(data_dicts,focus_patient,additional_data_dicts):
            try:

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
                    if int(k[-2])==1: # dynamic node
                        focus_keys_dynamic.append(k)
                
                data=self.df_eval[[focus_patient+f"_{k}" for k in ["10000000"]+focus_keys_dynamic]].tail(20).rolling(w).mean()
                data_corr=data.corr()[focus_patient+"_10000000"]
                # 一番相関が高い4000番台の因子を抜き出す
                data_corr_4000=data_corr[[focus_patient+"_"+k for k in focus_keys_dynamic]]
                most_corr_key=list(data_corr_4000.keys())[data_corr_4000.argmax()]
                most_corr_node=most_corr_key.replace(focus_patient+"_","")
                additional_data_dicts["dynamic_factor"]["most_significant_node"]=most_corr_node
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                self.logger.error(f"line {exc_tb.tb_lineno}: {e}")

            return additional_data_dicts,most_corr_node

            
            pass

        def guess_dynamic_factor_depreciated(data_dicts,focus_patient,additional_data_dicts):
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
        
        def judge_notification_necessity(data_dicts,most_risky_patient,dynamic_factor_node):
            def judge_time_interval(notify_history,data_dicts,alert_type):
                df=notify_history[notify_history["type"]==alert_type].reset_index()
                self.logger.info(f"\n{self.notify_history}\n{notify_history}")
                if len(df)==0:
                    if len(notify_history)==0:
                        self.logger.info("時間間隔制約：充足【初回】")
                        return True
                    if alert_type=="help":
                        df=notify_history[notify_history["type"]=="notice"].reset_index()
                        if self.timestamp-df.loc[len(df)-1,"timestamp"]>self.notify_interval_dict["notice2help"]:
                            return True
                        else:
                            return False
                    else:
                        self.logger.info("時間間隔制約：充足【よくわからん】")
                        return True

                if self.timestamp-df.loc[len(df)-1,"timestamp"]>self.notify_interval_dict[alert_type]:
                    if alert_type=="help":
                        df=notify_history[notify_history["type"]=="notice"].reset_index()
                        if self.timestamp-df.loc[len(df)-1,"timestamp"]>self.notify_interval_dict["notice2help"]:
                            return True
                        else:
                            return False
                    else:
                        self.logger.info(f"時間間隔制約：充足 ({self.timestamp-df.loc[len(df)-1,'timestamp']})")
                        return True
                else:
                    self.logger.info(f"時間間隔制約：非充足 ({self.timestamp-df.loc[len(df)-1,'timestamp']})")
                    return False
            def judge_rank_change(most_risky_patient):
                if len(self.notify_history)>0:
                    rank_change=most_risky_patient!=self.previous_risky_patient
                elif len(self.notify_history)==0:
                    rank_change=True
                    self.previous_risky_patient=most_risky_patient
                return rank_change
            def judge_above_dynamic_thre(most_risky_patient,dynamic_factor_node):
                
                node_val=self.df_eval.loc[len(self.df_eval)-1,most_risky_patient+"_"+dynamic_factor_node]
                if node_val>self.notify_threshold_by_dinamic_factor[dynamic_factor_node]:
                    return True
                else:
                    return False
                
            try:
                # 通知の必要性判断
                need_notify=False
                need_help=False
                ## A 前回通知からの時間経過
                tf_interval_notify=judge_time_interval(self.notify_history,data_dicts,"notice")
                # tf_interval_help=judge_time_interval(self.notify_history,data_dicts,"help")
                ## B 順位入れ替えの発生
                tf_rank_change=judge_rank_change(most_risky_patient)
                ## C dynamic_factor_node毎の通知基準値を超越しているか
                tf_dynamic_node=judge_above_dynamic_thre(most_risky_patient,dynamic_factor_node)

                if tf_interval_notify and tf_rank_change and tf_dynamic_node:
                    self.logger.warning(f"経過時間：{tf_interval_notify}　順位変更：{tf_rank_change}　ノード閾値以上：{tf_dynamic_node}　➡　通知実行")
                    return True
                else:
                    self.logger.warning(f"経過時間：{tf_interval_notify}　順位変更：{tf_rank_change}　ノード閾値以上：{tf_dynamic_node}　➡　通知見送り")
                    return False
                # elif (not tf_interval_notify) and tf_rank_change:
                #     self.logger.warning("経過時間：×　順位変更：有　➡　通知見送り")
                #     return False
                # elif tf_interval_notify and (not tf_rank_change):
                #     self.logger.warning("経過時間：〇　順位変更：×　➡　通知見送り")
                #     return False
                # elif (not tf_interval_notify) and (not tf_rank_change):
                #     self.logger.warning("経過時間：×　順位変更：×　➡　通知見送り")
                #     return False
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                self.logger.error(f"line {exc_tb.tb_lineno}: {e}")
                
                
            
            pass

        def get_alert_sentence(most_risky_patient,static_factor_node,dynamic_factor_node):
            if static_factor_node=="":
                text_dynamic=self.default_graph["node_dict"][dynamic_factor_node]["description_ja"]
                alert_text=f"{most_risky_patient}さんが，{text_dynamic}ので，危険です．"
            else:
                text_static=self.default_graph["node_dict"][static_factor_node]["description_ja"]
                text_dynamic=self.default_graph["node_dict"][dynamic_factor_node]["description_ja"]
                alert_text=f"{most_risky_patient}さんが，元々{text_static}のに，{text_dynamic}ので，危険です．"
            return alert_text

        try:
            # 順位の算出
            additional_data_dicts["rank"]={}
            patients=list(data_dicts.keys())
            total_risks=[]
            for patient in patients:
                # total_risks.append(data_dicts[patient]["10000000"])
                total_risks.append(self.df_eval[patient+"_10000000"].tail(5).mean())
            patients_rank=(-np.array(total_risks)).argsort()

            # 最も高リスクな患者の決定
            most_risky_patient=patients[np.array(total_risks).argmax()]

            # 静的・動的要因の算出
            additional_data_dicts,static_factor_node=guess_static_factor(data_dicts,most_risky_patient,additional_data_dicts)
            # additional_data_dicts=guess_dynamic_factor(data_dicts,most_risky_patient,additional_data_dicts)
            additional_data_dicts,dynamic_factor_node=guess_dynamic_factor(data_dicts,most_risky_patient,additional_data_dicts)
            
            # 通知の必要性を判断
            notice_necessary=judge_notification_necessity(data_dicts=data_dicts,most_risky_patient=most_risky_patient,dynamic_factor_node=dynamic_factor_node)
            text=get_alert_sentence(most_risky_patient,
                                    static_factor_node=additional_data_dicts["static_factor"]["most_significant_node"],
                                    dynamic_factor_node=additional_data_dicts["dynamic_factor"]["most_significant_node"],
                                    )
            if notice_necessary:
                # ["notificationId","timestamp","relativeTimestamp","patient","sentence","type","10000000"]
                self.notify_history.loc[len(self.notify_history),:]=[self.notification_id,self.timestamp,self.timestamp,most_risky_patient,text,"notice",data_dicts[most_risky_patient]["10000000"]]
                self.notification_id+=1
            
            additional_data_dicts["alert"]=text

            for patient,rank in zip(patients,patients_rank):
                additional_data_dicts["rank"][patient]={}
                additional_data_dicts["rank"][patient]["10000000"]=rank
            return additional_data_dicts
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            self.logger.error(f"line {exc_tb.tb_lineno}: {e}")
            return additional_data_dicts

    def draw_bbox(self,elp_img,json_latest_data,patients):
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
        
    def draw_timestamp(self,elp_img,json_latest_data):
        bbox_info=[
            (0,0),
            (250,40),
        ]
        cv2.rectangle(elp_img,bbox_info[0],bbox_info[1],color=(255,255,255),thickness=cv2.FILLED)
        cv2.putText(
            img=elp_img,
            text="Time: "+str(np.round(self.timestamp,2))+" [s]",
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=1,
            org=(0,30),
            color=(0,0,255),
            thickness=2,
            )
        return elp_img
    
    def draw_rank(self,img,data_dicts,additional_data_dicts,patients):
        def get_bbox_info(rank):
            x_width=125
            y_interval=50
            y_width=40
            bbox_info=[
                (0,int(100+rank*y_interval)),
                (x_width,int(100+rank*y_interval+y_width))
                ]
            return bbox_info
        
        # 背景色の白
        cv2.rectangle(img,(0,40),(250,50*(len(patients)+3)),color=(255,255,255),thickness=cv2.FILLED)
        cv2.putText(
                img=img,
                # text=f"No.{int(rank)+1}: "+"ID_"+patient,
                text=f"Priority Order",
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=1,
                org=(0,100-20),
                color=(0,0,0),
                thickness=2,
        )

        for patient in patients:
            # patient_str=self.patient_dict[patient]
            patient_str=patient
            rank=additional_data_dicts["rank"][patient]["10000000"]
            if np.isnan(rank):
                continue
            bbox_info=get_bbox_info(rank)
            # 患者カラーの帯
            cv2.rectangle(img,(bbox_info[0][0]+90,bbox_info[0][1]),bbox_info[1],color=colors[int(patient)],thickness=cv2.FILLED)
            cv2.putText(
                img=img,
                # text=f"No.{int(rank)+1}: "+"ID_"+patient,
                text=f"No.{int(rank)+1}: "+patient_str,
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=1,
                org=(bbox_info[0][0],bbox_info[1][1]-10),
                color=(0,0,0),
                thickness=2,
                )            
        #     # 通知中か判定
        #     notify_for_the_patient_data=self.notification_data[self.notification_data.fillna(99999)["patient"]==patient_str]
        #     for j,_ in notify_for_the_patient_data.iterrows():
        #         if (notify_for_the_patient_data.loc[j,"timestamp"]<self.rank_data.loc[i,"timestamp"]) and (self.rank_data.loc[i,"timestamp"]<=notify_for_the_patient_data.loc[j,"timestamp"]+8):
        #             img[bbox_info[0][1]:bbox_info[0][1]+self.speaker_red_img.shape[0],bbox_info[1][0]:bbox_info[1][0]+self.speaker_red_img.shape[1]]=self.speaker_red_img
        #     # raise NotImplementedError
        # # 応援要請の状況
        # notify_help_data=self.notification_data[self.notification_data["type"]=="help"]
        # bbox_info=get_bbox_info(len(patients))
        # for j,_ in notify_help_data.iterrows():
        #     if (notify_help_data.loc[j,"timestamp"]<self.rank_data.loc[i,"timestamp"]) and (self.rank_data.loc[i,"timestamp"]<=notify_help_data.loc[j,"timestamp"]+7):
        #         cv2.putText(
        #                 img=img,
        #                 text="Help:",
        #                 fontFace=cv2.FONT_HERSHEY_DUPLEX,
        #                 fontScale=1,
        #                 org=(bbox_info[0][0],bbox_info[1][1]),
        #                 color=(255,0,0),
        #                 thickness=2,
        #                 )
        #         img[bbox_info[0][1]:bbox_info[0][1]+self.speaker_blue_img.shape[0],bbox_info[1][0]:bbox_info[1][0]+self.speaker_blue_img.shape[1]]=self.speaker_blue_img
        #         break
        #     else:
        #         cv2.putText(
        #                 img=img,
        #                 text="Help:",
        #                 fontFace=cv2.FONT_HERSHEY_DUPLEX,
        #                 fontScale=1,
        #                 org=(bbox_info[0][0],bbox_info[1][1]),
        #                 color=(200,200,200),
        #                 thickness=2,
        #                 )

        #     # 通知中なら，スピーカーアイコンを描く

        return img
    
    def draw_pos(self,data_dicts,patients):
        map_ax=copy.deepcopy(self.map_ax)
        for patient in patients:
            map_ax.scatter(data_dicts[patient]["60010000"],data_dicts[patient]["60010001"],s=100,marker="o",c=[colors_01[int(patient)]],label=patient)
        plt.legend()
        plt.title(self.timestamp)
        plt.savefig(self.data_dir_dict["mobilesensing_dir_path"]+f"/jpg/map/{os.path.basename(self.elp_img_path)}")
        
        
    def draw_export_img(self,elp_img_path,elp_img,json_latest_data,data_dicts,additional_data_dicts,patients):
        # ELP画像
        # bbox情報を用意（rank連携要検討）
        # 描画
        elp_img=self.draw_bbox(elp_img=elp_img,json_latest_data=json_latest_data,patients=patients)
        elp_img=self.draw_timestamp(elp_img=elp_img,json_latest_data=json_latest_data)
        # rankの追記
        elp_img=self.draw_rank(img=elp_img,data_dicts=data_dicts,additional_data_dicts=additional_data_dicts,patients=patients)
        # 保存
        cv2.imwrite(self.data_dir_dict["mobilesensing_dir_path"]+"/jpg/bbox/"+os.path.basename(elp_img_path),elp_img)
        pass


    
    def evaluate_main(self):
        try:
            self.start=time.time()
            self.logger.info(f"RealtimeEvaluator called. Time: {np.round(time.time()-self.start,4)}")

            # 情報の読込
            self.logger.info(f"Loading info...")
            json_latest_path,json_latest_data=self.load_data()

            # 特徴量の算出
            self.logger.info(f"Calculating feature values...Time: {np.round(time.time()-self.start,4)}")
            data_dicts=self.get_features()

            # 危険性評価
            self.logger.info(f"Evaluating risk...Time: {np.round(time.time()-self.start,4)}")
            cls_master=Master(data_dicts,strage=self.strage)
            data_dicts=cls_master.evaluate()
            export_data={
                "sources":json_latest_data,
                "results":data_dicts,
            }
            Manager().write_json(export_data,json_path=os.path.split(json_latest_path)[0]+"/data_dicts_eval.json")
            data_dicts_flatten = self.flatten_dict(data_dicts)
            if len(list(data_dicts_flatten.keys()))>0:
                data_dicts_flatten["timestamp"]=json_latest_data[f"{list(data_dicts.keys())[0]}_timestamp"]
            self.df_eval=pd.concat([self.df_eval,pd.DataFrame(data_dicts_flatten,index=[0])],axis=0)
            self.df_eval.reset_index(inplace=True,drop=True)

            # 順位付け・通知文生成
            self.logger.info(f"Analyzing results...Time: {np.round(time.time()-self.start,4)}")
            patients=list(data_dicts.keys())
            if len(patients)>0:
                additional_data_dicts={}
                # リスク優先順位付け
                additional_data_dicts=self.get_rank_and_text(data_dicts,additional_data_dicts)
                additional_data_dicts_flatten=self.flatten_dict(additional_data_dicts)
                self.df_post=pd.concat([self.df_post,pd.DataFrame(additional_data_dicts_flatten,index=[0])],axis=0)
                self.df_post.reset_index(inplace=True,drop=True)            
            Manager().write_json(additional_data_dicts,json_path=os.path.split(json_latest_path)[0]+"/additional_data_dicts.json")

            # 可視化
            if draw_bbox:
                self.logger.info(f"Drawing bbox image...Time: {np.round(time.time()-self.start,4)}")
                patients=list(data_dicts.keys())
                self.draw_export_img(self.elp_img_path,self.elp_img,json_latest_data,data_dicts,additional_data_dicts,patients)
            if draw_map:
                self.logger.info(f"Drawing map image...Time: {np.round(time.time()-self.start,4)}")
                self.draw_pos(data_dicts=data_dicts,patients=patients)
            self.logger.info(f"All process finished...Time: {np.round(time.time()-self.start,4)}")
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            self.logger.error(f"line {exc_tb.tb_lineno}: {e}")

        

        
    def save(self):
        self.df_eval.sort_index(axis=1,inplace=True)
        self.df_eval.to_csv(self.data_dir_dict["mobilesensing_dir_path"]+"/csv/df_eval.csv",index=False)
        self.df_post.sort_index(axis=1,inplace=True)
        self.df_post.to_csv(self.data_dir_dict["mobilesensing_dir_path"]+"/csv/df_post.csv",index=False)
    
        self.notify_history.sort_index(axis=1,inplace=True)
        self.notify_history.to_csv(self.data_dir_dict["mobilesensing_dir_path"]+"/csv/notify_history.csv",index=False)

if __name__=="__main__":
    trial_name="20250220MinimumNotification"
    strage="local"
    json_dir_path="/catkin_ws/src/database"+"/"+trial_name+"/json"


    cls=RealtimeEvaluator(trial_name=trial_name,strage=strage)
    print("RealtimeEvaluator has woken up")
    cls_zokusei=PreprocessZokusei()
    print("PreprocessBlip has woken up")
    cls_yolo=PreprocessYolo()
    print("PreprocessYolo has woken up")
    cls_object=PreprocessObject()
    print("PreprocessObject has woken up")

    cls.evaluate_main()
    # path = "/media/hayashide/MobileSensing/20250207Dev/json"  # 監視するディレクトリ
    event_handler = JSONFileChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, json_dir_path, recursive=False)
    observer.start()    
    print("Observation started")

    try:
        while True:
            time.sleep(1)  # イベントループ
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    atexit.register(cls.save)
