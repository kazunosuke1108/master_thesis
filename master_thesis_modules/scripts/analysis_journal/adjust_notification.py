import os
import sys
import copy
import time
from glob import glob
import random
random.seed(42)  # 例えば 42 に固定

import json
import cv2
import atexit
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
from scripts.master_v4 import Master
from scripts.network.graph_manager_v4 import GraphManager
# from scripts.preprocess.preprocess_blip_snapshot import PreprocessBlip
# from scripts.preprocess.preprocess_zokusei_snapshot import PreprocessZokusei
# from scripts.preprocess.preprocess_yolo_snapshot import PreprocessYolo
# from scripts.preprocess.preprocess_objects_snapshot import PreprocessObject


class NotificationAdjust(Manager,GraphManager):
    def __init__(self,trial_name,strage,attempt_name=""):
        super().__init__()
        self.trial_name=trial_name
        self.strage=strage
        self.attempt_name=attempt_name
        self.data_dir_dict=self.get_database_dir(trial_name=self.trial_name,strage=self.strage)
        self.default_graph=self.get_default_graph()

        # path
        self.df_eval_csv_path=self.data_dir_dict["mobilesensing_dir_path"]+"/csv/df_eval.csv"
        self.name_dict_path=self.data_dir_dict["mobilesensing_dir_path"]+"/json/name_dict.json"
        # self.df_after_reid=pd.read_csv(self.df_after_reid_csv_path,header=0)

        # df
        self.df_eval=pd.read_csv(self.df_eval_csv_path,header=0)
        self.notify_history=pd.DataFrame(columns=["notificationId","timestamp","relativeTimestamp","patient","sentence","type","10000000","dynamicVal","staticVal"])

        # logger
        self.logger=self.prepare_log(self.data_dir_dict["mobilesensing_dir_path"]+"/log")

        # map
        self.map_fig,self.map_ax=self.plot_map_matplotlib()

        # notification params
        self.notify_interval_dict={"general":10,"notice":5,"help":10,"notice2help":5}
        self.notify_threshold_by_dinamic_factor={
            "40000000":0.3,
            "40000001":0.4,
            "40000010":0.6,
            "40000011":0.9,
            "40000012":0.9,
            "40000013":0.3,
            "40000014":0.6,
            "40000015":0.5,
            "40000016":0.9,
            "40000100":0.6,
            "40000101":0.6,
            "40000102":0.6,
            "40000110":0.5,
            "40000111":0.5,
        }
        self.total_risk_threshold=0.275
        self.notification_id=0
        self.previous_risky_patient=""

        # parameters
        self.luminance_threshold=0.5
        self.smoothing_w=20

        self.structure_dict={
            "ivPole":[
                np.array([-10,9]), # muに相当
                np.array([-4,9]),
                ],
            "wheelchair":[
                np.array([-9,12]),
                np.array([-5,12]),
                ],
            "handrail":{
                "xrange":[-10,-4],
                "yrange":[9,15]
                # "xrange":[6,15],
                # "yrange":[-11,-4]
                },
            "staff_station":{
                "pos":[-8,7],
                "direction":[0,0.1]
                # "pos":[6,-7.5],
                # "direction":[0,0.1]
                }
        }

    def get_rank_and_text(self,data_dicts,iteration_idx):
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
                # 見守り状況を追加
                focus_keys_static.append("40000110")
                focus_keys_static.append("40000111")

                additional_data_dicts["static_factor"]["significance"]={}
                # 顕著なノードの検出
                patients=list(data_dicts.keys())
                
                for patient in patients:
                    # if patient==focus_patient:
                    #     continue
                    additional_data_dicts["static_factor"]["significance"][patient]={}
                    for k in focus_keys_static:
                        if k in ["40000000","40000001"]: # TFNが格納されている場合
                            val_focus_patient=np.array([eval(v)[1] for v in self.df_eval.loc[iteration_idx:iteration_idx+self.smoothing_w,focus_patient+f"_{k}"].dropna()]).mean()
                            val_patient=np.array([eval(v)[1] for v in self.df_eval.loc[iteration_idx:iteration_idx+self.smoothing_w,patient+f"_{k}"].dropna()]).mean()
                            additional_data_dicts["static_factor"]["significance"][patient][k]=val_focus_patient-val_patient
                        else:
                            additional_data_dicts["static_factor"]["significance"][patient][k]=self.df_eval.loc[iteration_idx:iteration_idx+self.smoothing_w,focus_patient+f"_{k}"].mean()-self.df_eval.loc[iteration_idx:iteration_idx+self.smoothing_w,patient+f"_{k}"].mean()
                
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
                    if (int(k)>=40000010) and (int(k)<=40000019):
                        focus_keys_dynamic.append(k)
                    # if int(k[-2])==1: # dynamic node
                    #     focus_keys_dynamic.append(k)
                
                data=self.df_eval.loc[iteration_idx:iteration_idx+self.smoothing_w,[focus_patient+f"_{k}" for k in ["10000000"]+focus_keys_dynamic]].rolling(self.smoothing_w).mean()
                data_corr=data.corr()[focus_patient+"_10000000"]
                # 一番相関が高い4000番台の因子を抜き出す
                data_corr_4000=data_corr[[focus_patient+"_"+k for k in focus_keys_dynamic]]
                data_corr_4000[focus_patient+"_40000010"]=data_corr_4000[focus_patient+"_40000010"]*0.49
                data_corr_4000[focus_patient+"_40000011"]=data_corr_4000[focus_patient+"_40000011"]*0.08
                data_corr_4000[focus_patient+"_40000012"]=data_corr_4000[focus_patient+"_40000012"]*0.16
                data_corr_4000[focus_patient+"_40000013"]=data_corr_4000[focus_patient+"_40000013"]*0.17
                data_corr_4000[focus_patient+"_40000014"]=data_corr_4000[focus_patient+"_40000014"]*0.03
                data_corr_4000[focus_patient+"_40000015"]=data_corr_4000[focus_patient+"_40000015"]*0.05
                data_corr_4000[focus_patient+"_40000016"]=data_corr_4000[focus_patient+"_40000016"]*0.03
                most_corr_key=list(data_corr_4000.keys())[data_corr_4000.argmax()]
                most_corr_node=most_corr_key.replace(focus_patient+"_","")
                additional_data_dicts["dynamic_factor"]["most_significant_node"]=most_corr_node
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                self.logger.error(f"line {exc_tb.tb_lineno}: {e}")

            return additional_data_dicts,most_corr_node

            
            pass

        def judge_notification_necessity(data_dicts,most_risky_patient,dynamic_factor_node):
            def judge_time_interval(notify_history,data_dicts,alert_type):
                """
                - 直前の通知との時間差
                - 同じ種類の通知との時間差
                """
                if len(notify_history)==0: # 通知自体が初回の場合
                    self.logger.info("時間間隔制約：充足【初回】")
                    return True
                else:
                    last_notice_timestamp=notify_history.loc[len(notify_history)-1,"timestamp"]
                    if self.timestamp-last_notice_timestamp>self.notify_interval_dict["general"]:
                        self.logger.info(f"時間間隔制約：充足【経過時間: {np.round(self.timestamp-last_notice_timestamp,1)} sec.】")
                        return True
                    else:
                        self.logger.info(f"時間間隔制約：非充足【経過時間: {np.round(self.timestamp-last_notice_timestamp,1)} sec.】")
                        return False

                # # 同じ種類の通知（通知・応援要請）を抜き出す
                # df=notify_history[notify_history["type"]==alert_type].reset_index()
                
                # if len(df)==0: # 同じ種類の通知の、前例がない場合
                #     if len(notify_history)==0:
                #         self.logger.info("時間間隔制約：充足【初回】")
                #         return True
                #     if alert_type=="help":
                #         df=notify_history[notify_history["type"]=="notice"].reset_index()
                #         if self.timestamp-df.loc[len(df)-1,"timestamp"]>self.notify_interval_dict["notice2help"]:
                #             return True
                #         else:
                #             return False
                #     else:
                #         self.logger.info("時間間隔制約：充足【よくわからん】")
                #         return True

                # if self.timestamp-df.loc[len(df)-1,"timestamp"]>self.notify_interval_dict[alert_type]:
                #     if alert_type=="help":
                #         df=notify_history[notify_history["type"]=="notice"].reset_index()
                #         if self.timestamp-df.loc[len(df)-1,"timestamp"]>self.notify_interval_dict["notice2help"]:
                #             return True
                #         else:
                #             return False
                #     else:
                #         self.logger.info(f"時間間隔制約：充足 ({self.timestamp-df.loc[len(df)-1,'timestamp']})")
                #         return True
                # else:
                #     self.logger.info(f"時間間隔制約：非充足 ({self.timestamp-df.loc[len(df)-1,'timestamp']})")
                #     return False

            def judge_rank_change(most_risky_patient):
                if len(self.notify_history)>0:
                    rank_change=most_risky_patient!=self.previous_risky_patient
                elif len(self.notify_history)==0:
                    rank_change=True
                self.logger.info(f"順位入れ替わり検知：{rank_change}【今回：{most_risky_patient} 前イテレーション：{self.previous_risky_patient}】")
                return rank_change

            def judge_rank_change2(most_risky_patient):
                if len(self.notify_history)>0:
                    rank_change=most_risky_patient!=self.notify_history.loc[len(self.notify_history)-1,"patient"]
                    self.logger.info(f"前回通知人物と違う人物が最上位と検知：{rank_change}【今回：{most_risky_patient} 前回通知：{self.notify_history.loc[len(self.notify_history)-1,'patient']}】")
                elif len(self.notify_history)==0:
                    self.logger.info(f"初回のため、通知人物と最上位人物の比較はなし")
                    rank_change=True
                # self.logger.info("順位変更のルールは一時的に休止中")
                # rank_change=True
                return rank_change
            
            def judge_total_risk(most_risky_patient,data_dicts):
                total_risk=data_dicts[most_risky_patient]["10000000"]
                if total_risk>self.total_risk_threshold:
                    self.logger.info(f"{most_risky_patient}の{10000000}: {total_risk}>{self.total_risk_threshold} ➡ 要件合致")
                    return True
                else:
                    self.logger.info(f"{most_risky_patient}の{10000000}: {total_risk}<{self.total_risk_threshold} ➡ 要件不合致")
                    return False

            def judge_above_dynamic_thre(most_risky_patient,dynamic_factor_node):
                # node_val=self.df_eval.loc[iteration_idx-1,most_risky_patient+"_"+dynamic_factor_node]
                node_val=self.data_dicts[most_risky_patient][dynamic_factor_node]
                if node_val>self.notify_threshold_by_dinamic_factor[dynamic_factor_node]:
                    self.logger.info(f"{most_risky_patient}の{dynamic_factor_node}: {node_val}>{self.notify_threshold_by_dinamic_factor[dynamic_factor_node]} ➡ 要件合致")
                    return True
                else:
                    self.logger.info(f"{most_risky_patient}の{dynamic_factor_node}: {node_val}<{self.notify_threshold_by_dinamic_factor[dynamic_factor_node]} ➡ 要件不合致")
                    return False
                
            try:
                ## A 前回通知からの時間経過
                tf_interval_notify=judge_time_interval(self.notify_history,data_dicts,"notice")
                # tf_interval_help=judge_time_interval(self.notify_history,data_dicts,"help")

                ## B 順位入れ替えの発生
                # tf_rank_change=judge_rank_change(most_risky_patient)

                ## B2 以前通知した人物と違う人物が最上位にいる
                tf_rank_change=judge_rank_change2(most_risky_patient)

                ## C dynamic_factor_node毎の通知基準値を超越しているか
                tf_dynamic_node=judge_above_dynamic_thre(most_risky_patient,dynamic_factor_node)

                ## D total riskが基準値以上か
                tf_total_risk=judge_total_risk(most_risky_patient=most_risky_patient,data_dicts=data_dicts)

                
                if tf_interval_notify and tf_rank_change and tf_dynamic_node and tf_total_risk: # すべての条件を充足したら通知を実行
                    self.logger.warning(f"経過時間：{tf_interval_notify}　順位変更：{tf_rank_change}　ノード閾値以上：{tf_dynamic_node}　総合危険度しきい値以上：{tf_total_risk}　➡　通知実行")
                    return True
                else:
                    self.logger.warning(f"経過時間：{tf_interval_notify}　順位変更：{tf_rank_change}　ノード閾値以上：{tf_dynamic_node}　総合危険度しきい値以上：{tf_total_risk}　➡　通知見送り")
                    return False

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
                # alert_text=f"{self.get_patient_name(patient_id=most_risky_patient)}さんが，元々{text_static}のに，{text_dynamic}ので，危険です．"
                alert_text=f"{text_static}{self.get_patient_name(patient_id=most_risky_patient)}さんが，{text_dynamic}．"
            return alert_text

        try:
            additional_data_dicts={}
            # 順位の算出
            additional_data_dicts["rank"]={}
            patients=list(data_dicts.keys())
            total_risks=[]
            for patient in patients:
                # total_risks.append(data_dicts[patient]["10000000"])
                total_risks.append(self.df_eval.loc[iteration_idx:iteration_idx+self.smoothing_w,patient+"_10000000"].mean())
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
            
            
            additional_data_dicts["alert"]=text
            additional_data_dicts["most_risky_patient"]=most_risky_patient

            if notice_necessary:
                notify_dict={
                    "notificationId":self.notification_id,
                    "timestamp":self.timestamp,
                    "relativeTimestamp":self.timestamp,
                    "patient":most_risky_patient,
                    "sentence":text,
                    "type":"notice",
                    "10000000":data_dicts[most_risky_patient]["10000000"],
                    "significantDynamicVal":data_dicts[most_risky_patient][additional_data_dicts["dynamic_factor"]["most_significant_node"]],
                    "significantStaticVal":data_dicts[most_risky_patient][additional_data_dicts["static_factor"]["most_significant_node"]],
                    }
                self.notify_history.loc[len(self.notify_history),:]=[self.notification_id,self.timestamp,self.timestamp,most_risky_patient,text,"notice",data_dicts[most_risky_patient]["10000000"],data_dicts[most_risky_patient][additional_data_dicts["dynamic_factor"]["most_significant_node"]],data_dicts[most_risky_patient][additional_data_dicts["static_factor"]["most_significant_node"]]]
                self.notification_id+=1
            else:
                notify_dict={}

            # 今回の危険患者をメモ
            self.previous_risky_patient=most_risky_patient

            for patient,rank in zip(patients,patients_rank):
                additional_data_dicts["rank"][patient]={}
                additional_data_dicts["rank"][patient]["10000000"]=rank
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            self.logger.error(f"line {exc_tb.tb_lineno}: {e}")
        return additional_data_dicts,notify_dict

    def renew_name_dict(self,patients=[]):
        # name_dictを作成または読み込む
        # 新しい患者が確認された場合に、keyを作成する
        try:
            name_dict=self.load_json(self.name_dict_path)
        except FileNotFoundError:
            name_dict={}

        for patient in patients:
            if patient not in list(name_dict.keys()):
                name_dict[patient]=self.num2alpha(int(patient))
        self.write_json(name_dict,self.name_dict_path)
        return name_dict

    def get_patient_name(self,patient_id):
        try:
            patient_name=self.name_dict[patient_id]
        except KeyError:
            patient_name=patient_id
        return patient_name
        
    def main(self):
        # 平滑化
        for k in self.df_eval.keys():
            try:
                self.df_eval[k]=self.df_eval[k].rolling(self.smoothing_w).mean()
            except pd.errors.DataError:
                pass
        print(self.df_eval)

        for iteration_idx,row in self.df_eval.iterrows():
            if iteration_idx<self.smoothing_w:
                continue

            # data_dictsの復元
            self.name_dict=self.renew_name_dict()
            self.timestamp=row["timestamp"]
            self.temp_data_dicts=row.to_dict()
            self.temp_data_dicts=self.nest_dict_original(flat_dict=self.temp_data_dicts)
            self.data_dicts={}
            for k in self.temp_data_dicts.keys():
                if k=="timestamp":
                    continue 
                if not np.isnan(float(self.temp_data_dicts[k]["10000000"])):
                    self.data_dicts[k]=self.temp_data_dicts[k]
            print(self.temp_data_dicts["timestamp"])
            additional_data_dicts,notify_dict=self.get_rank_and_text(data_dicts=self.data_dicts,iteration_idx=iteration_idx)
            save_notify_history=self.notify_history.sort_index(axis=1)
            save_notify_history.to_csv(self.data_dir_dict["mobilesensing_dir_path"]+f"/csv/notify_history_postProcess.csv",index=False)
        pass

    def save(self):
        save_notify_history=self.notify_history.sort_index(axis=1)
        save_notify_history.to_csv(self.data_dir_dict["mobilesensing_dir_path"]+f"/csv/notify_history_postProcess.csv",index=False)
        print("saved")

if __name__=="__main__":
    trial_name="20250307postAnalysis3"
    strage="NASK"
    # json_dir_path="/catkin_ws/src/database"+"/"+trial_name+"/json"


    cls=NotificationAdjust(trial_name=trial_name,strage=strage)
    cls.main()
    atexit.register(cls.save)