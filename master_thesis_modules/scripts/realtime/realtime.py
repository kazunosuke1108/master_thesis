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
from scripts.preprocess.preprocess_blip_snapshot import PreprocessBlip
from scripts.preprocess.preprocess_yolo_snapshot import PreprocessYolo
from scripts.preprocess.preprocess_handrail_snapshot import PreprocessHandrail


# 定数
WATCHED_FILES = ["dict_after_reid.json","dict_after_reid_old.json"]
dayroom_structure_dict={"xrange":[6,15],"yrange":[-11,-4]} # 11月
# dayroom_structure_dict={"xrange":[-4,6],"yrange":[5,10]} # 08月
ss_structure_dict={"pos":[6,-7.5],"direction":[0,0.1]} # 11月
# ss_structure_dict={"pos":[-4,7.5],"direction":[0.1,0]} # 08月


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
        self.logger=self.prepare_log(self.data_dir_dict["mobilesensing_dir_path"]+"/log")
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
                elp_img_path=elp_img_paths[abs(np.array([float(os.path.basename(p).split("_")[1][:-len(".jpg")])-self.timestamp for p in elp_img_paths])).argmin()]
                self.logger.info(f"解析ELP: {elp_img_path}")
                self.elp_img=cv2.imread(elp_img_path)
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
        # 1人ずつ評価
        self.logger.info(f"特徴量の算出開始")
        data_dicts={}
        self.logger.info(f"評価開始前のpatients: {self.patients}")
        for patient in self.patients:
            # bbox img
            t,b,l,r=self.json_latest_data[patient+"_bboxLowerY"],self.json_latest_data[patient+"_bboxHigherY"],self.json_latest_data[patient+"_bboxLowerX"],self.json_latest_data[patient+"_bboxHigherX"]
            try:
                t,b,l,r=int(t),int(b),int(l),int(r)
            except ValueError: # NaNが入っていた場合
                # self.patients.remove(patient)
                self.logger.info(f"bbox情報が不正のため削除 {patient}")
                continue
            bbox_img=self.elp_img[t:b,l:r]

            data_dicts[patient]={}
            # luminance (7)

            # position (6)
            data_dicts[patient]["60010000"]=self.json_latest_data[patient+"_x"]
            data_dicts[patient]["60010001"]=self.json_latest_data[patient+"_y"]

            # feature (5)
            ## BLIP系 属性・物体(手すり以外)
            data_dicts[patient]=cls_blip.blip_snapshot(data_dicts[patient],self.elp_img,t,b,l,r,)
            ## 動作
            data_dicts[patient],success_flag=cls_yolo.yolo_snapshot(data_dicts[patient],self.elp_img,t,b,l,r,)
            if not success_flag:
                self.logger.info(f"姿勢推定が不正のため削除 {patient}")
                # self.patients.remove(patient)
                del data_dicts[patient]
                continue
            ## 物体(手すり)
            data_dicts[patient]=cls_handrail.handrail_snapshot(data_dicts[patient],dayroom_structure_dict)

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

        # スタッフがいるかどうかを判定
        staff=[patient for patient in self.patients if data_dicts[patient]["50000000"]=="no"]
        if len(staff)>0:
            print(staff)
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
        
        def get_alert_sentence(most_risky_patient,static_factor_node,dynamic_factor_node):
            if static_factor_node=="":
                text_dynamic=self.default_graph["node_dict"][dynamic_factor_node]["description_ja"]
                alert_text=f"{most_risky_patient}さんが，{text_dynamic}ので，危険です．"
            else:
                text_static=self.default_graph["node_dict"][static_factor_node]["description_ja"]
                text_dynamic=self.default_graph["node_dict"][dynamic_factor_node]["description_ja"]
                alert_text=f"{most_risky_patient}さんが，元々{text_static}のに，{text_dynamic}ので，危険です．"
            return alert_text

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
    
    


    
    def evaluate_main(self):
        s=time.time()
        self.logger.info(f"RealtimeEvaluator called. Time: {np.round(time.time()-s,4)}")
        # 情報の読込
        self.logger.info(f"Loading info...")
        json_latest_path,json_latest_data=self.load_data()
        # 特徴量の算出
        self.logger.info(f"Calculating feature values...Time: {np.round(time.time()-s,4)}")
        data_dicts=self.get_features()
        # 危険性評価
        self.logger.info(f"Evaluating risk...Time: {np.round(time.time()-s,4)}")
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
        
        self.logger.info(f"Analyzing results...Time: {np.round(time.time()-s,4)}")
        patients=list(data_dicts.keys())
        if len(patients)>0:
            additional_data_dicts={}
            # リスク優先順位付け
            additional_data_dicts=self.get_rank_and_text(data_dicts,additional_data_dicts)
            additional_data_dicts_flatten=self.flatten_dict(additional_data_dicts)
            self.df_post=pd.concat([self.df_post,pd.DataFrame(additional_data_dicts_flatten,index=[0])],axis=0)
            self.df_post.reset_index(inplace=True,drop=True)
        
        self.logger.info(f"Drawing bbox image...Time: {np.round(time.time()-s,4)}")
        self.draw_bbox_img()
        

        
    def save(self):
        self.df_eval.sort_index(axis=1,inplace=True)
        self.df_eval.to_csv(self.data_dir_dict["trial_dir_path"]+"/df_eval.csv",index=False)
        self.df_post.sort_index(axis=1,inplace=True)
        self.df_post.to_csv(self.data_dir_dict["trial_dir_path"]+"/df_post.csv",index=False)
    

if __name__=="__main__":
    trial_name="20250215NagasakiShort5"
    strage="local"
    json_dir_path="/catkin_ws/src/database"+"/"+trial_name+"/json"


    cls=RealtimeEvaluator(trial_name=trial_name,strage=strage)
    print("RealtimeEvaluator has woken up")
    cls_blip=PreprocessBlip()
    print("PreprocessBlip has woken up")
    cls_yolo=PreprocessYolo()
    print("PreprocessYolo has woken up")
    cls_handrail=PreprocessHandrail()
    print("PreprocessHandrail has woken up")

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
