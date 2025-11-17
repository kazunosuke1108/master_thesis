import os
import sys
import copy
import dill
from glob import glob

from icecream import ic
from pprint import pprint

import numpy as np
import pandas as pd

sys.path.append(".")
sys.path.append("..")
sys.path.append(os.path.expanduser("~")+"/kazu_ws/master_thesis/master_thesis_modules")
from scripts.management.manager import Manager
from scripts.network.graph_manager import GraphManager
from scripts.AHP.get_comparison_mtx import getConsistencyMtx
from scripts.fuzzy.fuzzy_reasoning import FuzzyReasoning
from scripts.entropy.entropy_weight_generator import EntropyWeightGenerator
from scripts.pseudo_data.pseudo_data_generator import PseudoDataGenerator

class Master(Manager,GraphManager,FuzzyReasoning,EntropyWeightGenerator):
    def __init__(self,trial_name,strage,scenario_dict={},runtype="simulation",staff_name=""):
        super().__init__()
        print("# Master開始 #")
        self.trial_name=trial_name
        self.strage=strage
        self.scenario_dict=scenario_dict
        self.runtype=runtype
        self.data_dir_dict=self.get_database_dir(self.trial_name,self.strage)
        self.staff_name=staff_name
        
        print("# default graph 定義 #")
        default_graph=self.get_default_graph()

        # parameters
        self.spatial_normalization_param=np.sqrt(2)*6
        self.fps=20
        self.throttling=False
        self.bg_differencing=False
        self.bg_differencing_thre=0.5
        self.fps_BLIP=10 #[Hz]
        self.fps_YOLO=50 #[Hz]

        if self.runtype=="simulation":
            self.patients=["A","B","C"]
            # 人数分のgraphを定義
            # シナリオの定義・DataFrameの生成
        elif self.runtype=="experiment":
            self.raw_csv_paths=sorted(glob(self.data_dir_dict["trial_dir_path"]+"/data_*_raw.csv"))
            self.patients=[os.path.basename(p)[len("data_"):-len("_raw.csv")] for p in self.raw_csv_paths]
        elif self.runtype=="basic_check":
            self.patients=["A"]
        
        print("# 人数分のgraph 定義 #")
        self.graph_dicts={}
        for patient in self.patients:
            self.graph_dicts[patient]=copy.deepcopy(default_graph)
            # dataに列が存在するかどうか確認

        if self.runtype=="simulation":
            print("# simulation用のデータを生成中 #")
            self.data_dicts=self.define_scenario(fps=self.fps,scenario_dict=scenario_dict)
            pass
        elif self.runtype=="experiment":
            print("# 実験データをロード中 #")
            self.data_dicts={}
            for patient,raw_csv_path in zip(self.patients,self.raw_csv_paths):
                self.data_dicts[patient]=pd.read_csv(raw_csv_path,header=0)#.fillna(method="bfill")
                renew_dict={}
                for col in self.data_dicts[patient].keys():
                    try:
                        new_col=int(col)
                        renew_dict[col]=new_col
                    except Exception:
                        continue
                    self.data_dicts[patient].rename(columns=renew_dict,inplace=True)
        elif self.runtype=="basic_check":
            print("# basic checkのデータをロード中 #")
            # self.data_dicts=PseudoDataGenerator(trial_name=self.trial_name,strage=self.strage).get_basic_check_data(graph_dicts=default_graph,patients=self.patients)
            df=pd.read_csv(self.data_dir_dict["trial_dir_path"]+"/data_A_eval.csv",header=0)
            df_columns=[int(k) for k in df.keys() if k!="timestamp"]
            df_columns=["timestamp"]+df_columns
            df.columns=df_columns
            self.data_dicts={"A":df}
            # raise NotImplementedError

        # data_dictsに不足した列がないか確認
        for col in self.graph_dicts[patient]["node_dict"].keys():
            try:
                self.data_dicts[patient][col]
            except KeyError:
                self.data_dicts[patient][col]=np.nan

        # staffごとのFuzzy推論のカスタマイズ
        TFN_csv_path=f"/media/hayashide/MasterThesis/common/TFN_{self.staff_name}.csv"
        TFN_data = pd.read_csv(TFN_csv_path,names=["l","c","r"])
        print(TFN_data)

        
        # 危険動作の事前定義
        self.risky_motion_dict={
            40000010:{
                "label":"standUp",
                "features":np.array([1,         0,      0,      1]),
                },
            40000011:{
                "label":"releaseBrake",
                "features":np.array([0,         0.5,    0.5,      0.5]),
                },
            40000012:{
                "label":"moveWheelchair",
                "features":np.array([0,         0.5,      0.5,      0.5]),
                },
            40000013:{
                "label":"loseBalance",
                "features":np.array([0,         1,      np.nan, np.nan]),
                },
            40000014:{
                "label":"moveHand",
                "features":np.array([np.nan,    0,      1,      np.nan]),
                },
            40000015:{
                "label":"coughUp",
                "features":np.array([np.nan,    0.5,      0.5,      np.nan]),
                },
            40000016:{
                "label":"touchFace",
                "features":np.array([np.nan,    0,      0.5,      np.nan]),
            },
        }

        # AHP 一対比較行列の作成
        if self.staff_name=="":
            array_type=1
        else:
            array_type=self.staff_name
        self.AHP_dict=getConsistencyMtx().get_all_comparison_mtx_and_weight(trial_name=self.trial_name,strage=self.strage,array_type=array_type)


    def pseudo_throttling(self,data_dicts):
        initial_fps=20
        entropy_window=5
        convert_dict={
            50000000:{"yes":1,"no":0},
            50000010:{"old":2,"middle":1,"young":0},
            }
        convert_dict_swap={
            50000000:{v: k for k, v in convert_dict[50000000].items()},
            50000010:{v: k for k, v in convert_dict[50000010].items()},
        }
        control_rule_dict={# control_rule_dict[s][ds]["result"] -> ans
            "+":{
                "+":{
                    "conditions":{"s":"+","ds":"+"},
                    "result":"++",
                },
                "o":{
                    "conditions":{"s":"+","ds":"o"},
                    "result":"+",
                },
                "-":{
                    "conditions":{"s":"+","ds":"-"},
                    "result":"o",
                },
            },
            "o":{
                "+":{
                    "conditions":{"s":"middle","ds":"+"},
                    "result":"+",
                },
                "o":{
                    "conditions":{"s":"middle","ds":"o"},
                    "result":"o",
                },
                "-":{
                    "conditions":{"s":"middle","ds":"-"},
                    "result":"-",
                },
            },
            "-":{
                "+":{
                    "conditions":{"s":"-","ds":"+"},
                    "result":"o",
                },
                "o":{
                    "conditions":{"s":"-","ds":"o"},
                    "result":"-",
                },
                "-":{
                    "conditions":{"s":"-","ds":"-"},
                    "result":"--",
                },
            }
        }
        s_thre_dict={
            "-":{"min":-np.inf,"max":1e-5},
            "o":{"min":1e-5,"max":1e-3},
            "+":{"min":1e-3,"max":np.inf},
        }
        ds_threshold=1e-7#0.1*(1/20) # thre[/s]*dt[s]
        ds_thre_dict={
            "-":{"min":-np.inf,"max":-ds_threshold},
            "o":{"min":-ds_threshold,"max":ds_threshold},
            "+":{"min":ds_threshold,"max":np.inf},
        }
        dfps_dict={
            "--":-2,
            "-":-1,
            "o":0,
            "+":2,
            "++":4,
        }
        fps_clip_dict={
            "min":1,
            "max":20,
        }
        dfps_powerup_by_bg_differencing_dict={
            "++":"++",
            "+":"++",
            "o":"+",
            "-":"+",
            "--":"o",
        }

        fps_history={}
        dod_history={}
        focus_keys=list(data_dicts[list(data_dicts.keys())[0]].keys())
        focus_keys.remove("timestamp")
        focus_keys=[int(k) for k in focus_keys]
        focus_keys_BLIP_zokusei= [50000000,50000001,50000010,50000011]
        focus_keys_BLIP_objects= [k for k in focus_keys if ((int(k)>=50001000) and (int(k)<=50001013))]
        focus_keys_YOLO= [50000100,50000101,50000102,50000103]
        focus_keys_others= [k for k in focus_keys if k not in (focus_keys_BLIP_zokusei+focus_keys_BLIP_objects+focus_keys_YOLO+[70000000])]
        
        def get_control_input(s_value,ds_value,bg_value=0):
            if np.isnan(s_value):
                s_value=0
            if np.isnan(ds_value):
                ds_value=0
            for key in s_thre_dict.keys():
                if (s_thre_dict[key]["min"]<=s_value) and (s_value<=s_thre_dict[key]["max"]):
                    s=key
            for key in ds_thre_dict.keys():
                if (ds_thre_dict[key]["min"]<=ds_value) and (ds_value<=ds_thre_dict[key]["max"]):
                    ds=key
            dfps=control_rule_dict[s][ds]["result"]
            if self.bg_differencing:
                if bg_value>self.bg_differencing_thre:
                    dfps=dfps_powerup_by_bg_differencing_dict[dfps]
            dfps_value=dfps_dict[dfps]
            return dfps_value

        # 輝度とその変化率に関するデータをまとめておく
        """
        70000000: 輝度値
        70000001: 輝度値の隣接時刻差分
        70000002: Fuzzy推論結果
        70000003: 患者間での正規化後の結果
        """
        event_df=pd.concat([data_dicts[list(data_dicts.keys())[0]]["timestamp"]]+[data_dicts[patient][70000000] for patient in data_dicts.keys()],axis=1)
        event_df.fillna(method="ffill",inplace=True)
        event_df.columns=["timestamp"]+[k+"_70000000" for k in list(data_dicts.keys())]
        # 隣接データの差分の計算 (Fuzzy推論のため，定義域を-0.5から0.5に制限)
        diff_clip_value=0.5
        for patient in data_dicts.keys():
            event_df[patient+"_70000001"]=event_df[patient+"_70000000"].diff()
            # 定義域制限
            event_df[patient+"_70000001"]=np.clip(event_df[patient+"_70000001"],-diff_clip_value,diff_clip_value)
            # オフセット
            event_df[patient+"_70000001"]+=0.5
        # ノイズ除去
        w=3
        event_df=event_df.rolling(w).mean()
        event_df.fillna(method="bfill",inplace=True)
        # Fuzzy推論による優先度決定
        for patient in data_dicts.keys():
            map_arg_1=[{70000000:arg01,70000001:arg02} for arg01,arg02 in zip(event_df[patient+"_70000000"],event_df[patient+"_70000001"])] # {40000110:0.5,40000111:0.5}の形を行数分格納したリスト
            map_arg_2=[70000000 for _ in event_df[patient+"_70000000"].values]
            event_df[patient+"_70000002"]=list(map(self.calculate_fuzzy,map_arg_1,map_arg_2))
            event_df[patient+"_70000002"]=event_df[patient+"_70000002"]**2
        # 正規化
        sum_70000002=event_df[[patient+"_70000002" for patient in data_dicts.keys()]].sum(axis=1)
        for patient in data_dicts.keys():
            event_df[patient+"_70000003"]=event_df[patient+"_70000002"]/sum_70000002

        # 間引き処理本体
        for patient in data_dicts.keys():
            fps_dict={k:{"entropy":np.nan,"fps":initial_fps,"previous_fps":initial_fps,"d":np.nan,"previous_d":np.nan} for k in focus_keys}
            fps_history[patient]=pd.DataFrame(data_dicts[patient]["timestamp"].values,columns=["timestamp"])
            dod_history[patient]=pd.DataFrame(data_dicts[patient]["timestamp"].values,columns=["timestamp"])
            for k in focus_keys:
                fps_history[patient][k]=np.nan
                dod_history[patient][k]=np.nan
            # 非floatのデータを置換
            data_dicts[patient][50000000].replace(convert_dict[50000000],inplace=True)
            data_dicts[patient][50000010].replace(convert_dict[50000010],inplace=True)

            # 時系列に沿って計算
            for i,row in data_dicts[patient].iterrows():
                if i<entropy_window:
                    continue
                # 各特徴量について，entropyを求める
                for k in focus_keys:
                    # 当該フレームがスキップ対象の場合，continue
                    if np.isnan(data_dicts[patient].loc[i,k]):
                        continue
                    # BLIP関連のFuzzy推論
                    if k in focus_keys_BLIP_zokusei:
                        fps_dict[k]["fps"]=1e-2 # 100sに1回確認（実質ゼロ）
                        pass
                    elif k in focus_keys_BLIP_objects:
                        fps_total=event_df.loc[i,patient+"_70000003"]*self.fps_BLIP
                        if k in [50001000,50001001,50001002,50001003]:#点滴を優先
                            fps_candidate=np.floor(fps_total/2).astype(int)+fps_total%2
                        elif k in [50001010,50001011,50001012,50001013]:
                            fps_candidate=np.floor(fps_total/2).astype(int)
                        if fps_candidate>1:
                            fps_dict[k]["fps"]=fps_candidate
                        else:
                            fps_dict[k]["fps"]=fps_total/2
                        pass
                    # YOLO関連のFuzzy推論
                    elif k in focus_keys_YOLO:
                        fps_dict[k]["fps"]=np.floor(event_df.loc[i,patient+"_70000003"]*self.fps_YOLO).astype(int)
                        pass
                    # EntropyによるFuzzy制御
                    elif k in focus_keys_others:
                        temp_data=data_dicts[patient].loc[:i,k]
                        temp_data=temp_data.dropna()
                        temp_data=temp_data.tail(entropy_window)
                        temp_data=temp_data/temp_data.sum()
                        # temp_data=temp_data.tail(entropy_window)
                        if temp_data.sum()==0:
                            e=1
                        else:
                            e=(-1/np.log(entropy_window)*temp_data*np.log(temp_data.astype(float))).sum()
                        fps_dict[k]["e"]=e
                        # entropyの値に基づいて，FPSを決める
                        d=1-fps_dict[k]["e"]+1e-10
                        new_fps=fps_dict[k]["fps"]+get_control_input(s_value=d,ds_value=d-fps_dict[k]["d"],bg_value=data_dicts[patient].loc[i,70000000])
                        new_fps=np.clip(new_fps,fps_clip_dict["min"],fps_clip_dict["max"])
                        fps_dict[k]["previous_fps"]=fps_dict[k]["fps"]
                        fps_dict[k]["previous_d"]=fps_dict[k]["d"]
                        fps_dict[k]["fps"]=new_fps
                        fps_dict[k]["d"]=d
                        dod_history[patient].loc[i,k]=d
                    
                    # FPSのメモ
                    fps_history[patient].loc[i,k]=fps_dict[k]["fps"]
                    
                    # FPSに基づいて，スキップする行を消す
                    try:
                        n_skip=int(self.fps/fps_dict[k]["fps"])-1
                    except Exception:
                        print(i,k,patient,event_df.loc[i,patient+"_70000000"],event_df.loc[i,patient+"_70000001"],event_df.loc[i,patient+"_70000002"],event_df.loc[i,patient+"_70000003"])
                        print(fps_total,fps_dict[k]["fps"])
                        print(data_dicts[patient].loc[i,k])
                        raise Exception
                    data_dicts[patient].loc[i+1:i+n_skip,k]=np.nan
                    # print(data_dicts[patient].loc[i+1:i+n_skip,1])
                    # if i>10:
                    #     print(data_dicts[patient].head(10))
                    #     raise NotImplementedError
                    # if d>0.1 and d<0.8:
                    #     raise NotImplementedError
                pass

        for patient in data_dicts.keys():
            # binary化したデータを直す
            data_dicts[patient][50000000].replace(convert_dict_swap[50000000],inplace=True)
            data_dicts[patient][50000010].replace(convert_dict_swap[50000010],inplace=True)

            # 間引き結果を保存
            data_dicts[patient].to_csv(self.data_dir_dict["trial_dir_path"]+"/data_"+patient+"_throttling_beforeInterp.csv",index=False)
            fps_history[patient].to_csv(self.data_dir_dict["trial_dir_path"]+"/fps_"+patient+".csv",index=False)
            dod_history[patient].to_csv(self.data_dir_dict["trial_dir_path"]+"/degreeOfDiversification_"+patient+".csv",index=False)
        
            # ffillでデータを補間して保存
            data_dicts[patient].interpolate(method="ffill",inplace=True)
            data_dicts[patient].to_csv(self.data_dir_dict["trial_dir_path"]+"/data_"+patient+"_throttling.csv",index=False)

        return data_dicts

    def define_scenario(self,fps,scenario_dict):
        print("# シナリオ生成 #")
        start_timestamp=0
        end_timestamp=10
        xrange=[0,6]
        yrange=[0,6]
        # fps=20
        patients=list(self.graph_dicts.keys())
        if len(scenario_dict)==0:
            general_dict={
                "start_timestamp":start_timestamp,
                "end_timestamp":end_timestamp,
                "fps":fps,
                "patients":patients,
                "xrange":xrange,
                "yrange":yrange,
            }
            zokusei_dict={
                "A":{
                    "patient":"yes",
                    "age":"old",
                },
                "B":{
                    "patient":"yes",
                    "age":"old",
                },
                "C":{
                    "patient":"yes",
                    "age":"old",
                },
                "NS":{
                    "patient":"no",
                    "age":"middle"
                }
            }
            position_dict={
                "A":(2,5),
                "B":(2,2),
                "C":(5,2),
                "NS":(5,5),
            }
            action_dict={
                "A":{
                    0:{
                        "label":"sit",
                        "start_timestamp":0,
                        "end_timestamp":end_timestamp,
                        },
                },
                "B":{
                    0:{
                        "label":"sit",
                        "start_timestamp":0,
                        "end_timestamp":2,
                        },
                    2:{
                        "label":"standup",
                        "start_timestamp":2,
                        "end_timestamp":4,
                        },
                    4:{
                        "label":"stand",
                        "start_timestamp":4,
                        "end_timestamp":6,
                        },
                    6:{
                        "label":"sitdown",
                        "start_timestamp":6,
                        "end_timestamp":8,
                        },
                    8:{
                        "label":"sit",
                        "start_timestamp":8,
                        "end_timestamp":end_timestamp,
                        },
                },
                "C":{
                    0:{
                        "label":"sit",
                        "start_timestamp":0,
                        "end_timestamp":end_timestamp,
                        },
                },
                "NS":{
                    0:{
                        "label":"work",
                        "start_timestamp":0,
                        "end_timestamp":5,
                    },
                    5:{
                        "label":"approach_B",
                        "start_timestamp":5,
                        "end_timestamp":7,
                        },
                    7:{
                        "label":"work_B",
                        "start_timestamp":7,
                        "end_timestamp":9,
                    },
                    9:{
                        "label":"leave_B",
                        "start_timestamp":9,
                        "end_timestamp":end_timestamp,
                        }
                }
            }
            surrounding_objects={
                "A":["wheelchair","ivPole"],
                "B":["wheelchair",],
                "C":[],
            }
        else:
            general_dict=scenario_dict["general_dict"]
            zokusei_dict=scenario_dict["zokusei_dict"]
            position_dict=scenario_dict["position_dict"]
            action_dict=scenario_dict["action_dict"]
            surrounding_objects=scenario_dict["surrounding_objects"]


        cls_PDG=PseudoDataGenerator(trial_name=self.trial_name,strage=self.strage)
        data_dicts=cls_PDG.get_pseudo_data(
            graph_dicts=self.graph_dicts,
            general_dict=general_dict,
            zokusei_dict=zokusei_dict,
            position_dict=position_dict,
            action_dict=action_dict,
            surrounding_objects=surrounding_objects,
        )
        return data_dicts

    def activation_func(self,val):
        return val

    def fuzzy_logic(self):
        def mu_yes():
            return (0.4,0.7,1.0)
        def mu_no():
            return (0.0,0.3,0.6)
        def mu_young():
            return (0.0,0.25,0.5)
        def mu_middle():
            return (0.25,0.5,0.75)
        def mu_old():
            return (0.5,0.75,1.0)
        def patient_or_not(val):
            if val=="yes":
                return mu_yes()
            elif val=="no":
                return mu_no()
            elif np.isnan(val):
                return np.nan
            else:
                raise Exception(f"Unexpected value in 内的・静的・患者判別: {val}")
        def age(val):
            if val=="young":
                return mu_young()
            elif val=="middle":
                return mu_middle()
            elif val=="old":
                return mu_old()
            elif np.isnan(val):
                return np.nan
            else:
                raise Exception(f"Unexpected value in 内的・静的・年齢: {val}")
        for patient in self.data_dicts.keys():
            self.data_dicts[patient][40000000]=list(map(patient_or_not,self.data_dicts[patient][50000000]))
            self.data_dicts[patient][40000001]=list(map(age,self.data_dicts[patient][50000010]))
        pass

    def pose_similarity(self):
        def get_similarity(risk,row):
            row=row[1]
            similarity=1-np.nanmean(abs(self.risky_motion_dict[risk]["features"]-row[[50000100,50000101,50000102,50000103]]))
            similarity=similarity**4
            # similarity=self.activation_func(similarity)
            return similarity
        for patient in self.data_dicts.keys():
            for risk in self.risky_motion_dict.keys():
                self.data_dicts[patient][risk]=list(map(get_similarity,[risk for i in range(len(self.data_dicts[patient]))],self.data_dicts[patient].iterrows()))
                self.data_dicts[patient][risk]=self.activation_func(self.data_dicts[patient][risk])
            print(patient,risk)
            
    def object_risk(self):
        for patient in self.data_dicts.keys():
            # 点滴
            self.data_dicts[patient][40000100]=1-np.clip(np.sqrt(
                (self.data_dicts[patient][50001000]-self.data_dicts[patient][60010000])**2+\
                (self.data_dicts[patient][50001001]-self.data_dicts[patient][60010001])**2
            )/self.spatial_normalization_param,0,1)
            self.data_dicts[patient][40000100]=self.activation_func(self.data_dicts[patient][40000100])

            # 車椅子
            self.data_dicts[patient][40000101]=1-np.clip(np.sqrt(
                (self.data_dicts[patient][50001010]-self.data_dicts[patient][60010000])**2+\
                (self.data_dicts[patient][50001011]-self.data_dicts[patient][60010001])**2
            )/self.spatial_normalization_param,0,1)
            self.data_dicts[patient][40000101]=self.activation_func(self.data_dicts[patient][40000101])

            # 手すり (逆数ではない)
            self.data_dicts[patient][40000102]=np.clip(np.sqrt(
                (self.data_dicts[patient][50001020]-self.data_dicts[patient][60010000])**2+\
                (self.data_dicts[patient][50001021]-self.data_dicts[patient][60010001])**2
            )/self.spatial_normalization_param,0,1)
            self.data_dicts[patient][40000102]=self.activation_func(self.data_dicts[patient][40000102])

    def staff_risk(self):
        def direction_risk(patient_x,patient_y,staff_x,staff_y,staff_vx,staff_vy):
            relative_pos=np.array([patient_x-staff_x,patient_y-staff_y])
            relative_vel=np.array([staff_vx,staff_vy])
            cos_theta=np.dot(relative_pos,relative_vel)/(np.linalg.norm(relative_pos)*np.linalg.norm(relative_vel))
            val=1-(cos_theta/2+0.5)
            # if cos_theta>1:
            #     val=0
            # elif cos_theta<0:
            #     val=1
            # else:
            #     val=1-cos_theta
            return val
        for patient in self.data_dicts.keys():
            # 距離リスク
            self.data_dicts[patient][40000110]=np.clip(np.sqrt(
                (self.data_dicts[patient][50001100]-self.data_dicts[patient][60010000])**2+\
                (self.data_dicts[patient][50001101]-self.data_dicts[patient][60010001])**2
            )/self.spatial_normalization_param,0,1)
            self.data_dicts[patient][40000110]=self.activation_func(self.data_dicts[patient][40000110])
            # 向きリスク
            self.data_dicts[patient][40000111]=list(map(direction_risk,
                                                        self.data_dicts[patient][60010000],
                                                        self.data_dicts[patient][60010001],
                                                        self.data_dicts[patient][50001100],
                                                        self.data_dicts[patient][50001101],
                                                        self.data_dicts[patient][50001110],
                                                        self.data_dicts[patient][50001111],
                                                        ))
            self.data_dicts[patient][40000111].fillna(method="ffill",inplace=True)
            self.data_dicts[patient][40000111].fillna(method="bfill",inplace=True)
            self.data_dicts[patient][40000111]=self.activation_func(self.data_dicts[patient][40000111])
            pass

    def fuzzy_multiply(self):
        def judge_left_or_right(tri1,tri2):
            # 型エラーの修正
            if (type(tri1)==float) or (type(tri2)==float):
                return [np.nan,np.nan,np.nan,],[np.nan,np.nan,np.nan,]
            if (type(tri1)==str):
                tri1=eval(tri1)
            if (type(tri2)==str):
                tri2=eval(tri2)
            
            if tri1[1]<tri2[1]:
                return tri1,tri2
            else:
                return tri2,tri1
        def calc_cross_point(left_tri,right_tri):
            x_cross=1/((right_tri[1]-right_tri[0])+(left_tri[2]-left_tri[1]))*(left_tri[2]*(right_tri[1]-right_tri[0])+right_tri[0]*(left_tri[2]-left_tri[1]))
            return x_cross
        for patient in self.data_dicts.keys():
            # 三角形の左右判別
            sort_tri=np.array(list(map(judge_left_or_right,self.data_dicts[patient][40000000],self.data_dicts[patient][40000001])))
            print(sort_tri)
            left_tri,right_tri=sort_tri[:,0,:],sort_tri[:,1,:]
            
            # 交点の計算
            x_cross=np.array(list(map(calc_cross_point,left_tri,right_tri)))

            # 重心の算出
            x_gravity=(left_tri[:,2]+right_tri[:,0]+x_cross)/3
            print(x_gravity)
            self.data_dicts[patient][30000000]=x_gravity
        pass

    def AHP_weight_sum(self,input_node_codes,output_node_code):
        def weight_sum(row):
            features=np.nan_to_num(row[1].values)
            w_sum=self.AHP_dict[output_node_code]["weights"]@features
            for i,k in enumerate(input_node_codes):
                self.graph_dicts[patient]["weight_dict"][output_node_code][k]=self.AHP_dict[output_node_code]["weights"][i]
            return w_sum
        for patient in self.data_dicts.keys():
            self.data_dicts[patient][output_node_code]=list(map(weight_sum,self.data_dicts[patient][input_node_codes].iterrows()))
        pass

    def simple_weight_sum(self,input_node_codes,output_node_code,weights):
        def weight_sum(row):
            features=np.nan_to_num(row[1].values)
            w_sum=np.array(weights)@features
            for i,k in enumerate(input_node_codes):
                self.graph_dicts[patient]["weight_dict"][output_node_code][k]=weights[i]
            return w_sum
        for patient in self.data_dicts.keys():
            self.data_dicts[patient][output_node_code]=list(map(weight_sum,self.data_dicts[patient][input_node_codes].iterrows()))
    
    def fuzzy_reasoning_master(self,input_node_codes,output_node_code):
        def ask_risk_to_calculator(row):
            features=row[1].fillna(0)
            input_nodes=features.to_dict()
            risk=self.calculate_fuzzy(input_nodes=input_nodes,output_node=output_node_code)
            return risk
        for patient in self.data_dicts.keys():
            self.data_dicts[patient][output_node_code]=list(map(ask_risk_to_calculator,self.data_dicts[patient][input_node_codes].iterrows()))

    def ewm_master(self,input_node_codes,output_node_code,dim="t"):
        horizon=100
        if dim=="t":
            for patient in self.data_dicts.keys():
                for i,row in self.data_dicts[patient].iterrows():
                    score_df=self.data_dicts[patient].loc[np.max(i-horizon,0):i,input_node_codes]
                    weight=self.get_entropy_weight_t(score_df=score_df)
                    self.data_dicts[patient].loc[i,output_node_code]=np.array(list(weight.values()))@self.data_dicts[patient].loc[i,input_node_codes].values
                    if self.data_dicts[patient].loc[i,output_node_code]>1:
                        raise Exception("なんかおかしい")
                    # print(weight)
                    # if i>105 and patient=="B":
                    #     raise NotImplementedError
                    pass
        elif dim=="p": # patient間
            for i,_ in self.data_dicts[list(self.data_dicts.keys())[0]].iterrows():
                score_df=pd.DataFrame()
                for patient in self.data_dicts.keys():
                    score_df=pd.concat([score_df,self.data_dicts[patient].loc[i,input_node_codes]],axis=1)
                score_df=score_df.T
                weight=self.get_entropy_weight(score_df=score_df)
                self.data_dicts[patient].loc[i,output_node_code]=np.array(list(weight.values()))@self.data_dicts[patient].loc[i,input_node_codes].values
                if self.data_dicts[patient].loc[i,output_node_code]>1:
                    raise Exception("なんかおかしい")

            pass
    
    def save_session(self):
        # 重み情報をnetworkxのGに追記していく
        for patient in self.patients:
            for node in self.graph_dicts[patient]["node_dict"].keys():
                node_code_from=node
                node_codes_to=self.graph_dicts[patient]["node_dict"][node_code_from]["node_code_to"]
                self.graph_dicts[patient]["G"].add_edges_from([(node_code_from,node_code_to,{"weight":self.graph_dicts[patient]["weight_dict"][node_code_from][node_code_to]}) for node_code_to in node_codes_to])
        print("# graph保存 #")
        self.write_picklelog(self.graph_dicts,self.data_dir_dict["trial_dir_path"]+"/graph_dicts.pickle")
        self.write_picklelog(self.data_dicts,self.data_dir_dict["trial_dir_path"]+"/data_dicts.pickle")
        self.write_picklelog(self.scenario_dict,self.data_dir_dict["trial_dir_path"]+"/scenario_dict.pickle")
        for patient in self.patients:
            del self.graph_dicts[patient]["G"]
        self.write_json(self.graph_dicts,self.data_dir_dict["trial_dir_path"]+"/graph_dicts.json")
        print("# DataFrame 保存 #")
        for patient in self.patients:
            self.data_dicts[patient].to_csv(self.data_dir_dict["trial_dir_path"]+"/data_"+patient+"_eval.csv",index=False)
    
    def main(self):
        if self.throttling:
            print("# スロットリングのためのデータ間引きを実施 #")
            self.data_dicts=self.pseudo_throttling(self.data_dicts)

        if self.runtype!="basic_check":
            print("# 5 -> 4層推論 #")
            # 内的・静的
            self.fuzzy_logic()
            # 内的・動的
            self.pose_similarity()
            # 外的・静的
            self.object_risk()
            # 外的・動的
            self.staff_risk()

        print(self.data_dicts["A"])

        print("# 4 -> 3層推論 #")
        # 内定・静的
        self.fuzzy_multiply()
        # 内的・動的
        self.AHP_weight_sum(input_node_codes=[40000010,40000011,40000012,40000013,40000014,40000015,40000016],output_node_code=30000001)
        # 外的・静的
        self.AHP_weight_sum(input_node_codes=[40000100,40000101,40000102],output_node_code=30000010)
        # 外的・動的
        self.fuzzy_reasoning_master(input_node_codes=["40000110","40000111"],output_node_code="30000011")

        print("# 3 -> 2層推論 #")
        # 内的
        # self.ewm_master(input_node_codes=[30000000,30000001],output_node_code=20000000,dim="p")
        # self.fuzzy_reasoning_master(input_node_codes=[30000000,30000001],output_node_code=20000000)
        self.simple_weight_sum(input_node_codes=[30000000,30000001],output_node_code=20000000,weights=[0.1,0.9])
        # 外的
        # self.ewm_master(input_node_codes=[30000010,30000011],output_node_code=20000001)
        self.fuzzy_reasoning_master(input_node_codes=["30000010","30000011"],output_node_code="20000001")
        
        print("# 2 -> 1層推論 #")
        # self.ewm_master(input_node_codes=[20000000,20000001],output_node_code=10000000)
        self.fuzzy_reasoning_master(input_node_codes=["20000000","20000001"],output_node_code="10000000")
        pass

if __name__=="__main__":
    staff_name="中村"
    trial_name=f"20251116{staff_name}_DevFuzzyCustomize"
    strage="NASK"
    runtype="basic_check"
    cls=Master(trial_name,strage,runtype=runtype,staff_name=staff_name)
    cls.main()
    cls.save_session()