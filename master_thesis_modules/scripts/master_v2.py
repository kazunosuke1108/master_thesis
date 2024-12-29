import os
import sys
import copy
import dill

from icecream import ic
from pprint import pprint

import numpy as np
import pandas as pd

sys.path.append(".")
sys.path.append("..")
sys.path.append(os.path.expanduser("~")+"/kazu_ws/master_thesis/master_thesis_modules")
from scripts.management.manager import Manager
from scripts.network.graph_manager import GraphManager
from scripts.pseudo_data.pseudo_data_generator import PseudoDataGenerator

class Master(Manager,GraphManager):
    def __init__(self,trial_name,strage):
        super().__init__()
        print("# Master開始 #")
        self.trial_name=trial_name
        self.strage=strage
        self.data_dir_dict=self.get_database_dir(self.trial_name,self.strage)
        self.patients=["A","B","C"]
        print("# default graph 定義 #")
        default_graph=self.get_default_graph()
        
        # 人数分のgraphを定義
        print("# 人数分のgraph 定義 #")
        self.graph_dicts={}
        for patient in self.patients:
            self.graph_dicts[patient]=copy.deepcopy(default_graph)

        # シナリオの定義・DataFrameの生成
        self.define_scenario()
        

        # DataFrameを定義
        print("# DataFrame 定義 #")
        # self.data_dicts={}
        # for patient in self.patients:
        #     self.data_dicts[patient]="hoge"

    def define_scenario(self):
        start_timestamp=0
        end_timestamp=10
        fps=20
        patients=list(self.graph_dicts.keys())
        general_dict={
            "start_timestamp":start_timestamp,
            "end_timestamp":end_timestamp,
            "fps":fps,
            "patients":patients,
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
            "D":{
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
                    "end_timestamp":2,
                },
                2:{
                    "label":"start_B",
                    "start_timestamp":2,
                    "end_timestamp":8,
                    },
                8:{
                    "label":"end_B",
                    "start_timestamp":8,
                    "end_timestamp":end_timestamp,
                    }
            }
        }
        surrounding_objects={
            "A":["wheelchair"],
            "B":["wheelchair","ivPole"],
            "C":[],
        }

        cls_PDG=PseudoDataGenerator(trial_name=self.trial_name,strage=self.strage)
        cls_PDG.get_pseudo_data(
            graph_dicts=self.graph_dicts,
            general_dict=general_dict,
            zokusei_dict=zokusei_dict,
            position_dict=position_dict,
            action_dict=action_dict,
            surrounding_objects=surrounding_objects,
        )
        
    
    def save_session(self):
        print("# graph保存 #")
        self.write_picklelog(self.graph_dicts,self.data_dir_dict["trial_dir_path"]+"/graph_dicts.pickle")
        for patient in self.patients:
            del self.graph_dicts[patient]["G"]
        self.write_json(self.graph_dicts,self.data_dir_dict["trial_dir_path"]+"/graph_dicts.json")


    def main(self):
        pass

if __name__=="__main__":
    trial_name="20241229BuildSimulator"
    strage="NASK"
    cls=Master(trial_name,strage)
    cls.main()
    cls.save_session()