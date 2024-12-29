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

        # DataFrameを定義
        print("# DataFrame 定義 #")
        
    
    def save_session(self):
        print("# session保存 #")
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