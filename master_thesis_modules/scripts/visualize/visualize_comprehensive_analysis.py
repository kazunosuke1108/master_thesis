import os
import sys
from glob import glob
sys.path.append(".")
sys.path.append("..")
sys.path.append(os.path.expanduser("~")+"/kazu_ws/master_thesis/master_thesis_modules")
from scripts.management.manager import Manager

import numpy as np
import pandas as pd

class Visualizer(Manager):
    def __init__(self,simulation_dir_path,strage):
        self.simulation_dir_path=self.get_database_dir(simulation_dir_path)["trial_dir_path"]
        self.strage=strage
        self.score_dict={}
        pass

    def check_standing_detection(self,trial_dir_path):
        trial_name=os.path.basename(trial_dir_path)
        csv_paths=sorted(glob(trial_dir_path+"/data_*_raw.csv"))
        start_timestamp=2 # 患者が立ち始める
        end_timestamp=5   # 看護師が動き始める
        self.score_dict[trial_name]={}
        for csv_path in csv_paths:
            all_data=pd.read_csv(csv_path,header=0)
            trial_no=os.path.basename(trial_dir_path)
            patient=os.path.basename(csv_path).split("_")[1]
            data=all_data[(all_data["timestamp"]>=start_timestamp) & (all_data["timestamp"]<=end_timestamp)]
            score=data["10000000"].mean()
            self.score_dict[trial_name]["risk25_"+patient]=score
        try:
            values_25=[self.score_dict[trial_name][f"risk25_{p}"] for p in ["A","B","C"]]
        except KeyError:
            print(self.score_dict[trial_name].keys())
        print(values_25)
        self.score_dict[trial_name]["risk25_max"]=["A","B","C"][np.argmax(values_25)]
        print(trial_no,self.score_dict[trial_name]["risk25_max"])
            
            


        pass

    def main(self):
        trial_dir_paths=sorted(glob(self.simulation_dir_path+"/*"))
        try:
            trial_dir_paths.remove(f"{self.simulation_dir_path}/common")
        except Exception:
            pass
        for trial_dir_path in trial_dir_paths:
            self.check_standing_detection(trial_dir_path)
        pass

if __name__=="__main__":
    simulation_dir_path="20250104SimulationPosition"
    strage="NASK"
    cls=Visualizer(simulation_dir_path=simulation_dir_path,strage=strage)
    cls.main()
