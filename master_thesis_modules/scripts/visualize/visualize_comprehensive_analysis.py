import os
import sys
from glob import glob
sys.path.append(".")
sys.path.append("..")
sys.path.append(os.path.expanduser("~")+"/kazu_ws/master_thesis/master_thesis_modules")
from scripts.management.manager import Manager

import numpy as np
import pandas as pd

from multiprocessing import cpu_count,Process


class Visualizer(Manager):
    def __init__(self,simulation_name,strage):
        self.simulation_name=simulation_name
        self.simulation_dir_path=self.get_database_dir(self.simulation_name,strage="NASK")["trial_dir_path"]
        self.simulation_common_dir_path=self.simulation_dir_path+"/common"
        os.makedirs(self.simulation_common_dir_path,exist_ok=True)
        self.strage=strage
        self.score_dict={}
        pass

    def check_standing_detection(self,trial_dir_path):
        trial_name=os.path.basename(trial_dir_path)
        print(trial_name)
        csv_paths=sorted(glob(trial_dir_path+"/data_*_raw.csv"))
        self.score_dict[trial_name]={}
        for csv_path in csv_paths:
            all_data=pd.read_csv(csv_path,header=0)
            trial_no=os.path.basename(trial_dir_path)
            patient=os.path.basename(csv_path).split("_")[1]
            # 立ち上がりの部分
            start_timestamp=2 # 患者が立ち始める
            end_timestamp=5   # 看護師が動き始める
            data=all_data[(all_data["timestamp"]>=start_timestamp) & (all_data["timestamp"]<=end_timestamp)]
            score=data["10000000"].mean()
            self.score_dict[trial_name]["risk25_"+patient]=score
            # 看護師対応中の部分
            start_timestamp=7 # 看護師が到着
            end_timestamp=9   # 看護師の対応終了
            data=all_data[(all_data["timestamp"]>=start_timestamp) & (all_data["timestamp"]<=end_timestamp)]
            score=data["10000000"].mean()
            self.score_dict[trial_name]["risk79_"+patient]=score
        try:
            values_25=[self.score_dict[trial_name][f"risk25_{p}"] for p in ["A","B","C"]]
            values_79=[self.score_dict[trial_name][f"risk79_{p}"] for p in ["A","B","C"]]
        except KeyError:
            print(self.score_dict[trial_name].keys())
            print(trial_dir_path)
            raise KeyError("患者の名前が見つからない")
        self.score_dict[trial_name]["risk25_max"]=["A","B","C"][np.argmax(values_25)]
        self.score_dict[trial_name]["risk79_max"]=["A","B","C"][np.argmax(values_79)]
        # self.write_csvlog([trial_name,])
        pass

    def main(self):
        nprocess=cpu_count()
        p_list=[]
        trial_dir_paths=[path for path in sorted(glob(self.simulation_dir_path+"/*")) if f"common" not in os.path.basename(path)]
        print(trial_dir_paths)
        for i,trial_dir_path in enumerate(trial_dir_paths):
            # self.check_standing_detection(trial_dir_path)
            p=Process(target=self.check_standing_detection,args=(trial_dir_path,))
            p_list.append(p)
            if len(p_list)==nprocess or i+1==len(trial_dir_paths):
                for p in p_list:
                    p.start()
                for p in p_list:
                    p.join()
                p_list=[]
        self.write_json(self.score_dict,json_path=self.simulation_common_dir_path+"/standing.json")

if __name__=="__main__":
    simulation_name="20250104SimulationPosition"
    strage="NASK"
    cls=Visualizer(simulation_name=simulation_name,strage=strage)
    cls.main()
