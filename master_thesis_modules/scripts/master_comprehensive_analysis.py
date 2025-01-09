import os
import sys
from pprint import pprint

from multiprocessing import cpu_count,Process

import copy
import numpy as np
import pandas as pd

sys.path.append(".")
sys.path.append("..")
sys.path.append(os.path.expanduser("~")+"/kazu_ws/master_thesis/master_thesis_modules")
from scripts.master_v2 import Master

class ComprehensiveAnalysis():
    def __init__(self):
        self.dict_format={
            "general_dict":{
                "start_timestamp":0,
                "end_timestamp":10,
                "fps":20,
                "patients":["A","B","C"],
                "xrange":[0,6],
                "yrange":[0,6],
            },
            "zokusei_dict":{
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
                },
            },
            "position_dict":{
                "A":(np.nan,np.nan),
                "B":(np.nan,np.nan),
                "C":(np.nan,np.nan),
                "NS":(np.nan,np.nan),
            },
            "action_dict":{
                "A":{
                },
                "B":{
                },
                "C":{
                },
            },
            "surrounding_objects":{
                "A":["wheelchair","ivPole",],
                "B":["wheelchair",],
                "C":[],
            },
        }
        pass

    def get_stand_scenario(self,standup_duration=1):
        action_dict={
            0:{
                "label":"sit",
                "start_timestamp":0,
                "end_timestamp":2,
            },
            2:{
                "label":"standup",
                "start_timestamp":2,
                "end_timestamp":2+standup_duration,
            },
            2+standup_duration:{
                "label":"stand",
                "start_timestamp":2+standup_duration,
                "end_timestamp":6
            },
            6:{
                "label":"sitdown",
                "start_timestamp":6,
                "end_timestamp":6+standup_duration
            },
            6+standup_duration:{
                "label":"sitdown",
                "start_timestamp":6+standup_duration,
                "end_timestamp":10
            },
        }
        return action_dict

    def get_sit_scenario(self):
        action_dict={
            0:{
                "label":"sit",
                "start_timestamp":0,
                "end_timestamp":10
            }
        }
        return action_dict

    def get_nurse_scenario(self,standup_patient):
        action_dict={
                    0:{
                        "label":"work",
                        "start_timestamp":0,
                        "end_timestamp":5,
                    },
                    5:{
                        "label":f"approach_{standup_patient}",
                        "start_timestamp":5,
                        "end_timestamp":7,
                        },
                    7:{
                        "label":f"work_{standup_patient}",
                        "start_timestamp":7,
                        "end_timestamp":9,
                    },
                    9:{
                        "label":f"leave_{standup_patient}",
                        "start_timestamp":9,
                        "end_timestamp":10,
                        }
                }
        return action_dict

    def generate_condition_dicts(self,simulation_name):
        patients=["A","B","C"]
        positions={
            "x":[1,3,5],
            "y":[1,3,5],
        }
        standups=[
            ["A"],
            # ["A","B"],
            # ["A","B","C"],
            ]
        standup_duration_list=[1]#,0.5,3]
        trial_no=0
        condition_dicts={}
        for x_A in positions["x"]:
            for y_A in positions["y"]:
                for x_B in positions["x"]:
                    for y_B in positions["y"]:
                        if (x_A==x_B) & (y_A==y_B):
                            continue
                        for x_C in positions["x"]:
                            for y_C in positions["y"]:
                                if ((x_A==x_C) & (y_A==y_C)) or ((x_B==x_C) & (y_B==y_C)):
                                    continue
                                for x_nurse in positions["x"]:
                                    for y_nurse in positions["y"]:
                                        if ((x_A==x_nurse) & (y_A==y_nurse)) or ((x_B==x_nurse) & (y_B==y_nurse)) or ((x_C==x_nurse) & (y_C==y_nurse)):
                                            continue
                                        for standup_list in standups:#実質1パターン
                                            for standup_duration in standup_duration_list:#実質1パターン
                                                condition_dict=copy.deepcopy(self.dict_format)
                                                # 位置
                                                condition_dict["position_dict"]["A"]=(x_A,y_A)
                                                condition_dict["position_dict"]["B"]=(x_B,y_B)
                                                condition_dict["position_dict"]["C"]=(x_C,y_C)
                                                condition_dict["position_dict"]["NS"]=(x_nurse,y_nurse)
                                                # 立ち座り
                                                for patient in patients:
                                                    if patient in standup_list:
                                                        # 立つシナリオを挿入
                                                        condition_dict["action_dict"][patient]=self.get_stand_scenario(standup_duration=standup_duration)
                                                    else:
                                                        # 座りっぱなしのシナリオを挿入
                                                        condition_dict["action_dict"][patient]=self.get_sit_scenario()
                                                condition_dict["action_dict"]["NS"]=self.get_nurse_scenario(standup_patient=standup_list[0])
                                                trila_name=f"{simulation_name}/"+f"no_{str(trial_no).zfill(5)}"
                                                condition_dicts[trila_name]=condition_dict
                                                trial_no+=1
        return condition_dicts
    
    def sample_process(self,text):
        print(text)
        return text
    def check_multiprocessor(self):
        nprocess=cpu_count()
        texts=np.arange(1000)
        p_list=[]
        for i,text in enumerate(texts):
            p=Process(target=self.sample_process,args=(text,))
            p_list.append(p)
            if len(p_list)==nprocess or i+1==len(texts):
                for p in p_list:
                    p.start()
                for p in p_list:
                    p.join()
                p_list=[]
        pass

    def main(self,trial_name,strage,condition_dict,runtype):       
        cls_master=Master(trial_name=trial_name,strage=strage,scenario_dict=condition_dict,runtype=runtype)
        cls_master.main()
        cls_master.save_session()

    def comprehensive_analysis_main(self):
        simulation_name="20250108SimulationPosition"
        strage="NASK"
        runtype="simulation"
        condition_dicts=self.generate_condition_dicts(simulation_name)
        nprocess=cpu_count()
        p_list=[]
        
        for i,(trial_name,condition_dict) in enumerate(condition_dicts.items()):
            p=Process(target=self.main,args=(trial_name,strage,condition_dict,runtype))
            p_list.append(p)
            if len(p_list)==nprocess or i+1==len(condition_dicts):
                for p in p_list:
                    p.start()
                for p in p_list:
                    p.join()
                p_list=[]
            
            # cls_master=Master(trial_name=trial_name,strage=strage,scenario_dict=condition_dict)
            # cls_master.main()
            # cls_master.save_session()



            # del cls_master

if __name__=="__main__":
    cls=ComprehensiveAnalysis()
    # cls.check_multiprocessor()
    cls.comprehensive_analysis_main()