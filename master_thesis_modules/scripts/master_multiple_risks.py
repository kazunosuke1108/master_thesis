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

    def get_stand_scenario(self,start_standup=2,standup_duration=1,stand_duration=4):
        action_dict={
            0:{
                "label":"sit",
                "start_timestamp":0,
                "end_timestamp":2,
            },
            start_standup:{
                "label":"standup",
                "start_timestamp":start_standup,
                "end_timestamp":start_standup+standup_duration,
            },
            start_standup+standup_duration:{
                "label":"stand",
                "start_timestamp":start_standup+standup_duration,
                "end_timestamp":start_standup+stand_duration
            },
            start_standup+stand_duration:{
                "label":"sitdown",
                "start_timestamp":start_standup+stand_duration,
                "end_timestamp":start_standup+stand_duration+standup_duration
            },
            start_standup+stand_duration+standup_duration:{
                "label":"sitdown",
                "start_timestamp":start_standup+stand_duration+standup_duration,
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
        trial_no=0
        condition_dicts={}

        # 立ち座り
        for standup_timing in ["same","different"]:
            if standup_timing=="same":
                # 同時に立ち上がる場合についての検証
                condition_dict=copy.deepcopy(self.dict_format)
                # 位置
                condition_dict["position_dict"]["A"]=(2,5)
                condition_dict["position_dict"]["B"]=(2,2)
                condition_dict["position_dict"]["C"]=(5,2)
                condition_dict["position_dict"]["NS"]=(5,5)
                # start_standup, standup_duration, stand_durationをそれぞれ定義
                start_standup=2
                standup_duration=1
                stand_duration=4
                # 立ち上がる組合せを定義
                standups=[
                    ["A"],
                    ["A","B"],
                    ["A","C"],
                    ["A","B","C"],
                ]
                everyone=["A","B","C"]
                for standup_people in standups:
                    # 各自にシナリオを入れていく
                    for person in everyone:
                        if person in standup_people:
                            condition_dict["action_dict"][person]=self.get_stand_scenario(start_standup=start_standup,standup_duration=standup_duration,stand_duration=stand_duration)
                        else:
                            condition_dict["action_dict"][person]=self.get_sit_scenario()
                    condition_dict["action_dict"]["NS"]=self.get_nurse_scenario(standup_patient=standup_people[0])
                    trila_name=f"{simulation_name}/"+f"no_{str(trial_no).zfill(5)}"
                    condition_dicts[trila_name]=condition_dict
                    trial_no+=1
                pass
            elif standup_timing=="different":
                # ばらばらに立ち上がる場合についての検証
                condition_dict=copy.deepcopy(self.dict_format)
                # 位置
                condition_dict["position_dict"]["A"]=(2,5)
                condition_dict["position_dict"]["B"]=(2,2)
                condition_dict["position_dict"]["C"]=(5,2)
                condition_dict["position_dict"]["NS"]=(5,5)

                # A-B-Cの順
                condition_dict["action_dict"]["A"]=self.get_stand_scenario(start_standup=2,standup_duration=1,stand_duration=4)
                condition_dict["action_dict"]["B"]=self.get_stand_scenario(start_standup=4,standup_duration=1,stand_duration=3)
                condition_dict["action_dict"]["C"]=self.get_stand_scenario(start_standup=6,standup_duration=1,stand_duration=2)
                condition_dict["action_dict"]["NS"]=self.get_nurse_scenario(standup_patient="A")
                trila_name=f"{simulation_name}/"+f"no_{str(trial_no).zfill(5)}"
                condition_dicts[trila_name]=condition_dict
                trial_no+=1

                # A-C-Bの順
                condition_dict["action_dict"]["A"]=self.get_stand_scenario(start_standup=2,standup_duration=1,stand_duration=4)
                condition_dict["action_dict"]["C"]=self.get_stand_scenario(start_standup=4,standup_duration=1,stand_duration=3)
                condition_dict["action_dict"]["B"]=self.get_stand_scenario(start_standup=6,standup_duration=1,stand_duration=2)
                condition_dict["action_dict"]["NS"]=self.get_nurse_scenario(standup_patient="A")
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
        simulation_name="20250109SimulationMultipleRisks"
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