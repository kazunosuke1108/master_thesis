import os
import sys
sys.path.append(".")
sys.path.append("..")
if "catkin_ws" in os.getcwd():
    sys.path.append("/catkin_ws/src/master_thesis_modules")
else:
    sys.path.append(os.path.expanduser("~")+"/kazu_ws/master_thesis/master_thesis_modules")

from icecream import ic
from pprint import pprint
from glob import glob

import numpy as np
import pandas as pd
import plotly.subplots as sp
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


from scripts.management.manager import Manager

class Visualizer(Manager):
    def __init__(self,trial_name,strage):
        super().__init__()
        self.trial_name=trial_name
        self.strage=strage
        self.data_dir_dict=self.get_database_dir(trial_name=trial_name,strage=strage)

    def delay_by_fps_control(self,trial_name_false,trial_name_true):
        self.data_dir_dict_false=self.get_database_dir(trial_name=trial_name_false,strage=strage)
        self.data_dir_dict_true=self.get_database_dir(trial_name=trial_name_true,strage=strage)
        delay_data=pd.DataFrame(columns=["timestamp_false","priority_order",])
        # Falseのときの切り替わりタイミング
        # Trueのときの切り替わりタイミング
        # 遅延を計算
        # latex形式でprint
        pass

if __name__=="__main__":
    # trial_name="20250113NormalSimulation"
    # trial_name="20250110SimulationMultipleRisks/no_00005"
    # trial_name="20250120FPScontrolFalse"
    trial_name="20250120FPScontrolTrue"
    # trial_name="20250108DevMewThrottlingExp"
    # trial_name="20250115PullWheelchairObaachan2"
    # trial_name="20250121ChangeCriteriaBefore"
    strage="NASK"
    cls=Visualizer(trial_name=trial_name,strage=strage)
    pass