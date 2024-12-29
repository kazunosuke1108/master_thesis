import os
import sys

import numpy as np
import pandas as pd

sys.path.append(".")
sys.path.append("..")
sys.path.append(os.path.expanduser("~")+"/kazu_ws/master_thesis/master_thesis_modules")
from scripts.management.manager import Manager

class PseudoDataGenerator(Manager):
    def __init__(self,trial_name,strage):
        super().__init__()
        self.trial_name=trial_name
        self.strage=strage
        self.data_dir_dict=self.get_database_dir(trial_name=self.trial_name,strage=self.strage)

        # DataFrameの定義
        self.t=np.arange(0,8.001,0.25)
        data=pd.DataFrame(self.t,columns=["timestamp"])

        columns=[1000,2000,2001,2002,3000,3001,3002,3003,3004,3005,4050,4051,4052,5510,5511,5512,5520,5521,5522]
        columns=columns+["active","fps"]
        for column in columns:
            data[column]=np.nan
        data["active"]=0
        data.loc[0,"active"]=1
        data.loc[0,"fps"]=1
        
        # 無視するデータ群を指定
        killed_columns=[2001,2002,3000,3001,3003,3004,4050,5521,5522]
        for killed_column in killed_columns:
            data[killed_column]=1e-5
        
        self.data_dict={
            "A":data.copy(),#A
            "B":data.copy(),#B
            "C":data.copy(),#C
        }        


    def main(self):
        
        pass
    

if __name__=="__main__":
    trial_name="BuildSimulator20241229"
    strage="NASK"
    cls=PseudoDataGenerator(trial_name=trial_name,strage=strage)