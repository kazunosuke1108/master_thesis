import os
from glob import glob
from pprint import pprint
import json
import yaml
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

class Manager():
    def __init__(self):
        super().__init__()
        plt.rcParams["figure.figsize"] = (15,10)
        plt.rcParams["figure.autolayout"] = True
        plt.rcParams["font.size"] = 24
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
        plt.rcParams["legend.edgecolor"] = 'black' # edgeの色を変更
        plt.rcParams["legend.handlelength"] = 1 # 凡例の線の長さを調節
        pass

    def get_module_path(self):
        if os.name == "nt": # Windows
            home = os.path.expanduser("~")
        else: # ubuntu
            home=os.environ['HOME']        
        # print("HOME: "+home)
        
        workspace_dir_name="kazu_ws"
        module_name="master_thesis_modules"
        module_dir_path=home+"/"+workspace_dir_name+"/"+module_name[:-len("_modules")]+"/"+module_name
        if os.path.isdir(module_dir_path):
            pass
        else:#dockerのとき
            module_dir_path=home+"/"+"catkin_ws/src"+"/"+module_name
            if not os.path.isdir(module_dir_path):
                raise FileNotFoundError("module directory not found: "+module_dir_path)
        
        return module_dir_path

    def get_database_dir(self,strage="NASK"):
        module_dir_path=self.get_module_path()

        if (strage=="NASK") or (strage=="nask"):
            if os.name=="nt": # Windows
                database_dir_path="//192.168.1.5/common/FY2024/01_M2/05_hayashide/MasterThesis_database"
                pass
            else: # Ubuntu
                # raise NotImplementedError("マウント処理を実装してください")
                if "catkin_ws" in module_dir_path: # docker
                    database_dir_path="/media/hayashide/MasterThesis"
                    pass
                else: # out of docker
                    database_dir_path="/media/hayashide/MasterThesis"
                    pass
        elif strage=="local":
            database_dir_path=module_dir_path+"/database"


        database_dir_dict={
            "module_dir_path":module_dir_path,
            "database_dir_path":database_dir_path,
        }

        for path in database_dir_dict.values():
            os.makedirs(path,exist_ok=True)
        
        return database_dir_dict

    def write_csvlog(self,output_data,csvpath,fmt="%s",dim=1):
        if dim==1:
            output_data=[output_data]
        else:
            pass
        try:
            with open(csvpath, 'a') as f_handle:
                np.savetxt(f_handle,output_data,delimiter=",")
        except TypeError:
            with open(csvpath, 'a') as f_handle:
                np.savetxt(f_handle,output_data,delimiter=",",fmt=fmt)    
        except FileNotFoundError:
            np.savetxt(csvpath,output_data,delimiter=",")
        pass  

    def write_json(self,dict_data,json_path):
        with open(json_path,mode="w") as f:
            json.dump(dict_data,f)

    def load_json(self,json_path):
        with open(json_path,encoding="utf-8") as f:
            data=json.load(f)
        return data

    def write_picklelog(self,output_dict,picklepath):
        with open(picklepath, mode='wb') as f:
            pickle.dump(output_dict,f)

    def load_picklelog(self,picklepath):
        with open(picklepath,mode="rb") as f:
            data=pickle.load(f)
        return data        
    
    def write_yaml(self,dict_data,yaml_path):
        with open(yaml_path,mode="w") as f:
            yaml.dump(dict_data,f)
    
    def load_yaml(self,yaml_path):
        with open(yaml_path,mode="r") as f:
            data=yaml.safe_load(f)
        return data
    
    def get_timestamp(self):
        import datetime
        current_time=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return current_time


if __name__=="__main__":
    cls=Manager()
    cls.get_database_dir("NASK")