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
        plt.rcParams["figure.figsize"] = (8/2.54,6/2.54)
        plt.rcParams["figure.autolayout"] = True
        plt.rcParams["font.size"] = 11
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
        plt.rcParams["legend.edgecolor"] = 'black' # edgeの色を変更
        plt.rcParams["legend.handlelength"] = 1 # 凡例の線の長さを調節
        pass

    def prepare_log(self,trial_dir_path):
        import os
        from datetime import datetime
        import logging

        logdir=trial_dir_path
        

        logger = logging.getLogger(os.path.basename(__file__))
        logger.setLevel(logging.DEBUG)
        format = "%(asctime)s [%(filename)s:%(funcName)s:%(lineno)d] %(levelname)-9s  %(message)s"
        st_handler = logging.StreamHandler()
        st_handler.setLevel(logging.WARNING)
        # StreamHandlerによる出力フォーマットを先で定義した'format'に設定
        st_handler.setFormatter(logging.Formatter(format))

        fl_handler = logging.FileHandler(filename=logdir+"/"+datetime.now().strftime('%Y%m%d_%H%M%S')+".log", encoding="utf-8")
        fl_handler.setLevel(logging.DEBUG)
        # FileHandlerによる出力フォーマットを先で定義した'format'に設定
        fl_handler.setFormatter(logging.Formatter(format))

        logger.addHandler(st_handler)
        logger.addHandler(fl_handler)
        return logger

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
                module_dir_path="/"+"catkin_ws/src"+"/"+module_name
            if not os.path.isdir(module_dir_path):
                raise FileNotFoundError("module directory not found: "+module_dir_path)
            
        
        return module_dir_path

    def get_database_dir(self,trial_name="NoTrialNameGiven",strage="NASK"):
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

        mobilesensing_dir_path=database_dir_path.replace("MasterThesis_database","MobileSensing")
        mobilesensing_dir_path=f"/catkin_ws/src/database/{trial_name}"

        if "/" in trial_name:
            os.makedirs(database_dir_path+"/"+trial_name.split("/")[0],exist_ok=True)
        trial_dir_path=database_dir_path+"/"+trial_name
        common_dir_path=database_dir_path+"/common"

        database_dir_dict={
            "mobilesensing_dir_path":mobilesensing_dir_path,
            "module_dir_path":module_dir_path,
            "database_dir_path":database_dir_path,
            "trial_dir_path":trial_dir_path,
            "common_dir_path":common_dir_path,
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
            try:
                data=pickle.load(f)
            except ModuleNotFoundError:
                # python2系列で書かれた場合
                data=pickle.load(f,fix_imports=True)
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
    
    def jpg2mp4(self,image_paths,mp4_path,size=(0,0),fps=30.0):
        import cv2
        # get size of the image
        img=cv2.imread(image_paths[0])
        if size[0]==0:
            size=(img.shape[1],img.shape[0])
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        print("mp4_path",mp4_path)
        print("fps",fps)
        print("size",size)
        video = cv2.VideoWriter(mp4_path,fourcc,fps,size)#(mp4_path,fourcc, fps, size)
        for idx,image in enumerate(image_paths):
            img=cv2.imread(image)
            video.write(img)
            print(f"now processing: {os.path.basename(image)} {idx}/{len(image_paths)}")
        video.release()

    def putText_japanese(img, text, point, size, color):
        from PIL import ImageFont, ImageDraw, Image

        #Notoフォントとする
        font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', size)

        #imgをndarrayからPILに変換
        img_pil = Image.fromarray(img)

        #drawインスタンス生成
        draw = ImageDraw.Draw(img_pil)

        #テキスト描画
        draw.text(point, text, fill=color, font=font)

        #PILからndarrayに変換して返す
        return np.array(img_pil)
    
    def flattern_dict(self,d):
        d_flatten={}
        for p in d.keys():
            for k in d[p].keys():
                if type(d[p][k]) in [list,tuple]:
                    d_flatten[f"{p}_{k}"]=str(d[p][k])
                else:
                    d_flatten[f"{p}_{k}"]=d[p][k]
        return d_flatten

if __name__=="__main__":
    cls=Manager()
    cls.get_database_dir("NASK")