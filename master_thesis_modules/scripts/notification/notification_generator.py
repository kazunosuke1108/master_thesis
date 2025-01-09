import os
import sys
sys.path.append(".")
sys.path.append("..")
if "catkin_ws" in os.getcwd():
    sys.path.append("/catkin_ws/src/master_thesis_modules")
else:
    sys.path.append(os.path.expanduser("~")+"/kazu_ws/master_thesis/master_thesis_modules")

from glob import glob
import pandas as pd
import numpy as np

from scripts.management.manager import Manager
from scripts.notification.notification import Notification

class NotificationGenerator(Manager):
    def __init__(self,trial_name,strage):
        super().__init__()
        self.trial_name=trial_name
        self.strage=strage
        self.data_dir_dict=self.get_database_dir(trial_name=self.trial_name,strage=self.strage)


    def main(self):
        # csvデータの読み込み
        csv_paths=sorted(glob("/media/hayashide/MasterThesis/20250108DevMewThrottlingExp/data_*_eval.csv"))
        data_dicts={}
        id_names=[]
        for csv_path in csv_paths:
            data=pd.read_csv(csv_path,header=0)
            id_name=os.path.basename(csv_path)[len("data_"):-len("_eval.csv")]
            id_names.append(id_name)
            data_dicts[id_name]=data

        # ランキングのデータを作成
        df_rank=pd.DataFrame(data_dicts[id_name]["timestamp"].values,columns=["timestamp"])
        for i in range(len(id_names)):
            df_rank[i]=np.nan
        
        temp_df=pd.DataFrame(data["timestamp"].values,columns=["timestamp"])
        for id_name in id_names:
            temp_df[id_name]=data_dicts[id_name]["10000000"].values
        
        for i,row in temp_df.iterrows():
            rank=np.argsort(-row.values[1:])
            # print(id_names)
            # print(row.values[1:])
            # print(rank)
            # print(np.array(id_names)[rank])
            df_rank.loc[i,list(np.arange(0,len(id_names)))]=np.array(id_names)[rank]
            # raise NotImplementedError
        print(df_rank)
            
            
        # csv_data=pd.read_csv(csv_path,header=0)
        # focus_keys=list(csv_data.keys())
        # focus_keys.remove("timestamp")
        # focus_keys=[k for k in focus_keys if int(k)<50000000]
        # focus_keys.remove("40000000")
        # focus_keys.remove("40000001")
        # diff_data=csv_data[focus_keys].diff()

        # 通知を行う行の指定
        # notify_index=300
        # roi_series=csv_data.loc[notify_index,:]
        # # roi_series=diff_data.loc[notify_index,:]
        # print(roi_series)

        # 通知内容の作成

        # 音声ファイルの発行
        pass

if __name__=="__main__":
    trial_name="20240109NotificationGenerator"
    strage="NASK"
    cls=NotificationGenerator(trial_name=trial_name,strage=strage)
    cls.main()
