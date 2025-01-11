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
from scripts.network.graph_manager import GraphManager
from scripts.notification.notification import Notification

class NotificationGenerator(Manager,GraphManager):
    def __init__(self,trial_name,strage,result_trial_name):
        super().__init__()
        self.trial_name=trial_name
        self.strage=strage
        self.result_trial_name=result_trial_name
        self.data_dir_dict=self.get_database_dir(trial_name=self.trial_name,strage=self.strage)

        self.w_average=40 # 平滑化のwindow
        self.increase_ratio_min=0 # 危険順位が1位に躍り出た患者の，危険度の上昇具合
        
        # csvデータの読み込み
        csv_paths=sorted(glob(f"{self.data_dir_dict['database_dir_path']}/{self.result_trial_name}/data_*_eval.csv"))
        self.data_dicts={}
        self.id_names=[]
        
        for csv_path in csv_paths:
            data=pd.read_csv(csv_path,header=0)
            for k in data.keys():
                if k =="timestamp":
                    continue
                try:
                    data[k]=data[k].rolling(self.w_average).mean()
                    data[k].fillna(method="bfill",inplace=True)
                except pd.errors.DataError:
                    pass
            id_name=os.path.basename(csv_path)[len("data_"):-len("_eval.csv")]
            self.id_names.append(id_name)
            self.data_dicts[id_name]=data
        
        self.default_graph=self.get_default_graph()

    def get_df_rank(self,data_dicts):
        # ランキングのデータを作成
        df_rank=pd.DataFrame(data_dicts[list(data_dicts.keys())[0]]["timestamp"].values,columns=["timestamp"])
        for i in range(len(self.id_names)):
            df_rank[i]=np.nan
        
        temp_df=pd.DataFrame(data_dicts[list(data_dicts.keys())[0]]["timestamp"].values,columns=["timestamp"])
        for id_name in self.id_names:
            temp_df[id_name]=data_dicts[id_name]["10000000"].values
        
        for i,row in temp_df.iterrows():
            rank=np.argsort(-row.values[1:])
            df_rank.loc[i,list(np.arange(0,len(self.id_names)))]=np.array(self.id_names)[rank]
        return df_rank
    
    def guess_dynamic_factor(self,data):
        # 当該時刻のデータ
        data["40000000"]=[eval(v)[1] for v in data["40000000"].values]
        data["40000001"]=[eval(v)[1] for v in data["40000001"].values]
        data.drop("timestamp",inplace=True,axis=1)
        data.drop([k for k in data.keys() if int(k)>=50000000],inplace=True,axis=1)
        data_corr=data.corr()["10000000"]
        # 相関が高い因子を抜き出す
        high_corr_nodes=[k for k in data_corr.keys() if ((data_corr[k]>0.8) and (k!="timestamp") and (k!="10000000"))]
        # for high_corr_node in high_corr_nodes:
        #     print(data_corr[high_corr_node],self.default_graph["node_dict"][int(high_corr_node)]["description_en"])
        # 一番相関が高い4000番台の因子を抜き出す
        data_corr_4000=data_corr[[k for k in data_corr.keys() if k[0]=="4"]]
        most_corr_node=list(data_corr_4000.keys())[data_corr_4000.argmax()]
        return most_corr_node,high_corr_nodes

    def guess_static_factor(self,data_dicts,most_risky_patient):
        focus_keys=[]
        for k in data_dicts[list(data_dicts.keys())[0]].keys():
            if k=="timestamp":
                continue
            elif (int(k[0])>=5) or (int(k[0])<=3):
                continue
            else:
                focus_keys.append(k)

        average_df=pd.DataFrame(index=focus_keys)
        # 4000番台の各項目について，患者間比較用の代表値を算出
        for patient in data_dicts.keys():
            for node_code in focus_keys:
                if node_code in ["40000000","40000001"]:
                    average_df.loc[node_code,patient]=np.mean([eval(v)[1] for v in data_dicts[patient][node_code].values])
                else:
                    average_df.loc[node_code,patient]=data_dicts[patient][node_code].mean()
        average_df["risky"]=average_df.idxmax(axis=1)
        average_df["significance"]=np.nan
        average_df["node_type"]=[self.default_graph["node_dict"][int(idx)]["node_type"] for idx in list(average_df.index)]
        for i,row in average_df.iterrows():
            patients=list(data_dicts.keys())
            total=row[patients].sum()
            others=total-row[row["risky"]]
            significance=abs(row[row["risky"]]-others/(len(patients)-1))
            average_df.loc[i,"significance"]=significance

        factor_df=average_df[(average_df["risky"]==most_risky_patient)].sort_values("significance")
        static_factor_df=factor_df[factor_df["node_type"]=="static"]
        static_factor_nodes=static_factor_df.index[static_factor_df["significance"]==static_factor_df["significance"].max()].tolist()
        if len(static_factor_nodes)>0:
            static_factor_node=static_factor_nodes[0]
        else:
            static_factor_node=""
        
        return static_factor_node,factor_df
        
    def get_alert_sentence(self,most_risky_patient,static_factor_node,dynamic_factor_node):
        if static_factor_node=="":
            text_dynamic=self.default_graph["node_dict"][int(dynamic_factor_node)]["description_ja"]
            alert_text=f"{most_risky_patient}さんが，{text_dynamic}ので，危険です．"
        else:
            text_static=self.default_graph["node_dict"][int(static_factor_node)]["description_ja"]
            text_dynamic=self.default_graph["node_dict"][int(dynamic_factor_node)]["description_ja"]
            alert_text=f"{most_risky_patient}さんが，元々{text_static}のに，{text_dynamic}ので，危険です．"
        return alert_text

    def main(self):
        # ランキングの作成
        self.df_rank=self.get_df_rank(data_dicts=self.data_dicts)

        # ランキングの入れ替わりを検知
        notification_history=pd.DataFrame(columns=["timestamp","sentence"])
        timestamp_list=[]
        sentence_list=[]
        increase_ratio_list=[]
        most_risky_patient=""
        for i,row in self.df_rank.iterrows():
            if i==0:
                most_risky_patient=self.df_rank.loc[i,0]
                continue
            if self.df_rank.loc[i,0]!=most_risky_patient:
                most_risky_patient=self.df_rank.loc[i,0]
                # 当該時刻における，危険度が上昇した患者のデータを準備
                w_roi=20
                data_of_risky_patient=self.data_dicts[self.df_rank.loc[i,0]].loc[i-w_roi:i+w_roi,:]
                data_dict_roi={k:v.loc[i-w_roi:i+w_roi,:] for k,v in self.data_dicts.items()}

                # 追い抜いた側が上昇したことによる入れ替わりか，追い抜かれた側が下降したことによる入れ替わりか，判別
                increase_ratio=data_of_risky_patient.corr().loc["timestamp","10000000"]
                print(increase_ratio)
                if increase_ratio>self.increase_ratio_min:
                    # ランキングの入れ替わりを起こすきっかけになった動的要因を探る（動作か周囲の人員配置．値の上昇が見られるものはどっちか考える）
                    dynamic_factor_node,_=self.guess_dynamic_factor(data_of_risky_patient)

                    # 元々の危険度を高める要因になっていた静的要因を探る（属性か周囲の物体）
                    static_factor_node,_=self.guess_static_factor(data_dict_roi,most_risky_patient)

                    alert_text=self.get_alert_sentence(most_risky_patient=most_risky_patient,dynamic_factor_node=dynamic_factor_node,static_factor_node=static_factor_node)

                    notification_mp3_path=self.data_dir_dict["trial_dir_path"]+f"/notification_{str(i).zfill(3)}.mp3"
                    # Notification().export_audio(text=alert_text,mp3_path=notification_mp3_path,chime_type=1)
                    
                    print("ranking changed")
                    print(self.df_rank.loc[i,"timestamp"])
                    print(self.df_rank.loc[i-1,0],"->",self.df_rank.loc[i,0])
                    print(dynamic_factor_node)
                    print(static_factor_node)
                    print(alert_text)
                    increase_ratio_list.append(increase_ratio)
                    timestamp_list.append(self.df_rank.loc[i,"timestamp"]-self.df_rank.loc[0,"timestamp"])
                    sentence_list.append(alert_text)
        notification_history["timestamp"]=timestamp_list
        notification_history["sentence"]=sentence_list
        notification_history["increase_ratio"]=increase_ratio_list

        print(notification_history)


            
        pass

if __name__=="__main__":
    # trial_name="20250111NotificationGenerator"
    # result_trial_name="20250108SimulationThrottlingTrue"
    trial_name="20250110NotificationGeneratorExp"
    result_trial_name="20250108DevMewThrottlingExp"
    strage="NASK"

    cls=NotificationGenerator(trial_name=trial_name,strage=strage,result_trial_name=result_trial_name)
    cls.main()
