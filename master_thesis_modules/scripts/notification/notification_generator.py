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
import matplotlib.pyplot as plt

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
        self.increase_ratio_min=0.8 # 危険順位が1位に躍り出た患者の，危険度の上昇具合
        self.decrease_ratio_max=0 # 危険順位が2位になった患者の，危険度の下降具合
        self.notify_interval_dict={"notice":2,"help":10}
        # self.notify_interval=2 # 2秒以内に前回の通知が発火していたら送信しない
        self.increase_corr_threshold=0.9 #
        self.increase_corr_threshold_help=0.9 #
        self.high_risk_threshold=0.4 #
        self.n_risky_patient_threshold=2

        # node別通知基準
        self.notify_threshold_by_dinamic_factor={
            "40000000":0.3,
            "40000001":0.4,
            "40000010":0.2,
            "40000011":0.3,
            "40000012":0.4,
            "40000013":0.6,
            "40000014":0.6,
            "40000015":0.5,
            "40000016":0.5,
            "40000100":0.6,
            "40000101":0.6,
            "40000102":0.6,
            "40000110":0.5,
            "40000111":0.5,
        }
        
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
        print(data["40000000"])
        try:
            data["40000000"]=[eval(v)[1] for v in data["40000000"].values]
            data["40000001"]=[eval(v)[1] for v in data["40000001"].values]
        except SyntaxError:
            data["40000000"]=data["40000000"]
            data["40000001"]=data["40000001"]
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
                    average_df.loc[node_code,patient]=np.mean([eval(v)[1] if not type(v)==float else np.nan for v in data_dicts[patient][node_code].values])

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
        print(average_df)
        print(factor_df)
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
    
    def get_help_sentence(self):
        help_text="デイルームで複数の患者さんの対応が必要です．デイルームに来てください．"
        return help_text

    def explain_risk(self,data_of_risky_patient,data_dict_roi,most_risky_patient):
        # ランキングの入れ替わりを起こすきっかけになった動的要因を探る（動作か周囲の人員配置．値の上昇が見られるものはどっちか考える）
        dynamic_factor_node,_=self.guess_dynamic_factor(data_of_risky_patient)

        # 元々の危険度を高める要因になっていた静的要因を探る（属性か周囲の物体）
        static_factor_node,_=self.guess_static_factor(data_dict_roi,most_risky_patient)

        alert_text=self.get_alert_sentence(most_risky_patient=most_risky_patient,dynamic_factor_node=dynamic_factor_node,static_factor_node=static_factor_node)
        return alert_text,dynamic_factor_node,static_factor_node
    
    def judge_time_interval(self,notify_history,row,alert_type):
        df=notify_history[notify_history["type"]==alert_type].reset_index()
        if len(df)==0:
            return True
        if row["timestamp"]-df.loc[len(df)-1,"timestamp"]>self.notify_interval_dict[alert_type]:
            return True
        else:
            return False
        
        pass

    def judge_top_increase(self,data_of_risky_patient):
        increase_ratio=data_of_risky_patient.loc[:,["timestamp","10000000"]].corr().loc["timestamp","10000000"]
        if increase_ratio>self.increase_corr_threshold:
            return True,increase_ratio
        else:
            return False,increase_ratio

    def judge_multi_patients(self,data_dict_roi):
        high_risk_increase_ratio_list=[]
        high_risk_average_list=[]
        high_risk_patients=[]
        for k,v in data_dict_roi.items():
            corr=data_dict_roi[k].loc[:,["timestamp","10000000"]].corr().loc["timestamp","10000000"]
            ave=data_dict_roi[k]["10000000"].mean()
            if (ave>self.high_risk_threshold) & (corr>self.increase_corr_threshold_help):
                high_risk_patients.append(k)
                high_risk_average_list.append(ave)
                high_risk_increase_ratio_list.append(corr)
        if len(high_risk_patients)>=self.n_risky_patient_threshold:
            return True
        else:
            return False

    def main_new(self):
        # ランキングの作成
        self.df_rank=self.get_df_rank(data_dicts=self.data_dicts)

        # 変数の初期化
        most_risky_patient=""
        previous_risky_patient=""
        notify_history=pd.DataFrame(columns=["timestamp","sentence","type","10000000"])
        
        for i,row in self.df_rank.iterrows():
            if i==0:
                most_risky_patient=self.df_rank.loc[i,0]
                previous_risky_patient=most_risky_patient
                continue

            # 情報収集
            most_risky_patient=self.df_rank.loc[i,0]
            previous_risky_patient=self.df_rank.loc[i-1,0]
            w_roi=20
            ## 個人のデータ
            data_of_risky_patient=self.data_dicts[most_risky_patient].loc[i-w_roi:i,:]
            data_of_previous_risky_patient=self.data_dicts[previous_risky_patient].loc[i-w_roi:i,:]
            ## 全員のデータ
            data_dict_roi={k:v.loc[i-w_roi:i,:] for k,v in self.data_dicts.items()}

            # 通知の必要性判断
            need_notify=False
            need_help=False
            ## A 前回通知からの時間経過
            tf_interval_notify=self.judge_time_interval(notify_history,row,"notice")
            tf_interval_help=self.judge_time_interval(notify_history,row,"help")

            ## B 順位入れ替えの発生
            tf_rankChange=True if (most_risky_patient!=previous_risky_patient) else False

            ## C 最上位患者の上昇
            tf_topIncrease,corr_top=self.judge_top_increase(data_of_risky_patient)

            ## D 複数人総合での危険性上昇
            tf_multiPatient=self.judge_multi_patients(data_dict_roi)

            ## E 本人の歴代リスクの中で最悪（値が最大値を更新）


            if tf_interval_notify and (tf_rankChange or tf_topIncrease):
                alert_text,dynamic_factor_node,static_factor_node=self.explain_risk(data_of_risky_patient,data_dict_roi,most_risky_patient)
                # 発火はdynamic node次第
                if self.notify_threshold_by_dinamic_factor[dynamic_factor_node]<=data_of_risky_patient["10000000"].values.mean():
                    need_notify=True
            if tf_interval_help and tf_multiPatient:
                need_help=True
            
            if need_notify:
                if alert_text not in notify_history["sentence"].values:
                    notify_history.loc[len(notify_history),:]=[self.df_rank.loc[i,"timestamp"],alert_text,"notice",data_of_risky_patient["10000000"].values.mean()]
            pass

            if need_help:
                alert_text=self.get_help_sentence()
                notify_history.loc[len(notify_history),:]=[self.df_rank.loc[i,"timestamp"],alert_text,"help",np.nan]

        # 通知文生成
        # 記録
        print(self.data_dir_dict["trial_dir_path"]+f"/{self.trial_name}_{self.result_trial_name}_notify_history.csv")
        notify_history.to_csv(self.data_dir_dict["trial_dir_path"]+f"/{self.trial_name}_{self.result_trial_name.replace('/','_')}_notify_history.csv",index=False)
        self.notify_history=notify_history

    def plot_timeseries_with_notification_point(self):
        for patient in self.id_names:
            plt.plot(self.data_dicts[patient]["timestamp"],self.data_dicts[patient]["10000000"],"-o",label=patient)

        for i,row in self.notify_history.iterrows():
            if row["type"]=="notice":
                color="red"
            elif row["type"]=="help":
                color="blue"
            plt.plot([row["timestamp"],row["timestamp"]],[0.1,0.7],color=color,linewidth=4,label=row["type"])
        plt.xlabel("Time [s]")
        plt.ylabel("Risk value")
        plt.legend()
        plt.grid()
        plt.savefig(self.data_dir_dict["trial_dir_path"]+f"/{self.trial_name}_{self.result_trial_name.replace('/','_')}_notify_history.png")
        plt.show()

    def main(self):
        # ランキングの作成
        self.df_rank=self.get_df_rank(data_dicts=self.data_dicts)

        # ランキングの入れ替わりを検知
        notification_history=pd.DataFrame(columns=["timestamp","sentence"])
        timestamp_list=[]
        sentence_list=[]
        increase_ratio_list=[]
        decrease_ratio_list=[]
        most_risky_patient=""
        previous_risky_patient=""
        for i,row in self.df_rank.iterrows():
            if i==0:
                most_risky_patient=self.df_rank.loc[i,0]
                previous_risky_patient=self.df_rank.loc[i,0]
                continue
            # 順位の入れ替えが発生した場合の通知
            if self.df_rank.loc[i,0]!=most_risky_patient:
                previous_risky_patient=most_risky_patient
                most_risky_patient=self.df_rank.loc[i,0]

                # 当該時刻における，危険度が上昇した患者のデータを準備
                w_roi=20
                data_of_risky_patient=self.data_dicts[most_risky_patient].loc[i-w_roi:i,:]
                data_of_previous_risky_patient=self.data_dicts[previous_risky_patient].loc[i-w_roi:i,:]
                data_dict_roi={k:v.loc[i-w_roi:i,:] for k,v in self.data_dicts.items()}
                # 追い抜いた側が上昇したことによる入れ替わりか，追い抜かれた側が下降したことによる入れ替わりか，判別
                increase_ratio=data_of_risky_patient.loc[:,["timestamp","10000000"]].corr().loc["timestamp","10000000"]
                if increase_ratio>self.increase_ratio_min:
                    # 追い抜かれた側（通知済み）の危険度が低下していない場合，応援が必要と判断
                    decrease_ratio=data_of_previous_risky_patient.loc[:,["timestamp","10000000"]].corr().loc["timestamp","10000000"]
                    if (len(timestamp_list)>0) & (decrease_ratio>self.decrease_ratio_max):
                        # 既に通知を飛ばしたことがあり，かつ今追い抜かれた患者も危険度が低下傾向にない場合，応援通知を飛ばす
                        help_text=self.get_help_sentence()
                        notification_mp3_path=self.data_dir_dict["trial_dir_path"]+f"/notification_{str(i).zfill(3)}_help.mp3"
                        # Notification().export_audio(text=help_text,mp3_path=notification_mp3_path,chime_type=2)
                        timestamp_list.append(self.df_rank.loc[i,"timestamp"]-self.df_rank.loc[0,"timestamp"])
                        sentence_list.append(help_text)
                        increase_ratio_list.append(increase_ratio)
                        decrease_ratio_list.append(decrease_ratio)
                        pass
                    
                    alert_text=self.explain_risk(data_of_risky_patient=data_of_risky_patient,data_dict_roi=data_dict_roi,most_risky_patient=most_risky_patient)

                    notification_mp3_path=self.data_dir_dict["trial_dir_path"]+f"/notification_{str(i).zfill(3)}.mp3"
                    # Notification().export_audio(text=alert_text,mp3_path=notification_mp3_path,chime_type=1)
                    
                    print("ranking changed")
                    print(self.df_rank.loc[i,"timestamp"])
                    print(self.df_rank.loc[i-1,0],"->",self.df_rank.loc[i,0])
                    print(alert_text)
                    increase_ratio_list.append(increase_ratio)
                    decrease_ratio_list.append(decrease_ratio)
                    timestamp_list.append(self.df_rank.loc[i,"timestamp"]-self.df_rank.loc[0,"timestamp"])
                    sentence_list.append(alert_text)
            else:
                # 最優先患者の状態変化をトリガーとした通知 
                # 動的リスクの増加傾向を相関係数で見る
                w=40
                corr_30000001=self.data_dicts[most_risky_patient].loc[i-w:i,["timestamp","30000001"]].corr().loc["timestamp","30000001"]
                corr_30000011=self.data_dicts[most_risky_patient].loc[i-w:i,["timestamp","30000011"]].corr().loc["timestamp","30000011"]
                print(corr_30000001,corr_30000011)
                if corr_30000001>0.95:
                    # 当該時刻における，危険度が上昇した患者のデータを準備
                    w_roi=20
                    data_of_risky_patient=self.data_dicts[most_risky_patient].loc[i-w_roi:i,:]
                    data_of_previous_risky_patient=self.data_dicts[previous_risky_patient].loc[i-w_roi:i,:]
                    data_dict_roi={k:v.loc[i-w_roi:i,:] for k,v in self.data_dicts.items()}
                    print(most_risky_patient,i)
                    print(self.data_dicts[most_risky_patient].loc[i,:])
                    alert_text=self.explain_risk(data_of_risky_patient=data_of_risky_patient,data_dict_roi=data_dict_roi,most_risky_patient=most_risky_patient)
                    print(alert_text)
                    # raise NotImplementedError
                pass
        
        notification_history["timestamp"]=timestamp_list
        notification_history["sentence"]=sentence_list
        notification_history["increase_ratio"]=increase_ratio_list
        notification_history["decrease_ratio"]=decrease_ratio_list

        print(notification_history)
        pass

if __name__=="__main__":
    trial_name="20250121PullWheelchairObaachan2Notification"
    # result_trial_name="20250113NormalSimulation"
    # result_trial_name="20250110SimulationMultipleRisks/no_00005"
    # result_trial_name="20250108DevMewThrottlingExp"
    result_trial_name="20250115PullWheelchairObaachan2"
    # trial_name="20250110NotificationGeneratorExp"
    # result_trial_name="20250121ChangeCriteriaBefore"
    strage="NASK"

    cls=NotificationGenerator(trial_name=trial_name,strage=strage,result_trial_name=result_trial_name)
    cls.main_new()
    cls.plot_timeseries_with_notification_point()
