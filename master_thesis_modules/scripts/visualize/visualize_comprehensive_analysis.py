import os
import sys
from glob import glob
from pprint import pprint
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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
        csv_paths=sorted(glob(trial_dir_path+"/data_*_eval.csv"))
        patients=[os.path.basename(p)[len("data_"):-len("_eval.csv")] for p in csv_paths]
        self.score_dict[trial_name]={}
        for csv_path in csv_paths:
            # データの読み込み
            all_data=pd.read_csv(csv_path,header=0)
            trial_no=os.path.basename(trial_dir_path)
            patient=os.path.basename(csv_path).split("_")[1]
            # 各登場人物の位置を記録
            self.score_dict[trial_name][f"pos_{patient}_x"]=all_data["60010000"].mean()
            self.score_dict[trial_name][f"pos_{patient}_y"]=all_data["60010001"].mean()
            self.score_dict[trial_name]["pos_NS_x"]=all_data["50001100"].values[-1]
            self.score_dict[trial_name]["pos_NS_y"]=all_data["50001101"].values[-1]
            
            # 立ち上がりの部分
            start_timestamp=2 # 患者が立ち始める
            end_timestamp=5   # 看護師が動き始める
            data_25=all_data[(all_data["timestamp"]>=start_timestamp) & (all_data["timestamp"]<=end_timestamp)]
            score=data_25["10000000"].mean()
            self.score_dict[trial_name]["risk25_"+patient]=score
            # 看護師対応中の部分
            start_timestamp=7 # 看護師が到着
            end_timestamp=9   # 看護師の対応終了
            data_79=all_data[(all_data["timestamp"]>=start_timestamp) & (all_data["timestamp"]<=end_timestamp)]
            score=data_79["10000000"].mean()
            self.score_dict[trial_name]["risk79_"+patient]=score
            # 外的・静的要因の記録
            self.score_dict[trial_name]["30000010_"+patient]=all_data["30000010"].mean()
            # 立上り検知時に見えているか
            self.score_dict[trial_name]["40000111_"+patient]=data_25["40000111"].min()
            

        try:
            # A,B,Cそれぞれの立ち上がり時のスコアを取得
            values_25=[self.score_dict[trial_name][f"risk25_{p}"] for p in patients]
            values_79=[self.score_dict[trial_name][f"risk79_{p}"] for p in patients]
        except KeyError:
            print(self.score_dict)
            raise KeyError("患者の名前が見つからない")
        # A,B,Cの中で最もスコアが高い人物を記録
        self.score_dict[trial_name]["risk25_max"]=patients[np.argmax(values_25)]
        self.score_dict[trial_name]["risk79_max"]=patients[np.argmax(values_79)]
        # self.write_csvlog([trial_name,])

        # risk25で誰がmaxであるべきなのか，追記する
        # 外的・静的要因からみた，最重症者の発見（点滴＋車椅子）
        most_serious_patient=patients[np.argmax([self.score_dict[trial_name][f"30000010_{p}"] for p in patients])]
        print("most_serious_patient:",most_serious_patient)
        # 最重症者が今回のシナリオを通じて，視野外になるのかどうかを確認
        visibility_of_most_serious_patient=self.score_dict[trial_name]["40000111_"+most_serious_patient]

        visible_binary=visibility_of_most_serious_patient<=(1-(np.cos(np.deg2rad(100))/2+0.5))
        print("visibility_of_most_serious_patient:",visibility_of_most_serious_patient)
        print("visible_binary:",visible_binary)
        # 一番視認性が悪い人の発見
        most_hidden_patient=patients[np.argmax([self.score_dict[trial_name][f"40000111_{p}"] for p in patients])]
        self.score_dict[trial_name]["risk25_hidden"]=most_hidden_patient
        # 視野外になるなら，その人が最上位であるべきだし，そうでないなら立ち上がった患者を通知するべき
        if visible_binary:
            # 見えているなら
            self.score_dict[trial_name]["risk25_truth"]=self.score_dict[trial_name]["risk25_max"]
        else:
            # 見えていないなら
            self.score_dict[trial_name]["risk25_truth"]=most_serious_patient

    def main(self):
        nprocess=cpu_count()
        p_list=[]
        trial_dir_paths=[path for path in sorted(glob(self.simulation_dir_path+"/*")) if f"common" not in os.path.basename(path)]
        print(trial_dir_paths)
        for i,trial_dir_path in enumerate(trial_dir_paths):
            self.check_standing_detection(trial_dir_path)
            # p=Process(target=self.check_standing_detection,args=(trial_dir_path,))
            # p_list.append(p)
            # if len(p_list)==nprocess or i+1==len(trial_dir_paths):
            #     for p in p_list:
            #         p.start()
            #     for p in p_list:
            #         p.join()
            #     p_list=[]
        self.write_json(self.score_dict,json_path=self.simulation_common_dir_path+"/standing.json")
    
    def check_json(self):
        data=self.load_json(self.simulation_common_dir_path+"/standing.json")
        self.count_dict={
            "risk25_max":{"A":0,"B":0,"C":0,},
            "risk79_max":{"A":0,"B":0,"C":0,},
        }
        self.pos_maps={ # key: 最も危険だと判断した人物
            "A":np.zeros((13,13,3)),
            "B":np.zeros((13,13,3)),
            "C":np.zeros((13,13,3)),
        } # z方向... 0:B, 1:C, 2:NS
        for trial_no,d in data.items():
            self.count_dict["risk25_max"][data[trial_no]["risk25_max"]]+=1
            self.count_dict["risk79_max"][data[trial_no]["risk79_max"]]+=1
            for i,id_name in enumerate(["B","C","NS"]):
                self.pos_maps[data[trial_no]["risk25_max"]][int(data[trial_no][f"pos_{id_name}_x"])-int(data[trial_no][f"pos_A_x"]),int(data[trial_no][f"pos_{id_name}_y"])-int(data[trial_no][f"pos_A_y"]),i]+=1
        for k in self.pos_maps.keys():
            self.pos_maps[k]=self.pos_maps[k]/len(list(data.keys()))*255
        print(self.count_dict)
        
    def draw_relative_pos_map(self):
        plt.imshow(self.pos_maps["A"])   # 患者Aが一番危険だと判定された際の、患者Aから見たB,C,NSの位置
        plt.title(f"Position of B(R) C(G) and Nurse(B) when A is the most risky (n={self.count_dict['risk25_max']['A']})")
        plt.xlabel("Position X (position of A = 6)")
        plt.ylabel("Position Y (position of A = 6)")
        plt.savefig(self.simulation_common_dir_path+"/result_fig_relativePosition_A_risky.jpg")
        plt.imshow(self.pos_maps["B"])
        plt.title(f"Position of B(R) C(G) and Nurse(B) when B is the most risky (n={self.count_dict['risk25_max']['B']})")
        plt.xlabel("Position X (position of A = 6)")
        plt.ylabel("Position Y (position of A = 6)")
        plt.savefig(self.simulation_common_dir_path+"/result_fig_relativePosition_B_risky.jpg")
        plt.imshow(self.pos_maps["C"])
        plt.title(f"Position of B(R) C(G) and Nurse(B) when B is the most risky (n={self.count_dict['risk25_max']['C']})")
        plt.xlabel("Position X (position of A = 6)")
        plt.ylabel("Position Y (position of A = 6)")
        plt.savefig(self.simulation_common_dir_path+"/result_fig_relativePosition_C_risky.jpg")
        pprint(self.count_dict)

    def draw_timeseries_with_categorization(self):
        # リスク波形を取得
        json_data=self.load_json(self.simulation_common_dir_path+"/standing.json")
        trial_dir_paths=[path for path in sorted(glob(self.simulation_dir_path+"/*")) if f"common" not in os.path.basename(path)]
        gs=GridSpec(nrows=3,ncols=1)

        draw_column="40000111"
        

        for i,trial_dir_path in enumerate(trial_dir_paths):
            trial_name=os.path.basename(trial_dir_path)
            csv_paths=sorted(glob(trial_dir_path+"/data_*_eval.csv"))
            print("now processing...",i,"/",len(trial_dir_paths))
            for j,csv_path in enumerate(csv_paths):
                csv_data=pd.read_csv(csv_path,header=0)
                if i==0 :
                    # df_dict[最も危険と判断された患者（most_dangerous_patient）][ノード番号]-> 列が1試行
                    # focus_keys=[k for k in list(csv_data.keys()) if ((k!="timestamp") and int(k)<)]
                    df_dict={
                        "A":{p:{k:pd.DataFrame(csv_data["timestamp"].values,columns=["timestamp"]) for k in csv_data.keys() if k!="timestamp"} for p in ["A","B","C"]},
                        "B":{p:{k:pd.DataFrame(csv_data["timestamp"].values,columns=["timestamp"]) for k in csv_data.keys() if k!="timestamp"} for p in ["A","B","C"]},
                        "C":{p:{k:pd.DataFrame(csv_data["timestamp"].values,columns=["timestamp"]) for k in csv_data.keys() if k!="timestamp"} for p in ["A","B","C"]},
                    }
                most_dangerous_patient=json_data[trial_name]["risk25_max"]
                patient=os.path.basename(csv_path)[len("data_"):-len("_eval.csv")]
                for k in csv_data.keys():
                    if k=="timestamp":
                        continue
                    df_dict[most_dangerous_patient][patient][k][trial_name]=csv_data[k].values
            # if i>100:
            #     break

        # 各データの代表値を求める
        for d_patient in df_dict.keys():
            for patient in df_dict[d_patient].keys():
                for k in csv_data.keys():
                    if k=="timestamp":
                        continue
                    try:
                        df_dict[d_patient][patient][k]["average"]=df_dict[d_patient][patient][k].drop("timestamp",axis=1).mean(axis=1)
                        df_dict[d_patient][patient][k]["std"]=df_dict[d_patient][patient][k].drop("timestamp",axis=1).std(axis=1)
                    except TypeError:
                        pass
        for k in [k for k in csv_data.keys() if k!="timestamp"]:
            if int(k)>=50000000:
                continue
            try:
                gs=GridSpec(nrows=3,ncols=1)
                plt.subplot(gs[0])
                plt.plot(df_dict["A"]["A"][k]["timestamp"],df_dict["A"]["A"][k]["average"],label="risky: A")
                plt.fill_between(df_dict["A"]["A"][k]["timestamp"],df_dict["A"]["A"][k]["average"]-df_dict["A"]["A"][k]["std"],df_dict["A"]["A"][k]["average"]+df_dict["A"]["A"][k]["std"],alpha=0.25)
                plt.plot(df_dict["B"]["A"][k]["timestamp"],df_dict["B"]["A"][k]["average"],label="risky: B")
                plt.fill_between(df_dict["B"]["A"][k]["timestamp"],df_dict["B"]["A"][k]["average"]-df_dict["B"]["A"][k]["std"],df_dict["B"]["A"][k]["average"]+df_dict["B"]["A"][k]["std"],alpha=0.25)
                plt.plot(df_dict["C"]["A"][k]["timestamp"],df_dict["C"]["A"][k]["average"],label="risky: C")
                plt.fill_between(df_dict["C"]["A"][k]["timestamp"],df_dict["C"]["A"][k]["average"]-df_dict["C"]["A"][k]["std"],df_dict["C"]["A"][k]["average"]+df_dict["C"]["A"][k]["std"],alpha=0.25)
                plt.xlabel("Time [s]")
                plt.ylabel("Risk Value")
                plt.title(k)
                plt.legend()
                plt.grid()
                plt.subplot(gs[1])
                plt.plot(df_dict["A"]["B"][k]["timestamp"],df_dict["A"]["B"][k]["average"],label="risky: A")
                plt.fill_between(df_dict["A"]["B"][k]["timestamp"],df_dict["A"]["B"][k]["average"]-df_dict["A"]["B"][k]["std"],df_dict["A"]["B"][k]["average"]+df_dict["A"]["B"][k]["std"],alpha=0.25)
                plt.plot(df_dict["B"]["B"][k]["timestamp"],df_dict["B"]["B"][k]["average"],label="risky: B")
                plt.fill_between(df_dict["B"]["B"][k]["timestamp"],df_dict["B"]["B"][k]["average"]-df_dict["B"]["B"][k]["std"],df_dict["B"]["B"][k]["average"]+df_dict["B"]["B"][k]["std"],alpha=0.25)
                plt.plot(df_dict["C"]["B"][k]["timestamp"],df_dict["C"]["B"][k]["average"],label="risky: C")
                plt.fill_between(df_dict["C"]["B"][k]["timestamp"],df_dict["C"]["B"][k]["average"]-df_dict["C"]["B"][k]["std"],df_dict["C"]["B"][k]["average"]+df_dict["C"]["B"][k]["std"],alpha=0.25)
                plt.xlabel("Time [s]")
                plt.ylabel("Risk Value")
                plt.legend()
                plt.grid()
                plt.legend()
                plt.subplot(gs[2])
                plt.plot(df_dict["A"]["C"][k]["timestamp"],df_dict["A"]["C"][k]["average"],label="risky: A")
                plt.fill_between(df_dict["A"]["C"][k]["timestamp"],df_dict["A"]["C"][k]["average"]-df_dict["A"]["C"][k]["std"],df_dict["A"]["C"][k]["average"]+df_dict["A"]["C"][k]["std"],alpha=0.25)
                plt.plot(df_dict["B"]["C"][k]["timestamp"],df_dict["B"]["C"][k]["average"],label="risky: B")
                plt.fill_between(df_dict["B"]["C"][k]["timestamp"],df_dict["B"]["C"][k]["average"]-df_dict["B"]["C"][k]["std"],df_dict["B"]["C"][k]["average"]+df_dict["B"]["C"][k]["std"],alpha=0.25)
                plt.plot(df_dict["C"]["C"][k]["timestamp"],df_dict["C"]["C"][k]["average"],label="risky: C")
                plt.fill_between(df_dict["C"]["C"][k]["timestamp"],df_dict["C"]["C"][k]["average"]-df_dict["C"]["C"][k]["std"],df_dict["C"]["C"][k]["average"]+df_dict["C"]["C"][k]["std"],alpha=0.25)
                plt.xlabel("Time [s]")
                plt.ylabel("Risk Value")
                plt.legend()
                plt.grid()
                plt.savefig(self.simulation_common_dir_path+f"/result_25_{k}_with_categorization.jpg")
                plt.close()
            except KeyError:
                plt.close()
                continue

            #     patient=os.path.basename(csv_path)[len("data_"):-len("_eval.csv")]
            #     print(trial_name,patient)
            #     if patient=="A":
            #         plt.subplot(gs[0])
            #     elif patient=="B":
            #         plt.subplot(gs[1])
            #     elif patient=="C":
            #         plt.subplot(gs[2])
            #     else:
            #         raise Exception
            #     plt.plot(csv_data["timestamp"],csv_data[draw_column],"red" if most_dangerous_patient==patient else "black")
            # if i%100==0:
            #     plt.pause(1)

                

        # Aのリスク波形を、A,B,Cの誰が最も危険とされたかに応じて色分けして表示

        pass

if __name__=="__main__":
    simulation_name="20250113SimulationPositionB"
    strage="NASK"
    cls=Visualizer(simulation_name=simulation_name,strage=strage)
    cls.main()
    # cls.check_json()
    # cls.draw_timeseries_with_categorization()
