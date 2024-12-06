import os

from icecream import ic
import numpy as np
import pandas as pd
import networkx as nx

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from network.graph_manager import GraphManager
from fuzzy.fuzzy_reasoning import FuzzyReasoning
from AHP.get_comparison_mtx import getConsistencyMtx
from pseudo_data.pseudo_data_generator_ABC import PseudoDataGenerator_ABC
from entropy.entropy_weight_generator import EntropyWeightGenerator
from management.manager import Manager


class Master(GraphManager,FuzzyReasoning,getConsistencyMtx,PseudoDataGenerator_ABC,EntropyWeightGenerator,Manager):
    def __init__(self):
        super().__init__()
        # 格納庫
        self.frames_dict={}
        self.fig_dict={}
        for name in self.data_dict.keys():
            self.fig_dict[name]=[]

        # params
        self.active_thre=0.5
        self.throttling=True

        # pseudo_dataが出来ていることを確認
        # pseudo_dataとgraph_dictのkey(A,B,C...)が合致しているか確認
        if len(list(self.data_dict.keys())) != len(list(self.graph_dict.keys())):
            raise Exception("擬似データの被験者数とグラフの被験者数が合致しません")

        # 重みの編集
        ## lv.4 -> 5 (AHP)
        ### 551
        criteria=[5510,5511,5512]
        A_551,eigvals,eigvecs,max_eigval,weights_551,CI_551=self.get_AHP_weight(criteria=[5510,5511,5512],comparison_answer=[7,9,5])
        new_weight_dict={4051:{criterion:w for criterion,w in zip(criteria,weights_551)}}
        for name in self.graph_dict.keys():
            self.update_weight(name=name,new_weight_dict=new_weight_dict)
        ### 552
        criteria=[5520,5521,5522]
        A_552,eigvals,eigvecs,max_eigval,weights_552,CI_552=self.get_AHP_weight(criteria=[5520,5521,5522],comparison_answer=[3,9,3])
        new_weight_dict={4052:{criterion:w for criterion,w in zip(criteria,weights_552)}}
        for name in self.graph_dict.keys():
            self.update_weight(name=name,new_weight_dict=new_weight_dict)
        ## lv.3 -> 4 (Fuzzy)
        ### 重み不要
        ## lv.2 -> 3 (Entropy)
        ### 実データで決まる

        # ネットワークが正しく作成できていることを確認
        # for name in self.graph_dict.keys():
        #     self.visualize_plotly(name=name)

    def main(self):
        for i,_ in list(self.data_dict.values())[0].iterrows():
            # 5000->4000 (AHP)
            for name in self.data_dict.keys():
                # 前iterationの1000がthre以上だった場合にのみ推論
                if i==0 or not(self.throttling):
                    ## 551系
                    w_vector=np.array([self.get_left_weight(name=name,node=key) for key in self.graph_dict[name]["G"].nodes() if str(551) in str(key)])
                    x_vector=self.data_dict[name].loc[i,[key for key in self.graph_dict[name]["G"].nodes() if str(551) in str(key)]].values
                    self.data_dict[name].loc[i,4051]=w_vector@x_vector
                    ## 552系
                    w_vector=np.array([self.get_left_weight(name=name,node=key) for key in self.graph_dict[name]["G"].nodes() if str(552) in str(key)])
                    x_vector=self.data_dict[name].loc[i,[key for key in self.graph_dict[name]["G"].nodes() if str(552) in str(key)]].values
                    self.data_dict[name].loc[i,4052]=w_vector@x_vector
                    self.data_dict[name].loc[i,"active"]=1
                elif self.data_dict[name].loc[i-1,1000]>=self.active_thre:
                    ## 551系
                    w_vector=np.array([self.get_left_weight(name=name,node=key) for key in self.graph_dict[name]["G"].nodes() if str(551) in str(key)])
                    x_vector=self.data_dict[name].loc[i,[key for key in self.graph_dict[name]["G"].nodes() if str(551) in str(key)]].values
                    self.data_dict[name].loc[i,4051]=w_vector@x_vector
                    ## 552系
                    w_vector=np.array([self.get_left_weight(name=name,node=key) for key in self.graph_dict[name]["G"].nodes() if str(552) in str(key)])
                    x_vector=self.data_dict[name].loc[i,[key for key in self.graph_dict[name]["G"].nodes() if str(552) in str(key)]].values
                    self.data_dict[name].loc[i,4052]=w_vector@x_vector
                    self.data_dict[name].loc[i,"active"]=1
                else:
                    self.data_dict[name].loc[i,4051]=self.data_dict[name].loc[i-1,4051]
                    self.data_dict[name].loc[i,4052]=self.data_dict[name].loc[i-1,4052]
                    self.data_dict[name].loc[i,"active"]=0


            # 4000->3000 (Fuzzy reasoning)
            for name in self.data_dict.keys():
                self.data_dict[name].loc[i,3005]=self.calculate_fuzzy({4051:self.data_dict[name].loc[i,4051],4052:self.data_dict[name].loc[i,4052],})

            # 3000->2000 (Entropy Weight Method)
            ## weightの更新
            w_source_data=self.data_dict[list(self.data_dict.keys())[0]].iloc[i]
            for i2,name in enumerate(self.data_dict.keys()):
                if i2==0:
                    continue
                w_source_data=pd.concat([w_source_data,self.data_dict[name].iloc[i]],axis=1)
            w_source_data=w_source_data.T[[key for key in self.graph_dict[name]["G"].nodes() if str(300) in str(key)]]
            w_dict={2000:self.get_entropy_weight(w_source_data)}
            for i2,name in enumerate(self.data_dict.keys()):
                self.update_weight(name=name,new_weight_dict=w_dict)
                w_vector=np.array([self.get_left_weight(name=name,node=key) for key in self.graph_dict[name]["G"].nodes() if str(300) in str(key)])
                x_vector=self.data_dict[name].loc[i,[key for key in self.graph_dict[name]["G"].nodes() if str(300) in str(key)]].values
                self.data_dict[name].loc[i,2000]=w_vector@x_vector

            # 2000 -> 1000
            for i2,name in enumerate(self.data_dict.keys()):
                self.data_dict[name].loc[i,1000]=self.data_dict[name].loc[i,2000]

            # scoreをグラフに反映
            for name in list(self.data_dict.keys()):
                new_score_dict=self.data_dict[name].iloc[i].to_dict()
                del new_score_dict["timestamp"]
                del new_score_dict["active"]
                self.update_score(name=name,new_score_dict=new_score_dict)
            
            for name in self.data_dict.keys():
                fig_data=self.visualize_plotly(name=name)
                self.fig_dict[name].append(fig_data)


    def draw_results(self):
        pass
        # anim
        # timestamps=self.data_dict[list(self.data_dict.keys())[0]]["timestamp"].values
        # for name in self.data_dict.keys():
        #     self.visualize_animation(name,self.fig_dict[name],timestamps)

        # matplotlibの時系列波形

    def save(self):
        self.data_dir_dict=self.get_database_dir("NASK")
        self.trial_timestamp=self.get_timestamp()
        self.trial_dir_path=self.data_dir_dict["database_dir_path"]+"/"+self.trial_timestamp
        os.makedirs(self.trial_dir_path,exist_ok=True)

        for i,name in enumerate(self.data_dict.keys()):
            self.data=self.data_dict[name]
            plt.plot(self.data["timestamp"],self.data[1000],label=name)
        plt.xlabel("Time [s]")
        plt.ylabel("Risk value")
        plt.legend()
        plt.grid()
        plt.savefig(self.trial_dir_path+"/"+self.trial_timestamp+".jpg")
        
        for name in self.data_dict.keys():
            fig_data=self.visualize_plotly(name=name)
            self.fig_dict[name].append(fig_data)

        pickle_data={}
        for name in self.data_dict.keys():
            pickle_data[name]=self.graph_dict[name]["G"]
        pickle_path=self.trial_dir_path+"/"+self.trial_timestamp+".pickle"
        self.write_picklelog(output_dict=pickle_data,picklepath=pickle_path)

        for name in self.data_dict.keys():
            self.data_dict[name].to_csv(self.trial_dir_path+"/"+self.trial_timestamp+"_"+name+".csv",index=False)

        json_data=self.graph_dict
        for name in self.data_dict.keys():
            del json_data[name]["G"]
        self.write_json(json_data,self.trial_dir_path+"/"+self.trial_timestamp+".json")

        # animation
        timestamps=self.data_dict[list(self.data_dict.keys())[0]]["timestamp"].values
        for name in self.data_dict.keys():
            self.visualize_animation(name,self.fig_dict[name],timestamps,show=True,save=False,trial_dir_path=self.trial_dir_path)

if __name__=="__main__":
    cls=Master()
    cls.main()
    cls.draw_results()
    cls.save()