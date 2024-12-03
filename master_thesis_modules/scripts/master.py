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

class Master(GraphManager,FuzzyReasoning,getConsistencyMtx,PseudoDataGenerator_ABC):
    def __init__(self):
        super().__init__()
        # pseudo_dataが出来ていることを確認
        ic(self.data_dict)

        # 重みの編集
        ## lv.4 -> 5 (AHP)
        ### 551
        criteria=[5510,5511,5512]
        A_551,eigvals,eigvecs,max_eigval,weights_551,CI_551=self.get_AHP_weight(criteria=[5510,5511,5512],comparison_answer=[7,9,5])
        new_weight_dict={4051:{criterion:w for criterion,w in zip(criteria,weights_551)}}
        self.update_weight(new_weight_dict=new_weight_dict)
        ### 552
        criteria=[5520,5521,5522]
        A_552,eigvals,eigvecs,max_eigval,weights_552,CI_552=self.get_AHP_weight(criteria=[5520,5521,5522],comparison_answer=[3,9,3])
        new_weight_dict={4052:{criterion:w for criterion,w in zip(criteria,weights_552)}}
        self.update_weight(new_weight_dict=new_weight_dict)
        ## lv.3 -> 4 (Fuzzy)
        ### 重み不要
        ## lv.2 -> 3 (Entropy)
        ### 実データで決まる

        # ネットワークが正しく作成できていることを確認
        self.visualize()

    def main(self,id="A"):
        self.data=self.data_dict[id]
        # スコア計算の実施 (forループ)
        nodes = list(self.G.nodes())[1:]
        weights = nx.get_edge_attributes(self.G, 'weight').values()
        weight_dict={node:weight for node,weight in zip(nodes,weights)}
        for i,row in self.data.iterrows():
            # 5000 -> 4000
            ## 551系
            w_vector=np.array([weight_dict[key] for key in nodes if str(551) in str(key)])
            x_vector=self.data.loc[i,[key for key in nodes if str(551) in str(key)]].values
            self.data.loc[i,4051]=w_vector@x_vector
            ## 552系
            w_vector=np.array([weight_dict[key] for key in nodes if str(552) in str(key)])
            x_vector=self.data.loc[i,[key for key in nodes if str(552) in str(key)]].values
            self.data.loc[i,4052]=w_vector@x_vector

            # 4000 -> 3000
            self.data.loc[i,3005]=self.calculate_fuzzy({4051:self.data.loc[i,4051],4052:self.data.loc[i,4052],})

            # 3000 -> 2000 # Entropy Weight Methodに変更予定．（複数候補にするときに．）
            ## 200系
            w_vector=np.array([weight_dict[key] for key in nodes if str(300) in str(key)])
            x_vector=self.data.loc[i,[key for key in nodes if str(300) in str(key)]].values
            self.data.loc[i,2000]=w_vector@x_vector

            # 2000 -> 1000
            self.data.loc[i,1000]=self.data.loc[i,2000]
            pass
        self.data_dict[id]=self.data

    def draw_results(self):
        gs=GridSpec(nrows=3,ncols=1)
        for i,id in enumerate(self.data_dict.keys()):
            plt.subplot(gs[i])
            self.data=self.data_dict[id]
            for key in self.data.keys():
                if key=="timestamp":
                    continue
                plt.plot(self.data["timestamp"],self.data[key],label=key)
            plt.xlabel("Time [s]")
            plt.ylabel("Risk value")
            plt.ylim()
            plt.legend()
            plt.grid()
        plt.show()

        for i,id in enumerate(self.data_dict.keys()):
            self.data=self.data_dict[id]
            plt.plot(self.data["timestamp"],self.data[1000],label=id)
        plt.xlabel("Time [s]")
        plt.ylabel("Risk value")
        plt.legend()
        plt.grid()
        plt.show()


if __name__=="__main__":
    cls=Master()
    for id in ["A","B","C"]:
        cls.main(id=id)
    cls.draw_results()
    ic(cls.data_dict["A"])
    ic(cls.data_dict["B"])
    ic(cls.data_dict["C"])