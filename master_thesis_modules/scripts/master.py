from icecream import ic
import numpy as np
import pandas as pd
import networkx as nx

import matplotlib.pyplot as plt

from network.graph_manager import GraphManager
from fuzzy.fuzzy_reasoning import FuzzyReasoning
from AHP.get_comparison_mtx import getConsistencyMtx
from pseudo_data.pseudo_data_generator import PseudoDataGenerator

class Master(GraphManager,FuzzyReasoning,getConsistencyMtx,PseudoDataGenerator):
    def __init__(self):
        super().__init__()
        # pseudo_dataが出来ていることを確認
        ic(self.data)

        # ネットワークが正しく作成できていることを確認
        self.visualize()

    def main(self):
        # 重みの編集
        ## lv.4 -> 5 (AHP)
        ### 551
        criteria=[5510,5511,5512]
        A_551,eigvals,eigvecs,max_eigval,weights_551,CI_551=self.get_AHP_weight(criteria=[5510,5511,5512],comparison_answer=[3,5,3])
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
        ic(self.data)

    def draw_results(self):
        for key in self.data.keys():
            if key=="timestamp":
                continue
            plt.plot(self.data["timestamp"],self.data[key],label=key)
        plt.xlabel("Time [s]")
        plt.ylabel("Risk value")
        plt.legend()
        plt.grid()
        plt.show()

if __name__=="__main__":
    cls=Master()
    cls.main()
    cls.draw_results()