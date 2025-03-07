import numpy as np
np.random.seed(0)
import pandas as pd

import matplotlib.pyplot as plt

class EntropyWeightGenerator():
    def __init__(self):
        super().__init__()

    def get_entropy_weight(self,score_df=pd.DataFrame(data=np.random.random((3,6)),columns=[3000,3001,3002,3003,3004,3005])):
        # 正規化
        score_df=score_df/score_df.sum()
        print(score_df)
        print(np.log(score_df.astype(float)))
        entropy=-1/np.log(len(score_df.index))*(score_df*np.log(score_df.astype(float))).sum()
        # entropy[np.isnan(entropy)]=1e-10
        degree_of_diversification=1-entropy
        weight=degree_of_diversification/degree_of_diversification.sum()
        for node in score_df.keys():
            if np.isnan(weight[node]) or abs(weight[node])==np.inf:
                for node in score_df.keys():
                    weight[node]=1/len(list(score_df.keys()))
                return weight.to_dict()
        # ic(score_df)
        # ic(weight.to_dict())
        return weight.to_dict()
    
    def get_entropy_weight_t(self,score_df=pd.DataFrame(data=np.random.random((3,6)),columns=[3000,3001,3002,3003,3004,3005])):
        # 正規化
        score_df=score_df/score_df.sum()
        # 列ごとにエントロピーを計算していく
        # ic(score_df.index)
        # ic(-1/np.log(len(score_df.index))*(score_df*np.log(score_df)))
        entropy=-1/np.log(len(score_df.index))*(score_df*np.log(score_df)).sum()
        degree_of_diversification=1-entropy+1e-5
        weight=degree_of_diversification/degree_of_diversification.sum()
        # NaNが含まれた場合の対応
        for node in score_df.keys():
            if np.isnan(weight[node]) or abs(weight[node])==np.inf:
                for node in score_df.keys():
                    weight[node]=1/len(list(score_df.keys()))
                return weight.to_dict()
        return weight.to_dict()


if __name__=="__main__":
    cls=EntropyWeightGenerator()
    cls.get_entropy_weight_t()
    pass