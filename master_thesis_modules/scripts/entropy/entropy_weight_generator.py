from icecream import ic

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
        entropy=-1/np.log(len(score_df.index))*(score_df*np.log(score_df)).sum()
        degree_of_diversification=1-entropy
        weight=degree_of_diversification/degree_of_diversification.sum()
        # ic(score_df)
        # ic(weight.to_dict())
        return weight.to_dict()


if __name__=="__main__":
    cls=EntropyWeightGenerator()
    cls.get_entropy_weight()
    pass