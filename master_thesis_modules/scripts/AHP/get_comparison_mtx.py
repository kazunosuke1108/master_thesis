import os
import sys
import numpy as np
import pandas as pd
from pprint import pprint

sys.path.append(".")
sys.path.append("..")
sys.path.append(os.path.expanduser("~")+"/kazu_ws/master_thesis/master_thesis_modules")
from scripts.management.manager import Manager

class getConsistencyMtx(Manager):
    def __init__(self) -> None:
        super().__init__()
        pass

    def get_comparison_mtx(self,criteria=[],comparison_answer=[]):
        n=len(criteria)
        A=np.eye(n)
        answer_count=0
        for i in range(n):
            for j in range(n):
                if i>=j:
                    continue
                if len(comparison_answer)!=0:
                    val=comparison_answer[answer_count]
                    answer_count+=1
                else:
                    val=float(input(f"{criteria[i]}行{criteria[j]}列の比較行列成分："))
                A[i,j]=val
                A[j,i]=1/val
        return A

    def input_comparison_mtx(self):
        n=3
        A=np.eye(n)
        for i in range(n):
            for j in range(n):
                if i>=j:
                    continue
                val=float(input(f"{i+1}行{j+1}列の比較行列成分："))
                A[i,j]=val
                A[j,i]=1/val
        # print("A\n",A)
        return A
    
    def input_feature_mtx(self,features):
        n=len(features)
        F=np.eye(n)
        for i in range(n):
            for j in range(n):
                if i>=j:
                    continue
                F[i,j]=features[j]/features[i]
                F[j,i]=1/F[i,j]
        print("F\n",F)
        return F

    def evaluate_mtx(self,A):
        n=A.shape[0]
        eigvals, eigvecs = np.linalg.eig(A)
        max_eigval = eigvals.real.max()
        weights = eigvecs[:, eigvals.real.argmax()].real
        weights = weights / weights.sum()        
        CI=(max_eigval-n)/(n-1)
        return eigvals,eigvecs,max_eigval,weights,CI
    
    def get_AHP_weight(self,criteria=[],comparison_answer=[]):
        A=self.get_comparison_mtx(criteria=criteria,comparison_answer=comparison_answer)
        eigvals,eigvecs,max_eigval,weights,CI=self.evaluate_mtx(A)
        return A,eigvals,eigvecs,max_eigval,weights,CI
    
    def get_all_comparison_mtx_and_weight(self,trial_name,strage,save_mtx=False):
        # 内的・動的（動作）30000001
        AHP_dict={}
        self.data_dir_dict=self.get_database_dir(trial_name=trial_name,strage=strage)
        
        AHP_dict[30000001]={}
        csv_path=self.data_dir_dict["common_dir_path"]+"/comparison_mtx_30000001.csv"
        data=pd.read_csv(csv_path,header=None).values
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if i==j:
                    data[i,j]=1
                    continue
                elif i>j:
                    data[i,j]=1/data[j,i]
        eigvals,eigvecs,max_eigval,weights,CI=self.evaluate_mtx(data)
        AHP_dict[30000001]["A"]=data
        AHP_dict[30000001]["CI"]=CI
        AHP_dict[30000001]["weights"]=weights
        data=pd.DataFrame(data)
        if save_mtx:
            data.to_csv(csv_path,index=False,header=False)
        print(weights)
        
        # 外的・静的（物体）30000010
        AHP_dict[30000010]={}
        csv_path=self.data_dir_dict["common_dir_path"]+"/comparison_mtx_30000010.csv"
        data=pd.read_csv(csv_path,header=None).values
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if i==j:
                    data[i,j]=1
                    continue
                elif i>j:
                    data[i,j]=1/data[j,i]
        eigvals,eigvecs,max_eigval,weights,CI=self.evaluate_mtx(data)
        AHP_dict[30000010]["A"]=data
        AHP_dict[30000010]["CI"]=CI
        AHP_dict[30000010]["weights"]=weights
        data=pd.DataFrame(data)
        if save_mtx:
            data.to_csv(csv_path,index=False,header=False)
        
        return AHP_dict


if __name__=="__main__":
    cls=getConsistencyMtx()
    AHP_dict=cls.get_all_comparison_mtx_and_weight(trial_name="20241229BuildSimulator",strage="NASK")
    print(AHP_dict[30000010])
    print(AHP_dict[30000010]["CI"])

    # A=cls.get_comparison_mtx(criteria=["a","b","c"])
    # eigvals,eigvecs,max_eigval,weights,CI=cls.evaluate_mtx(A)
    # print(A)
    # print(eigvals,eigvecs,max_eigval,weights,CI)
    # 比較行列
    # A=cls.input_comparison_mtx()
    # _,_,_,weights_cmpr,_=cls.evaluate_mtx(A)
    # print(weights_cmpr)
    # # 評価基準A（点滴）に対する重み
    # distances=np.array([4,2,0.5])
    # F=cls.input_feature_mtx(features=distances)
    # _,_,_,weights_A,_=cls.evaluate_mtx(F)
    # print(weights_A)
    # # 評価基準B（手すり）に対する重み
    # distances=np.array([6,6,2])
    # distances=1/distances
    # F=cls.input_feature_mtx(features=distances)
    # _,_,_,weights_B,_=cls.evaluate_mtx(F)
    # print(weights_B)
    # # 評価基準C（車椅子）に対する重み
    # distances=np.array([3.5,0.1,2.5])
    # F=cls.input_feature_mtx(features=distances)
    # _,_,_,weights_C,_=cls.evaluate_mtx(F)
    # print(weights_C)
    # W=np.vstack((weights_A,weights_B,weights_C)).T
    # print(W@weights_cmpr)
