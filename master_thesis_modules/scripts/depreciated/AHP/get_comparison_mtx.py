import numpy as np

class getConsistencyMtx():
    def __init__(self) -> None:
        pass

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


if __name__=="__main__":
    cls=getConsistencyMtx()
    # 比較行列
    A=cls.input_comparison_mtx()
    _,_,_,weights_cmpr,_=cls.evaluate_mtx(A)
    print(weights_cmpr)
    # 評価基準A（点滴）に対する重み
    distances=np.array([4,2,0.5])
    F=cls.input_feature_mtx(features=distances)
    _,_,_,weights_A,_=cls.evaluate_mtx(F)
    print(weights_A)
    # 評価基準B（手すり）に対する重み
    distances=np.array([6,6,2])
    distances=1/distances
    F=cls.input_feature_mtx(features=distances)
    _,_,_,weights_B,_=cls.evaluate_mtx(F)
    print(weights_B)
    # 評価基準C（車椅子）に対する重み
    distances=np.array([3.5,0.1,2.5])
    F=cls.input_feature_mtx(features=distances)
    _,_,_,weights_C,_=cls.evaluate_mtx(F)
    print(weights_C)
    W=np.vstack((weights_A,weights_B,weights_C)).T
    print(W@weights_cmpr)
