import numpy as np
import pandas as pd
from icecream import ic # pip install icecream

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class ThrottlingLab():
    def __init__(self):
        t=np.arange(0,10,0.1)
        self.data=pd.DataFrame(t,columns=["timestamp"])
        self.weights_dict={}
        self.threshold_dict={
            "total_score_thre":0.5,
        }
        self.status_color={"active":"red","inactive":"blue"}


    def define_observed_values(self):
        def add_risk_curve(feature_points,col_name):
            x_values = feature_points[:, 0]
            y_values = feature_points[:, 1]
            interpolated_values = np.interp(self.data["timestamp"], x_values, y_values)
            self.data[col_name]=interpolated_values
        fp_xA=np.array([
            [0,0.1],
            [1,0.1],
            [2,0.9],
            [4,0.9],
            [6,0.7],
            [8,0.9],
            [9,0.1],
            [10,0.1]
        ])
        fp_xB1=np.array([
            [0,0.9],
            [4,0.9],
            [7,0.1],
            [10,0.1],
        ])
        fp_xB2=np.array([
            [0,0.7],
            [10,0.7],
        ])
        add_risk_curve(fp_xA,"xA") # イベントカメラの値
        add_risk_curve(fp_xB1,"xB1") # スタッフとの距離（遠さ）
        add_risk_curve(fp_xB2,"xB2") # 点滴との距離（近さ）
        pass

    def define_weights_dict(self):
        self.weights_dict["lv_03"]={
            "wA":{"value":0.5,"level":"lv_03","description":"HOW"},
            "wB":{"value":0.5,"level":"lv_03","description":"WHERE"},
        }
        self.weights_dict["lv_04"]={
            "wA":{},
            "wB":{
                "wB1":{"value":0.5,"level":"lv_04","description":"スタッフとの距離"},
                "wB2":{"value":0.5,"level":"lv_04","description":"点滴との距離"},
            }
        }
        pass

    def get_score(self,method="sum"):
        if method=="sum":
            self.score_A=self.weights_dict["lv_03"]["wA"]["value"]*self.data["xA"]
            self.score_B=self.weights_dict["lv_03"]["wB"]["value"]*(self.weights_dict["lv_04"]["wB"]["wB1"]["value"]*self.data["xB1"]+self.weights_dict["lv_04"]["wB"]["wB2"]["value"]*self.data["xB2"])
            self.data["score"]=self.score_A+self.score_B
        elif method=="throttling":
            self.data["score_A"]=np.nan
            self.data["score_B"]=np.nan
            self.data["score"]=np.nan
            self.data["throttling_status"]=""
            score_lv_04_B=np.nan
            throttling_status="inactive"
            for i,row in self.data.iterrows():
                # lv_04の計算
                self.data.loc[i,"throttling_status"]=throttling_status
                ic(throttling_status)
                if i==0:
                    score_lv_04_B=self.weights_dict["lv_04"]["wB"]["wB1"]["value"]*self.data.loc[i,"xB1"]+self.weights_dict["lv_04"]["wB"]["wB2"]["value"]*self.data.loc[i,"xB2"]
                elif throttling_status=="active":
                    score_lv_04_B=self.weights_dict["lv_04"]["wB"]["wB1"]["value"]*self.data.loc[i,"xB1"]+self.weights_dict["lv_04"]["wB"]["wB2"]["value"]*self.data.loc[i,"xB2"]
                elif throttling_status=="inactive":
                    pass
                
                # scoreの計算
                self.data.loc[i,"score_A"]=self.weights_dict["lv_03"]["wA"]["value"]*self.data.loc[i,"xA"]
                self.data.loc[i,"score_B"]=self.weights_dict["lv_03"]["wB"]["value"]*score_lv_04_B
                self.data.loc[i,"score"]=self.data.loc[i,"score_A"]+self.data.loc[i,"score_B"]

                # statusの判定
                if self.data.loc[i,"score"]>self.threshold_dict["total_score_thre"]:
                    throttling_status="active"
                else:
                    throttling_status="inactive"
            pass
        else:
            raise NotImplementedError
        pass

    def visualize(self):
        gs=GridSpec(nrows=3,ncols=1)
        plt.subplot(gs[0])
        plt.plot(self.data["timestamp"],self.data["xA"],label="xA (move)")
        plt.plot(self.data["timestamp"],self.data["xB1"],label="xB1 (staff distance)")
        plt.plot(self.data["timestamp"],self.data["xB2"],label="xB2 (infusion distance)")
        plt.xlim([self.data["timestamp"].min()-1,self.data["timestamp"].max()+1])
        plt.xlabel("Time [s]")
        plt.ylabel("Risk value")
        plt.legend()
        plt.grid()

        plt.subplot(gs[1])
        # plt.plot(self.data["timestamp"],self.score_A,"--",label="score_A")
        # plt.plot(self.data["timestamp"],self.score_B,"--",label="score_B")
        plt.plot(self.data["timestamp"],self.data["score"],"--",label="score")
        plt.xlim([self.data["timestamp"].min()-1,self.data["timestamp"].max()+1])
        plt.xlabel("Time [s]")
        plt.ylabel("Risk value")
        plt.legend()
        plt.grid()

        plt.subplot(gs[2])
        for i,row in self.data.iterrows():
            # ic(self.data.loc[i,"throttling_status"])
            plt.barh(y=0,left=self.data.loc[i,"timestamp"],width=0.1,color=self.status_color[self.data.loc[i,"throttling_status"]])
        plt.xlim([self.data["timestamp"].min()-1,self.data["timestamp"].max()+1])
        plt.xlabel("Time [s]")
        plt.ylabel("Throttling status")
        plt.legend()
        plt.grid()
        plt.show()

    def main(self):
        # define observed values
        self.define_observed_values()
        ic(self.data)

        # define weight
        self.define_weights_dict()
        ic(self.weights_dict)

        # simulation
        self.get_score(method="throttling")

        # visualize
        self.visualize()

if __name__=="__main__":
    cls=ThrottlingLab()
    cls.main()
    pass