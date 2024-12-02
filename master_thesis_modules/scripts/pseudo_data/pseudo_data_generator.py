import numpy as np
import pandas as pd
from icecream import ic

class PseudoDataGenerator():
    def __init__(self):
        super().__init__()
        t=np.arange(0,10,0.1)
        self.data=pd.DataFrame(t,columns=["timestamp"])
        columns=[1000,2000,2001,2002,3000,3001,3002,3003,3004,3005,4050,4051,4052,5510,5511,5512,5520,5521,5522]
        for column in columns:
            self.data[column]=np.nan
        
        # 無視するデータ群を指定
        killed_columns=[2001,2002,3000,3001,3003,3004,4050]
        for killed_column in killed_columns:
            self.data[killed_column]=0
        
        self.define_observed_values()

    def define_observed_values(self):
        def add_risk_curve(feature_points,col_name):
            x_values = feature_points[:, 0]
            y_values = feature_points[:, 1]
            interpolated_values = np.interp(self.data["timestamp"], x_values, y_values)
            self.data[col_name]=interpolated_values
        x_3002=np.array([
            [0,0.1],
            [1,0.1],
            [2,0.9],
            [4,0.9],
            [6,0.7],
            [8,0.9],
            [9,0.1],
            [10,0.1]
        ])
        x_5510=np.array([
            [0,0.9],
            [10,0.9],
        ])
        x_5511=x_5512=np.array([
            [0,0.7],
            [10,0.7],
        ])
        x_5520=np.array([
            [0,0.8],
            [4,0.8],
            [7,0.1],
            [10,0.1],
        ])
        x_5521=np.array([
            [0,0.8],
            [10,0.8],
        ])
        x_5522=np.array([
            [0,0],
            [10,0],
        ])
        add_risk_curve(x_3002,3002) # イベントカメラの値
        add_risk_curve(x_5510,5510) # 点滴との距離（近さ）
        add_risk_curve(x_5511,5511) # 車椅子との距離（近さ）
        add_risk_curve(x_5512,5512) # 手すりとの距離（遠さ）
        add_risk_curve(x_5520,5520) # 看護師との距離（遠さ）
        add_risk_curve(x_5521,5521) # 介護士との距離（遠さ）
        add_risk_curve(x_5522,5522) # 面会者との距離（近さ）
        pass
    
    # def get_pseudo_data(self):
    #     self.define_observed_values()
    #     return self.data

if __name__=="__main__":
    cls=PseudoDataGenerator()
    # cls.define_observed_values()
    ic(cls.data)
    pass