import numpy as np
import pandas as pd
from icecream import ic

class PseudoDataGenerator_ABC():
    def __init__(self):
        super().__init__()
        t=np.arange(0,10,2)
        data=pd.DataFrame(t,columns=["timestamp"])
        columns=[1000,2000,2001,2002,3000,3001,3002,3003,3004,3005,4050,4051,4052,5510,5511,5512,5520,5521,5522]
        for column in columns:
            data[column]=np.nan
        
        # 無視するデータ群を指定
        killed_columns=[2001,2002,3000,3001,3003,3004,4050,5521,5522]
        for killed_column in killed_columns:
            data[killed_column]=0
        
        self.data_dict={
            "A":data.copy(),#A
            "B":data.copy(),#B
            "C":data.copy(),#C
        }

        self.data_dict["A"]=self.define_observed_values_A(self.data_dict["A"])
        self.data_dict["B"]=self.define_observed_values_B(self.data_dict["B"])
        self.data_dict["C"]=self.define_observed_values_C(self.data_dict["C"])

    def define_observed_values_A(self,data):
        def add_risk_curve(feature_points,col_name):
            x_values = feature_points[:, 0]
            y_values = feature_points[:, 1]
            interpolated_values = np.interp(data["timestamp"], x_values, y_values)
            data[col_name]=interpolated_values
        x_3002=np.array([
            [0,0.2],
            [8,0.2]
        ])
        x_5510=np.array([
            [0,0.2],
            [8,0.2],
        ])
        x_5511=np.array([
            [0,0.2],
            [8,0.2],
        ])
        x_5512=np.array([
            [0,0.6],
            [8,0.6],
        ])
        x_5520=np.array([
            [0,0.4],
            [2,0.4],
            [4,0.5],
            [6,0.6],
            [8,0.4],
        ])
        add_risk_curve(x_3002,3002) # イベントカメラの値
        add_risk_curve(x_5510,5510) # 点滴との距離（近さ）
        add_risk_curve(x_5511,5511) # 車椅子との距離（近さ）
        add_risk_curve(x_5512,5512) # 手すりとの距離（遠さ）
        add_risk_curve(x_5520,5520) # 看護師との距離（遠さ）
        return data

    def define_observed_values_B(self,data):
        def add_risk_curve(feature_points,col_name):
            x_values = feature_points[:, 0]
            y_values = feature_points[:, 1]
            interpolated_values = np.interp(data["timestamp"], x_values, y_values)
            data[col_name]=interpolated_values
        x_3002=np.array([
            [0,0.2],
            [2,0.8],
            [4,0.6],
            [6,0.8],
            [8,0.2],
        ])
        x_5510=np.array([
            [0,0.4],
            [8,0.4],
        ])
        x_5511=np.array([
            [0,0.6],
            [8,0.6],
        ])
        x_5512=np.array([
            [0,0.4],
            [8,0.4],
        ])
        x_5520=np.array([
            [0,0.9],
            [2,0.9],
            [4,0.5],
            [6,0.1],
            [8,0.9],
        ])
        add_risk_curve(x_3002,3002) # イベントカメラの値
        add_risk_curve(x_5510,5510) # 点滴との距離（近さ）
        add_risk_curve(x_5511,5511) # 車椅子との距離（近さ）
        add_risk_curve(x_5512,5512) # 手すりとの距離（遠さ）
        add_risk_curve(x_5520,5520) # 看護師との距離（遠さ）
        return data
 
    def define_observed_values_C(self,data):
        def add_risk_curve(feature_points,col_name):
            x_values = feature_points[:, 0]
            y_values = feature_points[:, 1]
            interpolated_values = np.interp(data["timestamp"], x_values, y_values)
            data[col_name]=interpolated_values
        x_3002=np.array([
            [0,0.2],
            [8,0.2]
        ])
        x_5510=np.array([
            [0,0.6],
            [8,0.6],
        ])
        x_5511=np.array([
            [0,0.4],
            [8,0.4],
        ])
        x_5512=np.array([
            [0,0.2],
            [8,0.2],
        ])
        x_5520=np.array([
            [0,0.6],
            [2,0.6],
            [4,0.5],
            [6,0.6],
            [8,0.6],
        ])
        add_risk_curve(x_3002,3002) # イベントカメラの値
        add_risk_curve(x_5510,5510) # 点滴との距離（近さ）
        add_risk_curve(x_5511,5511) # 車椅子との距離（近さ）
        add_risk_curve(x_5512,5512) # 手すりとの距離（遠さ）
        add_risk_curve(x_5520,5520) # 看護師との距離（遠さ）
        return data
     
    # def get_pseudo_data(self):
    #     self.define_observed_values()
    #     return self.data

if __name__=="__main__":
    cls=PseudoDataGenerator_ABC()
    # cls.define_observed_values()
    ic(cls.data_dict["B"])
    pass