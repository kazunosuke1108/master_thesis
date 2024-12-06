import numpy as np
import pandas as pd
from icecream import ic

class PseudoDataGenerator_ABC():
    def __init__(self):
        super().__init__()
        self.t=np.arange(0,10,0.25)
        data=pd.DataFrame(self.t,columns=["timestamp"])
        columns=[1000,2000,2001,2002,3000,3001,3002,3003,3004,3005,4050,4051,4052,5510,5511,5512,5520,5521,5522]
        columns=columns+["active"]
        for column in columns:
            data[column]=np.nan
        
        # 無視するデータ群を指定
        killed_columns=[2001,2002,3000,3001,3003,3004,4050,5521,5522]
        for killed_column in killed_columns:
            data[killed_column]=1e-5
        
        self.data_dict={
            "A":data.copy(),#A
            "B":data.copy(),#B
            "C":data.copy(),#C
        }

        self.data_dict["A"]=self.define_observed_values_A(self.data_dict["A"])
        self.data_dict["B"]=self.define_observed_values_B(self.data_dict["B"])
        self.data_dict["C"]=self.define_observed_values_C(self.data_dict["C"])

    def add_risk_curve(self,data,feature_points,col_name):
        x_values = feature_points[:, 0]
        y_values = feature_points[:, 1]
        interpolated_values = np.interp(data["timestamp"], x_values, y_values)
        data[col_name]=interpolated_values
        return data

    def define_observed_values_A(self,data):
        x_3002=np.array([
            [0,0.4],
            [8,0.4]
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
            [0,0.7],
            [2,0.7],
            [4,0.5],
            [6,0.43],
            [8,0.7],
        ])
        data=self.add_risk_curve(data,x_3002,3002) # イベントカメラの値
        data=self.add_risk_curve(data,x_5510,5510) # 点滴との距離（近さ）
        data=self.add_risk_curve(data,x_5511,5511) # 車椅子との距離（近さ）
        data=self.add_risk_curve(data,x_5512,5512) # 手すりとの距離（遠さ）
        data=self.add_risk_curve(data,x_5520,5520) # 看護師との距離（遠さ）
        return data

    def define_observed_values_B(self,data):
        x_3002=np.array([
            [0,0.3],
            [1,0.3],
            [2,0.8],
            [4,0.5],
            [6,0.7],
            [8,0.3],
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
        data=self.add_risk_curve(data,x_3002,3002) # イベントカメラの値
        data=self.add_risk_curve(data,x_5510,5510) # 点滴との距離（近さ）
        data=self.add_risk_curve(data,x_5511,5511) # 車椅子との距離（近さ）
        data=self.add_risk_curve(data,x_5512,5512) # 手すりとの距離（遠さ）
        data=self.add_risk_curve(data,x_5520,5520) # 看護師との距離（遠さ）
        return data
 
    def define_observed_values_C(self,data):
        x_3002=np.array([
            [0,0.4],
            [8,0.4]
        ])
        x_5510=np.array([
            [0,0.9],
            [8,0.9],
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
            [0,0.7],
            [2,0.7],
            [4,0.5],
            [6,0.43],
            [8,0.7],
        ])
        data=self.add_risk_curve(data,x_3002,3002) # イベントカメラの値
        data=self.add_risk_curve(data,x_5510,5510) # 点滴との距離（近さ）
        data=self.add_risk_curve(data,x_5511,5511) # 車椅子との距離（近さ）
        data=self.add_risk_curve(data,x_5512,5512) # 手すりとの距離（遠さ）
        data=self.add_risk_curve(data,x_5520,5520) # 看護師との距離（遠さ）
        return data
     
    # def get_pseudo_data(self):
    #     self.define_observed_values()
    #     return self.data

    def input_position_history(self):
        def get_sparse_traj(name):
            if name=="A":
                traj=np.array([
                    [0,2,5],
                    [8,2,5],
                ])
                return traj
            elif name=="B":
                traj=np.array([
                    [0,2,2],
                    [8,2,2],
                ])
                return traj
            elif name=="C":
                traj=np.array([
                    [0,5,2],
                    [8,5,2],
                ])
                return traj
            elif name=="nurse":
                traj=np.array([
                    [0,5,5],
                    [2,5,5],
                    [4,3,3],
                    [6,2.2,2.2],
                    [8,5,5],
                ])
                return traj
            elif name=="infusion":
                traj=np.array([
                    [0,4.5,2.5],
                    [8,4.5,2.5],
                ])
                return traj
            elif name=="wheelchair":
                traj=np.array([
                    [0,1.8,2.1],
                    [8,1.8,2.1],
                ])
                return traj
            elif name=="handrail":
                traj=np.array([
                    [0,6,3],
                    [8,6,3],
                ])
                return traj
            
        # 患者の位置情報
        patients_position_dict={}
        patients=["A","B","C"]
        for name in patients:
            patient_data=pd.DataFrame(self.t,columns=["timestamp"])
            patient_data["x"]=self.add_risk_curve(patient_data,feature_points=get_sparse_traj(name)[:,[0,1]],col_name="x").values[:,1]
            patient_data["y"]=self.add_risk_curve(patient_data,feature_points=get_sparse_traj(name)[:,[0,2]],col_name="y").values[:,2]
            patients_position_dict[name]=patient_data
            del patient_data
        
        # 
        surroundings_position_dict={}
        surroundings=["nurse","infusion","wheelchair","handrail"]
        for name in surroundings:
            surrounding_data=pd.DataFrame(self.t,columns=["timestamp"])
            surrounding_data["x"]=self.add_risk_curve(surrounding_data,feature_points=get_sparse_traj(name)[:,[0,1]],col_name="x").values[:,1]
            surrounding_data["y"]=self.add_risk_curve(surrounding_data,feature_points=get_sparse_traj(name)[:,[0,2]],col_name="y").values[:,2]
            surroundings_position_dict[name]=surrounding_data
            del surrounding_data
        ic(patients_position_dict)
        ic(surroundings_position_dict)


        

if __name__=="__main__":
    cls=PseudoDataGenerator_ABC()
    cls.input_position_history()
    # cls.define_observed_values()
    pass