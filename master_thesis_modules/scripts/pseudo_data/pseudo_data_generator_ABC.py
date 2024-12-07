import numpy as np
import pandas as pd
from icecream import ic

class PseudoDataGenerator_ABC():
    def __init__(self):
        super().__init__()
        self.t=np.arange(0,8.001,0.25)
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
            elif name==5520:
                traj=np.array([
                    [0,5.1,5.1],
                    [2,5,5],
                    [4,3,3],
                    [6,2.2,2.2],
                    [8,5,5],
                ])
                return traj
            elif name==5510:
                traj=np.array([
                    [0,4.5,2.5],
                    [8,4.5,2.5],
                ])
                return traj
            elif name==5511:
                traj=np.array([
                    [0,1.8,2.1],
                    [8,1.8,2.1],
                ])
                return traj
            elif name==5512:
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
        surroundings=[5520,5510,5511,5512]
        for name in surroundings:
            surrounding_data=pd.DataFrame(self.t,columns=["timestamp"])
            surrounding_data["x"]=self.add_risk_curve(surrounding_data,feature_points=get_sparse_traj(name)[:,[0,1]],col_name="x").values[:,1]
            surrounding_data["y"]=self.add_risk_curve(surrounding_data,feature_points=get_sparse_traj(name)[:,[0,2]],col_name="y").values[:,2]
            surroundings_position_dict[name]=surrounding_data
            del surrounding_data
        return patients_position_dict,surroundings_position_dict

    def position_to_features(self,patients_position_dict,surroundings_position_dict):
        def membership_distance(distance):
            max_val=7
            if distance>7:
                return 1
            else:
                return distance/7
        def membership_view(theta):
            if theta>90:
                return 1
            else:
                return theta/90
            
        def get_theta(object_vel,relative_pos):
            temp=np.dot(object_vel,relative_pos)/(np.linalg.norm(object_vel)*np.linalg.norm(relative_pos))
            if temp>1:
                temp=1
            elif temp<-1:
                temp=-1
            theta=np.rad2deg(np.arccos(temp))
            return theta
        for patient_name in patients_position_dict.keys():
            for surrounding_name in surroundings_position_dict.keys():
                # print(patients_position_dict[patient_name].loc[:,"x"]-surroundings_position_dict[surrounding_name].loc[:,"x"])
                distance2=(patients_position_dict[patient_name].loc[:,"x"]-surroundings_position_dict[surrounding_name].loc[:,"x"])**2+ \
                    (patients_position_dict[patient_name].loc[:,"y"]-surroundings_position_dict[surrounding_name].loc[:,"y"])**2
                distance=np.sqrt(distance2)
                print(surrounding_name)
                if surrounding_name==5512:
                    self.data_dict[patient_name][surrounding_name]=1-np.array(list(map(membership_distance,distance)))
                else:
                    self.data_dict[patient_name][surrounding_name]=np.array(list(map(membership_distance,distance)))
                    # ic(self.data_dict[patient_name][surrounding_name])
                if surrounding_name==5520: # 看護師の距離を算出済みの場合
                    # 速度
                    object_vel=surroundings_position_dict[surrounding_name][["x","y"]].diff().values
                    relative_pos=patients_position_dict[patient_name][["x","y"]].values-surroundings_position_dict[surrounding_name][["x","y"]].values
                    theta=np.array(list(map(get_theta,object_vel,relative_pos)))
                    self.data_dict[patient_name][5521]=np.array(list(map(membership_view,theta)))
                    self.data_dict[patient_name][5521].fillna(value=0,inplace=True)
                    ic(self.data_dict[patient_name])
        pass

        

if __name__=="__main__":
    cls=PseudoDataGenerator_ABC()
    patients_position_dict,surroundings_position_dict=cls.input_position_history()
    cls.position_to_features(patients_position_dict,surroundings_position_dict)
    # cls.define_observed_values()
    pass