import os
import sys
import copy
from glob import glob

from icecream import ic
from pprint import pprint

import numpy as np
import pandas as pd

sys.path.append(".")
sys.path.append("..")
sys.path.append(os.path.expanduser("~")+"/kazu_ws/master_thesis/master_thesis_modules")
from scripts.management.manager import Manager
from scripts.network.graph_manager_v3 import GraphManager
from scripts.AHP.get_comparison_mtx_v3 import getConsistencyMtx
from scripts.fuzzy.fuzzy_reasoning_v3 import FuzzyReasoning
from scripts.entropy.entropy_weight_generator import EntropyWeightGenerator
from scripts.pseudo_data.pseudo_data_generator import PseudoDataGenerator

class Master(Manager,GraphManager,FuzzyReasoning,EntropyWeightGenerator):
    def __init__(self,data_dicts,strage="NASK"):
        super().__init__()
        self.strage=strage
        default_graph=self.get_default_graph()

        # parameters
        self.spatial_normalization_param=np.sqrt(2)*6
        self.AHP_array_type=1

        self.data_dicts=data_dicts
        self.patients=list(self.data_dicts.keys())
        self.graph_dicts={}
        for patient in self.patients:
            self.graph_dicts[patient]=copy.deepcopy(default_graph)
            # dataに列が存在するかどうか確認

        # 危険動作の事前定義
        self.risky_motion_dict={
            "40000010":{
                "label":"standUp",
                "features":np.array([1,         0,      0,      1]),
                },
            "40000011":{
                "label":"releaseBrake",
                "features":np.array([0,         0.5,    0.5,      0.5]),
                },
            "40000012":{
                "label":"moveWheelchair",
                "features":np.array([0,         0.5,      0.5,      0.5]),
                },
            "40000013":{
                "label":"loseBalance",
                "features":np.array([0,         1,      np.nan, np.nan]),
                },
            "40000014":{
                "label":"moveHand",
                "features":np.array([np.nan,    0,      1,      np.nan]),
                },
            "40000015":{
                "label":"coughUp",
                "features":np.array([np.nan,    0.5,      0.5,      np.nan]),
                },
            "40000016":{
                "label":"touchFace",
                "features":np.array([np.nan,    0,      0.5,      np.nan]),
            },
        }
        # AHP 一対比較行列の作成
        self.AHP_dict=getConsistencyMtx().get_all_comparison_mtx_and_weight(trial_name="",strage=self.strage,array_type=self.AHP_array_type)

    def activation_func(self,val):
        return val

    def fuzzy_logic(self):
        def mu_yes():
            return (0.4,0.7,1.0)
        def mu_no():
            return (0.0,0.3,0.6)
        def mu_young():
            return (0.0,0.25,0.5)
        def mu_middle():
            return (0.25,0.5,0.75)
        def mu_old():
            return (0.5,0.75,1.0)
        def patient_or_not(val):
            print("val:",val)
            if val=="yes":
                return mu_yes()
            elif val=="no":
                return mu_no()
            elif np.isnan(val):
                return np.nan
            else:
                raise Exception(f"Unexpected value in 内的・静的・患者判別: {val}")
        def age(val):
            if val=="young":
                return mu_young()
            elif val=="middle":
                return mu_middle()
            elif val=="old":
                return mu_old()
            elif np.isnan(val):
                return np.nan
            else:
                raise Exception(f"Unexpected value in 内的・静的・年齢: {val}")
        for patient in self.data_dicts.keys():
            self.data_dicts[patient]["40000000"]=patient_or_not(self.data_dicts[patient]["50000000"])#list(map(patient_or_not,self.data_dicts[patient]["50000000"]))
            self.data_dicts[patient]["40000001"]=age(self.data_dicts[patient]["50000010"])#list(map(age,self.data_dicts[patient]["50000010"]))
        pass

    def pose_similarity(self):
        def get_similarity(risk,data_dict):
            print(data_dict)
            similarity=1-np.nanmean(abs(self.risky_motion_dict[risk]["features"]-np.array([data_dict[k] for k in ["50000100","50000101","50000102","50000103"]])))
            similarity=similarity**4
            # similarity=self.activation_func(similarity)
            return similarity
        for patient in self.data_dicts.keys():
            for risk in self.risky_motion_dict.keys():
                self.data_dicts[patient][risk]=get_similarity(risk,self.data_dicts[patient])
                # self.data_dicts[patient][risk]=list(map(get_similarity,[risk for i in range(len(self.data_dicts[patient]))],self.data_dicts[patient].iterrows()))
                self.data_dicts[patient][risk]=self.activation_func(self.data_dicts[patient][risk])
            
    def object_risk(self):
        for patient in self.data_dicts.keys():
            # 点滴
            self.data_dicts[patient]["40000100"]=1-np.clip(np.sqrt(
                (self.data_dicts[patient]["50001000"]-self.data_dicts[patient]["60010000"])**2+\
                (self.data_dicts[patient]["50001001"]-self.data_dicts[patient]["60010001"])**2
            )/self.spatial_normalization_param,0,1)
            self.data_dicts[patient]["40000100"]=self.activation_func(self.data_dicts[patient]["40000100"])

            # 車椅子
            self.data_dicts[patient]["40000101"]=1-np.clip(np.sqrt(
                (self.data_dicts[patient]["50001010"]-self.data_dicts[patient]["60010000"])**2+\
                (self.data_dicts[patient]["50001011"]-self.data_dicts[patient]["60010001"])**2
            )/self.spatial_normalization_param,0,1)
            self.data_dicts[patient]["40000101"]=self.activation_func(self.data_dicts[patient]["40000101"])

            # 手すり (逆数ではない)
            self.data_dicts[patient]["40000102"]=np.clip(np.sqrt(
                (self.data_dicts[patient]["50001020"]-self.data_dicts[patient]["60010000"])**2+\
                (self.data_dicts[patient]["50001021"]-self.data_dicts[patient]["60010001"])**2
            )/self.spatial_normalization_param,0,1)
            self.data_dicts[patient]["40000102"]=self.activation_func(self.data_dicts[patient]["40000102"])

    def staff_risk(self):
        def direction_risk(patient_x,patient_y,staff_x,staff_y,staff_vx,staff_vy):
            relative_pos=np.array([patient_x-staff_x,patient_y-staff_y])
            relative_vel=np.array([staff_vx,staff_vy])
            cos_theta=np.dot(relative_pos,relative_vel)/(np.linalg.norm(relative_pos)*np.linalg.norm(relative_vel))
            val=1-(cos_theta/2+0.5)
            # if cos_theta>1:
            #     val=0
            # elif cos_theta<0:
            #     val=1
            # else:
            #     val=1-cos_theta
            return val
        for patient in self.data_dicts.keys():
            # 距離リスク
            self.data_dicts[patient]["40000110"]=np.clip(np.sqrt(
                (self.data_dicts[patient]["50001100"]-self.data_dicts[patient]["60010000"])**2+\
                (self.data_dicts[patient]["50001101"]-self.data_dicts[patient]["60010001"])**2
            )/self.spatial_normalization_param,0,1)
            self.data_dicts[patient]["40000110"]=self.activation_func(self.data_dicts[patient]["40000110"])
            # 向きリスク
            self.data_dicts[patient]["40000111"]=direction_risk(
                                                        self.data_dicts[patient]["60010000"],
                                                        self.data_dicts[patient]["60010001"],
                                                        self.data_dicts[patient]["50001100"],
                                                        self.data_dicts[patient]["50001101"],
                                                        self.data_dicts[patient]["50001110"],
                                                        self.data_dicts[patient]["50001111"],
            )
            self.data_dicts[patient]["40000111"]=self.activation_func(self.data_dicts[patient]["40000111"])
            pass

    def fuzzy_multiply(self):
        def judge_left_or_right(tri1,tri2):
            # 型エラーの修正
            if (type(tri1)==float) or (type(tri2)==float):
                return [np.nan,np.nan,np.nan,],[np.nan,np.nan,np.nan,]
            if (type(tri1)==str):
                tri1=eval(tri1)
            if (type(tri2)==str):
                tri2=eval(tri2)
            
            if tri1[1]<tri2[1]:
                return tri1,tri2
            else:
                return tri2,tri1
        def calc_cross_point(left_tri,right_tri):
            x_cross=1/((right_tri[1]-right_tri[0])+(left_tri[2]-left_tri[1]))*(left_tri[2]*(right_tri[1]-right_tri[0])+right_tri[0]*(left_tri[2]-left_tri[1]))
            return x_cross
        for patient in self.data_dicts.keys():
            # 三角形の左右判別
            sort_tri=judge_left_or_right(self.data_dicts[patient]["40000000"],self.data_dicts[patient]["40000001"])#np.array(list(map(judge_left_or_right,self.data_dicts[patient]["40000000"],self.data_dicts[patient]["40000001"])))
            left_tri,right_tri=sort_tri[0],sort_tri[1]
            # sort_tri=np.array(list(map(judge_left_or_right,self.data_dicts[patient]["40000000"],self.data_dicts[patient]["40000001"])))
            # left_tri,right_tri=sort_tri[:,0,:],sort_tri[:,1,:]
            
            # 交点の計算
            x_cross=np.array(calc_cross_point(left_tri,right_tri))#np.array(list(map(calc_cross_point,left_tri,right_tri)))
            # x_cross=np.array(list(map(calc_cross_point,left_tri,right_tri)))

            # 重心の算出
            x_gravity=(left_tri[2]+right_tri[0]+x_cross)/3
            # x_gravity=(left_tri[:,2]+right_tri[:,0]+x_cross)/3
            self.data_dicts[patient]["30000000"]=x_gravity
        pass

    def AHP_weight_sum(self,input_node_codes,output_node_code):
        def weight_sum(input_node_codes):
            features=np.nan_to_num(input_node_codes)
            w_sum=self.AHP_dict[output_node_code]["weights"]@features
            for i,k in enumerate(input_node_codes):
                self.graph_dicts[patient]["weight_dict"][output_node_code][k]=self.AHP_dict[output_node_code]["weights"][i]
            return w_sum
        for patient in self.data_dicts.keys():
            self.data_dicts[patient][output_node_code]=weight_sum([self.data_dicts[patient][input_node_code] for input_node_code in input_node_codes])#list(map(weight_sum,self.data_dicts[patient][input_node_codes].iterrows()))
        pass

    def simple_weight_sum(self,input_node_codes,output_node_code,weights):
        def weight_sum(input_node_codes):
            features=np.nan_to_num(input_node_codes)
            w_sum=np.array(weights)@features
            for i,k in enumerate(input_node_codes):
                self.graph_dicts[patient]["weight_dict"][output_node_code][k]=weights[i]
            return w_sum
        for patient in self.data_dicts.keys():
            self.data_dicts[patient][output_node_code]=weight_sum([self.data_dicts[patient][input_node_code] for input_node_code in input_node_codes])#list(map(weight_sum,self.data_dicts[patient][input_node_codes].iterrows()))
    
    def fuzzy_reasoning_master(self,input_node_codes,output_node_code):
        def ask_risk_to_calculator(input_nodes):
            for k in input_nodes.keys():
                if np.isnan(input_nodes[k]):
                    input_nodes[k]=0
            risk=self.calculate_fuzzy(input_nodes=input_nodes,output_node=output_node_code)
            return risk
        for patient in self.data_dicts.keys():
            self.data_dicts[patient][output_node_code]=ask_risk_to_calculator({input_node_code:self.data_dicts[patient][input_node_code] for input_node_code in input_node_codes})#list(map(ask_risk_to_calculator,self.data_dicts[patient][input_node_codes].iterrows()))
            # self.data_dicts[patient][output_node_code]=list(map(ask_risk_to_calculator,self.data_dicts[patient][input_node_codes].iterrows()))

    def ewm_master(self,input_node_codes,output_node_code,dim="t"):
        horizon=100
        if dim=="t":
            for patient in self.data_dicts.keys():
                for i,row in self.data_dicts[patient].iterrows():
                    score_df=self.data_dicts[patient].loc[np.max(i-horizon,0):i,input_node_codes]
                    weight=self.get_entropy_weight_t(score_df=score_df)
                    self.data_dicts[patient].loc[i,output_node_code]=np.array(list(weight.values()))@self.data_dicts[patient].loc[i,input_node_codes].values
                    if self.data_dicts[patient].loc[i,output_node_code]>1:
                        raise Exception("なんかおかしい")
                    # print(weight)
                    # if i>105 and patient=="B":
                    #     raise NotImplementedError
                    pass
        elif dim=="p": # patient間
            for i,_ in self.data_dicts[list(self.data_dicts.keys())[0]].iterrows():
                score_df=pd.DataFrame()
                for patient in self.data_dicts.keys():
                    score_df=pd.concat([score_df,self.data_dicts[patient].loc[i,input_node_codes]],axis=1)
                score_df=score_df.T
                weight=self.get_entropy_weight(score_df=score_df)
                self.data_dicts[patient].loc[i,output_node_code]=np.array(list(weight.values()))@self.data_dicts[patient].loc[i,input_node_codes].values
                if self.data_dicts[patient].loc[i,output_node_code]>1:
                    raise Exception("なんかおかしい")

            pass
    
    def save_session(self):
        # 重み情報をnetworkxのGに追記していく
        for patient in self.patients:
            for node in self.graph_dicts[patient]["node_dict"].keys():
                node_code_from=node
                node_codes_to=self.graph_dicts[patient]["node_dict"][node_code_from]["node_code_to"]
                self.graph_dicts[patient]["G"].add_edges_from([(node_code_from,node_code_to,{"weight":self.graph_dicts[patient]["weight_dict"][node_code_from][node_code_to]}) for node_code_to in node_codes_to])
        print("# graph保存 #")
        self.write_picklelog(self.graph_dicts,self.data_dir_dict["trial_dir_path"]+"/graph_dicts.pickle")
        self.write_picklelog(self.data_dicts,self.data_dir_dict["trial_dir_path"]+"/data_dicts.pickle")
        self.write_picklelog(self.scenario_dict,self.data_dir_dict["trial_dir_path"]+"/scenario_dict.pickle")
        for patient in self.patients:
            del self.graph_dicts[patient]["G"]
        self.write_json(self.graph_dicts,self.data_dir_dict["trial_dir_path"]+"/graph_dicts.json")
        print("# DataFrame 保存 #")
        for patient in self.patients:
            self.data_dicts[patient].to_csv(self.data_dir_dict["trial_dir_path"]+"/data_"+patient+"_eval.csv",index=False)
    
    def evaluate(self):
        # raise NotImplementedError
        print("# 5 -> 4層推論 #")
        # 内的・静的
        self.fuzzy_logic()
        # 内的・動的
        self.pose_similarity()
        # 外的・静的
        self.object_risk()
        # 外的・動的
        self.staff_risk()

        print("# 4 -> 3層推論 #")
        # 内定・静的
        self.fuzzy_multiply()
        # 内的・動的
        self.AHP_weight_sum(input_node_codes=["40000010","40000011","40000012","40000013","40000014","40000015","40000016"],output_node_code="30000001")
        # 外的・静的
        self.AHP_weight_sum(input_node_codes=["40000100","40000101","40000102"],output_node_code="30000010")
        # 外的・動的
        self.fuzzy_reasoning_master(input_node_codes=["40000110","40000111"],output_node_code="30000011")

        print("# 3 -> 2層推論 #")
        # 内的
        # self.ewm_master(input_node_codes=["30000000","30000001"],output_node_code=20000000,dim="p")
        # self.fuzzy_reasoning_master(input_node_codes=["30000000","30000001"],output_node_code=20000000)
        self.simple_weight_sum(input_node_codes=["30000000","30000001"],output_node_code="20000000",weights=[0.1,0.9])
        # 外的
        # self.ewm_master(input_node_codes=["30000010","30000011"],output_node_code=20000001)
        self.fuzzy_reasoning_master(input_node_codes=["30000010","30000011"],output_node_code="20000001")
        
        print("# 2 -> 1層推論 #")
        # self.ewm_master(input_node_codes=[20000000,20000001],output_node_code=10000000)
        self.fuzzy_reasoning_master(input_node_codes=["20000000","20000001"],output_node_code="10000000")
        return self.data_dicts

if __name__=="__main__":
    trial_name="20250126EnvCheck"
    strage="NASK"
    runtype="simulation"
    cls=Master(trial_name,strage,runtype=runtype)
    cls.evaluate()
    cls.save_session()