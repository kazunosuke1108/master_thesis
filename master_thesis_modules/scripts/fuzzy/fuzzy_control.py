import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from icecream import ic

class FuzzyControl():
    def __init__(self):
        super().__init__()
        self.define_control_rules()
        pass

    def define_control_rules(self):
        self.control_rule_dict={# self.control_rule_dict[s][ds]["result"] -> ans
            "+":{
                "+":{
                    "conditions":{"s":"+","ds":"+"},
                    "result":"++",
                },
                "o":{
                    "conditions":{"s":"+","ds":"o"},
                    "result":"+",
                },
                "-":{
                    "conditions":{"s":"+","ds":"-"},
                    "result":"o",
                },
            },
            "o":{
                "+":{
                    "conditions":{"s":"middle","ds":"+"},
                    "result":"+",
                },
                "o":{
                    "conditions":{"s":"middle","ds":"o"},
                    "result":"o",
                },
                "-":{
                    "conditions":{"s":"middle","ds":"-"},
                    "result":"-",
                },
            },
            "-":{
                "+":{
                    "conditions":{"s":"-","ds":"+"},
                    "result":"o",
                },
                "o":{
                    "conditions":{"s":"-","ds":"o"},
                    "result":"-",
                },
                "-":{
                    "conditions":{"s":"-","ds":"-"},
                    "result":"--",
                },
            }
        }
        self.s_thre_dict={
            "-":{"min":0,"max":1/3},
            "o":{"min":1/3,"max":2/3},
            "+":{"min":2/3,"max":1},
        }
        ds_threshold=0.2*0.25 # thre[/s]*dt[s]
        self.ds_thre_dict={
            "-":{"min":-np.inf,"max":-ds_threshold},
            "o":{"min":-ds_threshold,"max":ds_threshold},
            "+":{"min":ds_threshold,"max":np.inf},
        }
        self.dfps_dict={
            "--":-2,
            "-":-1,
            "o":0,
            "+":1,
            "++":2,
        }
        self.fps_clip_dict={
            "min":0.25,
            "max":4,
        }

    def get_control_input(self,data,i,evaluate_col=1000):
        s_value=data.loc[i,evaluate_col]
        if i==0:
            ds_value=0
        else:
            ds_value=data.loc[i,evaluate_col]-data.loc[i-1,evaluate_col]
        for key in self.s_thre_dict.keys():
            if (self.s_thre_dict[key]["min"]<=s_value) and (s_value<=self.s_thre_dict[key]["max"]):
                s=key
        for key in self.ds_thre_dict.keys():
            if (self.ds_thre_dict[key]["min"]<=ds_value) and (ds_value<=self.ds_thre_dict[key]["max"]):
                ds=key
        dfps=self.control_rule_dict[s][ds]["result"]
        dfps_value=self.dfps_dict[dfps]
        return dfps_value
    
    def update_active(self,data,i,fps):
        next_timestamp=data.loc[i,"timestamp"]+1/fps
        closest_index=np.argmin(abs(data["timestamp"]-next_timestamp))
        if closest_index==-1:
            raise Exception(f"何かおかしい: i={i} next_timestamp={next_timestamp} fps={fps}")
        data.loc[closest_index,"active"]=1
        data.loc[closest_index,"fps"]=fps
        return data
    

    def main(self):
        fps=1
        dt=0.25
        t=np.arange(0,8.1,dt)
        self.data=pd.DataFrame(t,columns=["timestamp"])
        self.data[1000]=np.random.random(t.shape)
        self.data["active"]=0
        self.data["fps"]=np.nan
        self.data.loc[0,"active"]=1
        for i in range(len(self.data)):
            if self.data.loc[i,"active"]==0:
                continue
            dfps=self.get_control_input(self.data,i,evaluate_col=1000)
            fps+=dfps
            fps=np.clip(fps,self.fps_clip_dict["min"],self.fps_clip_dict["max"])
            self.data=self.update_active(self.data,i,fps)


if __name__=="__main__":
    cls=FuzzyControl()
    cls.main()