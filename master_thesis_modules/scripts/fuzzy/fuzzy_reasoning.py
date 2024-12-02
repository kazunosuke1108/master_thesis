import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from icecream import ic

class FuzzyReasoning():
    def __init__(self):
        self.define_rules()
        pass

    def define_rules(self):
        self.rule_dict={
            405:{
                405.1:{
                    "conditions":{4051:"up",4052:"up"},
                    "result":"high",
                },
                405.2:{
                    "conditions":{4051:"up",4052:"down"},
                    "result":"middle",
                },
                405.3:{
                    "conditions":{4051:"down",4052:"up"},
                    "result":"middle",
                },
                405.4:{
                    "conditions":{4051:"down",4052:"down"},
                    "result":"low",
                },
            }
        }

    def membership_func(self,x,type="up"):
        def up(x):
            y=x
            return y
        def down(x):
            y=1-x
            return y
        
        if type=="up":
            return up(x)
        elif type=="down":
            return down(x)
    
    def triangle_func(self,height,result):
        if result=="low":
            peak=0
        elif result=="middle":
            peak=0.5
        elif result=="high":
            peak=1
        else:
            raise KeyError
        return (peak,height)
    
    def calculate_fuzzy(self,values={4051:0.2,4052:0.6}):
        rule_id=int(str(list(values.keys())[0])[:3])
        reasoning_result=0
        for proposition_id in self.rule_dict[rule_id]:
            height=1
            for condition in self.rule_dict[rule_id][proposition_id]["conditions"].keys():
                h=self.membership_func(x=values[condition],type=self.rule_dict[rule_id][proposition_id]["conditions"][condition])
                height=height*h
            peak,height=self.triangle_func(height,result=self.rule_dict[rule_id][proposition_id]["result"])
            ic(peak,height)
            reasoning_result+=peak*height
        print(reasoning_result)


    

    def main(self):
        self.calculate_fuzzy(values={4051:0.2,4052:0.6})
        pass


if __name__=="__main__":
    cls=FuzzyReasoning()
    cls.main()