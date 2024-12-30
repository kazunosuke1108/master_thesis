import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from icecream import ic

class FuzzyReasoning():
    def __init__(self):
        super().__init__()
        self.define_rules()
        pass

    def define_rules(self):
        self.reasoning_rule_dict={
            30000011:{
                1:{
                    "conditions":{40000110:"high",40000111:"high"},
                    "result":"high",
                },
                # 2:{
                #     "conditions":{40000110:"middle",40000111:"high"},
                #     "result":"high",
                # },
                3:{
                    "conditions":{40000110:"low",40000111:"high"},
                    "result":"high",
                },
                # 4:{
                #     "conditions":{40000110:"high",40000111:"middle"},
                #     "result":"middle",
                # },
                # 5:{
                #     "conditions":{40000110:"middle",40000111:"middle"},
                #     "result":"middle",
                # },
                # 6:{
                #     "conditions":{40000110:"low",40000111:"middle"},
                #     "result":"low",
                # },
                7:{
                    "conditions":{40000110:"high",40000111:"low"},
                    "result":"middle",
                },
                # 8:{
                #     "conditions":{40000110:"middle",40000111:"low"},
                #     "result":"low",
                # },
                9:{
                    "conditions":{40000110:"low",40000111:"low"},
                    "result":"low",
                },
            },
        }

    def membership_func(self,x,type="high"):
        def high(x):
            y=x
            return y
        # def middle(x):
        #     if x<=0.5:
        #         y=2*x
        #         return y
        #     elif x>0.5:
        #         y=-2*x+2
        #         return y
        def low(x):
            y=1-x
            return y
        
        if type=="high":
            return high(x)
        # elif type=="middle":
        #     return middle(x)
        elif type=="low":
            return low(x)
    
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
    
    def calculate_fuzzy(self,input_nodes={40000110:0.5,40000111:0.5},output_node=30000011):
        rule_id=output_node
        reasoning_result=0
        for proposition_id in self.reasoning_rule_dict[rule_id]:
            height=1
            for condition in self.reasoning_rule_dict[rule_id][proposition_id]["conditions"].keys():
                h=self.membership_func(x=input_nodes[condition],type=self.reasoning_rule_dict[rule_id][proposition_id]["conditions"][condition])
                height=height*h
            peak,height=self.triangle_func(height,result=self.reasoning_rule_dict[rule_id][proposition_id]["result"])
            # ic(peak,height)
            reasoning_result+=peak*height
        return reasoning_result


    

    def main(self):
        ans=self.calculate_fuzzy(input_nodes={40000110:0.5,40000111:0.5},output_node=30000011)
        print(ans)
        pass


if __name__=="__main__":
    cls=FuzzyReasoning()
    cls.main()