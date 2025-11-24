# 無印から派生。同じく無印から派生したV3とは互換性がない。

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from icecream import ic

class FuzzyReasoning():
    def __init__(self):
        super().__init__()
        self.define_rules()
        pass

    def define_custom_rules(self,TFN_data):
        self.reasoning_rule_dict={
            30000011:{
                1:{
                    "conditions":{40000110:"high",40000111:"high"},
                    "result":"high",
                },
                2:{
                    "conditions":{40000110:"low",40000111:"high"},
                    "result":"high",
                },
                3:{
                    "conditions":{40000110:"high",40000111:"low"},
                    "result":"middle",
                },
                4:{
                    "conditions":{40000110:"low",40000111:"low"},
                    "result":"low",
                },
            },
            20000000:{
                1:{
                    "conditions":{30000000:"high",30000001:"high"},
                    "result":TFN_data.loc[0,"c"],
                },
                2:{
                    "conditions":{30000000:"high",30000001:"low"},
                    "result":TFN_data.loc[1,"c"],
                },
                3:{
                    "conditions":{30000000:"low",30000001:"high"},
                    "result":TFN_data.loc[2,"c"],
                },
                4:{
                    "conditions":{30000000:"low",30000001:"low"},
                    "result":TFN_data.loc[3,"c"],
                },
            },
            20000001:{
                1:{
                    "conditions":{30000010:"high",30000011:"high"},
                    "result":TFN_data.loc[4,"c"],
                },
                2:{
                    "conditions":{30000010:"high",30000011:"low"},
                    "result":TFN_data.loc[5,"c"],
                },
                3:{
                    "conditions":{30000010:"low",30000011:"high"},
                    "result":TFN_data.loc[6,"c"],
                },
                4:{
                    "conditions":{30000010:"low",30000011:"low"},
                    "result":TFN_data.loc[7,"c"],
                },
            },
            10000000:{
                1:{
                    "conditions":{20000000:"high",20000001:"high"},
                    "result":TFN_data.loc[8,"c"],
                },
                2:{
                    "conditions":{20000000:"high",20000001:"low"},
                    "result":TFN_data.loc[9,"c"],
                },
                3:{
                    "conditions":{20000000:"low",20000001:"high"},
                    "result":TFN_data.loc[10,"c"],
                },
                4:{
                    "conditions":{20000000:"low",20000001:"low"},
                    "result":TFN_data.loc[11,"c"],
                },
            },
            70000000:{
                1:{
                    "conditions":{70000000:"high",70000001:"high"},
                    "result":"high",
                },
                2:{
                    "conditions":{70000000:"high",70000001:"low"},
                    "result":"middle",
                },
                3:{
                    "conditions":{70000000:"low",70000001:"high"},
                    "result":"middle",
                },
                4:{
                    "conditions":{70000000:"low",70000001:"low"},
                    "result":"low",
                },
            },
        }
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
            20000000:{
                1:{
                    "conditions":{30000000:"high",30000001:"high"},
                    "result":"high",
                },
                2:{
                    "conditions":{30000000:"high",30000001:"low"},
                    "result":"high",
                },
                3:{
                    "conditions":{30000000:"low",30000001:"high"},
                    "result":"middle",
                },
                4:{
                    "conditions":{30000000:"low",30000001:"low"},
                    "result":"low",
                },
            },
            20000001:{
                1:{
                    "conditions":{30000010:"high",30000011:"high"},
                    "result":"high",
                },
                2:{
                    "conditions":{30000010:"high",30000011:"low"},
                    "result":"middle",
                },
                3:{
                    "conditions":{30000010:"low",30000011:"high"},
                    "result":"middle",
                },
                4:{
                    "conditions":{30000010:"low",30000011:"low"},
                    "result":"low",
                },
            },
            10000000:{
                1:{
                    "conditions":{20000000:"high",20000001:"high"},
                    "result":"high",
                },
                2:{
                    "conditions":{20000000:"high",20000001:"low"},
                    "result":"middle",
                },
                3:{
                    "conditions":{20000000:"low",20000001:"high"},
                    "result":"middle",
                },
                4:{
                    "conditions":{20000000:"low",20000001:"low"},
                    "result":"low",
                },
            },
            70000000:{
                1:{
                    "conditions":{70000000:"high",70000001:"high"},
                    "result":"high",
                },
                2:{
                    "conditions":{70000000:"high",70000001:"low"},
                    "result":"middle",
                },
                3:{
                    "conditions":{70000000:"low",70000001:"high"},
                    "result":"middle",
                },
                4:{
                    "conditions":{70000000:"low",70000001:"low"},
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
            if type(result)!=str:
                peak=result
            else:
                raise KeyError
            # raise KeyError
        return (peak,height)
    
    def calculate_fuzzy(self,input_nodes={40000110:0.5,40000111:0.5},output_node=30000011):
        rule_id=output_node
        reasoning_result=0
        try:
            proposition_ids=self.reasoning_rule_dict[rule_id]
        except KeyError:
            proposition_ids=self.reasoning_rule_dict[int(rule_id)]
        for proposition_id in proposition_ids:
            proposition_id=int(proposition_id)
            height=1
            for condition in proposition_ids[proposition_id]["conditions"].keys():
                print(input_nodes)
                print(condition)
                print(input_nodes[condition])
                h=self.membership_func(x=input_nodes[condition],type=self.reasoning_rule_dict[rule_id][proposition_id]["conditions"][condition])
                height=height*h
            peak,height=self.triangle_func(height,result=self.reasoning_rule_dict[rule_id][proposition_id]["result"])
            # ic(peak,height)
            reasoning_result+=peak*height
        return reasoning_result

    def main(self):
        # ans=self.calculate_fuzzy(input_nodes={40000110:0.5,40000111:0.5},output_node=30000011)
        ans=self.calculate_fuzzy(input_nodes={30000000:0.5,30000001:0.5},output_node=20000000)
        print(ans)
        pass


if __name__=="__main__":
    cls=FuzzyReasoning()
    # cls.main()
    import pandas as pd
    staff_name="中村"
    TFN_csv_path=f"/media/hayashide/MasterThesis/common/TFN_{staff_name}.csv"
    TFN_data = pd.read_csv(TFN_csv_path,names=["l","c","r"])
    cls.define_custom_rules(TFN_data)
    cls.main()
    print(TFN_data)