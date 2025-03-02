import os
import sys
from pprint import pprint

import numpy as np
import pandas as pd
import scipy.stats as stats


sys.path.append(".")
sys.path.append("..")
if "catkin_ws" in os.getcwd():
    sys.path.append("/catkin_ws/src/master_thesis_modules")
else:
    sys.path.append(os.path.expanduser("~")+"/kazu_ws/master_thesis/master_thesis_modules")
from scripts.management.manager import Manager


csv_path="C:/Users/hyper/kazu_ws/master_thesis/master_thesis_modules/scripts/exp_analysis/preference_order/preference.csv"
data=pd.read_csv(csv_path,header=0,index_col=0).T
print(data)

result_dict={}
result_table_dict={}

case_list=sorted(list(set([k.split("_")[0] for k in data.keys() if "事例" in k])))
condition_list=sorted(list(set([k.split("_")[1] for k in data.keys() if "事例" in k])))

# Friedman検定
alpha=0.05
result_dict["friedman"]={}

for case_name in case_list:
    # データの抽出
    ranks=data[sorted([k for k in data.keys() if case_name in k])].values.astype(int)
    # Friedman検定の実行
    stat, p = stats.friedmanchisquare(*zip(*ranks))

    print(f"統計量: {stat:.4f}, p値: {p:.4f}")

    if p < alpha:
        print(f"{case_name}: 方法間に有意な差がある")
    else:
        print(f"{case_name}: 方法間に有意な差はない")

    result_dict["friedman"][case_name]={}
    result_dict["friedman"][case_name]["統計量"]=np.round(stat,4)
    result_dict["friedman"][case_name]["p値"]=np.round(p,4)
    result_dict["friedman"][case_name]["alpha"]=alpha
    result_dict["friedman"][case_name]["有意差"]=p<alpha


# Wilcoxonの符号付順位和検定
alpha=0.05
result_dict["wilcoxon"]={}
for case_name in case_list:
    result_dict["wilcoxon"][case_name]={}
    result_table_dict[case_name]=pd.DataFrame(index=condition_list,columns=condition_list)
    comparison_pair_list=[
        [case_name+"_"+condition_list[0],case_name+"_"+condition_list[1]],
        [case_name+"_"+condition_list[0],case_name+"_"+condition_list[2]],
        [case_name+"_"+condition_list[1],case_name+"_"+condition_list[2]],
    ]
    for comparison_pair in comparison_pair_list:
        try:
            result_dict["wilcoxon"][case_name][comparison_pair[0]]
        except KeyError:
            result_dict["wilcoxon"][case_name][comparison_pair[0]]={}
        ranks_0=data[comparison_pair[0]].values.astype(int)
        ranks_1=data[comparison_pair[1]].values.astype(int)
        # Wilcoxon符号付き順位検定
        stat, p = stats.wilcoxon(ranks_0, ranks_1)

        print(f"Wilcoxon 検定統計量: {stat:.4f}")
        print(f"p値: {p:.4f}")

        # 有意水準 α=0.05 で判断
        if p < 0.05:
            print(f"{comparison_pair}: 有意な差がある")
        else:
            print(f"{comparison_pair}: 有意な差はない")
        pass
        result_dict["wilcoxon"][case_name][comparison_pair[0]][comparison_pair[1]]={}
        result_dict["wilcoxon"][case_name][comparison_pair[0]][comparison_pair[1]]["alpha"]=alpha
        result_dict["wilcoxon"][case_name][comparison_pair[0]][comparison_pair[1]]["p値"]=np.round(p,4)
        result_dict["wilcoxon"][case_name][comparison_pair[0]][comparison_pair[1]]["有意差"]=p<alpha
        result_dict["wilcoxon"][case_name][comparison_pair[0]][comparison_pair[1]]["統計量"]=np.round(stat,4)
        result_table_dict[case_name].loc[comparison_pair[0].split("_")[1],comparison_pair[1].split("_")[1]]=result_dict["wilcoxon"][case_name][comparison_pair[0]][comparison_pair[1]]["有意差"]
        

pprint(result_dict)

Manager().write_json(dict_data=Manager().convert_np_types(result_dict),json_path=os.path.split(csv_path)[0]+"/result.json")

for case_name in case_list:
    result_table_dict[case_name].to_csv(os.path.split(csv_path)[0]+f"/wilcoxon_{case_name}.csv")