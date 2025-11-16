import pandas as pd
import os
import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append(os.path.expanduser("~")+"/kazu_ws/master_thesis/master_thesis_modules")
from scripts.AHP.get_comparison_mtx_v3 import *

csv_path = "/home/hayashide/kazu_ws/master_thesis/master_thesis_modules/scripts_202511/2_中村さんと百武さんのAHPとFuzzy/questionaire_1a.csv"
data = pd.read_csv(csv_path,index_col="1a",header=0)
print(data)

# データの数値の読み替え (1->9, 5->5)
convert_dict={
    1:9,
    2:7,
    3:5,
    4:3,
    5:1,
    6:1/3,
    7:1/5,
    8:1/7,
    9:1/9,

}
for staff_name in data.columns:
    for i in data.index:
        print(i,staff_name)
        data.loc[i,staff_name]=convert_dict[data.loc[i,staff_name]]

print(data)

# 内的・動的（動作）30000001
criteria = [
    "立つ",
    "車椅子のブレーキを解除する",
    "車椅子を動かす",
    "姿勢を崩す",
    "挙手する",
    "せき込む",
    "顔を触る",
]
for staff_name in data.columns:
    comparison_answer = data[staff_name].tolist()[:21]
    cls = getConsistencyMtx()
    A=cls.get_comparison_mtx(criteria,comparison_answer)
    eigvals,eigvecs,max_eigval,weights,CI = cls.evaluate_mtx(A)
    comparison_mtx_csv_path=f"/home/hayashide/kazu_ws/master_thesis/master_thesis_modules/database/common/comparison_mtx_30000001_{staff_name}.csv"
    pd.DataFrame(A).to_csv(comparison_mtx_csv_path,index=False,header=False)
    print(CI)

# 外的・静的（物体）30000010
criteria = [
    "点滴の近くにいること",
    "点滴の近くにいること",
    "手すりから離れていること",
]
for staff_name in data.columns:
    comparison_answer = data[staff_name].tolist()[21:24]
    cls = getConsistencyMtx()
    A=cls.get_comparison_mtx(criteria,comparison_answer)
    eigvals,eigvecs,max_eigval,weights,CI = cls.evaluate_mtx(A)
    comparison_mtx_csv_path=f"/home/hayashide/kazu_ws/master_thesis/master_thesis_modules/database/common/comparison_mtx_30000010_{staff_name}.csv"
    pd.DataFrame(A).to_csv(comparison_mtx_csv_path,index=False,header=False)
    print(CI)