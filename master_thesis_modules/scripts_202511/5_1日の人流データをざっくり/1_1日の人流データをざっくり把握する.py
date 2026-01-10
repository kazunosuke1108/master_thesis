from glob import glob 
import pandas as pd
import sys
import os
from collections import defaultdict
sys.path.append(".")
sys.path.append("..")
sys.path.append("...")
sys.path.append(os.path.expanduser("~")+"/kazu_ws/master_thesis/master_thesis_modules")
from scripts.visualize.visualizer_v5 import Visualizer

csv_dir_path="/media/hayashide/MobileSensing/Nagasaki20240827/csv/raw"

data_list=[]
csv_paths=sorted(glob(csv_dir_path+"/*"))
"""
csvのファイル名は(実験名)_(raw)_(センサ名)_(日付8桁)_(時刻6桁).csv
CSVファイルはおよそ30分おきに1度生成される。
全てのセンサについて、1時間分のデータをまとめて1枚の図に可視化する。（例：12時台の全センサのデータ）
"""


# hour (YYYYMMDD_HH) -> list of dataframes for that hour
hourly_data=defaultdict(list)

for csv_path in csv_paths:
    print(csv_path)
    data=pd.read_csv(csv_path,header=0)
    # _x, _yで終わる列が全てNaNの行を削除
    data=data.dropna(subset=[col for col in data.columns if col.endswith(("_x","_y"))],how="all")
    basename=os.path.basename(csv_path)[:-4].split("_")
    date_str=basename[-2]
    hour_str=basename[-1][:2]
    hour_key=f"{date_str}_{hour_str}"
    hourly_data[hour_key].append(data)

# 1時間ごとに全センサのデータを縦結合して渡す
hour_labels=[]
data_list=[]
for hour_key in sorted(hourly_data.keys()):
    concat_df=pd.concat(hourly_data[hour_key],ignore_index=True,sort=False)
    hour_labels.append(hour_key)
    data_list.append(concat_df)

visualizer=Visualizer(trial_name="20260105_test",strage="NASK")
visualizer.draw_positions2(data_list=data_list,labels=hour_labels)
