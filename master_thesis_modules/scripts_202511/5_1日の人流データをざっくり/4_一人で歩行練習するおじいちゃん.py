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
import argparse


start_from_scratch=False
if start_from_scratch:
    # csv_dir_path="/media/hayashide/MobileSensing/Nagasaki20240827/csv/raw"
    csv_dir_path="C:/Users/hayashide/Downloads/Nagasaki20240827_夜の一部/csv/raw"

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
    # parse optional start/end unix seconds from command line
    start_timestamp=1724752800
    end_timestamp=1724753800
    hour_labels=[]
    data_list=[]
    for hour_key in sorted(hourly_data.keys()):
        concat_df=pd.concat(hourly_data[hour_key],ignore_index=True,sort=False)
        # if timestamp filtering requested, coerce and apply
        if (start_timestamp is not None) or (end_timestamp is not None):
            if "timestamp" in concat_df.columns:
                concat_df["timestamp"] = pd.to_numeric(concat_df["timestamp"], errors="coerce")
                mask = pd.Series([True]*len(concat_df))
                if start_timestamp is not None:
                    mask &= concat_df["timestamp"] >= start_timestamp
                if end_timestamp is not None:
                    mask &= concat_df["timestamp"] <= end_timestamp
                concat_df = concat_df[mask]
            else:
                print("Warning: 'timestamp' column not found; skipping time filter")
        # only include non-empty dataframes
        if len(concat_df) == 0:
            continue
        hour_labels.append(hour_key)
        data_list.append(concat_df)


    visualizer=Visualizer(trial_name="20260111_test",strage="local")
    for i,data in enumerate(data_list):
        data.to_csv(visualizer.data_dir_dict["trial_dir_path"]+f"/{os.path.basename(visualizer.data_dir_dict['trial_dir_path'])}_concat_df_{i}.csv",index=False)


visualizer=Visualizer(trial_name="20260111_test",strage="local")
csv_paths=sorted(glob(visualizer.data_dir_dict["trial_dir_path"]+f"/*_concat_df_*.csv"))
data_list=[]
for csv_path in csv_paths:
    data=pd.read_csv(csv_path,header=0)
    data_list.append(data)

visualizer.draw_positions2(data_list=data_list,labels=["20240827_18","20240827_19"])
# raise NotImplementedError