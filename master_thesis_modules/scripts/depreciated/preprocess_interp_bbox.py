import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_path="/media/hayashide/MobileSensing/PullWheelchairObaachan/csv/annotation/PullWheelchairObaachan_annotation_ytnpc2021h_20240827_192540New_fullimagePath.csv"

data=pd.read_csv(csv_path,header=0)
plt.plot(data["timestamp"],data["ID_00000_bbox_lowerX"],"-o")
for k in data.keys():
    if "bbox" not in k:
        continue
    data[k]=data[k].interpolate()
data.to_csv(csv_path,index=False)
