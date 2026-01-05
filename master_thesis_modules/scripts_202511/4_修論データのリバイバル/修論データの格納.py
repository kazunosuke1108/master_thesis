import os
from glob import glob
import pandas as pd
import pickle

csv_dir_path="/media/hayashide/MasterThesis/20251211_MasterThesisData"

csv_paths=sorted(glob(csv_dir_path+"/*.csv"))
data_dicts={}
for csv_path in csv_paths:
    data=pd.read_csv(csv_path,header=0)
    data.index=[int(k) for k in data.index]
    print(type(data.index[0]))
    patient=os.path.basename(csv_path).split("_")[1]
    print(patient)
    data_dicts[patient]=data

picklepath=csv_dir_path+"/data_dicts.pickle"
with open(picklepath, mode='wb') as f:
    pickle.dump(data_dicts,f)