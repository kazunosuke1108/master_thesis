from glob import glob 
import pandas as pd
import sys
import os
from multiprocessing import cpu_count,Process

sys.path.append(".")
sys.path.append("..")
sys.path.append("...")
sys.path.append(os.path.expanduser("~")+"/kazu_ws/master_thesis/master_thesis_modules")
from scripts.visualize.visualizer_v5 import Visualizer

extract_dir_path="/media/hayashide/MobileSensing/Nagasaki20240827/jpg/extract"
# event_dir_paths=sorted(glob(extract_dir_path+"/N*"))
# raise NotImplementedError
print("A")
extract_dir_path="/media/hayashide/MobileSensing/Nagasaki20240827/jpg/extract"
print("B")
pattern = extract_dir_path + "/N*"
print("pattern:", pattern)
print("C: before glob")
event_dir_paths = sorted(glob(pattern))
print("D: after glob", len(event_dir_paths))
raise NotImplementedError

visualizer=Visualizer(trial_name="20260110_findEvent",strage="NASK")
trial_dir_path=visualizer.data_dir_dict["trial_dir_path"]

def create_mp4(event_dir_path):
    event_name=os.path.basename(event_dir_path)
    jpg_paths=sorted(glob(event_dir_path+"/*.jpg"))
    visualizer.jpg2mp4(image_paths=jpg_paths,mp4_path=trial_dir_path+"/"+event_name+".mp4",fps=20)

nprocess=cpu_count()
p_list=[]

for i,event_dir_path in enumerate(event_dir_paths):
    p=Process(target=create_mp4,args=(event_dir_path,))
    p_list.append(p)
    print(f"Prepared process for {i}/{len(event_dir_paths)}: {event_dir_path}")
    if len(p_list)==nprocess or i+1==len(event_dir_paths):
        for p in p_list:
            p.start()
        for p in p_list:
            p.join()
        p_list=[]