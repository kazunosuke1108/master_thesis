import os
import shutil
from pprint import pprint

from glob import glob

from multiprocessing import cpu_count,Process


rosbag_paths=sorted(glob("/media/hayashide/YTHDD2024C/rosbag/*-19-*.bag"))
pprint(rosbag_paths)

def copy_rosbag(p):
    print(f"now processing... {os.path.basename(p)}")
    os.system(f"cp {rosbag_path} /media/hayashide/ExtremePro/rosbag/{os.path.basename(p)}")
    # shutil.copy(rosbag_path,"/media/hayashide/ExtremePro/rosbag/"+os.path.basename(p))
    pass

n_process=cpu_count()
p_list=[]
for i,rosbag_path in enumerate(rosbag_paths):
    p=Process(target=copy_rosbag,args=(rosbag_path,))
    p_list.append(p)
    if (len(p_list)==n_process) or (i+1==len(rosbag_paths)):
        for p in p_list:
            p.start()
        for p in p_list:
            p.join()
        p_list=[]