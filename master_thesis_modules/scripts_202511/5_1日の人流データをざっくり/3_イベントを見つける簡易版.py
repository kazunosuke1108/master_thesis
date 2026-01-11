import os
import sys
from glob import glob

# --- import Visualizer path ---
sys.path.append(".")
sys.path.append("..")
sys.path.append("...")
sys.path.append(os.path.expanduser("~") + "/kazu_ws/master_thesis/master_thesis_modules")
from scripts.visualize.visualizer_v5 import Visualizer

jpg_paths = sorted(glob("/home/hayashide/kazu_ws/master_thesis/master_thesis_modules/database/extract/*.jpg"))
mp4_path = "/home/hayashide/kazu_ws/master_thesis/master_thesis_modules/database/extract/summary.mp4"

visualizer_parent = Visualizer(trial_name="20260111_findEvent", strage="NASK")
trial_dir_path = visualizer_parent.data_dir_dict["trial_dir_path"]


visualizer_parent.jpg2mp4_basename(image_paths=jpg_paths, mp4_path=mp4_path, fps=2)

# from glob import glob
# import os
# import sys
# import shutil
# import traceback
# from multiprocessing import cpu_count, Process, Queue
# from datetime import datetime

# # --- logging helper ---
# def log(msg: str):
#     print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# # --- import Visualizer path ---
# sys.path.append(".")
# sys.path.append("..")
# sys.path.append("...")
# sys.path.append(os.path.expanduser("~") + "/kazu_ws/master_thesis/master_thesis_modules")
# from scripts.visualize.visualizer_v5 import Visualizer

# EXTRACT_DIR = "/media/hayashide/MobileSensing/Nagasaki20240827/jpg/extract"
# PATTERN = EXTRACT_DIR + "/N*"

# event_dir_paths = sorted(glob(PATTERN))
# print(event_dir_paths)

# visualizer_parent = Visualizer(trial_name="20260110_findEvent", strage="NASK")
# trial_dir_path = visualizer_parent.data_dir_dict["trial_dir_path"]

# for event_dir_path in event_dir_paths:
#     event_name=os.path.basename(event_dir_path)
#     l_jpg_paths = sorted(glob(event_dir_path+"/*_L_*.jpg"))
#     l_pickup_index = int(len(l_jpg_paths)/2)
#     r_jpg_paths = sorted(glob(event_dir_path+"/*_R_*.jpg"))
#     r_pickup_index = int(len(r_jpg_paths)/2)
#     l_jpg_path = l_jpg_paths[l_pickup_index]
#     r_jpg_path = r_jpg_paths[r_pickup_index]
#     shutil.copy(l_jpg_path,trial_dir_path+f"/{event_name}_{os.path.basename(l_jpg_path)}")
#     shutil.copy(r_jpg_path,trial_dir_path+f"/{event_name}_{os.path.basename(r_jpg_path)}")
#     raise NotImplementedError