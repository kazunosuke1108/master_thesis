import os
import sys
from glob import glob

sys.path.append(".")
sys.path.append("..")
if "catkin_ws" in os.getcwd():
    sys.path.append("/catkin_ws/src/master_thesis_modules")
else:
    sys.path.append(os.path.expanduser("~")+"/kazu_ws/master_thesis/master_thesis_modules")
from scripts.management.manager import Manager

bbox_image_paths=sorted(glob("/home/hayashide/ytlab_ros_ws/ytlab_handheld_sensoring_system/ytlab_handheld_sensoring_system_modules/database/20250227sleep6/jpg/bbox/*.jpg"))
mp4_path="/home/hayashide/ytlab_ros_ws/ytlab_handheld_sensoring_system/ytlab_handheld_sensoring_system_modules/database/20250227sleep6/mp4/bbox.mp4"

Manager().jpg2mp4(image_paths=bbox_image_paths,mp4_path=mp4_path,fps=2)