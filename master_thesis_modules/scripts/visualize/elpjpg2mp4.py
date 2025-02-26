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

elp_image_dir_paths=sorted(glob("/home/hayashide/ytlab_ros_ws/ytlab_handheld_sensoring_system/ytlab_handheld_sensoring_system_modules/database/20250226ELP0*/jpg/elp/right"))[34:]
print(elp_image_dir_paths)

for elp_image_dir_path in elp_image_dir_paths:
    elp_image_paths=sorted(glob(elp_image_dir_path+"/*.jpg"))
    mp4_path=elp_image_dir_path[:-len("/jpg/elp/right")]+"/mp4/elp.mp4"
    
    Manager().jpg2mp4(image_paths=elp_image_paths,mp4_path=mp4_path)