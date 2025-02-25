import os
import sys
import copy
import time
from glob import glob
import json
import cv2
import atexit
from icecream import ic
from pprint import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# pip install watchdog
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

sys.path.append(".")
sys.path.append("..")
if "catkin_ws" in os.getcwd():
    sys.path.append("/catkin_ws/src/master_thesis_modules")
else:
    sys.path.append(os.path.expanduser("~")+"/kazu_ws/master_thesis/master_thesis_modules")
from scripts.management.manager import Manager
from scripts.notification.notification import Notification

# 定数
WATCHED_FILES = ["notify_dict.json"]

class JSONFileChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.is_directory:
            return
        if os.path.basename(event.src_path) in WATCHED_FILES:
            print(f"File changed: {event.src_path}")
            cls.play_sound()
            cls.play_voice()
            
class NotificationPlayer(Manager):
    def __init__(self,trial_name,strage):
        super().__init__()
        self.trial_name=trial_name
        self.strage=strage
        self.data_dir_dict=self.get_database_dir(trial_name=self.trial_name,strage=self.strage)
        self.cls_notification=Notification(trial_name=self.trial_name,strage=self.strage)
        self.cls_notification.play_only_chime()

    def play_sound(self):
        self.cls_notification.play_only_chime()
        pass

    def play_voice(self):
        json_path=self.data_dir_dict["mobilesensing_dir_path"]+"/json/notify_dict.json"
        notify_dict=self.load_json(json_path=json_path)
        text=notify_dict["sentence"]
        mp3_path=self.data_dir_dict["mobilesensing_dir_path"]+f"/mp3/{str(notify_dict['notificationId']).zfill(3)}.mp3"
        self.cls_notification.export_audio(text=text,mp3_path=mp3_path)
        self.cls_notification.play_mp3(mp3_path=mp3_path)
        pass
        
    


if __name__=="__main__":
    trial_name="20250224NameDict"
    strage="local"
    json_dir_path="/catkin_ws/src/database"+"/"+trial_name+"/json"
    cls=NotificationPlayer(trial_name,strage)
    
    # path = "/media/hayashide/MobileSensing/20250207Dev/json"  # 監視するディレクトリ
    event_handler = JSONFileChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, json_dir_path, recursive=False)
    while True:
        try:
            observer.start()
            break
        except FileNotFoundError:
            print("notify_dict.json not found")
            time.sleep(0.1)
            continue        
    print("Observation started")

    try:
        while True:
            time.sleep(1)  # イベントループ
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

