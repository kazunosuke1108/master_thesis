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

import tkinter
from PIL import Image, ImageTk
from glob import glob

sys.path.append(".")
sys.path.append("..")
if "catkin_ws" in os.getcwd():
    sys.path.append("/catkin_ws/src/master_thesis_modules")
else:
    sys.path.append(os.path.expanduser("~")+"/kazu_ws/master_thesis/master_thesis_modules")
from scripts.management.manager import Manager

class GUI(Manager):
    def __init__(self):
        super().__init__()
        self.root=tkinter.Tk()
        self.root.title("モニタリング コンソール")
        self.root.geometry("1000x700")

        self.layout_dict={
            "elp":{"t":0,"b":360,"l":360,"r":1000},
        }

    def read_image(self,img_path):
        image_bgr=cv2.imread(img_path)
        return image_bgr

    def convert_cv2tk(self,image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # imreadはBGRなのでRGBに変換
        image_pil = Image.fromarray(image_rgb) # RGBからPILフォーマットへ変換
        image_tk  = ImageTk.PhotoImage(image_pil) # ImageTkフォーマットへ変換
        return image_tk


    def draw_elp(self,bbox=False):
        print("drawing elp")
        w=self.layout_dict["elp"]["r"]-self.layout_dict["elp"]["l"]
        h=self.layout_dict["elp"]["b"]-self.layout_dict["elp"]["t"]
        s=time.time()
        jpg_path=sorted(glob("//NASK/common/FY2024/09_MobileSensing/20250207Dev/jpg/elp/left/*.jpg"))[-1] # これめちゃめちゃ遅い．path指定にしたい
        print(time.time()-s)
        elp_image_bgr=self.read_image(jpg_path) 
        print(time.time()-s)
        elp_image_bgr=cv2.resize(elp_image_bgr,(w,h))
        self.elp_img_tk=self.convert_cv2tk(image_bgr=elp_image_bgr)
        print("image ready")
        if bbox:
            raise NotImplementedError
        else:
            canvas=tkinter.Canvas(bg="black",width=w,height=h)
            canvas.create_image(0,0,image=self.elp_img_tk,anchor=tkinter.NW)
            canvas.place(x=self.layout_dict["elp"]["l"],y=self.layout_dict["elp"]["t"])


    def main(self):
        self.draw_elp()
        self.root.mainloop()
        pass

if __name__=="__main__":
    cls=GUI()
    cls.main()
    pass
