import os
import cv2
from ultralytics import YOLO

img_path="C:/Users/hyper/OneDrive - keio.jp/M2_研究/06_修論/図表/png/yolo_sample.png"
model=YOLO("yolo11x-pose.pt")

img=cv2.imread(img_path)
result=model(img)[0].plot()
cv2.imwrite(img_path[:-4]+"_skeleton.png",result)