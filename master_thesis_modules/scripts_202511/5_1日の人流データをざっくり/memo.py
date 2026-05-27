import os
import cv2
jpg_path="//192.168.1.5/common/FY2024/09_MobileSensing/Nagasaki20240827/jpg/hmnpc2023e/NoTrialNameGiven_hmnpc2023e_1724753174.6143372_001.jpg"

img=cv2.imread(jpg_path)
cv2.imshow("bbox",img)
cv2.waitKey(0)
# 画像の中央のピクセルの色を取得
height, width, _ = img.shape
center_pixel = img[height // 2, width // 2]
print("Center pixel BGR value:", center_pixel)
print(os.path.isfile(jpg_path))