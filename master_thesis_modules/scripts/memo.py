import cv2
import matplotlib.pyplot as plt

img=cv2.imread("/catkin_ws/src/database/20250215NagasakiShort3/jpg/elp/left/l_1733394820.7874.jpg")
plt.imshow(img)
plt.show()
t,b,l,r=256,669,465,584