import tkinter
from PIL import Image, ImageTk
from glob import glob

root=tkinter.Tk()
root.title("test")
root.geometry("2000x1000")
jpg_path=sorted(glob("//192.168.1.5/common/FY2024/09_MobileSensing/20250207Dev/jpg/elp/left/*.jpg"))[-1]
img=Image.open(jpg_path)
img = img.convert("RGB")  # RGBモードに変換
img=ImageTk.PhotoImage(img)
canvas=tkinter.Canvas(bg="black",width=1000,height=1000)
canvas.place(x=1000,y=0)
canvas.create_image(0,0,image=img,anchor=tkinter.NW)
root.mainloop()