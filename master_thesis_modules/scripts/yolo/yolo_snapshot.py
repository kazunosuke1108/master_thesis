import os
import cv2
from PIL import Image
import logging
logging.getLogger().setLevel(logging.ERROR)

from ultralytics import YOLO # pip install ultralytics
from multiprocessing import cpu_count,Process
import logging
logging.getLogger().setLevel(logging.ERROR)

class mp42pose():
    def __init__(self,original_mp4_path,temporal_strage_directory_path,output_mp4_path):
        self.original_mp4_path=original_mp4_path
        self.temporal_strage_directory_path=temporal_strage_directory_path
        self.output_mp4_path=output_mp4_path
        self.models={}
        # self.models["pose"]=YOLO("yolov8n-pose.pt")
        # self.models["bbox"]=YOLO("yolo11x.pt")

    def mp42jpg(self,mp4_path,jpg_dir_path):
        cap = cv2.VideoCapture(mp4_path)
        if not cap.isOpened():
            raise Exception(f"動画を読み込めませんでした。mp4_path: {mp4_path}")
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps=int(cap.get(cv2.CAP_PROP_FPS))
        n_frame=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        jpg_paths=[]
        for i in range(n_frame):
            if i>100:
                break
            if i%10==0:
                print(i,"/",n_frame)
            ret,frame=cap.read()
            jpg_path=jpg_dir_path+f"/{os.path.basename(jpg_dir_path)}_{str(i).zfill(5)}.jpg"
            cv2.imwrite(jpg_path,frame)
            jpg_paths.append(jpg_path)
        return jpg_paths

    def jpg2mp4(self,image_paths,mp4_path,size=[0,0],fps=30.0):
        # get size of the image
        img=cv2.imread(image_paths[0])
        if size[0]==0:
            size=[img.shape[1],img.shape[0]]
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video = cv2.VideoWriter(mp4_path,fourcc, fps, size)
        for idx,image in enumerate(image_paths):
            img=cv2.imread(image)
            video.write(img)
            print(f"now processing: {os.path.basename(image)} {idx}/{len(image_paths)}")
        video.release()

    def jpg2pose(self,jpg_paths,jpg_pose_dir_path):
        

        plotted_jpg_paths=[]

        def estimate_pose(jpg_path):
            img=Image.open(jpg_path)
            plotted_frame=self.models["pose"](img)[0].plot()
            plotted_jpg_path=jpg_pose_dir_path+"/"+os.path.basename(jpg_path)[:-4]+"_pose.jpg"
            cv2.imwrite(plotted_jpg_path,plotted_frame)
            plotted_jpg_paths.append(plotted_jpg_path)

        p_list=[]
        nProcess=cpu_count()
        for i,jpg_path in enumerate(jpg_paths):
            estimate_pose(jpg_path)
            # if i%10==0:
            #     print(i,"/",len(jpg_paths))
            
            # p=Process(target=estimate_pose,args=(jpg_path,))
            # p_list.append(p)

            # if len(p_list)==nProcess:
            #     for p in p_list:
            #         p.start()
            #     for p in p_list:
            #         p.join()
            #     p_list=[]
            # if i+1==len(jpg_paths):
            #     for p in p_list:
            #         p.start()
            #     for p in p_list:
            #         p.join()
            #     p_list=[]
        plotted_jpg_paths=sorted(plotted_jpg_paths)
        return plotted_jpg_paths
    
    def main(self):
        # mp4->jpg
        self.jpg_paths=self.mp42jpg(mp4_path=self.original_mp4_path,jpg_dir_path=self.temporal_strage_directory_path)

        # jpg -> pose jpg
        self.jpg_pose_dir_path=self.temporal_strage_directory_path+"/pose_jpg"
        os.makedirs(self.jpg_pose_dir_path,exist_ok=True)
        plotted_jpg_paths=self.jpg2pose(self.jpg_paths,self.jpg_pose_dir_path)

        # pose jpg -> mp4
        self.jpg2mp4(plotted_jpg_paths,self.output_mp4_path,fps=30)

    def main_enjoy_yolo(self):
        def estimate_pose(img):
            plotted_frame=self.models["pose"](img)[0].plot()
            return plotted_frame
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()  # フレームを取得
            plotted_frame=estimate_pose(frame)
            cv2.imshow('Webカメラ映像', plotted_frame)  # 映像を表示
            if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q'キーで終了
                break
        cap.release()  # カメラを解放
        cv2.destroyAllWindows()  # ウィンドウを閉じる


    def detect_and_draw_boxes(self,image_path):
        # YOLOv8モデルのロード
        model = YOLO('yolo11x.pt')  # 'yolov8n.pt'はYOLOv8のNanoモデルを指します

        # 画像の読み込み
        image = cv2.imread(image_path)
        if image is None:
            print(f"画像を読み込めませんでした: {image_path}")
            return

        # 画像をRGB形式に変換
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # モデルで推論を実行
        results = model.predict(rgb_image, conf=0.1)

        # 検出結果の処理
        import matplotlib.pyplot as plt
        import random

        # self.colors_01 = plt.get_cmap("tab10").colors
        # self.colors = [(int(b*255), int(g*255), int(r*255)) for r, g, b in self.colors_01]
        # self.colors+=self.colors
        # self.colors+=self.colors

        num_colors = 100  # 100通りの色を用意
        cmap = plt.get_cmap("gist_ncar")  # カラーマップを選択

        self.colors_01 = [cmap(i / num_colors) for i in range(num_colors)]
        self.colors = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b, _ in self.colors_01]
        random.shuffle(self.colors)

        for result in results:
            boxes = result.boxes  # 検出されたバウンディングボックス
            for i,box in enumerate(boxes):
                cls = int(box.cls[0])  # クラスID
                if cls == 0:  # クラスIDが0の場合、'person'を指します
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # バウンディングボックスの座標
                    # バウンディングボックスを画像に描画
                    cv2.rectangle(image, (x1, y1), (x2, y2), self.colors[i], 2)

        # 結果の画像を保存
        output_path = image_path.replace('.jpg', '_detected.jpg')
        cv2.imwrite(output_path, image)
        print(f"検出結果の画像を保存しました: {output_path}")


if __name__=="__main__":
    original_mp4_path=r"C:\Users\kimura\Posture_Estimation\YOLO\movie\魚眼うろうろ.MP4"
    temporal_strage_directory_path=r"C:\Users\kimura\Posture_Estimation\YOLO\result\pose_images\魚眼うろうろ"
    output_mp4_path=r"C:\Users\kimura\Posture_Estimation\YOLO\result\processed_魚眼うろうろ.mp4"
    
    cls=mp42pose(original_mp4_path=original_mp4_path,temporal_strage_directory_path=temporal_strage_directory_path,output_mp4_path=output_mp4_path)
    # cls.main()
    # cls.main_enjoy_yolo()
    from glob import glob
    img_paths=sorted(glob("/catkin_ws/src/master_thesis_modules/database/20250226SampleVoice/*.jpg"))
    print(img_paths)
    for img_path in img_paths:
        if "detected" not in img_path:
            print(img_path)
            cls.detect_and_draw_boxes(img_path)