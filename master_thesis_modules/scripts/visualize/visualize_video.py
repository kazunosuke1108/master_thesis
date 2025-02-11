import os
import sys
from glob import glob
from pprint import pprint
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.append(".")
sys.path.append("..")
sys.path.append(os.path.expanduser("~")+"/kazu_ws/master_thesis/master_thesis_modules")
from scripts.management.manager import Manager

import numpy as np
import pandas as pd

import cv2

from multiprocessing import cpu_count,Process


class VideoVisualizer(Manager):
    def __init__(self,sensing_trial_name,evaluation_trial_name,notification_trial_name,visualize_trial_name,):
        super().__init__()
        self.sensing_trial_name=sensing_trial_name
        self.evaluation_trial_name=evaluation_trial_name
        self.notification_trial_name=notification_trial_name
        self.visualize_trial_name=visualize_trial_name
        self.sensing_dir_dict=self.get_database_dir(trial_name=sensing_trial_name,strage="NASK")
        self.evaluation_dir_dict=self.get_database_dir(trial_name=evaluation_trial_name,strage="NASK")
        self.notification_dir_dict=self.get_database_dir(trial_name=notification_trial_name,strage="NASK")
        self.visualize_dir_dict=self.get_database_dir(trial_name=visualize_trial_name,strage="NASK")
        os.makedirs(self.visualize_dir_dict["trial_dir_path"]+"/temp",exist_ok=True)
        
        speaker_icon_red_path=self.notification_dir_dict["common_dir_path"]+"/icon_speaker_red.png"
        self.speaker_red_img=cv2.resize(cv2.imread(speaker_icon_red_path), (50,50),fx=0,fy=0)
        speaker_icon_blue_path=self.notification_dir_dict["common_dir_path"]+"/icon_speaker_blue.png"
        self.speaker_blue_img=cv2.resize(cv2.imread(speaker_icon_blue_path), (50,50),fx=0,fy=0)

        self.patient_dict={
            "00000":"A",
            "00001":"B",
            "00002":"C",
            "00003":"D",
            "00004":"E",
            "00005":"F",
            "00006":"G",
            "00007":"H",
            "00008":"I",
            "00009":"J",
            "00010":"K",
            "00011":"L"
        }

        # parameters
        smoothing_window=20
        
        # MobileSensing系のデータ
        # sensing_csv_path="/media/hayashide/MobileSensing/Nagasaki20241205193158/csv/annotation/Nagasaki20241205193158_annotation_ytpc2024j_20241205_193158_fixposition.csv"
        sensing_csv_path=sorted(glob(f"/media/hayashide/MobileSensing/{self.sensing_trial_name}/csv/annotation/*_fixposition.csv"))[0]
        # sensing_csv_path="/media/hayashide/MobileSensing/PullWheelchairObaachan/csv/annotation/PullWheelchairObaachan_annotation_ytnpc2021h_20240827_192540New_fixposition.csv"
        # sensing_csv_path="//NASK/common/FY2024/09_MobileSensing/Nagasaki20241205193158/csv/annotation/Nagasaki20241205193158_annotation_ytpc2024j_20241205_193158_fixposition.csv"
        self.sensing_data=pd.read_csv(sensing_csv_path,header=0).interpolate(method="bfill")

        # リスク評価のデータ
        evaluation_csv_paths=sorted(glob(self.evaluation_dir_dict["trial_dir_path"]+"/data_*_eval.csv"))
        print(evaluation_csv_paths)
        self.patients=[os.path.basename(k)[len("data_"):-len("_eval.csv")] for k in evaluation_csv_paths]

        self.evaluation_data_dict={k:pd.read_csv(path) for k,path in zip(self.patients,evaluation_csv_paths)}
        ## 平滑化処理を入れておく（window幅は通知側と揃えないとまずい）
        smoothing_cols=[]
        for k in self.evaluation_data_dict[self.patients[0]].keys():
            try:
                int(k)
            except ValueError:
                continue
            if int(k)<40000000 or ((int(k)>=40000010) and int(k)<50000000):
                smoothing_cols.append(k)
        for patient in self.patients:
            print(patient)
            self.evaluation_data_dict[patient]=self.evaluation_data_dict[patient][smoothing_cols].rolling(smoothing_window).mean()

        # 通知のデータ
        notification_csv_path=sorted(glob(self.notification_dir_dict["trial_dir_path"]+"/*_notify_history.csv"))[0]
        self.notification_data=pd.read_csv(notification_csv_path,header=0)

        print(self.sensing_data)
        print(self.evaluation_data_dict)
        print(self.notification_data)
        # raise NotImplementedError

        if len(self.sensing_data)!=len(self.evaluation_data_dict[self.patients[0]]):
            raise Exception("MobileSensingとリスク評価でデータの長さが一致しない")
        
        # リスク評価の順位情報を用意しておく
        self.rank_data=self.get_rank_data()
        # rank_00: 危険度第1位患者の名称, 00000_rank: 患者ID00000の全体順位

        # 患者のイメージカラー
        self.patient_color_dict={p:c for p,c in zip(self.patients,self.get_colors())}


        
    def get_rank_data(self):
        rank_data=pd.DataFrame(self.sensing_data["timestamp"].values,columns=["timestamp"])
        # 全評価結果から10000000を取ってくる
        for patient in self.patients:
            rank_data[patient+"_risk"]=self.evaluation_data_dict[patient]["10000000"]
        # ランキングを作る
        # 各行ごとに順位付け

        for i,row in rank_data.iterrows():
            risks=np.array(row.values[1:])
            if np.isnan(risks[0]):
                continue
            rank_list=np.array(self.patients)[np.argsort(-risks)]
            rank_data.loc[i,[f"rank_{str(n).zfill(2)}" for n in range(len(self.patients))]]=rank_list

        for patient in self.patients:
            rank_data[patient+"_rank"]=np.nan
        for i,row in rank_data.iterrows():
            rank_list=rank_data.loc[i,[f"rank_{str(n).zfill(2)}" for n in range(len(self.patients))]]
            for rank,patient_name in enumerate(rank_list):
                try:
                    int(patient_name)
                except ValueError:
                    continue
                rank_data.loc[i,[f"{patient_name}_rank"]]=rank

        return rank_data

    def get_colors(self):
        # tab10カラーマップの上位10色を取得
        colors = plt.get_cmap("tab10").colors
        colors = [(int(b*255), int(g*255), int(r*255)) for r, g, b in colors]
        return colors


    def draw_bbox(self,i,img):
        for patient in self.patients:
            if np.isnan(self.sensing_data.loc[i,f"ID_{patient}_bbox_higherX"]):
                continue
            # patient_rank
            bbox_info=[
                (int(self.sensing_data.loc[i,f"ID_{patient}_bbox_higherX"]),int(self.sensing_data.loc[i,f"ID_{patient}_bbox_higherY"])),
                (int(self.sensing_data.loc[i,f"ID_{patient}_bbox_lowerX"]),int(self.sensing_data.loc[i,f"ID_{patient}_bbox_lowerY"])),
            ]
            if not np.isnan(self.rank_data.loc[i,patient+"_rank"]):
                thickness=len(self.patients)-int(self.rank_data.loc[i,patient+"_rank"])
            else:
                thickness=1
            cv2.rectangle(img,bbox_info[0],bbox_info[1],self.patient_color_dict[patient], thickness=thickness)
        return img
        
    def draw_timestamp(self,i,img):
        bbox_info=[
            (0,0),
            (250,40),
        ]
        cv2.rectangle(img,bbox_info[0],bbox_info[1],color=(255,255,255),thickness=cv2.FILLED)
        cv2.putText(
            img=img,
            text="Time: "+str(np.round(self.sensing_data.loc[i,"timestamp"]-self.sensing_data.loc[0,"timestamp"],2))+" [s]",
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=1,
            org=(0,30),
            color=(0,0,255),
            thickness=2,
            )
        return img
    
    def draw_rank(self,i,img):
        def get_bbox_info(rank):
            x_width=125
            y_interval=50
            y_width=40
            bbox_info=[
                (0,int(100+rank*y_interval)),
                (x_width,int(100+rank*y_interval+y_width))
                ]
            return bbox_info
        
        # 背景色の白
        cv2.rectangle(img,(0,40),(250,50*(len(self.patients)+3)),color=(255,255,255),thickness=cv2.FILLED)
        cv2.putText(
                img=img,
                # text=f"No.{int(rank)+1}: "+"ID_"+patient,
                text=f"Priority Order",
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=1,
                org=(0,100-20),
                color=(0,0,0),
                thickness=2,
        )
        for patient in self.patients:
            patient_str=self.patient_dict[patient]
            rank=self.rank_data.loc[i,patient+"_rank"]
            if np.isnan(rank):
                continue
            bbox_info=get_bbox_info(rank)
            # 患者カラーの帯
            cv2.rectangle(img,(bbox_info[0][0]+90,bbox_info[0][1]),bbox_info[1],color=self.patient_color_dict[patient],thickness=cv2.FILLED)
            cv2.putText(
                img=img,
                # text=f"No.{int(rank)+1}: "+"ID_"+patient,
                text=f"No.{int(rank)+1}: "+patient_str,
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=1,
                org=(bbox_info[0][0],bbox_info[1][1]-10),
                color=(0,0,0),
                thickness=2,
                )            
            # 通知中か判定
            notify_for_the_patient_data=self.notification_data[self.notification_data.fillna(99999)["patient"]==patient_str]
            for j,_ in notify_for_the_patient_data.iterrows():
                if (notify_for_the_patient_data.loc[j,"timestamp"]<self.rank_data.loc[i,"timestamp"]) and (self.rank_data.loc[i,"timestamp"]<=notify_for_the_patient_data.loc[j,"timestamp"]+8):
                    img[bbox_info[0][1]:bbox_info[0][1]+self.speaker_red_img.shape[0],bbox_info[1][0]:bbox_info[1][0]+self.speaker_red_img.shape[1]]=self.speaker_red_img
            # raise NotImplementedError
        # 応援要請の状況
        notify_help_data=self.notification_data[self.notification_data["type"]=="help"]
        bbox_info=get_bbox_info(len(self.patients))
        for j,_ in notify_help_data.iterrows():
            if (notify_help_data.loc[j,"timestamp"]<self.rank_data.loc[i,"timestamp"]) and (self.rank_data.loc[i,"timestamp"]<=notify_help_data.loc[j,"timestamp"]+7):
                cv2.putText(
                        img=img,
                        text="Help:",
                        fontFace=cv2.FONT_HERSHEY_DUPLEX,
                        fontScale=1,
                        org=(bbox_info[0][0],bbox_info[1][1]),
                        color=(255,0,0),
                        thickness=2,
                        )
                img[bbox_info[0][1]:bbox_info[0][1]+self.speaker_blue_img.shape[0],bbox_info[1][0]:bbox_info[1][0]+self.speaker_blue_img.shape[1]]=self.speaker_blue_img
                break
            else:
                cv2.putText(
                        img=img,
                        text="Help:",
                        fontFace=cv2.FONT_HERSHEY_DUPLEX,
                        fontScale=1,
                        org=(bbox_info[0][0],bbox_info[1][1]),
                        color=(200,200,200),
                        thickness=2,
                        )

            # 通知中なら，スピーカーアイコンを描く

        return img

    def create_mp4_with_audio(self):
        from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips,CompositeAudioClip
        mp4_path=self.visualize_dir_dict["trial_dir_path"]+"/output.mp4"
        video = VideoFileClip(mp4_path)
        # 動画のもともとの音声を削除（音声がなければ無視してOK）
        video = video.without_audio()
        audio_clips = []
        for i,row in self.notification_data.iterrows():
            mp3_path=sorted(glob(self.notification_dir_dict["trial_dir_path"]+f"/*_{str(i).zfill(5)}.mp3"))[0]
            audio = AudioFileClip(mp3_path).with_start(row["relativeTimestamp"])
            audio_clips.append(audio)
        composite_audio = CompositeAudioClip(audio_clips)

        # 音声の長さを動画に合わせて調整（必要に応じてトリミング）
        # if composite_audio.duration > video.duration:
        #     composite_audio = composite_audio.subclipped(0, video.duration)
        
        video_with_audio = video.with_audio(composite_audio)

        # 出力ファイルとして保存
        video_with_audio.write_videofile(mp4_path.replace("output","output_with_audio"), codec="libx264", audio_codec="aac")

        pass

    def main(self):
        # 画像を1枚ずつ取り出す
        def draw(i):

            if i%10==0:
                print(f"Now processing... {i}/{len(self.sensing_data)}")
            # print(self.sensing_data.loc[i,"fullrgb_imagePath"])
            elp_img_path="/media/hayashide/MobileSensing"+"/"+self.sensing_data.loc[i,"fullrgb_imagePath"]
            if "//" in elp_img_path:
                elp_img_path=self.sensing_data.loc[i,"fullrgb_imagePath"]
            # elp_img_path="//NASK/common/FY2024/09_MobileSensing"+"/"+self.sensing_data.loc[i,"fullrgb_imagePath"]
            elp_img=cv2.imread(elp_img_path)
            # bounding boxを描く
            elp_img=self.draw_bbox(i,elp_img)
            # timestampを描く
            elp_img=self.draw_timestamp(i,elp_img)
            # rankを描く
            elp_img=self.draw_rank(i,elp_img)
            # 危険pop upを描く
            cv2.imwrite(self.visualize_dir_dict["trial_dir_path"]+f"/temp/{str(i).zfill(5)}.jpg",elp_img)
            # print(elp_img_path)
            # cv2.imshow("ELP",elp_img)
            # cv2.waitKey(0)
        
        n_cpu=cpu_count()
        p_list=[]
        import time
        for i,row in self.sensing_data.iterrows():
            start=time.time()
            draw(i)
            print(np.round(1/(time.time()-start),2),"Hz")
            # p=Process(target=draw,args=(i,))
            # p_list.append(p)
            # if (len(p_list)>=n_cpu) or (i+1==len(self.sensing_data)):
            #     for p in p_list:
            #         p.start()
            #     for p in p_list:
            #         p.join()
            #     p_list=[]
        

        
        self.jpg2mp4(sorted(glob(self.visualize_dir_dict["trial_dir_path"]+"/temp/*.jpg")),mp4_path=self.visualize_dir_dict["trial_dir_path"]+"/output.mp4",fps=20)

        self.create_mp4_with_audio()
        pass

if __name__=="__main__":
    # 
    # sensing_trial_name="Nagasaki20241205193158"
    # evaluation_trial_name="20250121ChangeCriteriaBefore"
    # notification_trial_name="20250202ChangeCriteriaBefore"
    # visualize_trial_name="20250201VisualizeVideo"
    # 
    # sensing_trial_name="Nagasaki20241205193158"
    # evaluation_trial_name="20250121ChangeCriteriaAfter"
    # notification_trial_name="20250202ChangeCriteriaAfter"
    # visualize_trial_name="20250202VisualizeVideoAfter"
    # 
    # sensing_trial_name="PullWheelchairObaachan"
    # evaluation_trial_name="20250115PullWheelchairObaachan2"
    # notification_trial_name="20250202NotifyPull"
    # visualize_trial_name="20250202VisualizeVideoPull"
    # 
    sensing_trial_name="Nagasaki20241205193158"
    evaluation_trial_name="20250121ChangeCriteriaBefore"
    notification_trial_name="20250204NotifyIntervalBefore"
    visualize_trial_name="20250204VisualizeVideoBefore"
    cls=VideoVisualizer(
        sensing_trial_name=sensing_trial_name,
        evaluation_trial_name=evaluation_trial_name,
        notification_trial_name=notification_trial_name,
        visualize_trial_name=visualize_trial_name,
    )
    cls.main()
    
    sensing_trial_name="Nagasaki20241205193158"
    evaluation_trial_name="20250121ChangeCriteriaAfter"
    notification_trial_name="20250204NotifyIntervalAfter"
    visualize_trial_name="20250204VisualizeVideoAfter"
    cls=VideoVisualizer(
        sensing_trial_name=sensing_trial_name,
        evaluation_trial_name=evaluation_trial_name,
        notification_trial_name=notification_trial_name,
        visualize_trial_name=visualize_trial_name,
    )
    cls.main()
    sensing_trial_name="PullWheelchairObaachan"
    evaluation_trial_name="20250115PullWheelchairObaachan2"
    notification_trial_name="20250204NotifyIntervalPull"
    visualize_trial_name="20250204VisualizeVideoPull"
    cls=VideoVisualizer(
        sensing_trial_name=sensing_trial_name,
        evaluation_trial_name=evaluation_trial_name,
        notification_trial_name=notification_trial_name,
        visualize_trial_name=visualize_trial_name,
    )
    cls.main()

    


"""
'ID_00003_x', 'ID_00003_y',
'ID_00003_bbox_lowerX', 'ID_00003_bbox_lowerY', 'ID_00003_bbox_higherX',
'ID_00003_bbox_higherY', 'ID_00003_object_id', 'ID_00003_imagePath',
'ID_00003_activeBinary',
"""