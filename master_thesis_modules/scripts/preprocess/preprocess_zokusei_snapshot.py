import os
import sys
import copy

import cv2
import numpy as np
import pandas as pd

sys.path.append(".")
sys.path.append("..")
if "catkin_ws" in os.getcwd():
    sys.path.append("/catkin_ws/src/master_thesis_modules")
else:
    sys.path.append(os.path.expanduser("~")+"/kazu_ws/master_thesis/master_thesis_modules")

from scripts.management.manager import Manager
# from scripts.preprocess.blipTools import blipTools

class PreprocessZokusei(Manager):#,blipTools):
    def __init__(self):

        # BLIPの起動
        # self.blip_processor,self.blip_model,self.device=self.activate_blip()

        # questions
        self.questions={
            "patient":{"query":"Question: Is this person wearing a red shirt? Answer:","node_code":"50000000"},
            "age":{"query":"Question: Is this person old, middle, or young? Answer:","node_code":"50000010"},
            "ivPole":{"query":"Question: Is there an IV pole in this image? Answer:","node_code":"50001000"},
            "wheelchair":{"query":"Question: Are there any wheelchair in this picture? Answer:","node_code":"50001010"},
        }

    def get_center_median_rgb(self,image):
        """
        画像全体から「赤・黒・薄ピンク」に最も近いピクセルのRGB値を取得する。

        :param image: OpenCVで読み込んだBGR画像
        :return: (B, G, R) のタプル
        """
        # 画像をHSVに変換
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # 赤の条件（Hue ≈ 0° or 170°）
        red_distance = np.minimum(np.abs(h - 0), np.abs(h - 170))
        red_score = (1 - red_distance / 20) * (s / 255) * (v / 255)

        # 黒の条件（低明度・低彩度）
        # black_score = (1 - v / 100) * (1 - s / 100)

        # 薄ピンクの条件（Hue 160~180°, 低彩度, 高明度）
        pink_mask = (160 <= h) & (h <= 180)
        pink_distance = np.abs(h - 170)
        pink_score = np.where(pink_mask, (1 - pink_distance / 10) * (1 - s / 180) * (v / 255), 0)

        # 赤・黒・ピンクの最大スコアを持つピクセルの座標を取得
        # combined_score = np.maximum(np.maximum(red_score, black_score), pink_score)
        combined_score = np.maximum(red_score, pink_score)
        max_idx = np.unravel_index(np.argmax(combined_score), combined_score.shape)

        # 対応するBGRの値を取得
        best_pixel_bgr = image[max_idx[0], max_idx[1]]

        return tuple(best_pixel_bgr)  # (B, G, R) の順で返す
        # h, w, _ = image.shape
        # size = min(10, h, w)  # 画像が10x10未満でも処理可能に
        
        # # 中央座標
        # cx, cy = w // 2, h // 2
        
        # # 切り出し範囲を決定
        # x1, x2 = cx - size // 2, cx + size // 2
        # y1, y2 = cy - size // 2, cy + size // 2
        
        # # 画像の範囲内に収まるように調整
        # x1, x2 = max(0, x1), min(w, x2)
        # y1, y2 = max(0, y1), min(h, y2)
        
        # # 指定範囲のピクセルを取得
        # center_region = image[y1:y2, x1:x2]
        
        # # RGBごとの中央値を計算
        # median_rgb = np.median(center_region, axis=(0, 1)).astype(np.uint8)
        
        # return tuple(median_rgb)  # (B, G, R) の順で返す
    
    def is_nurse_color(self,rgb):
        """
        中央のRGB値が「看護師（赤・黒）」または「介護福祉士（薄ピンク）」の可能性があるかを判定し、信頼度も返す。

        :param rgb: (B, G, R) のタプル
        :return: (True / False, confidence (0.0~1.0))
        """

        # RGB → HSV 変換
        bgr = np.uint8([[list(rgb)]])  # OpenCVはBGRなのでこの順番
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0]

        h, s, v = hsv  # Hue, Saturation, Value

        # 赤っぽいスコア
        red_distance = min(abs(h - 0), abs(h - 170))  # 0° or 170° に近いほどスコアが高い
        red_conf = max(1 - red_distance / 20, 0) * (s / 255) * (v / 255)

        # 黒っぽいスコア
        # black_conf = (1 - v / 100) * (1 - s / 100)  # 明度と彩度が低いほど黒っぽい

        # 薄ピンクっぽいスコア
        if 160 <= h <= 180:  # Hueが160°〜180°の範囲
            pink_conf = (1 - (abs(h - 170) / 10)) * (1 - s / 180) * (v / 255)
        else:
            pink_conf = 0

        # 信頼度の最大値を採用
        # confidence = max(red_conf, black_conf, pink_conf)
        confidence = max(red_conf, pink_conf)

        # 「看護師または介護福祉士の可能性あり」と判定する基準
        is_nurse_or_caregiver = confidence > 0.25  # 50%以上なら該当

        return is_nurse_or_caregiver, confidence

    
    def zokusei_snapshot(self,data_dict,rgb_img,t,b,l,r,):
        # 患者判別
        t,b,l,r=int(t),int(b),int(l),int(r)
        bbox_rgb_img=rgb_img[t:b,l:r]
        # 中央画素の値を取得
        median_rgb=self.get_center_median_rgb(image=bbox_rgb_img)
        
        # 赤っぽい（または黒っぽい）ことを判別
        nurse,conf=self.is_nurse_color(median_rgb)
        if nurse:
            data_dict["50000000"]="no"# =NS,Cwである
        else:
            data_dict["50000000"]="yes"# =患者である
        data_dict["50000001"]=conf

        # 年齢
        answer="old"
        confidence=1
        data_dict["50000010"]=answer
        data_dict["50000011"]=confidence

        # 物体
        ## 点滴
        # 【移設】
        ## 車椅子
        # 【移設】
        
        return data_dict
        

    def blip_snapshot(self,data_dict,rgb_img,t,b,l,r,fps_control_dict,extend_ratio_tb=0.2,extend_ratio_lr=0.2):
        # bounding boxの切り出し
        t,b,l,r=int(t),int(b),int(l),int(r)
        bbox_rgb_img=rgb_img[t:b,l:r]
        # 拡張bounding boxの切り出し
        t_e,b_e,l_e,r_e,=np.max([t-extend_ratio_tb*rgb_img.shape[0],0]),np.min([b+extend_ratio_tb*rgb_img.shape[0],rgb_img.shape[0]]),np.max([l-extend_ratio_lr*rgb_img.shape[1],0]),np.min([r+extend_ratio_lr*rgb_img.shape[1],rgb_img.shape[0]])
        t_e,b_e,l_e,r_e=int(t_e),int(b_e),int(l_e),int(r_e)
        extended_bbox_rgb_img=rgb_img[t_e:b_e,l_e:r_e]

        # BLIP
        for q_title,q_info in self.questions.items():
            if ((q_title=="patient") and fps_control_dict[q_info["node_code"]]):
                # BLIP 患者？
                # answer,confidence=self.get_vqa(blip_processor=self.blip_processor,blip_model=self.blip_model,device=self.device,image=bbox_rgb_img,question=q_info["query"],confidence=True)
                answer="no"
                confidence=1
                # 答えを加工
                if (answer=="yes") or (answer=="Yes"):
                    answer="no"
                else:
                    answer="yes"
                # 答えを記録
                data_dict[self.questions[q_title]["node_code"]]=answer
                data_dict["50000001"]=confidence
            elif ((q_title=="age") and fps_control_dict[q_info["node_code"]]):
                # BLIP 年齢？
                # answer,confidence=self.get_vqa(blip_processor=self.blip_processor,blip_model=self.blip_model,device=self.device,image=bbox_rgb_img,question=q_info["query"],confidence=True)
                answer="old"
                confidence=1
                data_dict[self.questions[q_title]["node_code"]]=answer
                data_dict["50000011"]=confidence
            elif ((q_title=="ivPole") and fps_control_dict[q_info["node_code"]]):
                # BLIP 点滴？
                # answer,confidence=self.get_vqa(blip_processor=self.blip_processor,blip_model=self.blip_model,device=self.device,image=extended_bbox_rgb_img,question=q_info["query"],confidence=True)
                answer="no"
                confidence=1
                if (answer=="yes") or (answer=="Yes"):
                    answer=1
                else:
                    answer=0
                data_dict["50001000"]=answer
                data_dict["50001001"]=answer
                data_dict["50001002"]=confidence
                data_dict["50001003"]=confidence
                
            elif ((q_title=="wheelchair") and fps_control_dict[q_info["node_code"]]):
                # BLIP 車椅子？
                # answer,confidence=self.get_vqa(blip_processor=self.blip_processor,blip_model=self.blip_model,device=self.device,image=extended_bbox_rgb_img,question=q_info["query"],confidence=True)
                answer="yes"
                confidence=1
                if (answer=="yes") or (answer=="Yes"):
                    answer=1
                else:
                    answer=0
                data_dict["50001010"]=answer
                data_dict["50001011"]=answer
                data_dict["50001012"]=confidence
                data_dict["50001013"]=confidence
        return data_dict

if __name__=="__main__":
    trial_name="20250219DevAlternative"
    strage="local"
    # cls=PreprocessBlip()
    # print(cls.gauss_func(np.array([2,2]),np.array([1,1]),r=3))