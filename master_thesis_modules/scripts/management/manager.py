import os
from glob import glob
from pprint import pprint
import json
import yaml
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class Manager():
    def __init__(self):
        super().__init__()
        plt.rcParams["figure.figsize"] = (15/2.54,10/2.54)
        plt.rcParams["figure.autolayout"] = True
        plt.rcParams["font.size"] = 11
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
        plt.rcParams["legend.edgecolor"] = 'black' # edgeの色を変更
        plt.rcParams["legend.handlelength"] = 1 # 凡例の線の長さを調節
        pass

    def prepare_log(self,trial_dir_path):
        import os
        from datetime import datetime
        import logging

        logdir=trial_dir_path
        try:
            os.makedirs(logdir,exist_ok=True)
        except Exception:
            os.makedirs(os.path.split(logdir)[0],exist_ok=True)
            os.makedirs(logdir,exist_ok=True)
        

        logger = logging.getLogger(os.path.basename(__file__))
        logger.setLevel(logging.DEBUG)
        format = "%(asctime)s [%(filename)s:%(funcName)s:%(lineno)d] %(levelname)-9s  %(message)s"
        st_handler = logging.StreamHandler()
        st_handler.setLevel(logging.WARNING)
        # StreamHandlerによる出力フォーマットを先で定義した'format'に設定
        st_handler.setFormatter(logging.Formatter(format))

        fl_handler = logging.FileHandler(filename=logdir+"/"+datetime.now().strftime('%Y%m%d_%H%M%S')+".log", encoding="utf-8")
        fl_handler.setLevel(logging.DEBUG)
        # FileHandlerによる出力フォーマットを先で定義した'format'に設定
        fl_handler.setFormatter(logging.Formatter(format))

        logger.addHandler(st_handler)
        logger.addHandler(fl_handler)
        return logger

    def get_module_path(self):
        if os.name == "nt": # Windows
            home = os.path.expanduser("~")
        else: # ubuntu
            home=os.environ['HOME']        
        # print("HOME: "+home)
        
        workspace_dir_name="kazu_ws"
        module_name="master_thesis_modules"
        module_dir_path=home+"/"+workspace_dir_name+"/"+module_name[:-len("_modules")]+"/"+module_name
        if os.path.isdir(module_dir_path):
            pass
        else:#dockerのとき
            module_dir_path=home+"/"+"catkin_ws/src"+"/"+module_name
            if not os.path.isdir(module_dir_path):
                module_dir_path="/"+"catkin_ws/src"+"/"+module_name
            if not os.path.isdir(module_dir_path):
                raise FileNotFoundError("module directory not found: "+module_dir_path)
            
        
        return module_dir_path

    def get_database_dir(self,trial_name="NoTrialNameGiven",strage="NASK"):
        module_dir_path=self.get_module_path()

        if (strage=="NASK") or (strage=="nask"):
            if os.name=="nt": # Windows
                database_dir_path="//192.168.1.5/common/FY2024/01_M2/05_hayashide/MasterThesis_database"
                pass
            else: # Ubuntu
                # raise NotImplementedError("マウント処理を実装してください")
                if "catkin_ws" in module_dir_path: # docker
                    database_dir_path="/media/hayashide/MasterThesis"
                    pass
                else: # out of docker
                    database_dir_path="/media/hayashide/MasterThesis"
                    pass
        elif strage=="local":
            database_dir_path=module_dir_path+"/database"

        mobilesensing_dir_path=database_dir_path.replace("MasterThesis_database","MobileSensing")
        mobilesensing_dir_path=f"/catkin_ws/src/database/{trial_name}"

        if "/" in trial_name:
            os.makedirs(database_dir_path+"/"+trial_name.split("/")[0],exist_ok=True)
        trial_dir_path=database_dir_path+"/"+trial_name
        common_dir_path=database_dir_path+"/common"

        database_dir_dict={
            "mobilesensing_dir_path":mobilesensing_dir_path,
            "module_dir_path":module_dir_path,
            "database_dir_path":database_dir_path,
            "trial_dir_path":trial_dir_path,
            "common_dir_path":common_dir_path,
        }

        for path in database_dir_dict.values():
            os.makedirs(path,exist_ok=True)
        
        return database_dir_dict
    
    def convert_np_types(self,obj):
        """
        任意の階層構造を持つdictの中のnumpyデータ型を標準のPythonデータ型に変換する。
        """
        if isinstance(obj, dict):
            return {key: self.convert_np_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_np_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def write_csvlog(self,output_data,csvpath,fmt="%s",dim=1):
        if dim==1:
            output_data=[output_data]
        else:
            pass
        try:
            with open(csvpath, 'a') as f_handle:
                np.savetxt(f_handle,output_data,delimiter=",")
        except TypeError:
            with open(csvpath, 'a') as f_handle:
                np.savetxt(f_handle,output_data,delimiter=",",fmt=fmt)    
        except FileNotFoundError:
            np.savetxt(csvpath,output_data,delimiter=",")
        pass  

    def write_json(self,dict_data,json_path):
        dict_data=self.convert_np_types(dict_data)
        with open(json_path,mode="w",encoding="utf-8") as f:
            json.dump(dict_data,f,ensure_ascii=False)

    def load_json(self,json_path):
        with open(json_path,encoding="utf-8") as f:
            data=json.load(f)
        return data

    def write_picklelog(self,output_dict,picklepath):
        with open(picklepath, mode='wb') as f:
            pickle.dump(output_dict,f)

    def load_picklelog(self,picklepath):
        with open(picklepath,mode="rb") as f:
            try:
                data=pickle.load(f)
            except ModuleNotFoundError:
                # python2系列で書かれた場合
                data=pickle.load(f,fix_imports=True)
        return data        
    
    def write_yaml(self,dict_data,yaml_path):
        with open(yaml_path,mode="w") as f:
            yaml.dump(dict_data,f)
    
    def load_yaml(self,yaml_path):
        with open(yaml_path,mode="r") as f:
            data=yaml.safe_load(f)
        return data
    
    def get_timestamp(self):
        import datetime
        current_time=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return current_time
    
    def jpg2mp4(self,image_paths,mp4_path,size=(0,0),fps=30.0):
        import cv2
        # get size of the image
        img=cv2.imread(image_paths[0])
        if size[0]==0:
            size=(img.shape[1],img.shape[0])
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        print("mp4_path",mp4_path)
        print("fps",fps)
        print("size",size)
        video = cv2.VideoWriter(mp4_path,fourcc,fps,size)#(mp4_path,fourcc, fps, size)
        for idx,image in enumerate(image_paths):
            img=cv2.imread(image)
            video.write(img)
            print(f"now processing: {os.path.basename(image)} {idx}/{len(image_paths)}")
        video.release()

    # def putText_japanese(img, text, point, size, color):
    #     #Notoフォントとする
    #     font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', size)

    #     #imgをndarrayからPILに変換
    #     img_pil = Image.fromarray(img)

    #     #drawインスタンス生成
    #     draw = ImageDraw.Draw(img_pil)

    #     #テキスト描画
    #     draw.text(point, text, fill=color, font=font)

    #     #PILからndarrayに変換して返す
    #     return np.array(img_pil)
    
    # def flattern_dict(self,d):
    #     d_flatten={}
    #     for p in d.keys():
    #         try:
    #             for k in d[p].keys():
    #                 if type(d[p][k]) in [list,tuple]:
    #                     d_flatten[f"{p}_{k}"]=str(d[p][k])
    #                 else:
    #                     d_flatten[f"{p}_{k}"]=d[p][k]
    #         except AttributeError:
    #             return d
        
    #     return d_flatten
    
    def flatten_dict(self, d, parent_key='', sep='_'):
        """
        ネストされた辞書をフラットにする。
        各キーは「上位キー_下位キー」の形式になる。
        
        :param d: 入れ子構造の辞書
        :param parent_key: 親キー（再帰時に使用）
        :param sep: キーを結合するセパレータ（デフォルトは'_'）
        :return: フラット化された辞書
        """
        flattened = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                flattened.update(self.flatten_dict(v, new_key, sep=sep))
            else:
                if type(v) in [list,tuple]:
                    flattened[new_key] = str(v)
                else:
                    flattened[new_key] = v
        return flattened
    
    def plot_map_matplotlib(self):
        import matplotlib.pyplot as plt
        from PIL import Image
        import yaml
        def cell_to_xy(x, y, map_config, map_height):
            """
            2D mapの画像座標系を地図座標系に変換する
            """
            cell_x = map_config["origin"][0] + x * map_config["resolution"]
            cell_y = (map_height - y) * map_config["resolution"] + map_config["origin"][1]
            return cell_x, cell_y
        # 読込
        common_dir_path=self.get_database_dir("","local")["common_dir_path"]
        map_yaml_path=common_dir_path+"/map/map2d.yaml"
        map_pgm_path=common_dir_path+"/map/map2d.pgm"

        map_yaml_data=self.load_yaml(map_yaml_path)
        map_pgm=Image.open(map_pgm_path)

        # map画像をpltで表示
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ## 地図画像を地図座標系の空間で描画するための設定
        map_initial_x, map_initial_y = cell_to_xy(0, map_pgm.height, map_yaml_data, map_pgm.height)
        map_end_x, map_end_y = cell_to_xy(map_pgm.width, 0, map_yaml_data, map_pgm.height)
        extent = [
            map_initial_x,
            map_end_x,
            map_initial_y,
            map_end_y
        ]

        plt.gray()
        ax.imshow(map_pgm, alpha=1.0, extent=extent)
        # ax.scatter(0, 0, s=100, c='r', marker='o', label="start")

        return fig,ax
        


    def plot_map_plotly(self,fig,dimension="2D"):
        from PIL import Image,ImageOps
        def cell_to_xy(x, y, map_config, map_height):
            """
            2D mapの画像座標系を地図座標系に変換する
            """
            cell_x = map_config["origin"][0] + x * map_config["resolution"]
            cell_y = (map_height - y) * map_config["resolution"] + map_config["origin"][1]
            return cell_x, cell_y

        map_img=Image.open(self.get_database_dir("","local")["common_dir_path"]+"/map/map2d.pgm")
        map_config=self.load_yaml(self.get_database_dir("","local")["common_dir_path"]+"/map/map2d.yaml")
        map_initial_x, map_initial_y = cell_to_xy(0, map_img.height, map_config, map_img.height)
        map_end_x, map_end_y = cell_to_xy(map_img.width, 0, map_config, map_img.height)

        # fig=go.Figure()
        if dimension=="2D":
            fig.add_layout_image(
                dict(
                    source=map_img,
                    xref="x",  # x座標系を指定
                    yref="y",  # y座標系を指定
                    x=map_initial_x,   # 画像の左下のx座標
                    y=map_end_y,   # 画像の左上のy座標 (y軸が逆転しているので注意)
                    sizex=map_end_x - map_initial_x,  # 画像の幅
                    sizey=map_end_y - map_initial_y,  # 画像の高さ
                    sizing="stretch",     # 画像を伸縮してフィットさせる
                    layer="below"         # 軌跡の下に表示
                )
            )
        else:
            map_img=Image.open(self.get_database_dir("","local")["common_dir_path"]+"/map/map2d.png").convert("RGB")
            map_img=ImageOps.flip(map_img)
            map_img_np=np.array(map_img)

            # 画像の幅と高さを取得
            img_height, img_width, _ = map_img_np.shape

            # 画像のRGBデータを(行, 列)に対応する色データとして抽出
            img_x = np.linspace(0, 1, img_width)  # x座標をスケール
            img_y = np.linspace(0, 1, img_height)  # y座標をスケール
            img_x = (map_end_x-map_initial_x)*img_x+map_initial_x
            img_y = (map_end_y-map_initial_y)*img_y+map_initial_y

            # 画像のRGBデータを1次元配列に変換してScatter3dで表示
            x_flat = np.tile(img_x, img_height)
            y_flat = np.repeat(img_y, img_width)
            z_flat = np.ones_like(x_flat)  # z=0平面に表示

            # 色情報を1次元配列に変換
            colors_flat = ['rgb({}, {}, {})'.format(r, g, b) for r, g, b in map_img_np.reshape(-1, 3)]

            # compress
            x_flat=x_flat[::5]
            y_flat=y_flat[::5]
            z_flat=z_flat[::5]
            colors_flat=colors_flat[::5]

            fig.add_trace(go.Scatter3d(
                x=x_flat,
                y=y_flat,
                z=z_flat,
                mode='markers',
                marker=dict(
                    size=5,  # 各ピクセルを点として表示
                    color=colors_flat
                ),
                showlegend=False  # 凡例を非表示
            ))
            # import skimage.io
            # import skimage.transform
            # map_img=skimage.io.imread(self.data_dir_dict["common_dir_path"]+"/map/map2d.png")
            # # map_img=skimage.transform.rotate(map_img,90)
            # map_img = np.flipud(map_img)
            # map_img= map_img.swapaxes(0, 1)[:, ::-1]
            # map_img_8 = Image.fromarray(map_img).convert('P', palette='WEB', dither=None)
            # idx_to_color = np.array(map_img_8.getpalette()).reshape((-1, 3))
            # colorscale=[[i/255.0, "rgb({}, {}, {})".format(*rgb)] for i, rgb in enumerate(idx_to_color)]
            # print(map_img_8.size)
            # # raise NotImplementedError
            # xrange=np.arange(map_initial_x,map_end_x)#,map_img_rgb.size[0])
            # yrange=np.arange(map_initial_y,map_end_y)#,map_img_rgb.size[1])
            # print(xrange)
            # # raise NotImplementedError
            # fig.add_trace(go.Surface(
            #     # z=[[z,z],[z,z]],
            #     z=np.ones_like(map_img),
            #     # x=xrange,
            #     # y=yrange,
            #     surfacecolor=np.array(map_img_8),  # 画像を色として設定
            #     cmin=0,
            #     cmax=255,
            #     colorscale=colorscale,
            #     contours_z=dict(show=True, project_z=True, highlightcolor="limegreen"),
            #     opacity=1.0,
            #     showscale=False  # カラーバーは非表示
            # ))
            # グラフの設定
            fig.update_layout(
                scene=dict(
                    xaxis=dict(title='X'),
                    yaxis=dict(title='Y'),
                    zaxis=dict(title='Z'),
                    aspectmode="manual",  # 画像のアスペクト比を手動で調整
                    aspectratio=dict(x=(map_end_x-map_initial_x)/(map_end_y-map_initial_y), y=1, z=1)  # 適切なアスペクト比を設定
                )
            )
        return fig
    
    def draw_japanese_text(self,img, text, position, text_color=(255, 255, 255), bg_color=None, font_path="NotoSansCJK-Regular.ttc", font_size=30):
        """
        OpenCVの画像に日本語テキストを描画する関数

        Parameters:
            img (numpy.ndarray): OpenCVで読み込んだ画像
            text (str): 描画する日本語テキスト
            position (tuple): テキストの左上座標 (x, y)
            text_color (tuple): 文字色（BGR）
            bg_color (tuple or None): 背景色（BGR）、Noneの場合は背景なし
            font_path (str): 使用するフォントのパス（日本語対応フォント）
            font_size (int): フォントサイズ

        Returns:
            numpy.ndarray: 日本語テキストを描画した画像
        """
        # OpenCVの画像をPillow形式に変換
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # フォントを設定
        font = ImageFont.truetype(font_path, font_size)

        # 背景色が指定されていれば背景を描画
        if bg_color:
            text_size = draw.textbbox(position, text, font=font)  # (left, top, right, bottom)
            draw.rectangle(text_size, fill=bg_color)

        # 日本語テキストを描画
        draw.text(position, text, font=font, fill=text_color)

        # Pillowの画像をOpenCV形式に変換して返す
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    def color_converter(self,rgb_array):
        bgr_array=(rgb_array[2],rgb_array[1],rgb_array[0])
        return bgr_array
    
if __name__=="__main__":
    cls=Manager()
    # cls.get_database_dir("NASK")
    cls.plot_map_matplotlib()