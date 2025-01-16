import os
import sys
sys.path.append(".")
sys.path.append("..")
if "catkin_ws" in os.getcwd():
    sys.path.append("/catkin_ws/src/master_thesis_modules")
else:
    sys.path.append(os.path.expanduser("~")+"/kazu_ws/master_thesis/master_thesis_modules")

from icecream import ic
from pprint import pprint
from glob import glob

import numpy as np
import pandas as pd
import plotly.subplots as sp
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


from scripts.management.manager import Manager

class Visualizer(Manager):
    def __init__(self,trial_name,strage):
        super().__init__()
        self.trial_name=trial_name
        self.strage=strage
        self.data_dir_dict=self.get_database_dir(trial_name=trial_name,strage=strage)
        
        # self.visualize_dir_path=sorted(glob(self.get_database_dir(strage="NASK")["database_dir_path"]+"/*"))[-1]
        self.data_paths=sorted(glob(self.data_dir_dict["trial_dir_path"]+"/*"))
        pass

    def visualize_graph(self,trial_name="20241229BuildSimulator",strage="NASK",name="A",show=False):
        import networkx as nx
        # self.colorize()
        symbol_dict={
            1:"circle",
            0:"x",
            np.nan:"x",
            "active":"circle",
            "inactive":"x",
        }
        self.data_dir_dict=self.get_database_dir(trial_name=trial_name,strage=strage)
        self.graph_dicts=self.load_picklelog(self.data_dir_dict["trial_dir_path"]+"/graph_dicts.pickle")
        # pprint(pkl_data)
        # raise NotImplementedError
        self.graph_dict=self.graph_dicts[name]

        # グラフの情報を取り出しておく
        nodes = self.graph_dict["G"].nodes()
        pos = self.graph_dict["pos_dict"] # key: ノード番号, value: [x,y]
        weights = nx.get_edge_attributes(self.graph_dict["G"], 'weight') # key:(node_from,node_to), value: weight
        status = {n:self.graph_dict["node_dict"][n]["status"] for n in nodes}
        scores = {n:self.graph_dict["node_dict"][n]["score"] for n in nodes}# key: ノード番号, value: 特徴量
        descriptions = {n:self.graph_dict["node_dict"][n]["description_en"] for n in nodes}

        # 色の準備
        cmap = cm.get_cmap('jet')

        # エッジ描画データ
        edge_traces=[]
        for edge,weight in weights.items():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x = [x0, x1, None]  # Noneは各エッジの終了を示す
            edge_y = [y0, y1, None]
            color=mcolors.rgb2hex(cmap(weight))

            # エッジのトレース
            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=5, color=color),
                customdata=[weight],
                hovertemplate="weight: %{customdata:.2f}<extra></extra>",
                mode='lines'
            )
            edge_traces.append(edge_trace)

        node_traces=[]
        for node,p in pos.items():
            # ひとつ手前側のエッジの重み情報
            previous_weight=np.nan#self.get_left_weight(name=name,node=node)
            # ノードの色
            color=mcolors.rgb2hex(cmap(scores[node]))
            # ノードのトレース
            node_trace = go.Scatter(
                x=[p[0]],
                y=[p[1]],
                mode='markers+text',
                text=node,
                customdata=[node,descriptions[node]],
                hovertemplate=f"No. {node}<br>"+
                                "description: "+descriptions[node]+"<br>"+
                                f"score: {np.round(scores[node],2)}<br>"+
                                f"left weight: {previous_weight}<extra></extra>",
                marker=dict(
                    symbol=symbol_dict[status[node]],
                    size=10,
                    line_color="black",
                    line_width=2,
                    color=color,
                    showscale=False
                ),
                textposition="top center"
            )
            node_traces.append(node_trace)

        # グラフを作成
        try:
            del fig
        except UnboundLocalError:
            pass
        fig = go.Figure(data=edge_traces+node_traces)
        fig.update_layout(
            showlegend=False,
            title=name,
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            plot_bgcolor='white',
            legend=dict(
                bgcolor='white',  # 背景色
                bordercolor='black',  # 枠線の色
                borderwidth=1.5    # 枠線の太さ
            ),
            font=dict(
                family='Times New Roman',  # 推奨フォント
                size=18,  # フォントサイズ
                color='black'  # フォント色
            ),
        )
        if show:
            fig.show()
        else:
            pass
        return fig.data
        
    def visualize_animation(self,name,fig_datas,timestamps,show=False,save=False,trial_dir_path=""):
        frames=[]
        for i, fig_data in enumerate(fig_datas):
            try:
                timestamp=timestamps[i]
            except IndexError:
                timestamp=timestamps[-1]+(timestamps[-1]-timestamps[-2])
            frames.append(go.Frame(
                data=fig_data,
                name=f"frame no.: {i}",
                layout=go.Layout(
                    title=f"Person: {name}  Frame: {i+1}  timestamp: {timestamp}"
                ),
                ))
        # レイアウト設定
        layout = go.Layout(
            title=f"Person: {name}  Frame: 0  timestamp: 0",
            xaxis=dict(showticklabels=False),  # X軸のメモリを非表示
            yaxis=dict(showticklabels=False),   # Y軸のメモリを非表示
            updatemenus=[{
                'buttons': [
                    {
                        'args': [None, {"frame": {"duration": 250, "redraw": True}, "fromcurrent": True}],
                        'label': 'Play',
                        'method': 'animate'
                    },
                    {
                        'args': [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                        'label': 'Pause',
                        'method': 'animate'
                    }
                ],
                'direction': 'left',
                'pad': {'r': 10, 't': 87},
                'showactive': False,
                'type': 'buttons',
                'x': 0.1,
                'xanchor': 'right',
                'y': 0,
                'yanchor': 'top'
            }],
            showlegend=False,
            plot_bgcolor='white',
            legend=dict(
                bgcolor='white',  # 背景色
                bordercolor='black',  # 枠線の色
                borderwidth=1.5    # 枠線の太さ
            ),
            font=dict(
                family='Times New Roman',  # 推奨フォント
                size=18,  # フォントサイズ
                color='black'  # フォント色
            ),
        )

        # Figure作成
        fig = go.Figure(data=fig_datas[0], layout=layout, frames=frames)

        # プロット
        if show:
            fig.show()
        else:
            pass

        # export
        if save:
            import shutil
            from glob import glob
            trial_temp_dir_path=trial_dir_path+f"/temp_{name}"
            try:
                shutil.rmtree(trial_temp_dir_path)
            except FileNotFoundError:
                pass
            os.makedirs(trial_temp_dir_path,exist_ok=True)
            print(name)
            for i,frame in enumerate(frames):
                try:
                    timestamp=timestamps[i]
                except IndexError:
                    timestamp=timestamps[-1]+(timestamps[-1]-timestamps[-2])
                fig.update(data=frame.data,)
                fig.update_layout(
                    title_text=f"Person:{name}  Frame:{i+1}  timestamp: {timestamp}",
                    autosize=False,
                    width=int(640*2),
                    height=int(480*1.5),
                    )
                print(i)
                fig.write_image(f"{trial_temp_dir_path}/{name}_{i:03d}.jpg",format='jpg', engine="auto")
            print("export")
            self.jpg2mp4(sorted(glob(trial_temp_dir_path+"/*")),mp4_path=trial_dir_path+f"/{name}.mp4",fps=4)
        return frames

    def draw_timeseries(self,data,name,label_name="",symbol=""):
        if label_name=="":
            try:
                label_name=self.node_dict[int(name)]["description_en"]
            except AttributeError:
                label_name=name
            except ValueError:
                label_name=name
        # if label_name=="A":
        #     color="blue"
        # elif label_name=="B":
        #     color="orange"
        # elif label_name=="C":
        #     color="green"
        # else:
        #     color="black"
        trace=go.Scatter(
            x=data["timestamp"],
            y=data[name],
            xaxis="x",
            yaxis="y",
            mode="markers+lines",
            marker=dict(
                # color=color,
                size=10,
            ),
            name=label_name
            # name=f"<i>{symbol}</i><sub>"+str(name)+"</sub>"#+" : "+label_name,
        )
        return trace

    def draw_3d(self,plot_data,data,name):
        # try:
        #     label_name=self.node_dict[int(name)]["description"]
        # except ValueError:
        label_name=name
        print(name)
        trace=go.Scatter3d(
            x=data["60010000"],
            y=data["60010001"],
            z=data["timestamp"]-data["timestamp"].values[0],
            mode="lines+markers",
            name=str(name)+":"+label_name,
        )
        plot_data.append(trace)
        return plot_data

    def customize_camera(self,fig):
        camera = dict(
            eye=dict(x=0, y=0, z=5)  # カメラの視点の位置（x, y, z）
        )

        # レイアウトの更新
        fig.update_layout(
            scene_camera=camera,  # カメラの視点を設定
            scene=dict(
                xaxis=dict(title='X',tickfont=dict(family="Times New Roman",size=36)),
                yaxis=dict(title='Y',tickfont=dict(family="Times New Roman",size=36)),
                zaxis=dict(title='UnixTime [s]',tickfont=dict(family="Times New Roman",size=36)),
            )
        )
        return fig
    
    def customize_layout(self,fig):
        fig.update_layout(
                paper_bgcolor='white',  # 図全体の背景
                plot_bgcolor='white',   # プロット領域の背景
                # title=dict(
                #     text=f"FPS"
                # ),
                xaxis=dict(
                    color='black',
                    showgrid=True,  # グリッドを表示
                    gridcolor='lightgray',  # グリッド線の色
                    linecolor='black',  # 軸線の色
                    ticks='outside',  # メモリを外側に
                    tickwidth=1.5,  # メモリの太さ
                    ticklen=5       # メモリの長さ
                ),
                yaxis=dict(
                    color='black',
                    showgrid=True,
                    gridcolor='lightgray',
                    linecolor='black',
                    ticks='outside',
                    tickwidth=1.5,
                    ticklen=5
                ),
                legend=dict(
                    bgcolor='white',  # 背景色
                    bordercolor='black',  # 枠線の色
                    borderwidth=1.5    # 枠線の太さ
                ),
                font=dict(
                    family='Times New Roman',  # 推奨フォント
                    size=36,  # フォントサイズ
                    color='black'  # フォント色
                ),
                margin=dict(
                    l=40,  # 左
                    r=20,  # 右
                    t=40,  # 上
                    b=40   # 下
                ),
        )
        return fig
    
    def customize_layouts(self,fig):
        axises=[k for k in list(fig.to_dict()["layout"].keys()) if (("template" not in k) and ("title" not in k))]
        fig.update_layout(
                paper_bgcolor='white',  # 図全体の背景
                plot_bgcolor='white',   # プロット領域の背景
                # title=dict(
                #     text=f"FPS"
                # ),
                legend=dict(
                    # x=0.01,          # ①：X座標
                    # y=-0.2,          # ①：Y座標
                    bgcolor='white',  # 背景色
                    bordercolor='black',  # 枠線の色
                    borderwidth=1.5,    # 枠線の太さ
                    # xanchor='left',
                    # yanchor='top',
                    # orientation='h',
                ),
                font=dict(
                    family='Times New Roman',  # 推奨フォント
                    size=18,  # フォントサイズ
                    color='black'  # フォント色
                ),
                margin=dict(
                    l=40,  # 左
                    r=20,  # 右
                    t=40,  # 上
                    b=40   # 下
                ),
        )
        for axis in axises:
            fig.update_layout({
                axis:dict(
                    color='black',
                    showgrid=True,  # グリッドを表示
                    gridcolor='lightgray',  # グリッド線の色
                    linecolor='black',  # 軸線の色
                    ticks='outside',  # メモリを外側に
                    tickwidth=1.5,  # メモリの太さ
                    ticklen=5       # メモリの長さ
                    ),
                }
            )
        return fig

    def plot_map(self,fig,dimension="2D"):
        from PIL import Image,ImageOps
        def cell_to_xy(x, y, map_config, map_height):
            """
            2D mapの画像座標系を地図座標系に変換する
            """
            cell_x = map_config["origin"][0] + x * map_config["resolution"]
            cell_y = (map_height - y) * map_config["resolution"] + map_config["origin"][1]
            return cell_x, cell_y

        map_img=Image.open(self.data_dir_dict["mobilesensing_dir_path"]+"/common"+"/map/map2d.pgm")
        map_config=self.load_yaml(self.data_dir_dict["mobilesensing_dir_path"]+"/common"+"/map/map2d.yaml")
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
            map_img=Image.open(self.data_dir_dict["mobilesensing_dir_path"]+"/common"+"/map/map2d.png").convert("RGB")
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
            # map_img=skimage.io.imread(self.data_dir_dict["mobilesensing_dir_path"]+"/common"+"/map/map2d.png")
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
    


    def draw_positions(self):
        csv_paths=[path for path in self.data_paths if (("data" in os.path.basename(path)) and ("_eval.csv" in os.path.basename(path)))]
        plot_data=[]

        for csv_path in csv_paths:
            name="ID_"+os.path.basename(csv_path)[:-4].split("_")[-2]
            data=pd.read_csv(csv_path,header=0)
            plot_data=self.draw_3d(plot_data,data,name)
        fig=go.Figure(data=plot_data)
        # fig=self.customize_camera(fig)
        fig.update_layout(
                scene_aspectmode='manual',
                scene_aspectratio=dict(x=5, y=5, z=3),
            )
        # fig=self.customize_layout(fig)
        fig=self.plot_map(fig,dimension="3D")
        fig.write_html(self.data_dir_dict["trial_dir_path"]+"/positions.html")
        fig.show()
        pass

    def draw_features(self):
        # csv_paths=[path for path in self.data_paths if (("feature" in os.path.basename(path)) and (".csv" in os.path.basename(path)))]
        csv_paths=sorted(glob(self.data_dir_dict["trial_dir_path"]+"/data_*yolo.csv"))
        plot_data=[]

        for csv_path in csv_paths:
            name=os.path.basename(csv_path)[:-4].split("_")[-2]
            data=pd.read_csv(csv_path,header=0)
            nodes=[k for k in list(data.keys()) if (("timestamp" not in k) and ("active" not in k) and ("fps" not in k))]
            categories=sorted(list(set([str(k)[:1] for k in nodes])))
            
            plot_data=[[] for i in range(len(categories))]
            fig=sp.make_subplots(rows=len(categories), cols=1,  # 2行1列
                        shared_xaxes=True,  # x軸を共有
                        vertical_spacing=0.1)  # グラフ間の間隔
            
            for node in nodes:
                row_no=categories.index(str(node)[:1])
                trace=self.draw_timeseries(data=data,name=node,symbol="x<sup>i</sup>")
                plot_data[row_no].append(trace)
            
            for i,traces in enumerate(plot_data):
                for trace in traces:
                    fig.add_trace(trace,row=i+1,col=1)
            
            # layout
            fig.update_layout(    
                title=dict(
                    text=f"Person: {name}  feature values"
                ),
            )
            for i in range(len(categories)):
                fig.update_xaxes(title=dict(text="Time [s]",font=dict(family="Times New Roman",size=36)),row=i+1,col=1,)
            for i in range(len(categories)):
                fig.update_yaxes(
                    title=dict(text=f"Features {categories[i]}",font=dict(family="Times New Roman",size=36)),
                    range=[-0.1,1.1],
                    row=i+1,
                    col=1,
                    )
            fig=self.customize_layouts(fig)
            fig.write_html(self.data_dir_dict["trial_dir_path"]+f"/features_{name}.html")
            fig.show()

        pass
    
    def draw_weight(self):
        csv_paths=[path for path in self.data_paths if (("weight" in os.path.basename(path)) and (".csv" in os.path.basename(path)))]
        plot_data=[]

        for csv_path in csv_paths:
            name=os.path.basename(csv_path)[:-4].split("_")[-1]
            data=pd.read_csv(csv_path,header=0)
            nodes=[k for k in list(data.keys()) if (("timestamp" not in k) and ("active" not in k))]
            categories=sorted(list(set([str(k)[:3] for k in nodes])))
            
            plot_data=[[] for i in range(len(categories))]
            fig=sp.make_subplots(rows=len(categories), cols=1,  # 2行1列
                        shared_xaxes=False,  # x軸を共有
                        vertical_spacing=0.1)  # グラフ間の間隔
            
            for node in nodes:
                row_no=categories.index(str(node)[:3])
                trace=self.draw_timeseries(data=data,name=node,symbol="w")
                plot_data[row_no].append(trace)
            
            for i,traces in enumerate(plot_data):
                for trace in traces:
                    fig.add_trace(trace,row=i+1,col=1)

            # layout
            fig.update_layout(    
                title=dict(
                    text=f"Person: {name}  Weights"
                ),

            )
            for i in range(len(categories)):
                fig.update_xaxes(title=dict(text="Time [s]",font=dict(family="Times New Roman",size=36)),row=i+1,col=1,)
            for i in range(len(categories)):
                fig.update_yaxes(
                    title=dict(text=f"Weights <i>w</i><sub>{categories[i]}X</sub>",font=dict(family="Times New Roman",size=36)),
                    range=[-0.1,1.1],
                    row=i+1,
                    col=1,
                    )
            fig=self.customize_layouts(fig)
            fig.write_html(self.visualize_dir_path+f"/weights_{name}.html")
            fig.show()
    
    def draw_weights(self):
        csv_paths=[path for path in self.data_paths if (("weight" in os.path.basename(path)) and (".csv" in os.path.basename(path)))]
        plot_data=[]

        for csv_path in csv_paths:
            name=os.path.basename(csv_path)[:-4].split("_")[-1]
            data=pd.read_csv(csv_path,header=0)
            nodes=[k for k in list(data.keys()) if (("timestamp" not in k) and ("active" not in k))]
            categories=sorted(list(set([str(k)[:3] for k in nodes])))
            
            plot_data=[[] for i in range(len(categories))]
            fig=sp.make_subplots(rows=len(categories), cols=1,  # 2行1列
                        shared_xaxes=True,  # x軸を共有
                        vertical_spacing=0.1)  # グラフ間の間隔
            
            for node in nodes:
                row_no=categories.index(str(node)[:3])
                trace=self.draw_timeseries(data=data,name=node,symbol="w")
                plot_data[row_no].append(trace)
            
            for i,traces in enumerate(plot_data):
                for trace in traces:
                    fig.add_trace(trace,row=i+1,col=1)

            # layout
            fig.update_layout(    
                title=dict(
                    text=f"Person: {name}  Weights"
                ),

            )
            fig.update_xaxes(title=dict(text="Time [s]",font=dict(family="Times New Roman",size=36)),row=len(categories),col=1,)
            for i in range(len(categories)):
                fig.update_yaxes(
                    title=dict(text=f"Weights {categories[i]}",font=dict(family="Times New Roman",size=36)),
                    range=[-0.1,1.1],
                    row=i+1,
                    col=1,
                    )
            fig=self.customize_layouts(fig)
            fig.write_html(self.visualize_dir_path+f"/weights_{name}.html")
            fig.show()

    def draw_fps(self):
        csv_paths=[path for path in self.data_paths if (("feature" in os.path.basename(path)) and (".csv" in os.path.basename(path)))]
        plot_data=[]
        # plot_data=[[] for i in range(len(csv_paths))]

        for i,csv_path in enumerate(csv_paths):
            name=os.path.basename(csv_path)[:-4].split("_")[-1]
            print(name)
            
            data=pd.read_csv(csv_path,header=0)
            data["fps"].interpolate(method="ffill",inplace=True)
            trace=self.draw_timeseries(data,name="fps",label_name=name)
            plot_data.append(trace)
        fig=go.Figure(data=plot_data)
        fig=self.customize_layout(fig)
        fig.update_xaxes(title=dict(text="Time [s]",font=dict(family="Times New Roman",size=36)))
        fig.update_yaxes(title=dict(text="fps [/s]",font=dict(family="Times New Roman",size=36)))
        fig.write_html(self.visualize_dir_path+f"/fps.html")
        fig.show()

    def draw_nActive(self):
        indexes=["①","②","③"]
        data=[99,40,51]
        colors=["blue","blue","red"]
        fig=go.Figure()
        trace=go.Bar(x=indexes,y=data,marker_color=colors)
        fig.add_trace(trace)
        fig=self.customize_layout(fig)
        fig.update_xaxes(title=dict(text="Method",font=dict(family="Times New Roman",size=36)))
        fig.update_yaxes(title=dict(text="Number of feature observation",font=dict(family="Times New Roman",size=36)))
        fig.show()

    def plot_matplotlib(self):
        csv_paths=sorted(glob(self.data_dir_dict["trial_dir_path"]+"/data_*_eval.csv"))
        data_dict={}
        id_names=[]
        for csv_path in csv_paths:
            data=pd.read_csv(csv_path,header=0)
            id_name=os.path.basename(csv_path).split("_")[1]
            id_names.append(id_name)
            data_dict[id_name]=data
        #     print(data)
        export_labels=[str(k) for k in data_dict[id_name].keys() if str(k)!="timestamp"]
        for export_label in export_labels:
            print(export_label)
            for id_name in id_names:
                # if int(export_label) in ["10000000","20000001"]:
                w=40
                try:
                    plt.plot(data_dict[id_name]["timestamp"],data_dict[id_name][export_label].rolling(w).mean(),"-o",label=id_name)
                    plt.title(export_label+f" window: {w}")
                except pd.errors.DataError:
                    try:
                        plt.plot(data_dict[id_name]["timestamp"],data_dict[id_name][export_label],"-o",label=id_name)
                    except TypeError:
                        pass
                    plt.title(export_label)
            plt.legend()
            plt.grid()
            plt.xlabel("Time [s]")
            plt.ylabel("Risk value")
            plt.savefig(self.data_dir_dict["trial_dir_path"]+f"/result_{export_label}.jpg")
            plt.close()

        # for csv_path in csv_paths:
        #     data=pd.read_csv(csv_path,header=0)
        #     # if "B" in os.path.basename(csv_path):
        #     plt.plot(data["timestamp"],data["10000000"],label=os.path.basename(csv_path))
        # plt.xlabel("Time [s]")
        # plt.ylabel("Risk value")
        # plt.legend()
        # plt.show()

    def plot_fps(self):
        csv_paths=sorted(glob(self.data_dir_dict["trial_dir_path"]+"/fps_*.csv"))
        id_names=[os.path.basename(p)[len("fps_"):-len(".csv")] for p in csv_paths]
        data_dicts={}
        for id_name,csv_path in zip(id_names,csv_paths):
            data=pd.read_csv(csv_path,header=0)
            data=data.fillna(method="ffill")
            data=data.rolling(4).mean()
            data_dicts[id_name]=data
        
        plot_nodes=["50000100","50001000","60010000"]
        marker_shapes=["o","^","s","x","v","*","+"]
        # edge_colors=["r","g","b","m","c","k"]
        for plot_node in plot_nodes:
            # for i,id_name in enumerate(id_names):
            #     plt.plot(data_dicts[id_name]["timestamp"],data_dicts[id_name][plot_node],"--"+marker_shapes[i],alpha=0.5,label=id_name)
            # plt.legend()
            # plt.grid()
            # plt.xlabel("Time [s]")
            # plt.ylabel("Frame rate [Hz]")
            # plt.savefig(self.data_dir_dict["trial_dir_path"]+f"/result_fps_{plot_node}.jpg")
            # plt.close()
            timestamps = data_dicts[id_names[0]]["timestamp"]
            stacked_data = [data_dicts[id_name][plot_node] for id_name in id_names]

            # 積み上げ折れ線グラフを描画
            plt.stackplot(
                timestamps, 
                *stacked_data,  # 各 id_name に対応するデータを展開
                labels=id_names, 
                alpha=1
            )

            # グラフの装飾
            plt.legend()
            plt.grid()
            plt.xlabel("Time [s]")
            plt.ylabel("Frame rate [Hz]")
            plt.savefig(self.data_dir_dict["trial_dir_path"]+f"/result_fps_{plot_node}.jpg")
            plt.show()
        print(csv_paths)
        pass
    def main(self):
        pass

if __name__=="__main__":
    # trial_name="20250113NormalSimulation"
    # trial_name="20250110SimulationMultipleRisks/no_00005"
    trial_name="20250115PullWheelchairObaachan2"
    strage="NASK"
    cls=Visualizer(trial_name=trial_name,strage=strage)
    # cls.visualize_graph(trial_name="20241229BuildSimulator",strage="NASK",name="A",show=True)
    cls.plot_matplotlib()
    # cls.plot_fps()
    # cls.draw_positions()
    # cls.draw_features()
    # cls.draw_weight()
    # cls.draw_fps()
    # cls.draw_nActive()
    pass