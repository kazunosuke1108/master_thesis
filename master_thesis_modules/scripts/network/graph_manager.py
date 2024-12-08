# pip install -U kaleido
import os
from icecream import ic
import copy
import numpy as np
import networkx as nx
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import plotly.graph_objects as go

class GraphManager():
    def __init__(self):
        # define network
        super().__init__()
        
        # node定義
        self.node_dict={
            1000:{
                "score":np.nan,
                "status":"active",
                "description":"最も行動が危険な人物の選定"
                },
            2000:{
                "score":np.nan,
                "status":"active",
                "description":"転倒事故防止"
                },
            2001:{
                "score":np.nan,
                "status":"active",
                "description":"スタッフの緊張緩和"
                },
            2002:{
                "score":np.nan,
                "status":"active",
                "description":"患者の主体性担保"
                },
            3000:{
                "score":np.nan,
                "status":"active",
                "description":"本人の属性"
                },
            3001:{
                "score":np.nan,
                "status":"active",
                "description":"本人の動作"
                },
            3002:{
                "score":np.nan,
                "status":"active",
                "description":"本人の様子"
                },
            3003:{
                "score":np.nan,
                "status":"active",
                "description":"本人の動機"
                },
            3004:{
                "score":np.nan,
                "status":"active",
                "description":"時間的文脈"
                },
            3005:{
                "score":np.nan,
                "status":"active",
                "description":"空間的文脈"
                },
            4050:{
                "score":np.nan,
                "status":"inactive",
                "description":"本人の居場所"
                },
            4051:{
                "score":np.nan,
                "status":"inactive",
                "description":"周囲の物体"
                },
            4052:{
                "score":np.nan,
                "status":"inactive",
                "description":"周囲の人物"
                },
            5510:{
                "score":np.nan,
                "status":"inactive",
                "description":"経管栄養・点滴の存在"
                },
            5511:{
                "score":np.nan,
                "status":"inactive",
                "description":"車椅子の存在"
                },
            5512:{
                "score":np.nan,
                "status":"inactive",
                "description":"手すりの不在"
                },
            5520:{
                "score":np.nan,
                "status":"inactive",
                "description":"スタッフが遠方"
                },
            5521:{
                "score":np.nan,
                "status":"inactive",
                "description":"スタッフの視野外"
                },
            5522:{
                "score":np.nan,
                "status":"inactive",
                "description":"面会者の存在"
                },
        }

        # 重み定義
        self.weight_dict={
            1000:{
                2000:1/3,
                2001:1/3,
                2002:1/3,
            },
            2000:{
                3000:1/6,
                3001:1/6,
                3002:1/6,
                3003:1/6,
                3004:1/6,
                3005:1/6,
            },
            # 2001:{},
            # 2002:{},
            # 3000:{},
            # 3001:{},
            # 3002:{},
            # 3003:{},
            # 3004:{},
            3005:{
                4050:1/3,
                4051:1/3,
                4052:1/3,
            },
            # 4050:{},
            4051:{
                5510:1/3,
                5511:1/3,
                5512:1/3,
            },
            4052:{
                5520:1/3,
                5521:1/3,
                5522:1/3,
            },
        }
        weight_history={}
        for node_from in self.weight_dict.keys():
            for node_to in self.weight_dict[node_from].keys():
                weight_history[node_to]=self.weight_dict[node_from][node_to]
        self.weight_history=pd.DataFrame(weight_history,index=[0])

        self.graph_dict={
            "A":{"G":"","node_dict":copy.deepcopy(self.node_dict),"weight_dict":copy.deepcopy(self.weight_dict),"weight_history":copy.deepcopy(self.weight_history),"pos":""},
            "B":{"G":"","node_dict":copy.deepcopy(self.node_dict),"weight_dict":copy.deepcopy(self.weight_dict),"weight_history":copy.deepcopy(self.weight_history),"pos":""},
            "C":{"G":"","node_dict":copy.deepcopy(self.node_dict),"weight_dict":copy.deepcopy(self.weight_dict),"weight_history":copy.deepcopy(self.weight_history),"pos":""},
        }
        for name in self.graph_dict.keys():
            self.define_graph(name)
    
    def add_weight_history(self,weight_history,weight_dict,timestamp=0):
        flattened_weight_dict={}
        for node_from in weight_dict.keys():
            for node_to in weight_dict[node_from].keys():
                flattened_weight_dict[node_to]=weight_dict[node_from][node_to]
        added_weight=pd.DataFrame(flattened_weight_dict,index=[0]).values
        weight_history.loc[timestamp]=list(added_weight.flatten())
        return weight_history

    def define_graph(self,name):
        self.graph_dict[name]["G"] = nx.Graph()
        # node定義
        self.graph_dict[name]["G"].add_nodes_from(self.graph_dict[name]["node_dict"].keys())
        # edge定義
        for lv in range(5):
            for node_code_from in [n for n in list(self.graph_dict[name]["node_dict"].keys()) if str(n)[0]==str(lv)]:
                if node_code_from=="description":
                    continue
                node_codes_to=[k for k in [j for j in list(self.graph_dict[name]["node_dict"].keys()) if str(j)[0]==str(lv+1)] if str(node_code_from)[2:] == str(k)[1:3]]
                self.graph_dict[name]["G"].add_edges_from([(node_code_from,node_code_to,{"weight":self.graph_dict[name]["weight_dict"][node_code_from][node_code_to]}) for node_code_to in node_codes_to])
        # 位置追記
        self.graph_dict[name]["pos"]={}
        previous_layer=0
        for node_code in self.graph_dict[name]["G"].nodes():
            if previous_layer!=int(str(node_code)[0]):
                y=0
            else:
                y-=1
            self.graph_dict[name]["pos"][node_code]=(int(str(node_code)[0]),y)
            previous_layer=int(str(node_code)[0])

    def update_score(self,name,new_score_dict):
        for node_code,score in new_score_dict.items():
            self.graph_dict[name]["node_dict"][node_code]["score"]=score
        self.define_graph(name)

    def update_weight(self,name,new_weight_dict,timestamp):
        for node_from in new_weight_dict.keys():
            for node_to in new_weight_dict[node_from].keys():
                self.graph_dict[name]["weight_dict"][node_from][node_to]=new_weight_dict[node_from][node_to]
        self.define_graph(name)
        self.graph_dict[name]["weight_history"]=self.add_weight_history(weight_history=self.graph_dict[name]["weight_history"],weight_dict=self.graph_dict[name]["weight_dict"],timestamp=timestamp)
    
    def update_lower_layer_status(self,name,new_status="active"):
        for node in self.graph_dict[name]["G"].nodes():
            if int(str(node)[0])>=4:
                self.graph_dict[name]["node_dict"][node]["status"]=new_status
        self.define_graph(name)
        # self.colorize(default=False)
        
    def get_left_weight(self,name,node):
        if node==1000:
            previous_weight=np.nan
        else:
            for node_from in self.graph_dict[name]["weight_dict"].keys():
                for node_to in self.graph_dict[name]["weight_dict"][node_from].keys():
                    if node_to==node:
                        previous_weight=np.round(self.graph_dict[name]["weight_dict"][node_from][node_to],2)
                        break
        return previous_weight
    
    def visualize_plotly(self,name="A",show=False):
        # self.colorize()
        symbol_dict={
            1:"circle",
            0:"x",
            np.nan:"x",
            "active":"circle",
            "inactive":"x",
        }
        # グラフの情報を取り出しておく
        nodes = self.graph_dict[name]["G"].nodes()
        pos = self.graph_dict[name]["pos"] # key: ノード番号, value: [x,y]
        weights = nx.get_edge_attributes(self.graph_dict[name]["G"], 'weight') # key:(node_from,node_to), value: weight
        status = {n:self.graph_dict[name]["node_dict"][n]["status"] for n in nodes}
        scores = {n:self.graph_dict[name]["node_dict"][n]["score"] for n in nodes}# key: ノード番号, value: 特徴量
        descriptions = {n:self.graph_dict[name]["node_dict"][n]["description"] for n in nodes}

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
                line=dict(width=2, color=color),
                customdata=[weight],
                hovertemplate="weight: %{customdata:.2f}<extra></extra>",
                mode='lines'
            )
            edge_traces.append(edge_trace)

        node_traces=[]
        for node,p in pos.items():
            # ひとつ手前側のエッジの重み情報
            previous_weight=self.get_left_weight(name=name,node=node)
            # ノードの色
            color=mcolors.rgb2hex(cmap(scores[node]))
            # ノードのトレース
            node_trace = go.Scatter(
                x=[p[0]],
                y=[p[1]],
                mode='markers+text',
                text=descriptions[node],
                customdata=[node,descriptions[node]],
                hovertemplate=f"No. {node}<br>"+
                                "description: "+descriptions[node]+"<br>"+
                                f"score: {np.round(scores[node],2)}<br>"+
                                f"left weight: {previous_weight}<extra></extra>",
                marker=dict(
                    symbol=symbol_dict[status[node]],
                    size=30,
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
            plot_bgcolor='white'
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
            # xaxis=dict(range=[0, 5]),
            # yaxis=dict(range=[0, 20]),
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
            trial_temp_dir_path=trial_dir_path+"/temp"
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
                fig.update_layout(title_text=f"Person:{name}  Frame:{i+1}  timestamp: {timestamp}")
                print(i)
                fig.write_image(f"{trial_temp_dir_path}/{name}_{i:03d}.jpg",format='jpg', engine="auto")
            print("export")
            self.jpg2mp4(sorted(glob(trial_temp_dir_path+"/*")),mp4_path=trial_dir_path+f"/{name}.mp4",fps=4)
        return frames

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

    def main(self):
        self.visualize_plotly(name="C",show=True)
        # self.colorize(default=False)
        # self.visualize()
        # self.update_lower_layer_status()
        # self.visualize()

if __name__=="__main__":
    cls=GraphManager()
    cls.main()