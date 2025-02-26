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
        
    def get_default_graph(self):
        # node定義
        node_dict={
            "10000000":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Overall risk ratio",
                    "layer":1,
                    "node_type":"dynamic",
                    "description_ja":"",
                    "node_code_to":["20000000","20000001"]
                    },
            "20000000":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Internal risk ratio",
                    "layer":2,
                    "node_type":"static",
                    "description_ja":"",
                    "node_code_to":["30000000","30000001"]
                    },
            "20000001":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"External risk ratio",
                    "layer":2,
                    "node_type":"dynamic",
                    "description_ja":"",
                    "node_code_to":["30000010","30000011"]
                    },
            "30000000":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Risk regarding their attribution",
                    "layer":3,
                    "node_type":"static",
                    "description_ja":"",
                    "node_code_to":["40000000","40000001"]
                    },
            "30000001":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Risk regarding their action",
                    "layer":3,
                    "node_type":"static",
                    "description_ja":"",
                    "node_code_to":["40000010","40000011","40000012","40000013","40000014","40000015","40000016"]
                    },
            "30000010":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Risk regarding surrounding objects",
                    "layer":3,
                    "node_type":"dynamic",
                    "description_ja":"",
                    "node_code_to":["40000100","40000101","40000102"]
                    },
            "30000011":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Risk regarding the surrounding staff",
                    "layer":3,
                    "node_type":"dynamic",
                    "description_ja":"",
                    "node_code_to":["40000110","40000111"]
                    },
            "40000000":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Probability of patient",
                    "layer":4,
                    "node_type":"static",
                    "description_ja":"患者である",
                    "node_code_to":["50000000","50000001"]
                    },
            "40000001":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Risk of their age",
                    "layer":4,
                    "node_type":"static",
                    "description_ja":"高齢である",
                    "node_code_to":["50000010","50000011"]
                    },
            "40000010":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Similarity of pose with standing",
                    "layer":4,
                    "node_type":"dynamic",
                    "description_ja":"立ち上がろうとしている",
                    "node_code_to":["50000100","50000101","50000102","50000103"]
                    },
            "40000011":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Similarity of pose with releasing brakes",
                    "layer":4,
                    "node_type":"dynamic",
                    "description_ja":"車椅子のブレーキを解除しようとしている",
                    "node_code_to":["50000100","50000101","50000102","50000103"],
                    },
            "40000012":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Similarity of pose with moving wheelchair",
                    "layer":4,
                    "node_type":"dynamic",
                    "description_ja":"車椅子を動かそうとしている",
                    "node_code_to":["50000100","50000101","50000102","50000103"],
                    },
            "40000013":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Similarity of pose with losing balance",
                    "layer":4,
                    "node_type":"dynamic",
                    "description_ja":"バランスを崩している",
                    "node_code_to":["50000100","50000101","50000102","50000103"],
                    },
            "40000014":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Similarity of pose with raising hands",
                    "layer":4,
                    "node_type":"dynamic",
                    "description_ja":"手を挙げている",
                    "node_code_to":["50000100","50000101","50000102","50000103"],
                    },
            "40000015":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Similarity of pose with coughing up",
                    "layer":4,
                    "node_type":"dynamic",
                    "description_ja":"せき込んでいる",
                    "node_code_to":["50000100","50000101","50000102","50000103"],
                    },
            "40000016":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Similarity of pose with touching face",
                    "layer":4,
                    "node_type":"dynamic",
                    "description_ja":"顔を触っている",
                    "node_code_to":["50000100","50000101","50000102","50000103"],
                    },
            "40000100":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Probability of ivPole's existance",
                    "layer":4,
                    "node_type":"static",
                    "description_ja":"点滴の近くにいる",
                    "node_code_to":["50001000","50001001","50001002","50001003","60010000"],
                    },
            "40000101":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Probability of wheelchair's existance",
                    "layer":4,
                    "node_type":"static",
                    "description_ja":"車椅子に乗っている",
                    "node_code_to":["50001000","50001001","50001002","50001003","60010000"],
                    },
            "40000102":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Distance from handrail",
                    "layer":4,
                    "node_type":"static",
                    "description_ja":"手すりから離れている",
                    "node_code_to":["50001000","50001001","50001002","50001003","60010000"],
                    },
            "40000110":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Risk regarding absense of medical staff",
                    "layer":4,
                    "node_type":"dynamic",
                    "description_ja":"スタッフがいない",
                    "node_code_to":["50001100","50001101","60010000","60010001"],
                    },
            "40000111":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Risk regarding absense of staff's eyes",
                    "layer":4,
                    "node_type":"dynamic",
                    "description_ja":"スタッフが見ていない",
                    "node_code_to":["50001100","50001101","50001110","50001111","60010000","60010001"],
                    },
            "50000000":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Risk ratio",
                    "layer":5,
                    "node_type":"static",
                    "description_ja":"",
                    "node_code_to":[]
                    },
            "50000001":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Risk ratio",
                    "layer":5,
                    "node_type":"static",
                    "description_ja":"",
                    "node_code_to":[]
                    },
            "50000010":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Risk ratio",
                    "layer":5,
                    "node_type":"static",
                    "description_ja":"",
                    "node_code_to":[]
                    },
            "50000011":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Risk ratio",
                    "layer":5,
                    "node_type":"static",
                    "description_ja":"",
                    "node_code_to":[]
                    },
            "50000100":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Risk ratio",
                    "layer":5,
                    "node_type":"static",
                    "description_ja":"",
                    "node_code_to":[]
                    },
            "50000101":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Risk ratio",
                    "layer":5,
                    "node_type":"static",
                    "description_ja":"",
                    "node_code_to":[]
                    },
            "50000102":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Risk ratio",
                    "layer":5,
                    "node_type":"static",
                    "description_ja":"",
                    "node_code_to":[]
                    },
            "50000103":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Risk ratio",
                    "layer":5,
                    "node_type":"static",
                    "description_ja":"",
                    "node_code_to":[]
                    },
            "50001000":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Risk ratio",
                    "layer":5,
                    "node_type":"static",
                    "description_ja":"",
                    "node_code_to":[]
                    },
            "50001001":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Risk ratio",
                    "layer":5,
                    "node_type":"static",
                    "description_ja":"",
                    "node_code_to":[]
                    },
            "50001002":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Risk ratio",
                    "layer":5,
                    "node_type":"static",
                    "description_ja":"",
                    "node_code_to":[]
                    },
            "50001003":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Risk ratio",
                    "layer":5,
                    "node_type":"static",
                    "description_ja":"",
                    "node_code_to":[]
                    },
            "50001010":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Risk ratio",
                    "layer":5,
                    "node_type":"static",
                    "description_ja":"",
                    "node_code_to":[]
                    },
            "50001011":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Risk ratio",
                    "layer":5,
                    "node_type":"static",
                    "description_ja":"",
                    "node_code_to":[]
                    },
            "50001012":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Risk ratio",
                    "layer":5,
                    "node_type":"static",
                    "description_ja":"",
                    "node_code_to":[]
                    },
            "50001013":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Risk ratio",
                    "layer":5,
                    "node_type":"static",
                    "description_ja":"",
                    "node_code_to":[]
                    },
            "50001020":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Risk ratio",
                    "layer":5,
                    "node_type":"static",
                    "description_ja":"",
                    "node_code_to":[]
                    },
            "50001021":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Risk ratio",
                    "layer":5,
                    "node_type":"static",
                    "description_ja":"",
                    "node_code_to":[]
                    },
            "50001022":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Risk ratio",
                    "layer":5,
                    "node_type":"static",
                    "description_ja":"",
                    "node_code_to":[]
                    },
            "50001023":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Risk ratio",
                    "layer":5,
                    "node_type":"static",
                    "description_ja":"",
                    "node_code_to":[]
                    },
            "50001100":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Risk ratio",
                    "layer":5,
                    "node_type":"static",
                    "description_ja":"",
                    "node_code_to":[]
                    },
            "50001101":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Risk ratio",
                    "layer":5,
                    "node_type":"static",
                    "description_ja":"",
                    "node_code_to":[]
                    },
            "50001110":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Risk ratio",
                    "layer":5,
                    "node_type":"static",
                    "description_ja":"",
                    "node_code_to":[]
                    },
            "50001111":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Risk ratio",
                    "layer":5,
                    "node_type":"static",
                    "description_ja":"",
                    "node_code_to":[]
                    },
            "60010000":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Risk ratio",
                    "layer":6,
                    "node_type":"static",
                    "description_ja":"",
                    "node_code_to":[]
                    },
            "60010001":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Risk ratio",
                    "layer":6,
                    "node_type":"static",
                    "description_ja":"",
                    "node_code_to":[]
                    },
            "60010002":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Risk ratio",
                    "layer":6,
                    "node_type":"static",
                    "description_ja":"",
                    "node_code_to":[]
                    },
            "70000000":{
                    "score":np.nan,
                    "status":"active",
                    "description_en":"Background differencing value",
                    "layer":6,
                    "node_type":"static",
                    "description_ja":"",
                    "node_code_to":[]
                    },
        }

        # 重み定義
        weight_dict={
        }
        for node in node_dict.keys():
            node_code_from=node
            weight_dict[node_code_from]={}
            node_codes_to=node_dict[node_code_from]["node_code_to"]
            for node_code_to in node_codes_to:
                weight_dict[node_code_from][node_code_to]=1/len(node_codes_to)         

        G = nx.Graph()
        # node定義
        G.add_nodes_from(node_dict.keys())
        # edge定義
        for node in node_dict.keys():
            node_code_from=node
            node_codes_to=node_dict[node_code_from]["node_code_to"]
            G.add_edges_from([(node_code_from,node_code_to,{"weight":weight_dict[node_code_from][node_code_to]}) for node_code_to in node_codes_to])
        # 位置追記
        pos_dict={}
        previous_layer=0
        max_layer=int(str(max(list(node_dict.keys())))[0])
        n_node_per_layer_dict={
            1:len([n for n in node_dict.keys() if node_dict[n] ["layer"]==1]),
            2:len([n for n in node_dict.keys() if node_dict[n] ["layer"]==2]),
            3:len([n for n in node_dict.keys() if node_dict[n] ["layer"]==3]),
            4:len([n for n in node_dict.keys() if node_dict[n] ["layer"]==4]),
            5:len([n for n in node_dict.keys() if node_dict[n] ["layer"]==5]),
            6:len([n for n in node_dict.keys() if node_dict[n] ["layer"]==6]),
        }
        graph_height=50
        # print(n_node_per_layer_dict)
        # raise NotImplementedError
        for node in node_dict.keys():
            h=graph_height/(n_node_per_layer_dict[node_dict[node]["layer"]]+1)
            if previous_layer!=node_dict[node]["layer"]:
                y=-h
            else:
                y-=h
            x=max_layer-node_dict[node]["layer"]
            pos_dict[node]=(x,y)
            previous_layer=node_dict[node]["layer"]

        graph_dict={
            "G":G,
            "node_dict":node_dict,
            "weight_dict":weight_dict,
            "pos_dict":pos_dict,
        }
        return graph_dict


        # for lv in range(5):
        #     for node_code_from in [n for n in list(graph_dict[name]["node_dict"].keys()) if str(n)[0]==str(lv)]:
        #         if node_code_from=="description":
        #             continue
        #         node_codes_to=[k for k in [j for j in list(self.graph_dict[name]["node_dict"].keys()) if str(j)[0]==str(lv+1)] if str(node_code_from)[2:] == str(k)[1:3]]
        #         self.G.add_edges_from([(node_code_from,node_code_to,{"weight":self.graph_dict[name]["weight_dict"][node_code_from][node_code_to]}) for node_code_to in node_codes_to])
        # # 位置追記
        # self.graph_dict[name]["pos"]={}
        # previous_layer=0
        # for node_code in self.G.nodes():
        #     if previous_layer!=int(str(node_code)[0]):
        #         y=0
        #     else:
        #         y-=1
        #     self.graph_dict[name]["pos"][node_code]=(int(str(node_code)[0]),y)
        #     previous_layer=int(str(node_code)[0])


        # weight_history={}
        # # for node_from in weight_dict.keys():
        # #     for node_to in weight_dict[node_from].keys():
        # #         weight_history[node_to]=weight_dict[node_from][node_to]
        # self.weight_history=pd.DataFrame(weight_history,index=[0])

        # self.graph_dict={
        #     "A":{"G":"","node_dict":copy.deepcopy(self.node_dict),"weight_dict":copy.deepcopy(weight_dict),"weight_history":copy.deepcopy(self.weight_history),"pos":""},
        #     "B":{"G":"","node_dict":copy.deepcopy(self.node_dict),"weight_dict":copy.deepcopy(weight_dict),"weight_history":copy.deepcopy(self.weight_history),"pos":""},
        #     "C":{"G":"","node_dict":copy.deepcopy(self.node_dict),"weight_dict":copy.deepcopy(weight_dict),"weight_history":copy.deepcopy(self.weight_history),"pos":""},
        # }
        # for name in self.graph_dict.keys():
        #     self.define_graph(name)
    
    def add_weight_history(self,weight_history,weight_dict,timestamp=0):
        flattened_weight_dict={}
        for node_from in weight_dict.keys():
            for node_to in weight_dict[node_from].keys():
                flattened_weight_dict[node_to]=weight_dict[node_from][node_to]
        added_weight=pd.DataFrame(flattened_weight_dict,index=[0]).values
        weight_history.loc[timestamp]=list(added_weight.flatten())
        return weight_history

    # def define_graph(self,name):
    #     self.graph_dict[name]["G"] = nx.Graph()
    #     # node定義
    #     self.graph_dict[name]["G"].add_nodes_from(self.graph_dict[name]["node_dict"].keys())
    #     # edge定義
    #     for lv in range(5):
    #         for node_code_from in [n for n in list(self.graph_dict[name]["node_dict"].keys()) if str(n)[0]==str(lv)]:
    #             if node_code_from=="description":
    #                 continue
    #             node_codes_to=[k for k in [j for j in list(self.graph_dict[name]["node_dict"].keys()) if str(j)[0]==str(lv+1)] if str(node_code_from)[2:] == str(k)[1:3]]
    #             self.graph_dict[name]["G"].add_edges_from([(node_code_from,node_code_to,{"weight":self.graph_dict[name]["weight_dict"][node_code_from][node_code_to]}) for node_code_to in node_codes_to])
    #     # 位置追記
    #     self.graph_dict[name]["pos"]={}
    #     previous_layer=0
    #     for node_code in self.graph_dict[name]["G"].nodes():
    #         if previous_layer!=int(str(node_code)[0]):
    #             y=0
    #         else:
    #             y-=1
    #         self.graph_dict[name]["pos"][node_code]=(int(str(node_code)[0]),y)
    #         previous_layer=int(str(node_code)[0])

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
        descriptions = {n:self.graph_dict[name]["node_dict"][n]["description_en"] for n in nodes}

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
                    size=60,
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
    # cls.main()
    cls.get_default_graph()