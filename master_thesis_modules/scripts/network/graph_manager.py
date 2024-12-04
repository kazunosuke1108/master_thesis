from icecream import ic
import copy
import numpy as np
import networkx as nx

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
                "description":"看護師の不在"
                },
            5521:{
                "score":np.nan,
                "status":"inactive",
                "description":"介護士の不在"
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


        self.graph_dict={
            "A":{"G":"","node_dict":copy.deepcopy(self.node_dict),"weight_dict":copy.deepcopy(self.weight_dict),"pos":""},
            "B":{"G":"","node_dict":copy.deepcopy(self.node_dict),"weight_dict":copy.deepcopy(self.weight_dict),"pos":""},
            "C":{"G":"","node_dict":copy.deepcopy(self.node_dict),"weight_dict":copy.deepcopy(self.weight_dict),"pos":""},
        }
        for name in self.graph_dict.keys():
            self.define_graph(name)
    
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
        ic(self.graph_dict[name]["node_dict"]) # ここではちゃんと書き換わってる

    def update_weight(self,name,new_weight_dict):
        for node_from in new_weight_dict.keys():
            for node_to in new_weight_dict[node_from].keys():
                self.graph_dict[name]["weight_dict"][node_from][node_to]=new_weight_dict[node_from][node_to]
        self.define_graph(name)
    
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
    
    # def update_graph(self,name):
    #     try:
    #         del self.G
    #     except AttributeError:
    #         pass
    #     self.define_graph()

    # def colorize(self,default=True):
    #     # 色追記
    #     self.nodecolor=[]
    #     if default:
    #         for node_code in self.G.nodes():
    #             self.nodecolor.append((1,0,0) if self.node_dict[node_code]["status"]=="active" else "gray")
    #     else:
    #         for node_code in self.G.nodes():
    #             node=self.node_dict[node_code]
    #             if np.isnan(node["score"]):
    #                 self.nodecolor.append((1,1,1) if node["status"]=="active" else "gray")
    #             else:
    #                 self.nodecolor.append((node["score"],0,0) if node["status"]=="active" else "gray")
    #         pass

    def visualize_plotly(self,name="A"):
        ic(self.graph_dict[name]["node_dict"][5520])
        # self.colorize()
        # グラフの情報を取り出しておく
        nodes = self.graph_dict[name]["G"].nodes()
        pos = self.graph_dict[name]["pos"] # key: ノード番号, value: [x,y]
        weights = nx.get_edge_attributes(self.graph_dict[name]["G"], 'weight') # key:(node_from,node_to), value: weight
        scores = {n:self.graph_dict[name]["node_dict"][n]["score"] for n in nodes}# key: ノード番号, value: 特徴量
        descriptions = {n:self.graph_dict[name]["node_dict"][n]["description"] for n in nodes}

        # 色の準備
        cmap = cm.get_cmap('jet')

        # # ノード描画データ
        # node_x = [pos[node][0] for node in nodes]
        # node_y = [pos[node][1] for node in nodes]

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
                                f"score: {self.node_dict[node]['score']}<br>"+
                                f"left weight: {previous_weight}<extra></extra>",
                marker=dict(
                    size=30,
                    color=color,
                    showscale=True
                ),
                textposition="top center"
            )
            node_traces.append(node_trace)

        # グラフを作成
        fig = go.Figure(data=node_traces+edge_traces)
        fig.update_layout(
            showlegend=False,
            title=name,
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            plot_bgcolor='white'
        )
        fig.show()
        pass
        
    # def visualize_matplotlib(self):
    #     self.colorize()
    #     weights = nx.get_edge_attributes(self.G, 'weight').values()
    #     nx.draw(self.G, self.pos, node_color=self.nodecolor, with_labels=True, edge_color = weights, edge_cmap=plt.cm.RdBu_r)
    #     plt.pause(1)
    #     plt.close()

    def main(self):
        self.visualize_plotly(name="C")
        # self.colorize(default=False)
        # self.visualize()
        # self.update_lower_layer_status()
        # self.visualize()

if __name__=="__main__":
    cls=GraphManager()
    cls.main()