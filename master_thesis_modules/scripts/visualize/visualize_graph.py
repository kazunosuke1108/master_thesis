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
import networkx as nx
import plotly.subplots as sp
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


from scripts.management.manager import Manager

class Visualizer(Manager):
    def __init__(self):
        super().__init__()

    def visualize_graph(self,trial_name="20250120DevBasicCheck",strage="NASK",patient="A",nrow=1,show_features=False,show=False,save=False):
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
        self.graph_dict=self.graph_dicts[patient]
        csv_path=self.data_dir_dict["trial_dir_path"]+f"/data_{patient}_eval.csv"
        self.data=pd.read_csv(csv_path,header=0)

        # グラフの情報を取り出しておく
        nodes = self.graph_dict["G"].nodes()
        pos = self.graph_dict["pos_dict"] # key: ノード番号, value: [x,y]
        weights = nx.get_edge_attributes(self.graph_dict["G"], 'weight') # key:(node_from,node_to), value: weight
        status = {n:self.graph_dict["node_dict"][n]["status"] for n in nodes}
        # scores = {n:self.graph_dict["node_dict"][n]["score"] for n in nodes}# key: ノード番号, value: 特徴量
        scores = self.data.loc[nrow,:].to_dict()
        for k,v in scores.items():
            if type(v)==str:
                scores[k]=eval(v)[1]
            if np.isnan(scores[k]):
                scores[k]=0
        descriptions = {n:self.graph_dict["node_dict"][n]["description_en"] for n in nodes}

        # 5層以下を消す
        if not show_features:
            nodes=[n for n in nodes if int(n)<50000000]
            pos={k:pos[k] for k in pos.keys() if int(k)<50000000}
            # raise NotImplementedError
            weights={k:weights[k] for k in weights.keys() if int(k[0])<40000000}

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
            try:
                color=mcolors.rgb2hex(cmap(self.sigmoid(scores[str(node)])))
            except TypeError:
                color=mcolors.rgb2hex(cmap(0))
            # ノードのトレース
            print(scores[str(node)])
            node_trace = go.Scatter(
                x=[p[0]],
                y=[p[1]],
                mode='markers+text',
                text=descriptions[node]+f"<br>{np.round(scores[str(node)],2)}" if scores[str(node)]>0.3 else None,
                customdata=[node,descriptions[node]],
                hovertemplate=f"No. {node}<br>"+
                                "description: "+descriptions[node]+"<br>"+
                                f"score: {np.round(scores[str(node)] if not np.isnan(scores[str(node)]) else 0,2)}<br>"+
                                f"left weight: {previous_weight}<extra></extra>",
                marker=dict(
                    symbol=symbol_dict[status[node]],
                    size=100*self.sigmoid(scores[str(node)]),
                    line_color=None,
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
            width=1500,  # 図の幅
            height=700,  # 図の高さ
            margin=dict(l=10, r=10, t=10, b=10),  # 余白の調整
            # title=patient,
            xaxis=dict(showgrid=False, zeroline=False,showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False,showticklabels=False),
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
        if save:
            fig.write_image(self.data_dir_dict["trial_dir_path"]+f"/{trial_name}_{patient}_{str(nrow).zfill(2)}.pdf")
        return fig.data
        # ある行のnodeと重みの状態が与えられたときに，色付きノードを出力する
        pass

    def sigmoid(self,x):
        k=10
        return 1/(1+np.exp(-k*(x-0.5)))

    def bar(self,trial_name="20250120DevBasicCheck",strage="NASK",patient="A",show_features=False,show=False,save=False):
        self.data_dir_dict=self.get_database_dir(trial_name=trial_name,strage=strage)
        self.graph_dicts=self.load_picklelog(self.data_dir_dict["trial_dir_path"]+"/graph_dicts.pickle")
        self.graph_dict=self.graph_dicts[patient]
        csv_path=self.data_dir_dict["trial_dir_path"]+f"/data_{patient}_eval.csv"
        self.data=pd.read_csv(csv_path,header=0)

        # グラフの情報を取り出しておく
        nodes = self.graph_dict["G"].nodes()
        pos = self.graph_dict["pos_dict"] # key: ノード番号, value: [x,y]
        weights = nx.get_edge_attributes(self.graph_dict["G"], 'weight') # key:(node_from,node_to), value: weight
        status = {n:self.graph_dict["node_dict"][n]["status"] for n in nodes}
        # scores = {n:self.graph_dict["node_dict"][n]["score"] for n in nodes}# key: ノード番号, value: 特徴量
        descriptions = {n:self.graph_dict["node_dict"][n]["description_en"] for n in nodes}
        
        fig=go.Figure()
        args=["default"]+[descriptions[k] for k in nodes if ((k<50000000) and (k>=40000000))]
        scores=self.data.loc[:,"10000000"].values
        print(args)
        print(scores)
        # raise NotImplementedError
        trace=go.Bar(x=args,y=scores)
        fig.add_trace(trace=trace)
        fig.write_image(self.data_dir_dict["trial_dir_path"]+f"/{trial_name}_bar.pdf")
        fig.show()

    def bar_per_branch(self,trial_name="20250129DevBasicCheck",strage="NASK",patient="A",show_features=False,show=False,save=False):
        
        plt.rcParams["figure.figsize"] = (8/2.54,12/2.54)
        self.data_dir_dict=self.get_database_dir(trial_name=trial_name,strage=strage)
        self.graph_dicts=self.load_picklelog(self.data_dir_dict["trial_dir_path"]+"/graph_dicts.pickle")
        self.graph_dict=self.graph_dicts[patient]
        csv_path=self.data_dir_dict["trial_dir_path"]+f"/data_{patient}_eval.csv"
        self.data=pd.read_csv(csv_path,header=0)

        # グラフの情報を取り出しておく
        nodes = self.graph_dict["G"].nodes()
        pos = self.graph_dict["pos_dict"] # key: ノード番号, value: [x,y]
        weights = nx.get_edge_attributes(self.graph_dict["G"], 'weight') # key:(node_from,node_to), value: weight
        status = {n:self.graph_dict["node_dict"][n]["status"] for n in nodes}
        # scores = {n:self.graph_dict["node_dict"][n]["score"] for n in nodes}# key: ノード番号, value: 特徴量
        descriptions = {n:self.graph_dict["node_dict"][n]["description_en"] for n in nodes}

        pprint(self.graph_dict)
        # raise NotImplementedError
        # 1000 ~ 3000まで，nodeをひとつずつ参照．下位に繋がってるノードを列記
        # 下位ノードと本人ノードの棒グラフを列記
        for i,row in self.data.iterrows():
            for node in self.graph_dict["node_dict"].keys():
                if int(node)>40000000:
                    continue
                node_code_to_list=self.graph_dict["node_dict"][node]["node_code_to"]
                node_code_to_description=[self.graph_dict["node_dict"][n]["description_en"] for n in node_code_to_list]+[self.graph_dict["node_dict"][node]["description_en"]]
                risk_values=[]
                for n in node_code_to_list+[node]:
                    if str(n) in ["40000000","40000001"]:
                        r=eval(row[str(n)])[1]
                    else:
                        r=row[str(n)]
                    risk_values.append(r)
                    
                colors=["blue" for n in node_code_to_list]+["red"]
                print(node)
                print(node_code_to_description)
                print(risk_values)
                plt.bar(node_code_to_description,risk_values,color=colors)
                plt.ylim([0,1.1])
                plt.xticks(rotation=90)
                plt.savefig(self.data_dir_dict["trial_dir_path"]+f"/bar_{node}_{str(i).zfill(2)}.pdf")
                plt.close()
                # raise NotImplementedError


        

    def main(self):
        for i in range(15):
            self.visualize_graph(nrow=i,save=True)
        pass

if __name__=="__main__":
    cls=Visualizer()
    # cls.main()
    # cls.bar(show=True)
    cls.bar_per_branch()