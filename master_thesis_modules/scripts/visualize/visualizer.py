import os
import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append(os.path.expanduser("~")+"/kazu_ws/master_thesis/master_thesis_modules")

from icecream import ic
from glob import glob

import numpy as np
import pandas as pd
import plotly.subplots as sp
import plotly.graph_objects as go

from scripts.master import Master

class Visualizer(Master):
    def __init__(self):
        super().__init__()
        self.visualize_dir_path=sorted(glob(self.get_database_dir(strage="NASK")["database_dir_path"]+"/*"))[-1]
        self.data_paths=sorted(glob(self.visualize_dir_path+"/*"))
        
        pass

    def draw_timeseries(self,data,name):
        try:
            label_name=self.node_dict[int(name)]["description"]
        except ValueError:
            label_name=name
        trace=go.Scatter(
            x=data["timestamp"],
            y=data[name],
            xaxis="x",
            yaxis="y",
            mode="markers+lines",
            name=str(name)+":"+label_name,
        )
        return trace

    def draw_3d(self,plot_data,data,name):
        try:
            label_name=self.node_dict[int(name)]["description"]
        except ValueError:
            label_name=name
        trace=go.Scatter3d(
            x=data["x"],
            y=data["y"],
            z=data["timestamp"],
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
                xaxis=dict(title='X',tickfont=dict(family="Times New Roman",size=18)),
                yaxis=dict(title='Y',tickfont=dict(family="Times New Roman",size=18)),
                zaxis=dict(title='UnixTime [s]',tickfont=dict(family="Times New Roman",size=18)),
            )
        )
        return fig

    def draw_positions(self):
        csv_paths=[path for path in self.data_paths if (("position" in os.path.basename(path)) and (".csv" in os.path.basename(path)))]
        plot_data=[]

        for csv_path in csv_paths:
            name=os.path.basename(csv_path)[:-4].split("_")[-1]
            data=pd.read_csv(csv_path,header=0)
            plot_data=self.draw_3d(plot_data,data,name)
        fig=go.Figure(data=plot_data)
        fig=self.customize_camera(fig)
        fig.update_layout(
                scene_aspectmode='manual',
                scene_aspectratio=dict(x=5, y=5, z=3),
                
            )
        fig.to_html(self.visualize_dir_path+"/positions.html")
        fig.show()
        pass

    def draw_features(self):
        csv_paths=[path for path in self.data_paths if (("feature" in os.path.basename(path)) and (".csv" in os.path.basename(path)))]
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
                trace=self.draw_timeseries(data=data,name=node)
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
            fig.update_xaxes(title=dict(text="Time [s]",font=dict(family="Times New Roman",size=18)),row=len(categories),col=1,)
            for i in range(len(categories)):
                fig.update_yaxes(
                    title=dict(text=f"Features {categories[i]}",font=dict(family="Times New Roman",size=18)),
                    range=[-0.1,1.1],
                    row=i+1,
                    col=1,
                    )
            fig.write_html(self.visualize_dir_path+f"/features_{name}.html")
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
                        shared_xaxes=True,  # x軸を共有
                        vertical_spacing=0.1)  # グラフ間の間隔
            
            for node in nodes:
                row_no=categories.index(str(node)[:3])
                trace=self.draw_timeseries(data=data,name=node)
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
            fig.update_xaxes(title=dict(text="Time [s]",font=dict(family="Times New Roman",size=18)),row=len(categories),col=1,)
            for i in range(len(categories)):
                fig.update_yaxes(
                    title=dict(text=f"Weights {categories[i]}",font=dict(family="Times New Roman",size=18)),
                    range=[-0.1,1.1],
                    row=i+1,
                    col=1,
                    )
            fig.write_html(self.visualize_dir_path+f"/weights_{name}.html")
            fig.show()

        pass


    def main(self):
        pass

if __name__=="__main__":
    cls=Visualizer()
    cls.draw_positions()
    cls.draw_features()
    cls.draw_weight()
    pass