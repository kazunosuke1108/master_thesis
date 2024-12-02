import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class GraphManager():
    def __init__(self):
        # define network
        super().__init__()
        self.G = nx.Graph()
        
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
        self.G.add_nodes_from(self.node_dict.keys())

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

        # edge定義
        for lv in range(5):
            for node_code_from in [n for n in list(self.node_dict.keys()) if str(n)[0]==str(lv)]:
                if node_code_from=="description":
                    continue
                node_codes_to=[k for k in [j for j in list(self.node_dict.keys()) if str(j)[0]==str(lv+1)] if str(node_code_from)[2:] == str(k)[1:3]]
                self.G.add_edges_from([(node_code_from,node_code_to,{"weight":self.weight_dict[node_code_from][node_code_to]}) for node_code_to in node_codes_to])

        # 位置追記
        self.pos={}
        previous_layer=0
        for node_code in self.G.nodes():
            if previous_layer!=int(str(node_code)[0]):
                y=0
            else:
                y-=1
            self.pos[node_code]=(int(str(node_code)[0]),y)
            previous_layer=int(str(node_code)[0])

    def update_score(self,new_score_dict):
        for node_code,score in new_score_dict.items():
            self.node_dict[node_code]["score"]=score

    def update_weight(self,new_weight_dict):
        for node_from in new_weight_dict.keys():
            for node_to in new_weight_dict[node_from].keys():
                self.weight_dict[node_from][node_to]=new_weight_dict[node_from][node_to]
    
    def update_lower_layer_status(self,new_status="active"):
        for node in self.G.nodes():
            if int(str(node)[0])>=4:
                self.node_dict[node]["status"]=new_status
        self.colorize(default=False)
        
    def colorize(self,default=True):
        # 色追記
        self.nodecolor=[]
        if default:
            for node_code in self.G.nodes():
                self.nodecolor.append((1,0,0) if self.node_dict[node_code]["status"]=="active" else "gray")
        else:
            for node_code in self.G.nodes():
                node=self.node_dict[node_code]
                if np.isnan(node["score"]):
                    self.nodecolor.append((1,1,1) if node["status"]=="active" else "gray")
                else:
                    self.nodecolor.append((node["score"],0,0) if node["status"]=="active" else "gray")
            pass
        
    def visualize(self):
        self.colorize()
        weights = nx.get_edge_attributes(self.G, 'weight').values()
        nx.draw(self.G, self.pos, node_color=self.nodecolor, with_labels=True, edge_color = weights, edge_cmap=plt.cm.RdBu_r)
        plt.pause(1)
        plt.close()

    def main(self):
        self.colorize(default=False)
        self.visualize()
        self.update_lower_layer_status()
        self.visualize()

if __name__=="__main__":
    cls=GraphManager()
    cls.main()