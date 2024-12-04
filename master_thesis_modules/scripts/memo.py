from icecream import ic

class A():
    def __init__(self):
        super().__init__()
        self.node_dict={"A":1,"B":2,"C":3}
        self.graph_dict={
            "A":{"G":"","node_dict":self.node_dict.copy(),"weight_dict":"","pos":""},
            "B":{"G":"","node_dict":self.node_dict.copy(),"weight_dict":"","pos":""},
            "C":{"G":"","node_dict":self.node_dict.copy(),"weight_dict":"","pos":""},
        }        
    
    def print_dict(self):
        ic(self.graph_dict)

class B(A):
    def __init__(self):
        super().__init__()
        self.graph_dict["A"]["node_dict"]["A"]=4
        self.print_dict()

cls=B()

