from icecream import ic
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from network.graph_manager import GraphManager
from fuzzy.fuzzy_reasoning import FuzzyReasoning
from pseudo_data.pseudo_data_generator import PseudoDataGenerator

class Master(GraphManager,FuzzyReasoning,PseudoDataGenerator):
    def __init__(self):
        super().__init__()
        # pseudo_dataが出来ていることを確認
        ic(self.data)

        # ネットワークが正しく作成できていることを確認
        self.visualize()

        

if __name__=="__main__":
    cls=Master()