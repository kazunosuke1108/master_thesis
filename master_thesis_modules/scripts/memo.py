import pandas as pd
import numpy as np
np.random.seed(1)

results={
    "A":0.4,
    "B":0.6,
    "C":0.3,
}
print(list(results.keys())[np.argmax(list(results.values()))])