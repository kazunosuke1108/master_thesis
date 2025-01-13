import pandas as pd
import numpy as np

df=pd.DataFrame(np.random.random((3,6)))
print(df)
print(df.loc[1:6,0])