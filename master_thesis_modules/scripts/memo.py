import pandas as pd
import numpy as np
np.random.seed(1)

data=pd.DataFrame(np.random.random((5,6)),columns=[0,1,2,3,4,5])
# data.loc[1:2,1]=np.nan
data[3]=0
print(data)
print(data.loc[:1,:])
# print(len(data.index))

# entropy_window=3

# for k in data.keys():
#     temp_data=data[k]
#     temp_data=temp_data.dropna()
#     temp_data=temp_data.tail(entropy_window)
#     if temp_data.sum()==0:
#         e=1
#     else:
#         e=(-1/np.log(entropy_window)*temp_data*np.log(temp_data.astype(float))).sum()
#     d=1-e
#     print(k)
#     print(d)
#     # print(temp_data.tail(3))

# n_skip=3
# i=0
# data.loc[i+1:i+n_skip,1]=np.nan
# print(data)