from tqdm import tqdm
import numpy as np
import pandas as pd
from os import listdir;
from os.path import isfile, join;
from sklearn.metrics import r2_score


import numpy as np
import pandas as pd
from tqdm import tqdm
import os;os.sys.path.append(os.path.expanduser('~') + "/cmem/codes/")
import os;os.sys.path.append(os.path.expanduser('~') + "/cmem/")
import os;os.sys.path.append("/homes/80/kang/cmem/")
import os;os.sys.path.append("/homes/80/kang/cmem/codes/")
from utils import *
from model import *
from params import *
from trainPred import *
from get_results import get_r2df
import multiprocessing
import time



path060000Files = readFromPath(path060000)
print(len(path060000Files))

path0702Files = readFromPath(path0702)
print(len(path0702Files))
path0702Files_filtered = list(filter(lambda x: x.endswith('.pkl'), path0702Files))
print(len(path0702Files_filtered))

def getSingleDfs(trainType):
    
    if trainType=="universal":
        df = pd.read_pickle(path0700+"universal.pkl")
        dfs=[df]
        num_of_stacked_stocks = 483
        return dfs, num_of_stacked_stocks
    
    elif trainType=="single":
        dfs=[]
        for i in range(len(path0702Files_filtered)):
            print(f">>> {i}")
            df=pd.read_pickle(path0702+path0702Files_filtered[i])
            dfs.append(df)
        num_of_stacked_stocks = 1
        return dfs, num_of_stacked_stocks
    else: raise NotImplementedError
        
    # return pd.read_pickle(path0701+"one_file.pkl")
    # return dfs

def print_mean(df3):
    print(f">>>> stock mean: \n",df3.mean(axis=0))  # stock
    print(f">>>> date mean: \n",df3.mean(axis=1))   # date
    print(f">>>> aggregate mean: \n",df3.mean(axis=1).mean())

if __name__=="__main__":
    
    regulator = "Lasso"
    # regulator = "XGB"

    # regulator = "cnnLstm"
    # regulator = "CNN"
    # regulator = "Inception"
    # regulator = "OLS"
    # regulator = "Ridge"
    # regulator = "CMEM"
    
    trainType = "universal"
    
    dfs,num_of_stacked_stocks = getSingleDfs(trainType)
    # for idex,df in enumerate(dfs):
    df3s=[];df33s=[]
    for idx, df in tqdm(enumerate(dfs), desc="get_r2df", total=len(dfs)):
        df3,df33 = get_r2df(num=num_of_stacked_stocks,regulator=regulator,df=df)
        total_r2 = df3.mean(axis=1).mean()
        print('total r2: ',df3.mean(axis=1).mean()) # all mean
        df3s.append(df3)
        df33s.append(df33)
        # name = path0702Files_filtered[idx][:-4]
        # df3.to_csv(path00 + "0802_r2df_single_day_"+str(1)+"_"+regulator+"_"+str(total_r2)[:6]+name+".csv", mode = 'w')
        # df33.to_csv(path00 + "0802_r2df_single_day_"+str(1)+"_"+regulator+"_"+str(total_r2)[:6]+'_values_'+name+".csv", mode = 'w')
    
    # num_stock=len(dfs)
    df3_ = pd.concat(df3s,axis=1)
    # df3_.columns=np.arange(num_stock)
    # pd.set_option('display.max_rows', None) 
    print(df3_.mean(axis=0))
    
    
    '''
    df3_.mean(axis=0)[df3_.mean(axis=0)>df3_.mean(axis=0).quantile(q=0.25)].mean()
    0.3993333265521051 the result is same to the previous research
    means that there is no error in the codes
    '''
    
    print(df3_.mean(axis=1).mean())
    df33_ = pd.concat(df33s,axis=0)
    print(df33_.mean(axis=1).mean())
    # r2_score(df33_.true,df33_.pred)
    df33_.stock_index=np.tile(np.arange(483).repeat(26),61)
    df33_.reset_index(drop=True)
    df3_.to_csv(path00 + "0802_r2df_"+trainType+"_day_"+str(num_of_stacked_stocks)+"_"+regulator+"_"+str(total_r2)[:6]+".csv", mode = 'w')
    df33_.to_csv(path00 + "0802_r2df_"+trainType+"_day_"+str(num_of_stacked_stocks)+"_"+regulator+"_"+str(total_r2)[:6]+'_values_'+".csv", mode = 'w')

# # %%
# import  pandas as pd
# df = pd.read_csv("/homes/80/kang/cmem/0802_r2df_universal_day_483_Lasso_0.4285_values_.csv")
# # %%
# lst=[]
# g=df.groupby(['date','stock_index'])
# for idx,itm in g:
#     r2=r2_score(itm['true'],itm['pred'])
#     lst.append([itm.date.iloc[0],itm.stock_index.iloc[0],r2])
# # %%
# df1=pd.DataFrame(lst,columns=['date','stock','r2'])
# df2=df1.pivot(index='date',columns='stock')
# # %%
# df2.mean(axis=0).mean()