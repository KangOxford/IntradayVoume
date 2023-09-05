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

def getSingleDfs():
    # df = pd.read_csv(path0700+"universal.csv",index_col=0)
    dfs=[]
    for i in range(len(path0702Files_filtered)):
        print(f">>> {i}")
        df=pd.read_pickle(path0702+path0702Files_filtered[i])
        dfs.append(df)
    # return pd.read_pickle(path0701+"one_file.pkl")
    return dfs

def print_mean(df3):
    print(f">>>> stock mean: \n",df3.mean(axis=0))  # stock
    print(f">>>> date mean: \n",df3.mean(axis=1))   # date
    print(f">>>> aggregate mean: \n",df3.mean(axis=1).mean())

if __name__=="__main__":
    # regulator = "Lasso"
    # regulator = "XGB"

    # regulator = "cnnLstm"
    # regulator = "CNN"
    regulator = "OLS"
    # regulator = "Ridge"
    # regulator = "None"
    
    dfs = getSingleDfs()
    # for idex,df in enumerate(dfs):
    df3s=[];df33s=[]
    for idx, df in tqdm(enumerate(dfs), desc="get_r2df", total=len(dfs)):
        name = path0702Files_filtered[idx][:-4]
        df3,df33 = get_r2df(num=1,regulator=regulator,df=df)
        total_r2 = df3.mean(axis=1).mean()
        print('total r2: ',df3.mean(axis=1).mean()) # all mean
        df3s.append(df3)
        df33s.append(df33)
        # df3.to_csv(path00 + "0802_r2df_single_day_"+str(1)+"_"+regulator+"_"+str(total_r2)[:6]+name+".csv", mode = 'w')
        # df33.to_csv(path00 + "0802_r2df_single_day_"+str(1)+"_"+regulator+"_"+str(total_r2)[:6]+'_values_'+name+".csv", mode = 'w')
        
    df3_ = pd.concat(df3s,axis=1)
    df3_.mean(axis=1).mean()
    df33_ = pd.concat(df33s,axis=0)
    df33_.mean(axis=1).mean()
    r2_score(df33_.true,df33_.pred)