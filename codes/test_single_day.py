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
import multiprocessing
import time

path0600_1Files = readFromPath(path0600_1)
print(len(path0600_1Files))



def process_df(index,df,num,regulator):
    print("+ ",index)
    return train_and_pred(index,df,num,regulator,tile_array=np.arange(num))

def getUniversalDf():
    # df = pd.read_csv(path0700+"universal.csv",index_col=0)
    return pd.read_pickle(path0700+"universal.pkl")


def get_r2df(num,regulator):
    df = getUniversalDf()
    print("universal data loaded")
    total_test_days, bin_size, train_size, test_size, x_list, y_list, original_space = param_define(df,num)

    # num_processes = multiprocessing.cpu_count()  # on local machine
    # num_processes = multiprocessing.cpu_count() -10 # on flair-node-03
    num_processes = 1 # Number of available CPU cores

    start = time.time()
    # with multiprocessing.Pool(processes=num_processes) as pool:
    results = []
    print("total_test_days",total_test_days)
    for index in range(total_test_days):
        print("+ ",index)
        # breakpoint()
        result = train_and_pred(index,df,num,regulator,tile_array=np.arange(num))
        results.append(result)
    end = time.time()

    r2arr = np.array(results).reshape(-1, 3)
    df1 = pd.DataFrame(r2arr)
    df1.columns = ['test_date', 'stock_index', 'r2']
    assert np.unique(df1['stock_index']).shape == (len(path0600_1Files),)
    df2 = df1.pivot(index="test_date", columns="stock_index", values="r2")

    print(f"time {(end-start)/60}")
    return df2
    
    '''
    r2arr = np.array(r2_list)
    df1 = pd.DataFrame(r2arr)
    df1.columns = ['test_date','stock_index','r2']
    df2 = df1.pivot(index="test_date",columns="stock_index",values="r2")
    print(df2)
    df2.mean(axis=0) # stock
    df2.mean(axis=1) # date
    df2.mean(axis=1).mean()
    return df2
    '''

def print_mean(df3):
    print(f">>>> stock mean: \n",df3.mean(axis=0))  # stock
    print(f">>>> date mean: \n",df3.mean(axis=1))   # date
    print(f">>>> aggregate mean: \n",df3.mean(axis=1).mean())

if __name__=="__main__":
    # regulator = "Lasso"
    # regulator = "XGB"

    # regulator = "cnnLstm"
    regulator = "OLS"
    # regulator = "Ridge"
    # regulator = "None"
    
    df3 = get_r2df(num=len(path0600_1Files),regulator=regulator)




    total_r2 = df3.mean(axis=1).mean()
    print('total r2: ',df3.mean(axis=1).mean()) # all mean
    df3.to_csv(path00 + "08_r2df_universal_day_"+str(len(path0600_1Files))+"_"+regulator+"_"+str(total_r2)[:6]+".csv", mode = 'w')
    
    