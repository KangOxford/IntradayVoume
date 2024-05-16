import time
import numpy as np
import pandas as pd
import os;os.sys.path.append(os.path.expanduser('~') + "/cmem/codes/")
import os;os.sys.path.append(os.path.expanduser('~') + "/cmem/")
import os;os.sys.path.append("/homes/80/kang/cmem/")
import os;os.sys.path.append("/homes/80/kang/cmem/codes/")
from trainPred import *
from utils import get_git_hash
from utils import check_GPU_memory
from tqdm import tqdm 


import ray
# @ray.remote(num_cpus=64)
@ray.remote
def get_r2df_ray(num,regulator,df):
    return get_r2df(num,regulator,df)

def get_r2df(num,regulator,df):
    
    print("universal data loaded")
    total_test_days, bin_size, train_size, test_size, x_list, y_list, original_space = param_define(df,num)
    print(f"num of stocks {num}, total test days {total_test_days}")
    # num_processes = multiprocessing.cpu_count()  # on local machine
    # num_processes = multiprocessing.cpu_count() -10 # on flair-node-03
    # num_processes = 1 # Number of available CPU cores
    
    config = {
        "num":num,
        "regulator":regulator,
        "bin_size": bin_size,
        "train_days":train_size//bin_size,
        # "tile_array":np.arange(num),
        "short_hash":get_git_hash()
    }
    
    # # suquentially
    # start = time.time()
    # # with multiprocessing.Pool(processes=num_processes) as pool:
    # r2results = [];oneday_dfs=[]
    # print("total_test_days",total_test_days)
    # # index=0
    # for index in range(total_test_days):     
    #     try:
    #         r2result,oneday_df = train_and_pred(index,df,config)     
    #         r2results.append(r2result)
    #         oneday_dfs.append(oneday_df)
    #     except:pass
    #     # print(r2results)
    #     # print(oneday_dfs)
    # end = time.time()
    
    # in parallel
    start = time.time()
    ray.shutdown()
    ray.init(num_cpus=64,object_store_memory=40*1e9)
    @ray.remote
    def train_and_pred_ray(index,df,config):
        print(f"+++ date index: {index}")
        return train_and_pred(index,df,config)
    ids=[train_and_pred_ray.remote(index,df,config) for index in tqdm(3)]
    # ids=[train_and_pred_ray.remote(index,df,config) for index in tqdm(range(total_test_days))]
    results = [ray.get(id_) for id_ in tqdm(ids)]
    r2results,oneday_dfs=zip(*results)
    end = time.time()
    print(f"get r2results,oneday_dfs time taken {end-start}")
    breakpoint()
    
    
    def get_r2df_from_results(r2results):
        r2arr = np.array(r2results).reshape(-1, 3)
        df1 = pd.DataFrame(r2arr)
        df1.columns = ['test_date', 'stock_index', 'r2']
        # assert np.unique(df1['stock_index']).shape == (len(path060000Files),)
        df2 = df1.pivot(index="test_date", columns="stock_index", values="r2")
        return df2
    # check_GPU_memory()
    df2 = get_r2df_from_results(r2results)
    # check_GPU_memory()
    df22 =pd.concat(oneday_dfs,axis=0)

    print(df2)
    print(df22)
    print(f"time {(end-start)/60}")
    return df2, df22



def get_git_hash():
    import subprocess
    try:
        # Execute the command to get the latest commit hash
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')

        # Slice the first 4 characters
        short_hash = git_hash[:4]

        return short_hash

    except subprocess.CalledProcessError:
        print("An error occurred while fetching the Git hash.")
        return None
    
    
    