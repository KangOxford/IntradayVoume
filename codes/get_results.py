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
# @ray.remote(num_cpus=32)
# @ray.remote
@ray.remote(num_gpus=1)
def get_r2df_ray(config,df):
# def get_r2df_ray(num,regulator,trainType,df):
    return get_r2df(config,df)
    # return get_r2df(num,regulator,trainType,df)

@ray.remote
def train_and_pred_ray(index,df,config):
    print(f">>> date index BEGIN: {index}")
    result = train_and_pred(index,df,config)
    print(f"+++ date index COMPLETE: {index}")
    return result

def get_r2df(config,df):
# def get_r2df(num,regulator,trainType,df):
    num=config['num']
    regulator=config['regulator']
    trainType=config['trainType']
    task_id = config['task_id']
    stock_symbol = config['stock_symbol']
    
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
        'trainType':trainType,
        # "tile_array":np.arange(num),
        "short_hash":get_git_hash(),
        'task_id':task_id
    }
    
    # suquentially
    start = time.time()
    # with multiprocessing.Pool(processes=num_processes) as pool:
    r2results = [];oneday_dfs=[]
    print("total_test_days",total_test_days)
    # index=0
    for index in range(total_test_days):     
        # try:
        #     r2result,oneday_df = train_and_pred(index,df,config)     
        #     r2results.append(r2result)
        #     oneday_dfs.append(oneday_df)
        # except Exception as e:
        #     print(f"An error occurred: {e}")
        
        r2result,oneday_df = train_and_pred(index,df,config)
        oneday_df['stock_symbol'] = stock_symbol
        oneday_df.drop(columns=['stock_index'],inplace=True)
        r2results.append(r2result)
        # if len(r2result) != 469:
        #     print()
        #     breakpoint()
        oneday_dfs.append(oneday_df)
        # print(r2results)
        # print(oneday_dfs)
    end = time.time()
    
    # # in parallel
    # start = time.time()
    # # ids=[train_and_pred_ray.remote(index,df,config) for index in tqdm(3)]
    # ids=[train_and_pred_ray.remote(index,df,config) for index in tqdm(range(total_test_days))]
    # results = [ray.get(id_) for id_ in tqdm(ids)]
    # r2results  = [result[0] for result in results]
    # oneday_dfs = [result[1] for result in results]
    # # r2results,oneday_dfs=zip(*results)
    # end = time.time()
    # print(f"get r2results,oneday_dfs time taken {end-start}")
    # # breakpoint()
    
    
    def get_r2df_from_results(r2results, stock_symbol):
        r2arr = np.array(r2results).reshape(-1, 3)
        df1 = pd.DataFrame(r2arr)
        df1.columns = ['test_date', 'stock_symbol', 'r2']
        df1['stock_symbol']=stock_symbol
        # assert np.unique(df1['stock_index']).shape == (len(path060000Files),)
        df2 = df1.pivot(index="test_date", columns="stock_symbol", values="r2")
        # TODO not valid for universal
        # for date,itm in df1.groupby('test_date'):
        #     pass
        #     if (itm.shape[0]!=469):
        #         print(date)
        #         print(itm)
        #         break
        return df2
    # check_GPU_memory()
    df2 = get_r2df_from_results(r2results, stock_symbol)
    # check_GPU_memory()
    df22 =pd.concat(oneday_dfs,axis=0)

    # print(df2)
    # print(df22)
    # print(f"time {(end-start)/60}")
    def _save_data(df2, df22):
        from datetime import datetime
        # Get current date and time
        current_time = datetime.now()
        # Format date and time to be used in the file name
        date_str = current_time.strftime("%Y-%m-%d")
        time_str = current_time.strftime("%H")
        # Combine date and time to form the file name
        # Get the short Git hash
        short_hash = config["short_hash"]
        idnetificator = f"_{task_id}_{short_hash}"
        # In subsequent iterations, append without the header
        df2.to_csv('/homes/80/kang/cmem/'+'data_summary_'+trainType+regulator+idnetificator+'.csv', mode='a', header=False, index=True)
        # test_df.to_csv('/homes/80/kang/cmem/'+'data_summary.csv', mode='a', header=False, index=True)
        # print(index,test_date,test_df.r2value.mean())
        df22.to_csv('/homes/80/kang/cmem/'+'data_allValues_'+trainType+regulator+idnetificator+'.csv', mode='a', header=False, index=False)
        # oneday_df.to_csv('/homes/80/kang/cmem/'+'data_all_values.csv', mode='a', header=False, index=False)
        print(f'results would be saved into {trainType+regulator+idnetificator}')
    _save_data(df2, df22)
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
    
    
    