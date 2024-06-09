import time
import numpy as np
import pandas as pd
import os;os.sys.path.append(os.path.expanduser('~') + "/cmem/codes/")
import os;os.sys.path.append(os.path.expanduser('~') + "/cmem/")
import os;os.sys.path.append("/homes/80/kang/cmem/")
import os;os.sys.path.append("/homes/80/kang/cmem/codes/")
# from trainPred import *
from utils import get_git_hash
from utils import check_GPU_memory
from tqdm import tqdm 
from model import *
import pandas as pd
from sklearn.metrics import r2_score

BIN_SIZE = 26
# BIN_SIZE = 2
# TRAIN_DAYS = 2
# TRAIN_DAYS = 5
# TRAIN_DAYS = 10
TRAIN_DAYS = 50

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
    dates = config['dates']
    
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
    test_dates = dates[-total_test_days:]
    config['total_test_days']=total_test_days
    for index in range(total_test_days):     
        config['test_date'] = test_dates[index]
        # try:
        #     r2result,oneday_df = train_and_pred(index,df,config)     
        #     r2results.append(r2result)
        #     oneday_dfs.append(oneday_df)
        # except Exception as e:
        #     print(f"An error occurred: {e}")
        
        r2result,oneday_df = train_and_pred(index,df,config)
        if trainType == 'universal':
            oneday_df_stock_symbol = np.repeat(stock_symbol, BIN_SIZE)
        elif trainType == 'single':
            oneday_df_stock_symbol = np.repeat(stock_symbol, 1)
        else: raise NotImplementedError
        
        oneday_df['stock_symbol'] = oneday_df_stock_symbol[0] if trainType == 'single' else oneday_df_stock_symbol
        oneday_df.drop(columns=['stock_index'],inplace=True)
        r2results.append(r2result)
        if len(r2result) != num:
            print("len(r2result) != num in get_results line 92")
            raise NotImplementedError
        oneday_dfs.append(oneday_df)
        print(pd.DataFrame(np.array(r2results)[0],columns=['date','stock','r2']))
        print(pd.DataFrame(np.array(r2results)[0],columns=['date','stock','r2']).mean())
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
    
    
    def get_r2df_from_results(r2results, stock_symbol, config):
        r2arr = np.array(r2results).reshape(-1, 3)
        df1 = pd.DataFrame(r2arr)
        df1.columns = ['test_date', 'stock_symbol', 'r2']
        df1['stock_symbol']=np.tile(stock_symbol,config['total_test_days'])
        # df1['stock_symbol']=np.tile(stock_symbol,config['num'])
        # assert np.unique(df1['stock_index']).shape == (len(path060000Files),)
        df2 = df1.pivot(index="test_date", columns="stock_symbol", values="r2")
        # TODO not valid for universal
        # breakpoint()
        # [print(date,itm) for date,itm in df1.groupby('test_date') if (itm.shape[0]!=469)]
        for date,itm in df1.groupby('test_date'):
            if (itm.shape[0]!=469):
                print(date,itm)
                raise NotImplementedError
        return df2
    # check_GPU_memory()
    df2 = get_r2df_from_results(r2results, stock_symbol,config)
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
    
    
    


def train_and_pred(index,df,config):
    
    num,regulator,trainType,task_id = config["num"],config["regulator"],config['trainType'],config['task_id']

    total_test_days, bin_size, train_size, test_size, x_list, y_list, original_space = param_define(df,num) 
    # here already wrong, need to check the codes
    
    bin_size = BIN_SIZE
    

    def get_X_train_y_train_X_test_original_images(df,num):
        train_start_index = (index * bin_size) * num
        train_end_index = (index * bin_size + train_size) * num
        test_start_index = train_end_index
        test_end_index = train_end_index + test_size * num

        def get_trainData(df):
            x_train = df.loc[:, x_list].iloc[train_start_index: train_end_index, :]
            y_train = df.loc[:, y_list].iloc[train_start_index: train_end_index, :]
            # x_train = df.iloc[train_start_index : train_end_index, x_list]
            # y_train = df.loc[train_start_index : train_end_index, y_list]
            return x_train, y_train

        def get_testData(df):
            x_test = df.loc[:, x_list].iloc[train_end_index:  test_end_index, :]
            y_test = df.loc[:, y_list].iloc[train_end_index: test_end_index, :]
            return x_test, y_test

        X_train, y_train = get_trainData(df)
        X_test, y_test = get_testData(df)
        original_images = df.loc[:, original_space].iloc[train_end_index:test_end_index, :]
        return X_train,y_train,X_test,y_test,original_images,train_end_index
    X_train,y_train,X_test,y_test,original_images,train_end_index=get_X_train_y_train_X_test_original_images(df,num)

    # already wrong at this stage, as the X_test is empty


    # breakpoint()
    # print(regulator)
    if regulator == "Inception":
        y_pred = model_nn(X_train, y_train, X_test, y_test, config)
        # y_pred = regularity_nn(X_train, y_train, X_test,y_test, regulator,num)
    else:
        y_pred = regularity_ols(X_train, y_train, X_test, config)
    # print(regulator+"_finished")
    min_limit, max_limit = y_train.min(), y_train.max()
    broadcast = lambda x: np.full(y_pred.shape[0], x.to_numpy())
    min_limit, max_limit = map(broadcast, [min_limit, max_limit])
    y_pred_clipped = np.clip(y_pred, min_limit, max_limit)
    if any('log' in x for x in x_list):
        y_pred_clipped = np.exp(y_pred)
        # y_pred_clipped = np.exp(y_pred_clipped) # TODO caution 
    # test_date = df.date.iloc[train_end_index]
    # df.date.iloc[train_end_index-1]
    # df.date.iloc[train_end_index+1]
    # # TODO determine the test date
    # df.date.iloc[train_end_index-1:train_end_index+10000]
    test_date = config['test_date']
    '''prob in the y_pred shapes'''

    # r2 = r2_score(y_test, y_pred_clipped)
    y_pred_clipped = pd.DataFrame(y_pred_clipped)
    y_pred_clipped.columns = ['pred']
    original_images.reset_index(inplace=True, drop=True)
    original_images.columns = ['true']

    original_images['date'] = test_date
    # stock_index = np.tile(tile_array, 26) # original [[bin,483] 26]
    stock_index = np.arange(num).repeat(bin_size)
    original_images['stock_index'] = stock_index
    oneday_df = pd.concat([original_images, y_pred_clipped], axis=1)[['date','stock_index','true','pred']]
    
    lst = []
    g = oneday_df.groupby(stock_index)
    for stock, item in g:
        try:
            r2value = r2_score(item['true'], item['pred'])
            lst.append([test_date, stock, r2value])
        except:
            print(f"trainPred Error {stock} {item.head()}")
    test_df = pd.DataFrame(lst,columns=["test_date", "stock", "r2value"])
    test_df = test_df.pivot(index='test_date',columns='stock')
    return lst,oneday_df



def param_define(df,num):
    # bin_size = 26
    bin_size = BIN_SIZE
    train_days = TRAIN_DAYS
    # bin_size = 2
    # train_days = 20
    # train_days = 50
    # train_days = 50
    # train_days = 20
    # train_days = 10
    train_size = train_days * bin_size
    test_size = 1 * bin_size
    # breakpoint()
    if type(df) == list:
        total_test_days = df[0].shape[0]//bin_size - train_days
    else:
        total_test_days = (df.shape[0]//num - train_size)//bin_size # reached
    # index_max  = int((df.shape[0] -(train_size + test_size))/bin_size)
    # index = 0 for index in range(0, index_max+1)
    # index = 0 for index in range(0, index_max+0) # not sure

    # x_list = ['x', 'eta*seas', 'eta', 'seas', 'mu']
    # y_list = ['turnover']

    # x_list = ['eta','seas','mu']
    # y_list = ['turnover']

    our_log_features = ['log_ntn', 'log_volBuyNotional', 'log_volSellNotional', 'log_nrTrades', 'log_ntr',
                        'log_volBuyNrTrades_lit', 'log_volSellNrTrades_lit', 'log_volBuyQty', 'log_volSellQty',
                        'log_daily_ntn', 'log_daily_volBuyNotional', 'log_daily_volSellNotional', 'log_daily_nrTrades',
                        'log_daily_ntr', 'log_daily_volBuyNrTrades_lit', 'log_daily_volSellNrTrades_lit',
                        'log_daily_volBuyQty', 'log_daily_volSellQty', 'log_daily_qty', 'log_intraday_ntn',
                        'log_intraday_volBuyNotional', 'log_intraday_volSellNotional', 'log_intraday_nrTrades',
                        'log_intraday_ntr', 'log_intraday_volBuyNrTrades_lit', 'log_intraday_volSellNrTrades_lit',
                        'log_intraday_volBuyQty', 'log_intraday_volSellQty', 'log_intraday_qty', 'log_ntn_2',
                        'log_volBuyNotional_2', 'log_volSellNotional_2', 'log_nrTrades_2', 'log_ntr_2',
                        'log_volBuyNrTrades_lit_2', 'log_volSellNrTrades_lit_2', 'log_volBuyQty_2', 'log_volSellQty_2',
                        'log_ntn_8', 'log_volBuyNotional_8', 'log_volSellNotional_8', 'log_nrTrades_8', 'log_ntr_8',
                        'log_volBuyNrTrades_lit_8', 'log_volSellNrTrades_lit_8', 'log_volBuyQty_8', 'log_volSellQty_8']
    x_list = ['log_x', 'log_eta*seas', 'log_eta', 'log_seas', 'log_mu']
    x_list = x_list + our_log_features
    y_list = ['log_turnover']
    # x_list = ['log_eta', 'log_seas', 'log_mu']
    # y_list = ['log_turnover']
    # x_list = ['x']
    # y_list = ['turnover']

    # x_list = ['log_x']
    # y_list = ['log_turnover']
    original_space = ['turnover']
    return total_test_days, bin_size, train_size, test_size, x_list, y_list, original_space



