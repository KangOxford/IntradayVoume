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
import multiprocessing
import time

path0600_1Files = readFromPath(path0600_1)
print(len(path0600_1Files))

def param_define():
    bin_size = 26
    train_size = 10 * 26
    test_size = 1 * 26
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
    return bin_size, train_size, test_size, x_list, y_list, original_space


def process_df(index,regulator):
    print("+ ",index)
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


    # breakpoint()
    print(regulator)
    y_pred = regularity_ols(X_train, y_train, X_test, regulator,num)
    print(regulator+"_finished")
    min_limit, max_limit = y_train.min(), y_train.max()
    broadcast = lambda x: np.full(y_pred.shape[0], x.to_numpy())
    min_limit, max_limit = map(broadcast, [min_limit, max_limit])
    y_pred_clipped = np.clip(y_pred, min_limit, max_limit)
    if any('log' in x for x in x_list):
        y_pred_clipped = np.exp(y_pred_clipped)
    test_date = df.date[train_end_index]
    '''prob in the y_pred shapes'''

    # r2 = r2_score(y_test, y_pred_clipped)
    y_pred_clipped = pd.DataFrame(y_pred_clipped)
    y_pred_clipped.columns = ['pred']
    original_images.reset_index(inplace=True, drop=True)
    original_images.columns = ['true']

    original_images['date'] = test_date
    stock_index = np.tile(np.arange(num), 26)
    original_images['stock_index'] = stock_index
    oneday_df = pd.concat([original_images, y_pred_clipped], axis=1)
    lst = []
    g = oneday_df.groupby(stock_index)
    for stock, item in g:
        pass
        r2value = r2_score(item['true'], item['pred'])
        lst.append([test_date, stock, r2value])
    test_df = pd.DataFrame(lst,columns=["test_date", "stock", "r2value"])
    print(test_df)
    print(index,test_date,test_df.r2value.mean())
    return lst

def getUniversalDf():
    # df = pd.read_csv(path0700+"universal.csv",index_col=0)
    return pd.read_pickle(path0700+"universal.pkl")

def getClusterDf():
    return 0

def get_universal(num_of_stocks,regulator):
    global num, df, bin_size, train_size, test_size, x_list, y_list, original_space, total_test_days, num_processes
    num = num_of_stocks
    df = getUniversalDf()
    # df = getClusterDf()
    print("universal data loaded")
    # breakpoint()
    bin_size, train_size, test_size, x_list, y_list, original_space = param_define()

    total_test_days = (df.shape[0]//num - train_size)//bin_size # reached
    # num_processes = multiprocessing.cpu_count()  # on local machine
    # num_processes = multiprocessing.cpu_count() -10 # on flair-node-03
    num_processes = 1 # Number of available CPU cores
    
    


    start = time.time()
    # with multiprocessing.Pool(processes=num_processes) as pool:
    results = []
    for i in range(total_test_days):
        results.append(process_df(i,regulator))
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
    regulator = "XGB"

    # regulator = "cnnLstm"
    # regulator = "OLS"
    # regulator = "Ridge"
    # regulator = "None"
    
    df3 = get_universal(num_of_stocks=len(path0600_1Files),regulator=regulator)




    total_r2 = df3.mean(axis=1).mean()
    print('total r2: ',df3.mean(axis=1).mean()) # all mean
    df3.to_csv(path00 + "08_r2df_universal_day_"+str(num)+"_"+regulator+"_"+str(total_r2)[:6]+".csv", mode = 'w')
    
    