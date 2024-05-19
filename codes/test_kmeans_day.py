import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from os import listdir;
from os.path import isfile, join;
from sklearn.metrics import r2_score

import os;os.sys.path.append(os.path.expanduser('~') + "/cmem/codes/")
import os;os.sys.path.append(os.path.expanduser('~') + "/cmem/")
import os;os.sys.path.append("/homes/80/kang/cmem/")
import os;os.sys.path.append("/homes/80/kang/cmem/codes/")
from utils import *
from utils import get_git_hash
from model import *
# from params import *
from kmeans import *
from kmeans import get_features,get_labels_byPCA
from universal import *
# from universal import get_df_list
from test_single_stock import getSingleDfs
from trainPred import *
# import multiprocessing
import ray

readFromPath = lambda data_path: sorted([f for f in listdir(data_path) if isfile(join(data_path, f)) and f != '.DS_Store'])

path0702Files = readFromPath(path0702)
print(len(path0702Files))
path0702Files_filtered = list(filter(lambda x: x.endswith('.pkl'), path0702Files))
print(len(path0702Files_filtered))

ONE_STOCK_SHAPE = 111*26 # 2886



def return_lst(list_, date_index,regulator,dfs):
    """Here, process each stock within all groups."""
    # print(f"\n+++@ return_lst() called\n")
    # This function is used to stack the feature matrices of clustered models.
    groupped_dfs = [dfs[i] for i in list_]
    gs=[[df for date, df in list(dflst.groupby("date") )] for dflst in groupped_dfs]
    
    
    dff=[]
    num_stocks = len(list_)
    num_days = dfs[0].shape[0]//BIN_SIZE
    for i in tqdm(range(num_days)):
        for j in range(num_stocks):
            group = gs[j][i].iloc[-BIN_SIZE:,:]    
            dff.append(group)
    df = pd.concat(dff, axis=0)
    df.reset_index(inplace=True, drop=True)
    # df: Vertically concatenate all stocks for each day, with each group having a size of 26 bins.
    
    
    # gs = [dflst.iterrows() for dflst in groupped_dfs]
    # dff = []
    # for i in range(ONE_STOCK_SHAPE):
    #     for g in gs:
    #         elem = next(g)[1].T
    #         dff.append(elem)
    # df = pd.concat(dff, axis=1).T
    # df.reset_index(inplace=True, drop=True)
    # # print(df.shape)


    config={'num':len(groupped_dfs), 'regulator':regulator,'short_hash':get_git_hash(),'bin_size':BIN_SIZE,'train_days':TRAIN_DAYS}
    result=train_and_pred(date_index,df,config)
    # result=train_and_pred(index,df,num,regulator,tile_array=list_)
    def update_stock_index(result, list_):
        summary,details=result
        stock_indicies = np.repeat(np.array(list_), BIN_SIZE)
        details.stock_index = stock_indicies
        for i in range(len(summary)):
            summary[i][1]=list_[i]
        return [summary,details]
    updated_result = update_stock_index(result, list_)
    return updated_result

# date_index=0;n_components=len(path0702Files_filtered)

def process_data(total_num_stocks,date_index,regulator,ratio_cumsum,n_clusters,features,dfs):
    """Process the data of all stocks within each date_index."""
    print(f">>> index BEGIN, {date_index}")
    
    # def classifyStocks(features):
    train_start_Index = (date_index * bin_size ) # for classification of stocks
    train_end_Index = (date_index * bin_size + train_size)  # for classification

    corr_matrix = get_corr_matrix(train_start_Index, train_end_Index, features)
    labels =get_labels_byPCA(corr_matrix,ratio_cumsum,n_components=len(path0702Files_filtered),n_clusters=n_clusters)
    # The labels used to indicate how to group.

    v = pd.DataFrame({"group_labels":labels,"stock_index": np.arange(total_num_stocks)})
    lst2 = [[i1,item.stock_index.values] for i1,item in tqdm(v.groupby("group_labels"))]  
    """Here, process each stock within all groups."""
    sub_r2_list = [return_lst(list_, date_index,regulator,dfs) for _, list_ in tqdm(lst2)]

    summaries = pd.DataFrame(np.concatenate([np.array(lst[0]) for lst in sub_r2_list]),columns=['date','stock_index','r2']).sort_values('stock_index')
    details_list = [lst[1] for lst in sub_r2_list]
    details = pd.concat(details_list).sort_values('stock_index')
    print(f"+++ index FINISH, {date_index}")
    return summaries,details



def get_r2df(total_num_stocks,n_clusters,ratio_cumsum,regulator,dfs):

    # # in sequential
    # start = time.time()
    # results = []
    # # for i in tqdm(range(2)): # for debug only
    # for date_index in tqdm(range(total_test_days)):
    #     # "Process the data of all stocks within each date_index."
    #     result = process_data(total_num_stocks,date_index,regulator,ratio_cumsum,n_clusters,features,dfs)
    #     results.append(result)
    #     # results.append(process_data(i,regulator))
    # end = time.time()
    # print(f"time {(end-start)/60}")
    
    
    # in parrallel
    start = time.time()
    ids=[process_data_ray.remote(total_num_stocks,date_index,regulator,ratio_cumsum,n_clusters,features,dfs) for date_index in tqdm(range(total_test_days))]
    results = [ray.get(id_) for id_ in ids]
    end = time.time()
    print(f"time {(end-start)/60}")

    summaries_list = [result[0] for result in results]
    details_list = [result[1] for result in results]
    details_df = pd.concat(details_list)
    df1 = pd.concat(summaries_list)
    df1.columns = ['test_date','stock_index','r2']
    assert np.unique(df1.stock_index).shape == (len(path0702Files_filtered),)
    df2 = df1.pivot(index="test_date",columns="stock_index",values="r2")
    return df2, details_df


@ray.remote
def process_data_ray(total_num_stocks,date_index,regulator,ratio_cumsum,n_clusters,features,dfs):
    return process_data(total_num_stocks,date_index,regulator,ratio_cumsum,n_clusters,features,dfs)

if __name__ == '__main__':
    START = time.time()

    ray.shutdown()
    ray.init(num_cpus=80,object_store_memory=40*1e9)

    regulator = "OLS"
    # regulator = "XGB"
    # regulator = "Lasso"
    regulator = "Inception"
    # n_clusters = 2
    # n_clusters = 5
    n_clusters = 1
    # n_clusters = 10
    # n_clusters = 20
    # n_clusters = 50


    # ratio_cumsum = 0.80
    # ratio_cumsum = 0.99
    # ratio_cumsum = 0.9999
    ratio_cumsum = 1.00
    
    trainType='clustered'
    print(f'trainType {trainType}, regulator {regulator}, n_clusters {n_clusters}, ratio_cumsum {ratio_cumsum}')
    
    dfs,num_of_stacked_stocks = getSingleDfs(trainType)
    print("dfs,num_of_stacked_stocks:",len(dfs),num_of_stacked_stocks)
    
    # new_dflst_lst,dflst = get_df_list(start_index=0, num = 481)


    # one_stock_shape = 3146
    # one_stock_shape = 109*26
    total_num_stocks = len(dfs)
    total_test_days, bin_size, train_size, test_size, x_list, y_list, original_space = param_define(dfs,total_num_stocks)    
    features = get_features(dfs,x_list,type="volume")
    # features = get_features(new_dflst_lst,x_list,type="features")
    print(features.shape)
    
    df2, details_df = get_r2df(total_num_stocks=total_num_stocks,n_clusters=n_clusters,ratio_cumsum=ratio_cumsum,regulator=regulator,dfs=dfs)
    

    print(df2)
    df2finalr2 = str(df2.mean(axis=1).mean())[:6]
    df2.to_csv("n_clusters_"+str(n_clusters)+"_"+df2finalr2+"_.csv")
    df2.mean(axis=0) # stock
    df2.mean(axis=1) # date
    print(df2.mean(axis=1).mean())

    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename_df2 = "07_2_kmeans_day_compare_test.py_" + str(len(df2.index)) + '_' + str(int(df2.index[0])) + "_" + str(int(df2.index[-1])) + "_" + current_datetime + "_.csv"
    df2.to_csv(filename_df2)

    filename_details_df = "07_2_kmeans_day_compare_test.py_" + str(len(df2.index)) + '_' + str(int(df2.index[0])) + "_" + str(int(df2.index[-1])) + "_" + current_datetime + "_all_values_.csv"
    details_df.to_csv(filename_details_df)
    END = time.time()
    
    print(f'trainType {trainType}, regulator {regulator}, n_clusters {n_clusters}, ratio_cumsum {ratio_cumsum}, time {END-START}')
    

