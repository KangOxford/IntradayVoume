import numpy as np
import pandas as pd
from os import listdir;
from os.path import isfile, join;

# path00 = "/Users/kang/CMEM/"
path00 = "/home/kanli/cmem/"
path01 = path00 + "data/01_raw/"
path01_1=path00 + "data/01.1_raw/"
path02 = path00 + "data/02_r_input/"
path04 = path00 + "r_output/04_r_output_raw_data/"
path05 = path00 + "r_output/05_r_output_raw_pkl/"
path06 = path00 + "r_output/06_r_output_raw_pkl/"

import os
os.sys.path.append("/home/kanli/cmem/src/")
from config import *

'''
I guess this file is wrong
due to 
        x_train = df.loc[train_start_index: train_end_index, x_list]
        y_train = df.loc[train_start_index: train_end_index, y_list]
        
        not using iloc here


'''
def tryMkdir(path):
    try: listdir(path)
    except:import os;os.mkdir(path)
    return 0
_,_ = map(tryMkdir,[path05,path06])

readFromPath = lambda data_path: sorted([f for f in listdir(data_path) if isfile(join(data_path, f)) and f != '.DS_Store'])
path01Files, path01_1Files, path02Files, path04Files, path05Files, path06Files =\
    map(readFromPath, [path01, path01_1, path02, path04, path05, path06])

array1 = np.concatenate( [np.arange(1,10,0.01), np.arange(10,50,0.1) ])
array2 = np.arange(1,0.001,-0.001)
combined_array = np.array(list(zip(array1, array2))).flatten()
# used for alphas

def regularity_ols(X_train, y_train, X_test, regulator):
    if regulator == "None":
        y_pred = X_test.to_numpy().flatten()
        return y_pred
    elif regulator == "OLS":
        # print("OLS")
        import statsmodels.api as sm
        def ols_with_summary(X, y):
            X = sm.add_constant(X, has_constant='add')
            results = sm.OLS(y, X).fit()
            return results

        model = ols_with_summary(X_train, y_train)
        X = sm.add_constant(X_test, has_constant='add')
        y_pred = model.predict(X).values
        # assert type(y_pred) == np.float64
        return y_pred
    elif regulator in ["Lasso", "Ridge"]:
        # print("LASSO / RIDGE")
        def find_best_regularity_alpha(X_train, y_train):
            if regulator == "Lasso":
                from sklearn.linear_model import LassoCV
                model = LassoCV(random_state=0, max_iter=10000000)
            if regulator == "Ridge":
                from sklearn.linear_model import RidgeCV
                model = RidgeCV(alphas=combined_array)
            model.fit(X_train, y_train)
            return model.alpha_

        best_regularity_alpha = find_best_regularity_alpha(X_train, y_train)
        # print(best_regularity_alpha) #$
        if regulator == "Lasso":
            from sklearn.linear_model import Lasso
            reg = Lasso(alpha=best_regularity_alpha, max_iter=10000000, tol=1e-2)
        if regulator == "Ridge":
            from sklearn.linear_model import Ridge
            reg = Ridge(alpha=best_regularity_alpha, max_iter=10000000, tol=1e-2)
        reg.fit(X_train, y_train)
        # X = pd.DataFrame(X_test).T
        # y_pred = reg.predict(X)
        y_pred = reg.predict(X_test)
        y_pred = y_pred.flatten()
        return y_pred
    else:
        raise NotImplementedError
# ================================
num = len(path06Files) # CAUTION do not forget to make the right number of stocks
# num = 100
# num = 90
# num = 80
# num = 70
# num = 60
# num = 50
# num = 40
# num = 30
# num = 20
# num = 10
# num = 3
# num = 2
# num = 1
def get_universal_df():
    df_lst = []
    from tqdm import tqdm
    for i in tqdm(range(num)): # on mac4
        df = pd.read_pickle(path06+path06Files[i])
        df_lst.append(df)

    new_dflst_lst = []
    for index, dflst in enumerate(df_lst):
        # assert dflst.shape[0] == 3172, f"index, {index}"
        if dflst.shape[0] == 3146:
            new_dflst_lst.append(dflst)

    gs = [dflst.iterrows() for dflst in new_dflst_lst]
    dff = []
    for i in tqdm(range(dflst.shape[0])):
        for g in gs:
            elem = next(g)[1].T
            dff.append(elem)
    df = pd.concat(dff, axis=1).T
    df.reset_index(inplace=True,drop=True)
    return df
df = get_universal_df()
# symbol = path06Files[i][:-4]



# ================================
bin_size = 26
train_size = 10 * 26
test_size = 1 * 26
index_max  = int((df.shape[0] -(train_size + test_size))/bin_size)
r2_list = []
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
x_list = x_list +  our_log_features
y_list = ['log_turnover']
# x_list = ['log_eta', 'log_seas', 'log_mu']
# y_list = ['log_turnover']
# x_list = ['x']
# y_list = ['turnover']

# x_list = ['log_x']
# y_list = ['log_turnover']
original_space = ['turnover']
# ================================



from tqdm import tqdm
# for index in tqdm(range(0, index_max + 1)):
#     print(index)
for index in tqdm(range(111)):
    train_start_index = index * bin_size * num
    train_end_index = (index * bin_size + train_size) * num -1
    test_start_index = train_end_index + 1
    test_end_index = train_end_index+test_size * num
    def get_trainData(df):
        x_train = df.loc[train_start_index: train_end_index, x_list]
        y_train = df.loc[train_start_index: train_end_index, y_list]
        return x_train, y_train
    def get_testData(df):
        x_test = df.loc[test_start_index:test_end_index , x_list]
        y_test = df.loc[test_start_index:test_end_index , y_list]
        return x_test, y_test
    original_images = df.loc[test_start_index:test_end_index , original_space]
    X_train, y_train = get_trainData(df)
    X_test, y_test = get_testData(df)
    def no_overlap_assert(y_test, y_train):
        assert y_test.index[0] > y_train.index[-1]
    no_overlap_assert(y_test, y_train)

    # regulator = "OLS"
    regulator = "Lasso"
    # regulator = "Ridge"
    # regulator = "None"
    y_pred = regularity_ols(X_train, y_train, X_test, regulator)
    min_limit, max_limit = y_train.min(), y_train.max()
    broadcast = lambda x:np.full(y_pred.shape[0], x.to_numpy())
    min_limit, max_limit= map(broadcast, [min_limit, max_limit])
    y_pred_clipped = np.clip(y_pred, min_limit, max_limit)
    if any('log' in x for x in x_list):
        y_pred_clipped = np.exp(y_pred_clipped)
    test_date = df.date[test_start_index]


    # r2 = r2_score(y_test, y_pred_clipped)
    y_pred_clipped = pd.DataFrame(y_pred_clipped)
    y_pred_clipped.columns = ['pred']
    original_images.reset_index(inplace=True,drop=True)
    original_images.columns = ['true']
    repeated = np.repeat(np.arange(num), 26)
    original_images.index = repeated
    y_pred_clipped.index = repeated
    test = pd.concat([original_images, y_pred_clipped], axis = 1)
    test.reset_index(inplace=True)
    from sklearn.metrics import r2_score
    g = test.groupby('index')
    lst = []
    for index, item in g:
        pass
        assert item.shape[0] == 26
        r2v = r2_score(item['true'], item['pred'])
        lst.append(r2v)
    r2 = np.mean(lst)
    assert type(r2) == np.float64
    r2_list.append([test_date,r2])
    # y_list.append([test_date, y_test, y_pred_clipped])
assert len(r2_list) == 111






r2arr = np.array(r2_list)
df1 = pd.DataFrame(r2arr)
df1.columns = ['test_date','r2']
df1
df1.r2.mean()
#
#
# df.test_date = df.test_date.astype(int)
# pivot_df = df.pivot(index='test_date', columns='symbol', values='r2')
# dflst.append(pivot_df)
#
#
#
# r2df = pd.concat(dflst,axis =1)
# r2df.to_csv(path00 + "07_r2df_"+regulator+"_.csv", mode = 'w')
