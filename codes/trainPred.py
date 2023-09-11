from model import *
from params import *
import pandas as pd
from sklearn.metrics import r2_score

def train_and_pred(index,df,num,regulator,tile_array):
    total_test_days, bin_size, train_size, test_size, x_list, y_list, original_space = param_define(df,num)
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
    # print(regulator)
    y_pred = regularity_ols(X_train, y_train, X_test, regulator,num)
    # y_pred = regularity_nn(X_train, y_train, X_test,y_test, regulator,num)
    # print(regulator+"_finished")
    min_limit, max_limit = y_train.min(), y_train.max()
    broadcast = lambda x: np.full(y_pred.shape[0], x.to_numpy())
    min_limit, max_limit = map(broadcast, [min_limit, max_limit])
    y_pred_clipped = np.clip(y_pred, min_limit, max_limit)
    if any('log' in x for x in x_list):
        y_pred_clipped = np.exp(y_pred_clipped)
    test_date = df.date.iloc[train_end_index]
    '''prob in the y_pred shapes'''

    # r2 = r2_score(y_test, y_pred_clipped)
    y_pred_clipped = pd.DataFrame(y_pred_clipped)
    y_pred_clipped.columns = ['pred']
    original_images.reset_index(inplace=True, drop=True)
    original_images.columns = ['true']

    original_images['date'] = test_date
    stock_index = np.tile(tile_array, 26)
    original_images['stock_index'] = stock_index
    oneday_df = pd.concat([original_images, y_pred_clipped], axis=1)[['date','stock_index','true','pred']]
    lst = []
    g = oneday_df.groupby(stock_index)
    for stock, item in g:
        pass
        r2value = r2_score(item['true'], item['pred'])
        lst.append([test_date, stock, r2value])
    test_df = pd.DataFrame(lst,columns=["test_date", "stock", "r2value"])
    # print(test_df)
    # print(index,test_date,test_df.r2value.mean())
    return lst,oneday_df