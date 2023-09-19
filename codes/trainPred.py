from model import *
from params import *
import pandas as pd
from sklearn.metrics import r2_score

def train_and_pred(index,df,num,regulator,tile_array):
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
    total_test_days, bin_size, train_size, test_size, x_list, y_list, original_space = param_define(df,num)
    X_train,y_train,X_test,y_test,original_images,train_end_index=get_X_train_y_train_X_test_original_images(df,num)

    # breakpoint()
    # print(regulator)
    if regulator == "Inception":
        y_pred = model_nn(X_train, y_train, X_test, y_test, regulator,num)
        # y_pred = regularity_nn(X_train, y_train, X_test,y_test, regulator,num)
    else:
        y_pred = regularity_ols(X_train, y_train, X_test, regulator,num)
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
    # stock_index = np.tile(tile_array, 26) # original [[bin,483] 26]
    stock_index = np.arange(num).repeat(26)
    original_images['stock_index'] = stock_index
    oneday_df = pd.concat([original_images, y_pred_clipped], axis=1)[['date','stock_index','true','pred']]
    lst = []
    g = oneday_df.groupby(stock_index)
    for stock, item in g:
        pass
        r2value = r2_score(item['true'], item['pred'])
        lst.append([test_date, stock, r2value])
    test_df = pd.DataFrame(lst,columns=["test_date", "stock", "r2value"])
    test_df = test_df.pivot(index='test_date',columns='stock')
    print(test_df)
    from datetime import datetime

    # Get current date and time
    current_time = datetime.now()

    # Format date and time to be used in the file name
    date_str = current_time.strftime("%Y-%m-%d")
    time_str = current_time.strftime("%H")

    # Combine date and time to form the file name
    # Get the short Git hash
    short_hash = get_git_hash()
    idnetificator = f"_{date_str}_{time_str}_{short_hash}"
    # In subsequent iterations, append without the header
    test_df.to_csv('/homes/80/kang/cmem/'+'data_summary_'+regulator+idnetificator+'.csv', mode='a', header=False, index=True)
    # test_df.to_csv('/homes/80/kang/cmem/'+'data_summary.csv', mode='a', header=False, index=True)
    # print(index,test_date,test_df.r2value.mean())
    oneday_df.to_csv('/homes/80/kang/cmem/'+'data_allValues_'+regulator+idnetificator+'.csv', mode='a', header=False, index=False)
    # oneday_df.to_csv('/homes/80/kang/cmem/'+'data_all_values.csv', mode='a', header=False, index=False)
    return lst,oneday_df


import subprocess
def get_git_hash():
    try:
        # Execute the command to get the latest commit hash
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')

        # Slice the first 4 characters
        short_hash = git_hash[:4]

        return short_hash

    except subprocess.CalledProcessError:
        print("An error occurred while fetching the Git hash.")
        return None


