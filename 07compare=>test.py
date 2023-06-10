import numpy as np
import pandas as pd
from os import listdir;
from os.path import isfile, join;

path00 = "/Users/kang/CMEM/"
path01 = "/Users/kang/CMEM/data/01_raw/"
path01_1 = "/Users/kang/CMEM/data/01.1_raw/"
path02 = "/Users/kang/CMEM/data/02_r_input/"
path04 = "/Users/kang/CMEM/r_output/04_r_output_raw_data/"
path05 = "/Users/kang/CMEM/r_output/05_r_output_raw_pkl/"
path06 = '/Users/kang/CMEM/r_output/06_r_output_raw_pkl/'
# path02 = "/Users/kang/CMEM/data/02_r_input_10/"
# path04 = "/Users/kang/CMEM/r_output/04_r_output_raw_data_10/"
# path05 = "/Users/kang/CMEM/r_output/05_r_output_raw_pkl_10/"
# path06 = '/Users/kang/CMEM/r_output/06_r_output_raw_pkl_10/'

def tryMkdir(path):
    try: listdir(path)
    except:import os;os.mkdir(path)
    return 0
_,_ = map(tryMkdir,[path05,path06])

readFromPath = lambda data_path: sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])
path01Files, path01_1Files, path02Files, path04Files, path05Files, path06Files =\
    map(readFromPath, [path01, path01_1, path02, path04, path05, path06])


def regularity_ols(X_train, y_train, X_test, regulator):
    if regulator == "OLS":
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
        X = pd.DataFrame(X_test).T
        y_pred = reg.predict(X)
        return y_pred[0]
    else:
        raise NotImplementedError


dflst = []
for i in range(100):
    print(f">>> i: {i}")
    df = pd.read_pickle(path06+path06Files[i])
    symbol = path06Files[i][:-4]
    bin_size = 26
    train_size = 10 * 26
    test_size = 1 * 26
    index_max  = int((df.shape[0] -(train_size + test_size))/bin_size)
    r2_list = []
    # index = 0 for index in range(0, index_max+1)
    # index = 0 for index in range(0, index_max+0) # not sure
    # x_list = ['eta','seas','mu']
    # y_list = ['turnover']
    x_list = ['log_x', 'log_eta*seas', 'log_eta', 'log_seas', 'log_mu']
    y_list = ['log_turnover']
    for index in range(0, index_max + 1):
        train_end_index = index * bin_size + train_size
        def get_trainData(df):
            x_train = df.loc[index * bin_size : index * bin_size + train_size,x_list]
            y_train = df.loc[index * bin_size : index * bin_size + train_size,y_list]
            return x_train, y_train
        def get_testData(df):
            x_test = df.loc[train_end_index:train_end_index+test_size,x_list]
            y_test = df.loc[train_end_index:train_end_index+test_size,y_list]
            return x_test, y_test
        X_train, y_train = get_trainData(df)
        X_test, y_test = get_testData(df)

        regulator = "OLS"
        y_pred = regularity_ols(X_train, y_train, X_test, regulator)
        min_limit, max_limit = y_train.min(), y_train.max()
        broadcast = lambda x:np.full(y_pred.shape[0], x.to_numpy())
        min_limit, max_limit= map(broadcast, [min_limit, max_limit])
        y_pred_clipped = np.clip(y_pred, min_limit, max_limit)
        test_date = df.date[train_end_index]
        from sklearn.metrics import r2_score
        r2 = r2_score(y_test, y_pred)
        r2_list.append([test_date,r2])
        # y_list.append([test_date, y_test, y_pred_clipped])
    r2arr = np.array(r2_list)
    df = pd.DataFrame(r2arr)
    # r2arr[:,1].mean()
    df.columns = ['test_date','r2']
    df['symbol'] = symbol
    df = df[['symbol','test_date','r2']]
    df.test_date = df.test_date.astype(int)
    pivot_df = df.pivot(index='test_date', columns='symbol', values='r2')
    dflst.append(pivot_df)
r2df = pd.concat(dflst,axis =1)
r2df.to_csv(path00 + "07_r2df.csv")
