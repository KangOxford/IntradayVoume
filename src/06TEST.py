path06 = '/home/kanli/cmem/r_output/06_r_output_raw_pkl/'
path = path06
import pandas as pd
import numpy as np
from os import listdir;from os.path import isfile,join
files = sorted([f for f in listdir(path) if isfile(join(path,f)) and f != '.DS_Store'])
r2df_lst = []
msedf_lst = []
from tqdm import tqdm
for i in tqdm(range(len(files))):
    file = path + files[i]
    symbol = files[i][:-4]
    df = pd.read_pickle(file)
    test = df[["date","turnover","x"]]
    # df.columns
    assert test.shape[0]/26 == test.shape[0]//26
    g = test.groupby('date')
    r2_list = []
    mse_list =[]
    for index , item in g:
        test_date = str(int(index))
        from sklearn.metrics import r2_score
        r2 = r2_score(item.turnover, item.x)
        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(item.turnover, item.x)
        r2_list.append([test_date, r2])
        mse_list.append([test_date, mse])
    r2df = pd.DataFrame(np.array(r2_list),columns=['date',symbol]).set_index("date")
    msedf = pd.DataFrame(np.array(mse_list),columns=['date',symbol]).set_index("date")
    r2df_lst.append(r2df)
    msedf_lst.append(msedf)
r2dfs = pd.concat(r2df_lst,axis=1)
msedfs = pd.concat(msedf_lst,axis=1)

r2dfs
