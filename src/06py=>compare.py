import numpy as np
import pandas as pd
from os import listdir;
from os.path import isfile, join;
import os
os.sys.path.append("/home/kanli/cmem/src/")
try: from config import *
except: from src.config import *


def tryMkdir(path):
    try: listdir(path)
    except:import os;os.mkdir(path)
    return 0
_,_ = map(tryMkdir,[path0500,path0600])

readFromPath = lambda data_path: sorted([f for f in listdir(data_path) if isfile(join(data_path, f)) and f != '.DS_Store'])
path01Files, path01_1Files, path0200Files, path0400Files, path0500Files = map(readFromPath, [path01, path01_1, path0200, path0400, path0500])

from tqdm import tqdm
for i in tqdm(range(len(path0500Files))):
    name = path0400Files[i]
    df = pd.read_csv(path0400 + name)[['date','original']]
    df['date'] = df['date'].str.replace('X', '').str.replace('.', '')
    df.columns = ['date','turnover']
    raw =df
    # raw = df.iloc[26:,:].loc[:,['date','turnover']].reset_index(drop=True)?
    # ==========
    raw['log_turnover'] = raw['turnover'].apply(np.log)
    # ==========
    fore = pd.read_pickle(path0500 + path0500Files[i]).reset_index(drop=True)
    forecast = fore.loc[:,['log_x', 'log_eta*seas', 'log_eta', 'log_seas', 'log_mu', 'x', 'eta*seas','eta', 'seas', 'mu']]
    assert forecast.shape[1] == 10
    feature = fore.iloc[:,10:]
    result = pd.concat([raw, forecast], axis = 1)
    shift_result = result.shift(-1)
    new_result = pd.concat([shift_result, feature],axis=1)
    new_result = new_result.dropna()
    frequency_counts = new_result.date.value_counts()
    unique_dates = sorted(frequency_counts[frequency_counts == 26].index.tolist())
    newResult = new_result[new_result['date'].isin(unique_dates)]
    newResult = newResult.reset_index(drop=True)
    newResult.to_pickle(path0600+path0400Files[i][:-3]+'pkl')
    newResult.columns
    # [print(col) for col in newResult.columns]
    def shift_check(newResult):
        columns = list(newResult.columns)
        columns.remove('log_qty')
        columns.remove('qty')
        columns.remove('date')
        columns.remove('turnover')
        columns.remove('log_turnover')
        newCol = ['date','turnover','qty','log_turnover','log_qty']+columns
        NewResult = newResult[newCol]
        return NewResult

    lst = []
    g=newResult.groupby("date")
    for index, item in g:
        pass
        itm = item[['turnover','x']]
        from sklearn.metrics import r2_score
        r2 = r2_score(itm.turnover,itm.x)
        lst.append([index,r2])
    newDf=pd.DataFrame(lst)
    assert newDf.mean()[1] >= 0.45, "the kf-cmem should have a oos r2 over 0.45"
