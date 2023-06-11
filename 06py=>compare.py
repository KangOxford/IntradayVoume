import numpy as np
import pandas as pd
from os import listdir;
from os.path import isfile, join;

path01 = "/Users/kang/CMEM/data/01_raw/"
path01_1 = "/Users/kang/CMEM/data/01.1_raw/"
# path02 = "/Users/kang/CMEM/data/02_r_input_10/"
# path04 = "/Users/kang/CMEM/r_output/04_r_output_raw_data_10/"
# path05 = "/Users/kang/CMEM/r_output/05_r_output_raw_pkl_10/"
# path06 = '/Users/kang/CMEM/r_output/06_r_output_raw_pkl_10/'
path02 = "/Users/kang/CMEM/data/02_r_input/"
path04 = "/Users/kang/CMEM/r_output/04_r_output_raw_data/"
path05 = "/Users/kang/CMEM/r_output/05_r_output_raw_pkl/"
path06 = '/Users/kang/CMEM/r_output/06_r_output_raw_pkl/'


def tryMkdir(path):
    try: listdir(path)
    except:import os;os.mkdir(path)
    return 0
_,_ = map(tryMkdir,[path05,path06])

readFromPath = lambda data_path: sorted([f for f in listdir(data_path) if isfile(join(data_path, f)) and f != '.DS_Store'])
path01Files, path01_1Files, path02Files, path04Files, path05Files = map(readFromPath, [path01, path01_1, path02, path04, path05])


for i in range(100):
    df = pd.read_csv(path02 + path02Files[i], delimiter='\t')
    raw = df.iloc[26:,:].loc[:,['date','turnover']].reset_index(drop=True)
    # ==========
    raw['log_turnover'] = raw['turnover'].apply(np.log)
    # ==========
    fore = pd.read_pickle(path05 + path05Files[i]).reset_index(drop=True)
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
    # newResult = new_result.iloc[:-25,:].dropna(axis = 0).reset_index(drop=True)
    # newResult.to_pickle(path06+path02Files[i][:-3]+'pkl')


