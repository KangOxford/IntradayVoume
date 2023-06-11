path00 = "/Users/kang/CMEM/"
path01 = "/Users/kang/CMEM/data/01_raw/"
path01_1 = "/Users/kang/CMEM/data/01.1_raw/"
path02 = "/Users/kang/CMEM/data/02_r_input/"
path04 = "/Users/kang/CMEM/r_output/04_r_output_raw_data/"
path04_1 = "/Users/kang/CMEM/r_output/04_1_rOuputFeatured/"
path04_2 = "/Users/kang/Volume-Forecasting/02_raw_component/"
path05 = "/Users/kang/CMEM/r_output/05_r_output_raw_pkl/"
path06 = '/Users/kang/CMEM/r_output/06_r_output_raw_pkl/'


def tryMkdir(path):
    try: listdir(path)
    except:import os;os.mkdir(path)
    return 0
_,_ = map(tryMkdir,[path01,path05])

from os import listdir;
from os.path import isfile, join;
import pandas as pd
import numpy as np


readFromPath = lambda data_path: sorted([f for f in listdir(data_path) if isfile(join(data_path, f)) and f != '.DS_Store'])
path04Files, path04_1Files, path04_2Files, path05Files = map(readFromPath, [path04, path04_1, path04_2, path05])

common_dates = np.load(path00 + 'common_dates.npy')[1:] # only the r output parts is in need

for i in range(len(path04Files)):
    df = pd.read_csv(path04 + path04Files[i],header=None, index_col=0).dropna(axis=1).reset_index(drop=True)
    df = df.apply(abs)
    df.columns = ['eta','seas','mu','x']
    df['eta*seas'] = df['eta'] * df['seas']
    # ============= milestone here ============
    df['log_eta'] = df['eta'].apply(np.log)
    df['log_seas'] = df['seas'].apply(np.log)
    df['log_mu'] = df['mu'].apply(np.log)
    df['log_x'] = df['x'].apply(np.log)
    df['log_eta*seas'] = df['eta*seas'].apply(np.log)
    new_df = df[['log_x','log_eta*seas','log_eta','log_seas','x','eta*seas','log_mu', 'eta','seas','mu']]
    # ============= milestone here ============
    ft = pd.read_pickle(path04_2 + path04_2Files[i]).reset_index(drop=True)
    # Select rows from the DataFrame based on common dates
    selected_rows = ft[ft['date'].isin(common_dates)].reset_index(drop=True)
    assert len(list(set(selected_rows.date))) == len(common_dates)
    assert selected_rows.shape[0] == 122 * 26
    features = selected_rows.columns[5:-2]
    for feature in features:
        new_ft = "log_"+feature
        selected_rows[new_ft] = selected_rows[feature].apply(np.log)
    new_features = list(features) + ["log_"+feature for feature in features]
    df_with_newFeatures = selected_rows[new_features]
    merged_df = pd.concat([new_df, df_with_newFeatures],axis = 1)
    # ============= milestone here ============
    merged_df.to_csv(path04_1 + path04Files[i])
