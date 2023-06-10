path00 = "/Users/kang/CMEM/"
path01 = "/Users/kang/CMEM/data/01_raw/"
path01_1 = "/Users/kang/CMEM/data/01.1_raw/"
path02 = "/Users/kang/CMEM/data/02_r_input/"
path04 = "/Users/kang/CMEM/r_output/04_r_output_raw_data/"
path04_1 = "/Users/kang/CMEM/r_output/04_1_rOuputFeatured/"
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
path04Files, path04_1Files, path05Files = map(readFromPath, [path04, path04_1, path05])
for i in range(len(path04Files)):
    df = pd.read_csv(path04 + path04Files[i],header=None, index_col=0).dropna(axis=1).reset_index(drop=True)
    df.columns = ['eta','seas','mu','x']
    df['eta*seas'] = df['eta'] * df['seas']
    # ============= milestone here ============
    df['log_eta'] = df['eta'].apply(np.log)
    df['log_seas'] = df['seas'].apply(np.log)
    df['log_mu'] = df['mu'].apply(np.log)
    df['log_x'] = df['x'].apply(np.log)
    df['log_eta*seas'] = df['eta*seas'].apply(np.log)
    new_df = df[['log_x','log_eta*seas','log_eta','log_seas','log_mu']]
    # ============= milestone here ============
    new_df.to_csv(path04_1 + path04Files[i])
