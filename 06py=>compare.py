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

readFromPath = lambda data_path: sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])
path01Files, path01_1Files, path02Files, path04Files, path05Files = map(readFromPath, [path01, path01_1, path02, path04, path05])


for i in range(100):
    df = pd.read_csv(path02 + path02Files[i], delimiter='\t')
    raw = df.iloc[26:,:].loc[:,['date','turnover']].reset_index(drop=True)
    # ==========
    raw['log_turnover'] = raw['turnover'].apply(np.log)
    # ==========
    fore = pd.read_pickle(path05 + path05Files[i]).reset_index(drop=True)
    # fore.columns = ['eta','seas','mu']
    result = pd.concat([raw, fore], axis = 1)
    # item = pd.read_csv(path01_1 + path01_1Files[i],header=None,index_col=0).dropna(axis=1)
    result.to_pickle(path06+path02Files[i][:-3]+'pkl')


