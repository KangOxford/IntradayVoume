import numpy as np

path01 = "/Users/kang/CMEM/data/01_raw/"
path01_1 = "/Users/kang/CMEM/data/01.1_raw/"
# path02 = "/Users/kang/CMEM/data/02_r_input_10/"
# path03 = "/Users/kang/CMEM/data/03_r_input_10/"
# path04 = "/Users/kang/CMEM/r_output/04_r_output_raw_data_10/"
# path05 = "/Users/kang/CMEM/r_output/05_r_output_raw_pkl_10/"
path02 = "/Users/kang/CMEM/data/02_r_input/"
path04 = "/Users/kang/CMEM/r_output/04_r_output_raw_data/"
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

readFromPath = lambda data_path: sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])
path01Files, path04Files, path05Files = map(readFromPath, [path01, path04, path05])

for i in range(len(path04Files)):
    item = pd.read_csv(path04+path04Files[i],header=None,index_col=0).dropna(axis=1)
    item.to_pickle(path05 + path04Files[i][:-3] +"pkl")
    assert item.shape[0] == 3172










