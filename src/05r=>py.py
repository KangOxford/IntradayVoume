import os
os.sys.path.append("/home/kanli/cmem/src/")
from config import *

def tryMkdir(path):
    from os import listdir
    try: listdir(path)
    except:import os;os.mkdir(path)
    return 0
_,_ = map(tryMkdir,[path01,path05])

from os import listdir;
from os.path import isfile, join;
import pandas as pd

readFromPath = lambda data_path: sorted([f for f in listdir(data_path) if isfile(join(data_path, f)) and f != '.DS_Store'])
path0400_1Files, path0500Files = map(readFromPath, [path0400_1, path0500])

from tqdm import tqdm
for i in tqdm(range(len(path0400_1Files))):
    item = pd.read_csv(path0400_1+path0400_1Files[i],index_col=0)
    item.to_pickle(path0500 + path0400_1Files[i][:-3] +"pkl")
    assert item.shape[0] == 110*26











