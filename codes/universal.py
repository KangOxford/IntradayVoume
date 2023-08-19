from tqdm import tqdm
import numpy as np
import pandas as pd
from os import listdir;
from os.path import isfile, join;
from sklearn.metrics import r2_score


import numpy as np
import pandas as pd
from tqdm import tqdm
import os;os.sys.path.append(os.path.expanduser('~') + "/cmem/codes/")
import os;os.sys.path.append(os.path.expanduser('~') + "/cmem/")
import os;os.sys.path.append("/homes/80/kang/cmem/")
import os;os.sys.path.append("/homes/80/kang/cmem/codes/")
from utils import *
from model import *
import multiprocessing
import time

path0600_1Files = readFromPath(path0600_1)
print(len(path0600_1Files))

def get_universal_df(start_index, num):
    df_lst = []
    from tqdm import tqdm
    for i in tqdm(range(start_index, start_index + num)):  # on mac4
        df = pd.read_csv(path0600_1 + path0600_1Files[i], index_col=0)
        df_lst.append(df)

    new_dflst_lst = []
    for index, dflst in enumerate(df_lst):
        # assert dflst.shape[0] == 3172, f"index, {index}"
        # if dflst.shape[0] == 3146:
        # assert  dflst.shape[0] == 2834, f"shape, {dflst.shape}"
        if dflst.shape[0] == 2834:
            new_dflst_lst.append(dflst)

    gs = [dflst.iterrows() for dflst in new_dflst_lst]
    dff = []
    for i in tqdm(range(dflst.shape[0])):
        for g in gs:
            elem = next(g)[1].T
            dff.append(elem)
    df = pd.concat(dff, axis=1).T
    df.reset_index(inplace=True, drop=True)
    print(">>> finish preparing the universal df")
    return df



if __name__=="__main__":    
    df = get_universal_df(start_index=0, num=len(path0600_1Files))
    tryMkdir(path0700)
    df.to_csv(path0700+"universal.csv")
    
    
    
    