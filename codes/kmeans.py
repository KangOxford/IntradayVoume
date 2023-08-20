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


def get_features(new_dflst_lst,type="volume"):
    if type =="volume":
        return get_volume_features(new_dflst_lst)
    elif type =="features":
        return get_features_features(new_dflst_lst)
    else:
        raise NotImplementedError

def get_volume_features(new_dflst_lst):
    feature_list = []
    for index, item in enumerate(new_dflst_lst):
        pass
        item.date = item.date.astype(np.int32)
        item = item.set_index('date')
        value = item.turnover
        value.name = path0600_1Files[index][:-4]
        feature_list.append(value)
    features = pd.concat(feature_list,axis = 1)
    return features

def get_features_features(new_dflst_lst):
    nfeatures = []
    for col in x_list:
        feature_list = []
        for index, item in enumerate(new_dflst_lst):
            pass
            item.columns
            item.date = item.date.astype(np.int32)
            item = item.set_index('date')
            value = item.loc[:,col]
            assert item.shape[1] == 108
            value.name = path0600_1Files[index][:-4]
            feature_list.append(value)
        features = pd.concat(feature_list,axis = 1)
        nfeatures.append(features)
    nfeatures = np.stack(nfeatures)
    len(nfeatures.shape)
    nfeatures.shape
    return nfeatures



# n_clusters = 2
# n_clusters = 5
n_clusters = 10
# n_clusters = 20
# n_clusters = 50


ratio_cumsum = 0.80
# ratio_cumsum = 0.99
# ratio_cumsum = 0.9999
# ratio_cumsum = 1.00
features = get_features(new_dflst_lst,type="volume")
# features = get_features(new_dflst_lst,type="features")
features.shape





groupped_dfs = [new_dflst_lst[i] for i in list_]
gs = [dflst.iterrows() for dflst in groupped_dfs]
dff = []
for i in range(one_stock_shape):
# for i in tqdm(range(one_stock_shape)):
    for g in gs:
        elem = next(g)[1].T
        dff.append(elem)
df = pd.concat(dff, axis=1).T
df.reset_index(inplace=True, drop=True)
# print(df.shape)

num = len(groupped_dfs)