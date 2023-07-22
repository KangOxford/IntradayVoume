import os
os.sys.path.append("/home/kanli/cmem/src/")
from config import *

def tryMkdir(path):
    from os import listdir
    try: listdir(path)
    except:import os;os.mkdir(path)
    return 0


from os import listdir;
from os.path import isfile, join;
import pandas as pd
import numpy as np

path04_2="/home/kanli/seventh/02_raw_component/"
path04_3='/home/kanli/cmem/r_output/04_3_positiveFiltered/'
_,_ = map(tryMkdir,[path04_3,path05])

readFromPath = lambda data_path: sorted([f for f in listdir(data_path) if isfile(join(data_path, f)) and f != '.DS_Store'])
# path04Files, path04_1Files, path05Files = map(readFromPath, [path04, path04_1, path05])
path04Files, path04_1Files, path04_2Files,path04_3Files, path05Files = map(readFromPath, [path04, path04_1, path04_2,path04_3, path05])

from tqdm import tqdm
for i in tqdm(range(len(path04_1Files))):
    df = pd.read_csv(path04_1+path04_1Files[i],index_col=0)
    # df.columns
    seleced = df[['eta','seas','mu']]
    if (seleced>0).all().all():
        df.to_csv(path04_3+path04_1Files[i])
