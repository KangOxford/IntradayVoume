path = "/home/kanli/cmem/data/01.1_raw/"
import pandas as pd
import numpy as np
from os import listdir;from os.path import isfile,join
files = sorted([f for f in listdir(path) if isfile(join(path,f)) and f != '.DS_Store'])

lst = []
for i in range(len(files)):

    file = path + files[i]
    df = pd.read_pickle(file)
    g = df.groupby("timeHMs")[['symbol','qty']].mean()
    g.columns = [files[i][:-4]]
    try:
        assert (g.index == [ 930.0,  945.0, 1000.0, 1015.0, 1030.0, 1045.0, 1100.0, 1115.0,
                  1130.0, 1145.0, 1200.0, 1215.0, 1230.0, 1245.0, 1300.0, 1315.0,
                  1330.0, 1345.0, 1400.0, 1415.0, 1430.0, 1445.0, 1500.0, 1515.0,
                  1530.0, 1545.0]).all()
        lst.append(g)
    except:
        print(f"{files[i]} not found")
        continue

len(lst)

df1 = pd.concat(lst,axis=1)
m=df1.mean(axis=1)
s=df1.std(axis=1)


