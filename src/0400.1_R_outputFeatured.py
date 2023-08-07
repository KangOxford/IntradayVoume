import os
os.sys.path.append("/home/kanli/cmem/src/")
try: from config import *
except: from src.config import *

def tryMkdir(path):
    from os import listdir
    try: listdir(path)
    except:import os;os.mkdir(path)
    return 0
_,_ = map(tryMkdir,[path01,path05])


from os import listdir;
from os.path import isfile, join;
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score

readFromPath = lambda data_path: sorted([f for f in listdir(data_path) if isfile(join(data_path, f)) and f != '.DS_Store'])
# path04Files, path04_1Files, path05Files = map(readFromPath, [path04, path04_1, path05])
path0400Files, path0400_1Files, path0400_2Files = map(readFromPath, [path0400, path0400_1, path0400_2])
path04Files, path04_1Files, path04_2Files = map(readFromPath, [path04, path04_1, path04_2])

# mkdir /home/kanli/cmem/02_raw_component/
# cp /home/kanli/seventh/02_raw_component/* /home/kanli/cmem/02_raw_component/
'''
# common_dates = np.load(path00 + 'common_dates.npy')[1:] # only the r output parts is in need
sample=pd.read_csv("/home/kanli/cmem/data/02_r_input/A.txt",sep="\t")
sample1=sorted(list(set(sample.date)))
common_dates0 = sample1[1:]
common_dates = [str(date) for date in common_dates0]
# len(common_dates)
'''

import pandas as pd
# Assuming 'df' is your DataFrame variable name
pd.set_option('display.max_columns', None)



# R2LST=[]
# i=0
# for i in tqdm(range(0,len(path04Files))):
#     df = pd.read_csv(path04 + path04Files[i]).dropna(axis=1).reset_index(drop=True)
#     print(f"days {df.shape[0]/26}")
#     # df = df.apply(abs)
#     df
#     # r2_score(df.iloc[:26,0],df.iloc[:26,1])
#
#     # df.r2.mean()
#     df['date'] = df['date'].str.replace('X', '').str.replace('.', '')
#     common_dates = sorted(list(set(df.date)))[1:-1] #the last/first one is for the shift -1
#     df = df[['date','daily','seasonal','dynamic','forecast_signal','original']]
#     # d1=df[['date','original']]
#     # d1
#     # d2=df[['daily','seasonal','dynamic','forecast_signal']]
#     # d2
#     # # df.shift(-1)
#     # df=pd.concat([d1,d2],axis=1)
#     # df
#     # df.columns = ['date','qty','eta','seas','mu','x',]
#     # df['eta*seas'] = df['eta'] * df['seas']
#     # # r2_score(df.original,df.forecast_signal)
#     # df
#     g = df.groupby('date')
#
#     r2lst=[]
#     for a,b in g:
#         # print(a)
#         pass
#         has_na = b.isna().any().any()
#         if has_na:
#             continue
#         r2 = r2_score(b.original,b.forecast_signal)
#         r2lst.append(r2)
#     R2LST.append(pd.Series(r2lst, name=path04Files[i][:-4]))
# r2df = pd.DataFrame(R2LST)
# r2df.mean(axis=0).mean()





#
#
#
# R2LST=[]
# i=0
# for i in tqdm(range(0,len(path0400Files))):
#     df = pd.read_csv(path0400 + path0400Files[i]).dropna(axis=1).reset_index(drop=True)
#     print(f"days {df.shape[0]/26}")
#     # df = df.apply(abs)
#     df
#     r2_score(df.iloc[:26,0],df.iloc[:26,1])
#
#
#     df.r2.mean()
#     df['date'] = df['date'].str.replace('X', '').str.replace('.', '')
#     common_dates = sorted(list(set(df.date)))[1:-1] #the last/first one is for the shift -1
#     df = df[['date','daily','seasonal','dynamic','forecast_signal','original']]
#     # d1=df[['date','original']]
#     # d1
#     # d2=df[['daily','seasonal','dynamic','forecast_signal']]
#     # d2
#     # # df.shift(-1)
#     # df=pd.concat([d1,d2],axis=1)
#     # df
#     # df.columns = ['date','qty','eta','seas','mu','x',]
#     # df['eta*seas'] = df['eta'] * df['seas']
#     # # r2_score(df.original,df.forecast_signal)
#     # df
#     g = df.groupby('date')
#
#     r2lst=[]
#     for a,b in g:
#         # print(a)
#         pass
#         has_na = b.isna().any().any()
#         if has_na:
#             continue
#         r2 = r2_score(b.original,b.forecast_signal)
#         r2lst.append(r2)
#     R2LST.append(pd.Series(r2lst, name=path0400Files[i][:-4]))
# r2df = pd.DataFrame(R2LST)
# r2df.mean(axis=0).mean()
#




R2LST=[]
R2LST2=[]
i=0
for i in tqdm(range(0,len(path0400Files))):
    df = pd.read_csv(path0400 + path0400Files[i]).dropna(axis=1).reset_index(drop=True)
    print(f"days {df.shape[0]/26}")
    # df = df.apply(abs)
    df
    r2_score(df.iloc[:26,0],df.iloc[:26,1])
    R2LST2.append(r2_score(df.original,df.forecast_signal))

    df.r2.mean()
    df['date'] = df['date'].str.replace('X', '').str.replace('.', '')
    common_dates = sorted(list(set(df.date)))[1:-1] #the last/first one is for the shift -1
    df = df[['date','daily','seasonal','dynamic','forecast_signal','original']]
    d1=df[['date','original']]
    d2=df[['daily','seasonal','dynamic','forecast_signal']]
    # df.shift(-1)
    df=pd.concat([d1,d2],axis=1)
    df.columns = ['date','qty','eta','seas','mu','x',]
    df['eta*seas'] = df['eta'] * df['seas']
    # ============= milestone here ============
    df['log_eta'] = df['eta'].apply(np.log)
    df['log_seas'] = df['seas'].apply(np.log)
    df['log_mu'] = df['mu'].apply(np.log)
    df['log_x'] = df['x'].apply(np.log)
    df['log_eta*seas'] = df['eta*seas'].apply(np.log)
    df['log_qty'] = df['qty'].apply(np.log)
    new_df = df[['date','log_qty','log_x','log_eta*seas','log_eta','log_seas','qty','x','eta*seas','log_mu', 'eta','seas','mu']]
    new_df0 = new_df.copy()
    new_df0.qty = new_df0.qty.shift(-1)
    g = new_df0.groupby('date')
    r2lst=[]
    for a,b in g:
        # print(a)
        pass
        has_na = b.isna().any().any()
        if has_na:
            continue
        r2 = r2_score(b.qty,b.x)
        r2lst.append(r2)
    R2LST.append(pd.Series(r2lst,name=path0400Files[i][:-4]))
r2df = pd.DataFrame(R2LST)
r2df.mean(axis=0).mean()
r2df.mean(axis=0)
r2df.mean(axis=1).mean()
r2df.mean(axis=1)

np.array(R2LST2).mean()


































#
# from tqdm import tqdm
# i=0
#
# R2LST=[]
# for i in tqdm(range(0,len(path0400Files))):
#     df = pd.read_csv(path0400 + path0400Files[i]).dropna(axis=1).reset_index(drop=True)
#     print(f"days {df.shape[0]/26}")
#     # df = df.apply(abs)
#     df
#     df.r2.mean()
#     df['date'] = df['date'].str.replace('X', '').str.replace('.', '')
#     common_dates = sorted(list(set(df.date)))[:-1] #the last one is for the shift -1
#     df = df[['date','daily','seasonal','dynamic','forecast_signal','original']]
#     d1=df[['date','original']][:-26]
#     d1
#     d2=df[['daily','seasonal','dynamic','forecast_signal']].shift(-1)[:-26]
#     d2
#     # df.shift(-1)
#     df=pd.concat([d1,d2],axis=1)
#     df
#     df.columns = ['date','qty','eta','seas','mu','x',]
#     df['eta*seas'] = df['eta'] * df['seas']
#     # ============= milestone here ============
#     df['log_eta'] = df['eta'].apply(np.log)
#     df['log_seas'] = df['seas'].apply(np.log)
#     df['log_mu'] = df['mu'].apply(np.log)
#     df['log_x'] = df['x'].apply(np.log)
#     df['log_eta*seas'] = df['eta*seas'].apply(np.log)
#     df['log_qty'] = df['qty'].apply(np.log)
#     new_df = df[['date','log_qty','log_x','log_eta*seas','log_eta','log_seas','qty','x','eta*seas','log_mu', 'eta','seas','mu']]
#     new_df0 = new_df.copy()
#     new_df0.qty = new_df0.qty.shift(1)
#     g = new_df0.groupby('date')
#     r2lst=[]
#     for a,b in g:
#         # print(a)
#         pass
#         has_na = b.isna().any().any()
#         if has_na:
#             continue
#         r2 = r2_score(b.qty,b.x)
#         r2lst.append(r2)
#     R2LST.append(pd.Series(r2lst,name=path0400Files[i][:-4]))
# r2df = pd.DataFrame(R2LST)
# r2df.mean(axis=0).mean()


# ============= milestone here ============
try:
    name = path0400Files[i][:-4] +".pkl"
    ft = pd.read_pickle(path0400_2 + name).reset_index(drop=True)
    # Select rows from the DataFrame based on common dates
    selected_rows = ft[ft['date'].isin(common_dates)].reset_index(drop=True)
except:
    print(f"no {name} file, continue")
    continue

try:
    assert len(list(set(selected_rows.date))) == len(common_dates)
    assert selected_rows.shape[0] == 109 * 26 # 3172
except:
    print(f"{path0400Files[i]} exist error, continue")
    continue
# ============= milestone here ============
features = ['qty'] + list(selected_rows.columns[5:-2])
# features = list(selected_rows.columns[5:-2])
assert type(features) == list
for feature in features:
    new_ft = "log_"+feature
    selected_rows[new_ft] = selected_rows[feature].apply(np.log)
new_features = features + ["log_"+feature for feature in features]
df_with_newFeatures = selected_rows[new_features]
merged_df = pd.concat([new_df, df_with_newFeatures],axis = 1)
# ============= milestone here ============
merged_df.to_csv(path0400_1 + path0400Files[i], mode = 'w+')





def r2_check_0400_1V2():
    lst = []
    for i in tqdm(range(len(path0400_1Files))):
        item = pd.read_csv(path0400_1 + path0400_1Files[i], index_col=0)
        item = item[['qty','turnover','x','date']]
        g = item.groupby("date")
        for index, it in g:
            pass
            true = it.qty
            pred = it.x
            from sklearn.metrics import r2_score
            r2 = r2_score(true, pred)
            lst.append([index, path0400_1Files[i][:-4],r2])
    df = pd.DataFrame(lst,columns=['date','stock','r2'])
    dff=df.pivot(index = 'date',columns='stock')
    r2 = dff.mean(axis=1).mean()
    assert r2 >= 0.45, f"the kf-cmem should have a oos r2:{r2} over 0.45"
r2_check_0400_1V2()

def r2_check_0400_1():
    lst = []
    for i in tqdm(range(len(path0400_1Files))):
        item = pd.read_csv(path0400_1 + path0400_1Files[i], index_col=0)
        item['log_qty_shift1']=item.log_qty.shift(-1)
        import numpy as np
        item['date'] = np.array([[i]*26 for i in range(item.shape[0]//26)]).reshape(-1)
        itm = item[:-26]
        g = itm.groupby("date")
        for index, it in g:
            pass
            true = it.log_qty_shift1.apply(np.exp)
            pred = it.log_x.apply(np.exp)
            from sklearn.metrics import r2_score
            r2 = r2_score(true, pred)
            lst.append([index, path0400_1Files[i][:-4],r2])

    df = pd.DataFrame(lst,columns=['date','stock','r2'])
    dff=df.pivot(index = 'date',columns='stock')
    r2 = dff.mean(axis=1).mean()
    assert r2 >= 0.45, f"the kf-cmem should have a oos r2:{r2} over 0.45"
r2_check_0400_1()
