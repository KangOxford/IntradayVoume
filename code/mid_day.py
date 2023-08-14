import numpy as np
import pandas as pd
from tqdm import tqdm
import os;os.sys.path.append(os.path.expanduser('~') + "/cmem/code")
import os;os.sys.path.append(os.path.expanduser('~') + "/cmem/")
from utils import *

path0400_1files = readFromPath(path0400_1)
path0400Files = readFromPath(path0400)


def selectMidDay(i):
    # i=0
    file = pd.read_csv(path0400_1+path0400_1files[i],index_col=0)
    intraday_interval = np.array([0]*2+[1]*22+[2]*2)
    intradayInterval = np.tile(intraday_interval,(file.shape[0]//26,))
    file['intradayInterval']=intradayInterval
    df=file[file['intradayInterval']==1]
    df = df.reset_index(drop=True)
    return df

def compare2test(i,frequency_counts=22):
    name = path0400_1files[i]
    fore = selectMidDay(i)

    fore['turnover'] = fore.qty.shift(-1)
    fore['log_turnover'] = fore['turnover'].apply(np.log)
    new_result = fore.dropna()
    frequency_counts = new_result.date.value_counts()
    unique_dates = sorted(frequency_counts[frequency_counts==frequency_counts].index.tolist())
    newResult = new_result[new_result['date'].isin(unique_dates)]
    '''
    after the dropna here, the days shrinked from 109 to 108
    there should be no 110 => 109 shrink. it is of no need
    '''

    newResult = newResult.reset_index(drop=True)
    newResult.to_csv(path0600+path0400Files[i][:-3]+'csv')
    newResult.columns

    lst = []
    g=newResult.groupby("date")
    for index, item in g:
        pass
        itm = item[['turnover','x']]
        from sklearn.metrics import r2_score
        r2 = r2_score(itm.turnover,itm.x)
        lst.append([index,r2])
    newDf=pd.DataFrame(lst,columns = ['date',name[:-4]])
    newDf = newDf.set_index('date')
    return newDf

newDflist = []
from tqdm import tqdm
for i in tqdm(range(len(path0400_1files))):
    newDf = compare2test(i)
    newDflist.append(newDf)
def check_NewDf(NewDf):
    NewDf = pd.concat(newDflist,axis=1)
    r2= NewDf.mean(axis=1).mean()
    print(f"the kf-cmem have a oos r2:{r2}")
    # assert r2 >= 0.45, "the kf-cmem should have a oos r2 over 0.45"
