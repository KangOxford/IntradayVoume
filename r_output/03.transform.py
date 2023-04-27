import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings;warnings.simplefilter("ignore", category=FutureWarning)
from os import listdir;from os.path import isfile, join
from r_output import Config
from sklearn.metrics import r2_score


pd.set_option('display.max_columns', None)

def get_predict():
    import platform # Check the system platform
    if platform.system() == 'Darwin':
        print("Running on MacOS")
        data_path = Config.r_output_datapath
        # out_path = Config.r_data_path
        # try:
        #     listdir(out_path)
        # except:
        #     import os;os.mkdir(out_path)
    elif platform.system() == 'Linux':
        print("Running on Linux")
        raise NotImplementedError
    else:print("Unknown operating system")

    onlyfiles = sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])
    pred = []
    for i in tqdm(range(len(onlyfiles))):
        file = onlyfiles[i]
        df = pd.read_csv(data_path + file,header=None)
        # set the first column as index
        df.set_index(df.columns[0], inplace=True)
        # extract the diagonal elements of the DataFrame as a numpy array
        diagonal_elements = np.diag(df.values)
        pred.append(diagonal_elements)
    predict = np.concatenate(pred,axis=0)
    return predict

def get_df():
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    import warnings;
    warnings.simplefilter("ignore", category=FutureWarning)
    from os import listdir;
    from os.path import isfile, join
    from data import Config

    pd.set_option('display.max_columns', None)

    import platform  # Check the system platform
    if platform.system() == 'Darwin':
        print("Running on MacOS")
        data_path = Config.r_data_path
    elif platform.system() == 'Linux':
        print("Running on Linux")
        raise NotImplementedError
    else:
        print("Unknown operating system")

    onlyfiles = sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])
    file = onlyfiles[0]
    df = pd.read_csv(data_path + file, sep='\t|\n', engine='python')
    return df

def get_result_data():
    df = get_df()
    df1 = df.iloc[26:,:]
    assert df1.shape[0] // 26 == df1.shape[0]/26
    predict = get_predict()
    df1['pred'] = predict
    selected_columns =['date', 'bin','turnover','pred']
    df1 = df1[selected_columns]
    return df1
df1 = get_result_data()
df2 = df1.copy()
df2['percentage'] = df2.pred/df2.turnover

r2_list = []
for i in range(df2.shape[0]-26):
    r2 = r2_score(df2.turnover[i:i+26], df2.pred[i:i+26])
    r2_list.append(r2)
r2_list += [np.nan] * 26
df2['r2'] = r2_list


r2_list = [ ]
groupped = df1.groupby('date')
for index, item in groupped:
    # if index == 20171006:
    #     print()
    r2_scored = r2_score(item.turnover,item.pred)
    r2_list.append([index,r2_scored])
r2_arr = np.array(r2_list)

