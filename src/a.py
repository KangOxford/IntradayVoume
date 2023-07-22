import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings;

warnings.simplefilter("ignore", category=FutureWarning)
from os import listdir;
from os.path import isfile, join
from r_output import Config
from sklearn.metrics import r2_score

pd.set_option('display.max_columns', None)


def platform():
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
        data_path = "/r_output/04_r_output_raw_data_10/"
    elif platform.system() == 'Linux':
        print("Running on Linux")
        raise NotImplementedError
    else:
        print("Unknown operating system")

    onlyfiles = sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])
    return data_path, onlyfiles

data_path, onlyfiles = platform()

for i in range(len(onlyfiles)):
    df = pd.read_csv(data_path + onlyfiles[i])




a = "/Users/kang/CMEM/data/02_r_input"
b = "/Users/kang/CMEM/r_output/04_r_output_raw_data"
c = '/Users/kang/CMEM/data/02_r_input_remained'
from os import listdir;from os.path import isfile, join
onlyfiles = lambda data_path: sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])
a1 = onlyfiles(a)
a2 = [file[:-4] for file in a1]
a2
b1 = onlyfiles(b)
b2 = [file[10:-4] for file in b1]
b2
c2 = [file  for file in a2 if file not in b2]
c1 = [file+".txt" for file in c2]
c1
# for file name in list c1 copy the file in dir a into dir c

import os
import shutil
for file_name in c1:
    try:
        src_file_path = os.path.join(a, file_name)
        dest_file_path = os.path.join(c, file_name)
        shutil.copy(src_file_path, dest_file_path)
    except:
        print(f"{file_name} not found")




import pandas as pd
a = "/home/kanli/cmem/r_output/04_1_rOuputFeatured_100/forecasts_A.csv"
a1 = pd.read_csv(a)
a1
b="/home/kanli/cmem/data/01.1_raw/A.pkl"
b1 = pd.read_pickle(b)
c = "/home/kanli/volume/02_raw_component/A.pkl"
c1 = pd.read_pickle(c)
c1
