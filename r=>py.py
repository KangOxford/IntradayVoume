import numpy as np

path01 = "/Users/kang/CMEM/data/01_raw/"
path04 = "/Users/kang/CMEM/r_output/r_output_raw_data_10/"

from os import listdir;
from os.path import isfile, join;
import pandas as pd

readFromPath = lambda data_path: sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])
path01Files, path04Files = map(readFromPath, [path01, path04])


date_list = []
i = 0
for i in range(len(path01Files)):
    item = pd.read_csv(path04+path04Files[i])
