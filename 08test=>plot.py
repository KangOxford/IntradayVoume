import numpy as np
import pandas as pd
from os import listdir;
from os.path import isfile, join;

path00 = "/Users/kang/CMEM/"
path01 = "/Users/kang/CMEM/data/01_raw/"
path01_1 = "/Users/kang/CMEM/data/01.1_raw/"
path02 = "/Users/kang/CMEM/data/02_r_input/"
path04 = "/Users/kang/CMEM/r_output/04_r_output_raw_data/"
path05 = "/Users/kang/CMEM/r_output/05_r_output_raw_pkl/"
path06 = '/Users/kang/CMEM/r_output/06_r_output_raw_pkl/'

r2df = pd.read_csv(path00 +"07_r2df.csv", index_col=0)
r2df['mean'] = r2df.mean(axis=1)
mean_row = pd.DataFrame(r2df.mean(axis=0), columns=['mean']).T
r2df = pd.concat([r2df, mean_row])
# r2df['mean'].mean()
mean_col = r2df['mean'][:-1]
mean_row = mean_row.iloc[:,:-1]
percentage = 0.95
# percentage = 0.90
# percentage = 0.85
# percentage = 0.80
# percentage = 0.10
# percentage = 0.05
top_95_percent_col = mean_col[mean_col >= mean_col.quantile(1 - percentage)]
date_mean = top_95_percent_col.mean()
top_95_percent_row = mean_row.apply(lambda row: row[row >= row.quantile(1 - percentage)], axis=1)
asset_mean = top_95_percent_row.mean(axis = 1)
print(date_mean, asset_mean)
