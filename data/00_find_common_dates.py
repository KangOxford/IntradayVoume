#


import numpy as np

import os; os.sys.path.append("/home/kanli/cmem/")
# path = "/home/kanli/cmem/data/01_raw/"
path = "/home/kanli/cmem/data/01.1_raw/"
import pandas as pd

pd.read_pickle("/home/kanli/cmem/data/01.1_raw/PKI.pkl")

a = pd.read_pickle(path + "AAP.pkl")
z = pd.read_pickle(path + "ZTS.pkl")
