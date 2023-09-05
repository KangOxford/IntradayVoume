from tqdm import tqdm
import numpy as np
import pandas as pd
from os import listdir;
from os.path import isfile, join;
from sklearn.metrics import r2_score


import numpy as np
import pandas as pd
from tqdm import tqdm
import os;os.sys.path.append(os.path.expanduser('~') + "/cmem/codes/")
import os;os.sys.path.append(os.path.expanduser('~') + "/cmem/")
import os;os.sys.path.append("/homes/80/kang/cmem/")
import os;os.sys.path.append("/homes/80/kang/cmem/codes/")
from utils import *
from model import *
from params import *
from trainPred import *
from get_results import get_r2df
import multiprocessing
import time