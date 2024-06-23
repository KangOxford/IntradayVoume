import numpy as np
from tqdm import tqdm
import pandas as pd
from os import listdir;
from os.path import isfile, join;
from sklearn.metrics import r2_score



import os;os.sys.path.append(os.path.expanduser('~') + "/cmem/codes/")
import os;os.sys.path.append(os.path.expanduser('~') + "/cmem/")
import os;os.sys.path.append("/homes/80/kang/cmem/")
import os;os.sys.path.append("/homes/80/kang/cmem/codes/")
from utils import *
from model import *
from dates import *
from get_results import BIN_SIZE, TRAIN_DAYS
import multiprocessing
import time




path0702Files = readFromPath(path0702)
print(len(path0702Files))
path0702Files_filtered = list(filter(lambda x: x.endswith('.pkl'), path0702Files))

def get_df_list(start_index, num):
    
    sector_stock = [
        ['CommunicationServices', ['TMUS', 'CHTR', 'ATVI', 'GOOGL', 'GOOG', 'FOXA', 'FOX', 'FB', 'NFLX', 'NWS', 'NWSA', 'OMC', 'EA', 'VZ', 'IPG', 'TWTR', 'DISH', 'DISCK', 'TTWO', 'CMCSA', 'DISCA', 'DIS', 'CTL', 'T']],
        ['ConsumerDiscretionary', ['HLT', 'HD', 'HBI', 'HAS', 'GRMN', 'GPS', 'GPC', 'GM', 'COTY', 'DRI', 'DHI', 'F', 'EXPE', 'HRB', 'EBAY', 'CMG', 'VFC', 'LB', 'KMX', 'TJX', 'TIF', 'TRIP', 'TSCO', 'SBUX', 'ROST', 'RL', 'RCL', 'PVH', 'UA', 'PHM', 'UAA', 'ORLY', 'NWL', 'NVR', 'NKE', 'ULTA', 'NCLH', 'MHK', 'MGM', 'MCD', 'MAR', 'M', 'LVS', 'LOW', 'LKQ', 'LEN', 'LEG', 'KSS', 'JWN', 'CCL', 'HOG', 'AMZN', 'BWA', 'AAP', 'WHR', 'BBY', 'AZO', 'WYNN']],
        ['ConsumerStaples', ['MNST', 'CPB', 'HRL', 'SJM', 'MKC', 'TSN', 'PEP', 'COST', 'EL', 'WBA', 'GIS', 'PG', 'DG', 'MDLZ', 'PM', 'ADM', 'LW', 'WMT', 'DLTR', 'HSY', 'TGT', 'TAP', 'SYY', 'K', 'CHD', 'KHC', 'CAG', 'MO', 'KMB', 'CL', 'STZ', 'CLX', 'KO', 'KR']],
        ['Energy', ['OXY', 'FANG', 'OKE', 'NOV', 'FTI', 'MPC', 'VLO', 'DVN', 'WMB', 'MRO', 'NBL', 'EOG', 'XEC', 'HES', 'HAL', 'SLB', 'PSX', 'HP', 'CVX', 'CXO', 'KMI', 'HFC', 'COG', 'APA', 'PXD']],
        ['Financials', ['KEY', 'WRB', 'FRC', 'USB', 'JPM', 'ICE', 'L', 'JKHY', 'IVZ', 'MS', 'AXP', 'WLTW', 'AON', 'MKTX', 'AMP', 'AMG', 'LNC', 'MA', 'ALL', 'HIG', 'MET', 'FITB', 'UNM', 'GPN', 'MCO', 'HBAN', 'GS', 'MMC', 'MSCI', 'CB', 'MTB', 'BLK', 'PYPL', 'FIS', 'DFS', 'RF', 'RJF', 'SCHW', 'SIVB', 'TRV', 'COF', 'TROW', 'SPGI', 'STT', 'CME', 'C', 'CMA', 'CINF', 'SYF', 'CFG', 'WFC', 'CBOE', 'BK', 'PRU', 'RE', 'AIG', 'ETFC', 'PBCT', 'NTRS', 'BAC', 'V', 'AJG', 'NDAQ', 'AFL', 'PFG', 'AIZ', 'ADS', 'BEN', 'PNC', 'PGR']],
        ['HealthCare', ['SYK', 'TFX', 'UHS', 'JNJ', 'ALGN', 'MYL', 'ABBV', 'MRK', 'MCK', 'PRGO', 'AMGN', 'LH', 'PFE', 'ABMD', 'LLY', 'ABT', 'RMD', 'MDT', 'REGN', 'PKI', 'UNH', 'ALXN', 'ISRG', 'ABC', 'A', 'BIIB', 'BMY', 'HUM', 'DHR', 'HSIC', 'CI', 'BAX', 'TMO', 'IDXX', 'EW', 'DGX', 'CNC', 'HCA', 'WCG', 'COO', 'CVS', 'GILD', 'VAR', 'WAT', 'HOLX', 'CERN', 'DVA', 'BDX', 'ANTM', 'VRTX', 'INCY', 'CAH', 'BSX', 'ILMN']],
        ['Industrials', ['BA', 'FAST', 'DE', 'RHI', 'DAL', 'FBHS', 'FDX', 'UTX', 'PWR', 'ROK', 'ROL', 'RSG', 'RTN', 'NLSN', 'ODFL', 'UAL', 'VRSK', 'EMR', 'ETN', 'FLS', 'ADP', 'DOV', 'EFX', 'PH', 'IR', 'BR', 'NSC', 'WAB', 'EXPD', 'PAYX', 'NOC', 'PCAR', 'TXT', 'MMM', 'CSX', 'LUV', 'CTAS', 'UNP', 'LMT', 'HON', 'SWK', 'LDOS', 'URI', 'KSU', 'IEX', 'CHRW', 'UPS', 'TDG', 'INFO', 'CAT', 'JCI', 'JBHT', 'AAL', 'ITW', 'CMI', 'AOS', 'AME', 'WM', 'HII', 'MAS', 'ALLE', 'FTV', 'PNR', 'ALK', 'GWW', 'ARNC', 'GE', 'SNA', 'GD', 'CPRT']],
        ['InformationTechnology', ['TEL', 'CTXS', 'ADBE', 'CDNS', 'TXN', 'CTSH', 'CDW', 'ROP', 'CRM', 'AAPL', 'WDC', 'SWKS', 'ACN', 'QCOM', 'QRVO', 'SNPS', 'ADI', 'STX', 'CSCO', 'IT', 'ANSS', 'ANET', 'MXIM', 'MU', 'FFIV', 'MSI', 'MSFT', 'FISV', 'FLIR', 'IBM', 'FLT', 'HPE', 'LRCX', 'AMD', 'FTNT', 'GLW', 'AMAT', 'AVGO', 'HPQ', 'MCHP', 'APH', 'AKAM', 'IPGP', 'ADSK', 'VRSN', 'JNPR', 'INTU', 'INTC', 'ORCL', 'KEYS', 'KLAC', 'NVDA', 'NTAP', 'NOW', 'WU']],
        ['Materials', ['CF', 'CE', 'LYB', 'IP', 'IFF', 'PKG', 'MOS', 'AVY', 'PPG', 'ECL', 'EMN', 'BLL', 'VMC', 'WRK', 'APD', 'NUE', 'ALB', 'FCX', 'FMC', 'SEE', 'SHW', 'MLM', 'NEM']],
        ['RealEstate', ['WY', 'VTR', 'VNO', 'UDR', 'IRM', 'SPG', 'EQIX', 'EQR', 'ESS', 'CCI', 'BXP', 'EXR', 'FRT', 'HST', 'KIM', 'DLR', 'MAA', 'MAC', 'AVB', 'ARE', 'SLG', 'SBAC', 'PSA', 'REG', 'AIV', 'O', 'AMT', 'PLD', 'DRE']],
        ['Utilities', ['AES', 'AWK', 'AEP', 'ATO', 'WEC', 'AEE', 'CMS', 'CNP', 'D', 'ES', 'DUK', 'SO', 'PPL', 'PNW', 'PEG', 'NRG', 'NI', 'DTE', 'NEE', 'FE', 'EXC', 'ETR', 'SRE', 'EIX', 'ED', 'LNT', 'XEL']],
    ]
    stock_names = [stock for sector,stocks in sector_stock for stock in stocks]
    num = len(stock_names)
    
    
    df_lst = []
    new_dflst_lst = []

    # for i in tqdm(range(start_index, start_index + num)):  # on mac4
    #     df = pd.read_pickle(path0702+path0702Files_filtered[i])
    #     df=df.reset_index(drop=True)
    #     df_lst.append(df)
        
    for stock in tqdm(stock_names):
        df=pd.read_pickle(path0702+stock+'.pkl')
        df=df.reset_index(drop=True)
        df_lst.append(df)
    

    d = generate_unusual_date(year=2017) 
    shape_lst = [df.shape[0] for df in df_lst]
    from statistics import mode
    try:
        mode_value = mode(shape_lst)
        print(f"The mode of the list is {mode_value}")
    except:
        print("No unique mode found")
    
    for index, dflst in enumerate(df_lst):
        if dflst.shape[0] == mode_value:
        # if dflst.shape[0] == 2834:
            dflst_filtered = dflst[~dflst['date'].isin(d)]
            new_dflst_lst.append(dflst_filtered)
    '''what is the meaning of dflst.shape to be 3146*109
    109 is the num of features
    dflst is for one single stock
    and across different dates
    26bins*121days==3146rows
    up to here, it is all right'''
    return new_dflst_lst



def get_universal_df(start_index, num):
    '''dflst_filtered this is a sample from the new_dflst_lst'''
    new_dflst_lst = get_df_list(start_index, num=469)
    gs=[[itm for date, itm in list(df.groupby("date") )] for df in new_dflst_lst]
    dflst_filtered =  new_dflst_lst[0]
    num_days = dflst_filtered.shape[0]//BIN_SIZE
    
    
    print(gs[468][0].iloc[-1,:])
    df =new_dflst_lst[468]
    
    '''this check seems to be unnecessary'''
    def truncate_df_wrt_bin_size(dflst_filtered):
        dflst_filtered = pd.concat([itm.iloc[-BIN_SIZE:,:] for idx,itm in dflst_filtered.groupby('date')])
        return dflst_filtered
    dflst_filtered =  truncate_df_wrt_bin_size(dflst_filtered)
    assert dflst_filtered.shape[0]//BIN_SIZE == dflst_filtered.shape[0]/BIN_SIZE
    '''this check seems to be unnecessary'''
    
    
    # num_days = dflst_filtered.shape[0]//26
    num_stocks = len(new_dflst_lst)
    # for i in tqdm(range(num_days+1)):
    dff = []
    for i in tqdm(range(num_days)):
        for j in range(num_stocks):
            group = gs[j][i].iloc[-BIN_SIZE:,:]    
            dff.append(group)
    df = pd.concat(dff, axis=0)
    df.reset_index(inplace=True, drop=True)
    print(">>> finish preparing the universal df")
    return df


    
        
    # gs = [dflst.iterrows() for dflst in new_dflst_lst]
    # dff = []
    # '''the way of stack is wrong, here it is stacked by perrows/bins.
    # but actually it should be stacked by per day'''
    # for i in tqdm(range(dflst_filtered.shape[0])):
    #     for g in gs:
    #         elem = next(g)[1].T
    #         dff.append(elem)
    # df = pd.concat(dff, axis=1).T
    # df.reset_index(inplace=True, drop=True)
    # print(">>> finish preparing the universal df")
    # return df



# if __name__=="__main__":    
#     df = get_universal_df(start_index=0, num=len(path060000Files))
#     tryMkdir(path0700)
#     df.to_csv(path0700+"universal.csv")
#     df.to_pickle(path0700+"universal.pkl")
def main1(path060000Files):
    df = get_universal_df(start_index=0, num=len(path060000Files))
    # breakpoint()
    # print("len(path060000Files):",len(path060000Files))
    tryMkdir(path0701)
    print("path0701:",path0701)
    df.to_csv(path0701+"one_file.csv")
    df.to_pickle(path0701+"one_file.pkl")
def main2(path060000Files):
    dfs,_ = get_df_list(start_index=0, num=len(path060000Files))
    print("len(path060000Files):",len(path060000Files))
    tryMkdir(path0702)
    print("path0702:",path0702)
    # breakpoint()
    # for idx,df in enumerate(dfs):
    for idx, df in tqdm(enumerate(dfs), desc="Processing files", total=len(dfs)):
        df.to_csv(path0702 + path060000Files[idx][:-4] + ".csv")
        df.to_pickle(path0702 + path060000Files[idx][:-4] + ".pkl")


if __name__=="__main__":   
    path0702Files = readFromPath(path0702)
    # path060000Files = readFromPath(path060000)
    print(len(path0702Files)) 
    main1(path0702Files)
    # path060000Files = readFromPath(path060000_fractional_shares)
    # # path060000Files = readFromPath(path060000)
    # print(len(path060000Files)) 
    # main1(path060000Files)
    # main2(path060000Files)
    
    
    