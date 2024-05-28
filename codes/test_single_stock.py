from tqdm import tqdm
import numpy as np
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
from trainPred import *
from get_results import get_r2df, get_r2df_ray
import multiprocessing
import time
import ray


def check_GPU_memory():
    import GPUtil
    # Get the list of GPU devices
    devices = GPUtil.getGPUs()
    # Loop through devices and print their memory usage
    for device in devices:
        print(f"Device: {device.id}, Free Memory: {device.memoryFree}MB, Used Memory: {device.memoryUsed}MB")

path060000Files = readFromPath(path060000_fractional_shares)
# path060000Files = readFromPath(path060000)
print(len(path060000Files))

path0702Files = readFromPath(path0702)
print(len(path0702Files))
path0702Files_filtered = list(filter(lambda x: x.endswith('.pkl'), path0702Files))
print(len(path0702Files_filtered))

def getSingleDfs(trainType):
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
    
    
    
    if trainType=="universal":
        print("getSingleDfs trainType:",trainType)
        df = pd.read_pickle(path0701+"one_file.pkl")
        dfs=[df]
        # num_of_stacked_stocks = 100
        num_of_stacked_stocks = 481
        return dfs, num_of_stacked_stocks
    # if trainType=="universal":
    #     print("getSingleDfs trainType:",trainType)
    #     df = pd.read_pickle(path0700+"universal.pkl")
    #     dfs=[df]
    #     # num_of_stacked_stocks = 100
    #     num_of_stacked_stocks = 483
    #     return dfs, num_of_stacked_stocks
    
    elif trainType=="single" or trainType=='clustered':
        print("getSingleDfs trainType:",trainType)
        dfs=[]
        # for i in tqdm(range(len(path0702Files_filtered))):
        #     # print(f">>> {i}")
        #     df=pd.read_pickle(path0702+path0702Files_filtered[i])
        for stock in tqdm(stock_names):
            # print(f">>> {i}")
            df=pd.read_pickle(path0702+stock+'.pkl')
            df=df.reset_index(drop=True)
            dfs.append(df)
        num_of_stacked_stocks = 1

        def truncate_by_bin_size(df):
            df = pd.concat([itm.iloc[-BIN_SIZE:,:] for idx, itm in df.groupby('date')]).reset_index(drop=True)
            return df
        dfs_truncated = [truncate_by_bin_size(df) for df in tqdm(dfs)]
        
        # stock_names=[itm[:-4] for itm in path0702Files_filtered]

        return dfs_truncated, num_of_stacked_stocks,stock_names
    else: raise NotImplementedError

def print_mean(df3):
    print(f">>>> stock mean: \n",df3.mean(axis=0))  # stock
    print(f">>>> date mean: \n",df3.mean(axis=1))   # date
    print(f">>>> aggregate mean: \n",df3.mean(axis=1).mean())

if __name__=="__main__":
    rayOn = True
    if rayOn:
        ray.shutdown()
        ray.init(num_cpus=32,object_store_memory=40*1e9)
    # /homes/80/kang/anaconda3/bin/python /homes/80/kang/cmem/codes/test_single_stock.py
    # regulator = "Lasso"
    # regulator = "XGB"

    regulator = "Inception"
    # regulator = "OLS"
    # regulator = "Ridge"
    # regulator = "CMEM"
    

    # trainType = "universal"
    trainType = "single"
    # trainType = "clustered"
    
    print(f'trainType {trainType}, regulator {regulator}')
    

    dfs,num_of_stacked_stocks = getSingleDfs(trainType)
    print("dfs,num_of_stacked_stocks:",len(dfs),num_of_stacked_stocks)
    
    if trainType=="single":
        rayOn = False
        if rayOn:

            ray.shutdown()
            ray.init(num_cpus=64,object_store_memory=40 * 1e9)    
            ids = [get_r2df_ray.remote(num=num_of_stacked_stocks,regulator=regulator,df=df) for idx, df in tqdm(enumerate(dfs), desc="get_r2df", total=len(dfs))]
            results = [ray.get(id_) for id_ in tqdm(ids)]
        else: 
            results=[]
            for idx, df in tqdm(enumerate(dfs), desc="get_r2df", total=len(dfs)):
                result = get_r2df(num=num_of_stacked_stocks,regulator=regulator,df=df)
                results.append(result)
        df3s=[result[0] for result in results]
        df33s=[result[1] for result in results]

        names = [name[:-4] for name in path0702Files_filtered]
        df3s_=pd.concat(df3s,axis=1)
        df3s_.columns = names
        for i in range(len(df33s)):df33s[i].stock_index = names[i] 
        df33s_=pd.concat(df33s,axis=0)
        print("df3_.mean(axis=0):",df3s_.mean(axis=0))
        print("df3_.mean(axis=0).mean():",df3s_.mean(axis=0).mean())
    elif trainType=="universal":
        df3s=[];df33s=[]
        # idx=0;df=dfs[0];num=num_of_stacked_stocks
        for idx, df in tqdm(enumerate(dfs), desc="get_r2df", total=len(dfs)):
            df3,df33 = get_r2df(num=num_of_stacked_stocks,regulator=regulator,df=df)
            total_r2 = df3.mean(axis=1).mean()
            print('total r2: ',df3.mean(axis=1).mean()) # all mean
            df3s.append(df3)
            df33s.append(df33)
        # num_stock=len(dfs)
        df3_ = pd.concat(df3s,axis=1)
        # df3_.columns=np.arange(num_sto ck)
        # pd.set_option('display.max_rows', None) 
        print("df3_.mean(axis=0):",df3_.mean(axis=0))
        print("df3_.mean(axis=0).mean():",df3_.mean(axis=0).mean())

        # Reload the list from the text file
    # elif trainType=='clustered':
        
    else: raise NotImplementedError
        
        
    
    # breakpoint()
    # print()
    
    # def get_df33_r2_close_interval_single_clipped_from26bin(df33s_):
    #     g=df33s_.groupby(['date','stock_index'])
    #     for (idx,idx2),itm in g:pass
    #     df33s_closeInterval = pd.concat([itm.iloc[-2:,:] for (idx,idx2),itm in df33s_.groupby(['date','stock_index'])]).reset_index(drop=True)
    #     r2_cross_dates_closeInterval=df33s_closeInterval.groupby('date').apply(lambda x:r2_score(x.true,x.pred))
    #     r2_cross_dates_closeInterval.mean()
    #     r2_cross_dates=df33s_.groupby('date').apply(lambda x:r2_score(x.true,x.pred))
    #     r2_cross_dates.mean()
    #     '''
    #     how is the performance, should save to csv and then
    #     compare with the df33 results comes from the single params
    #     '''
    #     path = '/homes/80/kang/cmem/output/df33s/'
    #     import os;os.system(f'mkdir {path}')
    #     df33s_.to_csv(path+'df33_single_close_interval_481_lasso_from26bin.csv')
    #     r2_cross_dates.to_csv(path+'r2_cross_dates_single_close_interval_481_lasso_from26bin.csv')
    #     df33s_closeInterval.to_csv(path+'df33_single_close_interval_481_lasso_clipped_from26bin.csv')
    #     r2_cross_dates_closeInterval.to_csv(path+'r2_cross_dates_single_close_interval_481_lasso_clipped_from26bin.csv')
    
    
    # def get_df33_r2_close_interval_single(df33s_):
    #     r2_cross_dates=df33s_.groupby('date').apply(lambda x:r2_score(x.true,x.pred))
    #     r2_cross_dates.mean()
    #     '''
    #     how is the performance, should save to csv and then
    #     compare with the df33 results comes from the single params
    #     '''
    #     path = '/homes/80/kang/cmem/output/df33s/'
    #     import os;os.system(f'mkdir {path}')
    #     df33s_.to_csv(path+'df33_single_close_interval_481_lasso.csv')
    #     r2_cross_dates.to_csv(path+'r2_cross_dates_single_close_interval_481_lasso.csv')

    
#     df3s=[];df33s=[]
#     for idx, df in tqdm(enumerate(dfs), desc="get_r2df", total=len(dfs)):
#         df3,df33 = get_r2df(num=num_of_stacked_stocks,regulator=regulator,df=df)
#         total_r2 = df3.mean(axis=1).mean()
#         print('total r2: ',df3.mean(axis=1).mean()) # all mean
#         df3s.append(df3)
#         df33s.append(df33)
#         # name = path0702Files_filtered[idx][:-4]
#         # df3.to_csv(path00 + "0802_r2df_single_day_"+str(1)+"_"+regulator+"_"+str(total_r2)[:6]+name+".csv", mode = 'w')
#         # df33.to_csv(path00 + "0802_r2df_single_day_"+str(1)+"_"+regulator+"_"+str(total_r2)[:6]+'_values_'+name+".csv", mode = 'w')
    
#     # num_stock=len(dfs)
#     df3_ = pd.concat(df3s,axis=1)
#     # df3_.columns=np.arange(num_sto ck)
#     # pd.set_option('display.max_rows', None) 
#     print("df3_.mean(axis=0):",df3_.mean(axis=0))
#     print("df3_.mean(axis=0).mean():",df3_.mean(axis=0).mean())
#     # Reload the list from the text file
    
#     def get_df33_r2_close_interval(df33):
#         r2_cross_dates=df33.groupby('date').apply(lambda x:r2_score(x.true,x.pred))
#         r2_cross_dates.mean()
#         '''
#         how is the performance, should save to csv and then
#         compare with the df33 results comes from the single params
#         '''
#         path = '/homes/80/kang/cmem/output/df33s/'
#         import os;os.system(f'mkdir {path}')
#         df33.to_csv(path+'df33_universal_close_interval_481_lasso.csv')
#         r2_cross_dates.to_csv(path+'r2_cross_dates_universal_close_interval_481_lasso.csv')
    
#     def give_name_and_save_file():
#         file_path = 'stock_names.txt'
#         reloaded_stock_names = []
#         with open(file_path, 'r') as f:
#             reloaded_stock_names = [line.strip() for line in f.readlines()]
#         reloaded_stock_names_truncated = reloaded_stock_names[:df3_.shape[1]]
#         df3_.columns = reloaded_stock_names_truncated
        
        
#         '''
#         df3_.mean(axis=0)[df3_.mean(axis=0)>df3_.mean(axis=0).quantile(q=0.25)].mean()
#         0.3993333265521051 the result is same to the previous research
#         means that there is no error in the codes
#         '''
        
#         df33_ = pd.concat(df33s,axis=0)
#         df33_.stock_index=np.tile(np.array(reloaded_stock_names_truncated).repeat(26),61)
#         # df33_.stock_index=np.tile(np.arange(df3_.shape[1]).repeat(26),61)
#         df33_.reset_index(drop=True)
#         from datetime import datetime
#         current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
#         df3_path = path00 + "0802_r2df_"+trainType+"_day_"+str(num_of_stacked_stocks)+"_"+regulator+"_"+str(total_r2)[:6]+f"_{current_datetime}"+".csv"
#         print("df3_path:",df3_path)
#         df3_.to_csv(df3_path, mode = 'w')
#         df33_path = path00 + "0802_r2df_"+trainType+"_day_"+str(num_of_stacked_stocks)+"_"+regulator+"_"+str(total_r2)[:6]+'_values_'+f"{current_datetime}"+".csv"
#         print("df33_path:",df33_path)
#         df33_.to_csv(df33_path, mode = 'w')

# # # %%
# # import  pandas as pd
# # df = pd.read_csv("/homes/80/kang/cmem/0802_r2df_universal_day_483_Lasso_0.4285_values_.csv")
# # # %%
# # lst=[]
# # g=df.groupby(['date','stock_index'])
# # for idx,itm in g:
# #     r2=r2_score(itm['true'],itm['pred'])
# #     lst.append([itm.date.iloc[0],itm.stock_index.iloc[0],r2])
# # # %%
# # df1=pd.DataFrame(lst,columns=['date','stock','r2'])
# # df2=df1.pivot(index='date',columns='stock')
# # # %%
# # df2.mean(axis=0).mean()
