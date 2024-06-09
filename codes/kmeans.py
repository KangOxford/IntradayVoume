import numpy as np
import pandas as pd
from tqdm import tqdm
import os;os.sys.path.append(os.path.expanduser('~') + "/cmem/codes/")
import os;os.sys.path.append(os.path.expanduser('~') + "/cmem/")
import os;os.sys.path.append("/homes/80/kang/cmem/")
import os;os.sys.path.append("/homes/80/kang/cmem/codes/")
from utils import *
from model import *
import multiprocessing
import time

path0600_1Files = readFromPath(path0600_1)
print(len(path0600_1Files))


def get_features(new_dflst_lst,x_list,stock_names,type="volume"):
    if type =="volume":
        return get_volume_features(new_dflst_lst,x_list,stock_names)
    elif type =="features":
        return get_features_features(new_dflst_lst,x_list,stock_names)
    else:
        raise NotImplementedError

def get_volume_features(new_dflst_lst,x_list,stock_names):
    feature_list = []
    for index, item in enumerate(new_dflst_lst):
        pass
        item.date = item.date.astype(np.int32)
        item = item.set_index('date')
        value = item.turnover
        value.name = path0600_1Files[index][:-4]
        feature_list.append(value)
    features = pd.concat(feature_list,axis = 1)
    return features

def get_features_features(new_dflst_lst,x_list,stock_names):
    nfeatures = []
    for col in x_list:
        feature_list = []
        for index, item in enumerate(new_dflst_lst):
            pass
            item.columns
            item.date = item.date.astype(np.int32)
            item = item.set_index('date')
            value = item.loc[:,col]
            # assert item.shape[1] == 108
            value.name = path0600_1Files[index][:-4]
            feature_list.append(value)
        features = pd.concat(feature_list,axis = 1)
        nfeatures.append(features)
    nfeatures = np.stack(nfeatures)
    len(nfeatures.shape)
    nfeatures.shape
    return nfeatures

def get_corr_matrix(train_start_Index, train_end_Index, features, stock_names):
    from scipy.stats import rankdata
    if len(features.shape) ==2:
        '''correlation matrix from volume'''
        f = features.iloc[train_start_Index:train_end_Index,:]
        fv=f.values
        print(f"shape of features: {features.shape}")
        
           
        # ms=[]        
        # for i in range(111):
        #     train_start_Index=26*i
        #     train_end_Index=26*(i+1)
        #     f = features.iloc[train_start_Index:train_end_Index,:]
        #     fv=f.values
        #     corr_matrix = np.corrcoef(np.apply_along_axis(rankdata, 0, fv), rowvar=False)
        #     ms.append(corr_matrix)
        # msa=np.array(ms)
        
        # corr_matrix = np.mean(msa,axis=0)
        # corr_matrix = np.median(msa,axis=0)
        
        
        # file_path = "/homes/80/kang/cmem/corr_matrix_volume.csv"
        # # file_path = "/homes/80/kang/cmem/corr_matrix_features.csv"
        # if os.path.exists(file_path):
        #     os.remove(file_path)
        # np.savetxt(file_path, corr_matrix, delimiter=",")
            
        
        # ranked_fv = np.apply_along_axis(rankdata, 1, fv)
        corr_matrix = np.corrcoef(np.apply_along_axis(rankdata, 0, fv), rowvar=False)
        
        # corr_matrix = np.corrcoef(fv, rowvar=False)
        # Print the shape of the correlation matrix
        print("Shape of correlation matrix:", corr_matrix.shape)
        return corr_matrix
    elif len(features.shape) ==3:
        '''correlation matrix from features'''
        print(f"shape of features: {features.shape}")
        nfeatures = features
        f = np.array([nfeatures[i,train_start_Index:train_end_Index,:] for i in range(nfeatures.shape[0])])
        f.shape
        
        
        
        
        # ms=[]        
        # for i in range(111):
        #     train_start_Index=26*i
        #     train_end_Index=26*(i+1)
        #     nfeatures = features
        #     f = np.array([nfeatures[i,train_start_Index:train_end_Index,:] for i in range(nfeatures.shape[0])])
        #     ncorr_matrix = np.array([np.corrcoef(np.apply_along_axis(rankdata, 0, fv), rowvar=False) for fv in f])
        #     corr_matrix = np.mean(ncorr_matrix,axis=0)
        #     ms.append(corr_matrix)
        # msa=np.array(ms)
        
        # corr_matrix = np.mean(msa,axis=0)
        # corr_matrix = np.median(msa,axis=0)
        
        
        # file_path = "/homes/80/kang/cmem/corr_matrix_features.csv"
        # if os.path.exists(file_path):
        #     os.remove(file_path)
        # np.savetxt(file_path, corr_matrix, delimiter=",")
        
        
        
        
        # ncorr_matrix = np.array([np.corrcoef(np.apply_along_axis(rankdata, 1, fv), rowvar=True) for fv in f])
        # ncorr_matrix = np.array([np.corrcoef(np.apply_along_axis(rankdata, 0, fv), rowvar=True) for fv in f])
        
        ncorr_matrix = np.array([np.corrcoef(np.apply_along_axis(rankdata, 0, fv), rowvar=False) for fv in f])
        ncorr_matrix.shape
        corr_matrix = np.mean(ncorr_matrix,axis=0)
        corr_matrix
        
        # Print the shape of the correlation matrix
        # print("Shape of correlation matrix:", corr_matrix.shape)
        return corr_matrix
    else:
        raise NotImplementedError
def get_labels_byPCA(corr_matrix,ratio_cumsum,n_components,n_clusters):
    from sklearn.decomposition import PCA
    pca = PCA()
    # Fit PCA on the correlation matrix
    # print(corr_matrix)
    pca.fit(corr_matrix)
    # Obtain the principal
    ratio=pca.explained_variance_ratio_
    # pca=PCA(n_components=40)
    # pca=PCA(n_components=100)
    # pca=PCA(n_components=np.argmax(ratio.cumsum() >= 0.9999))
    # pca=PCA(n_components=np.argmax(ratio.cumsum() >= 0.99))
    def provide_eigen_values(pca):
        # Get the eigenvalues
        eigenvalues = pca.explained_variance_
        with open('/homes/80/kang/cmem/output/eigenvalues_features.txt', 'a') as f:
            f.write(', '.join(map(str, eigenvalues.tolist())))
            f.write('\n')
    provide_eigen_values(pca)
    
    if ratio_cumsum == 1.00:
        pca = PCA(n_components=n_components)
        print("n_components 100")
    else:
        pca = PCA(n_components=np.argmax(ratio.cumsum() >= ratio_cumsum))
        print(f"n_components {np.argmax(ratio.cumsum() >= ratio_cumsum)}")
    pca.fit(corr_matrix)
    scores_pca = pca.transform(corr_matrix)
    print(f'the shape of the scores_pca is {scores_pca.shape}')

    from sklearn.cluster import KMeans
    kmeans_pca = KMeans(n_clusters=n_clusters, init="k-means++",random_state=42)
    
    kmeans_pca.fit(scores_pca)
    assert kmeans_pca.labels_.shape == (n_components,)
    labels = kmeans_pca.labels_
    return labels

# # n_clusters = 2
# # n_clusters = 5
# n_clusters = 10
# # n_clusters = 20
# # n_clusters = 50


# ratio_cumsum = 0.80
# # ratio_cumsum = 0.99
# # ratio_cumsum = 0.9999
# # ratio_cumsum = 1.00
# features = get_features(new_dflst_lst,type="volume")
# # features = get_features(new_dflst_lst,type="features")
# features.shape





# groupped_dfs = [new_dflst_lst[i] for i in list_]
# gs = [dflst.iterrows() for dflst in groupped_dfs]
# dff = []
# for i in range(one_stock_shape):
# # for i in tqdm(range(one_stock_shape)):
#     for g in gs:
#         elem = next(g)[1].T
#         dff.append(elem)
# df = pd.concat(dff, axis=1).T
# df.reset_index(inplace=True, drop=True)
# # print(df.shape)

# num = len(groupped_dfs)