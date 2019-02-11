import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import sparse
import hdbscan
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from collections import Counter
from tslearn.metrics import sigma_gak, cdist_gak

def GenerateSimilarityMatrix(self,series):
    nrows,ncols = series.shape
    # now, compute the whole matrix of similarities 
    print("Computing similarity matrix...")
    try:
            distances = [[fastdtw(series[j,:], series[i,:],dist=euclidean)[0] for i in range(j, nrows)] for j in np.arange(nrows)]
    except Exception as e:
        print(e)
        pass

    SimilarityMatrix = np.array([[0]*(nrows-len(i)) + i for i in distances])
    SimilarityMatrix[np.tril_indices(nrows,-1)] = SimilarityMatrix.T[np.tril_indices(nrows,-1)]
    print("DONE!")

    return SimilarityMatrix

def ClusterSimilarityMatrix(self,SimilarityMatrix,eps,min_samples):
    # perform DBSCAN clustering
    db = DBSCAN(eps=eps,min_samples=min_samples,metric='precomputed')
    db.fit(SimilarityMatrix)
    labels = db.labels_
    nclusters = len(set(labels))-(1 if -1 in labels else 0)
    cnt = Counter()
    for label in list(labels):
        cnt[label] += 1

    return nclusters, labels, cnt

def HClusterSimilarityMatrix(self,SimilarityMatrix,min_samples,PLOT=False):
    # perform DBSCAN clustering
    hdb = hdbscan.HDBSCAN(min_cluster_size=min_samples,min_samples=min_samples,metric='precomputed')
    labels = hdb.fit_predict(SimilarityMatrix)
    nclusters = len(set(labels))-(1 if -1 in labels else 0)
    cnt = Counter()
    for label in list(labels):
        cnt[label] += 1
    if(PLOT):
        plt.figure()
        hdb.condensed_tree_.plot()
        plt.figure()
        hdb.single_linkage_tree_.plot(cmap='viridis',colorbar=True)

    return nclusters, labels, cnt

def SaveSimilarityMatrix(self,SimilarityMatrix,filename):
    np.save(filename,SimilarityMatrix)

def SaveSparseSimilarityMatrix(self,SimilarityMatrix,filename):
    # sometimes the following may make sense - create a sparse representation
    SimilarityMatrixSparse = sparse.csr_matrix(SimilarityMatrix)
    with open(filename,'wb') as outfile:
        pickle.dump(SimilarityMatrixSparse,outfile,pickle.HIGHEST_PROTOCOL)

def LoadSimilarityMatrix(self,filename):
    SimilarityMatrix = np.load(filename+'.npy')
    return SimilarityMatrix

# algorithm specifies which kmeans clustering algorithm to use form tslearn
# options are 'GlobalAlignmentKernelKMeans' and 'TimeSeriesKMeans'
def ClusterSeriesKMeans(self,series,n_clusters,algorithm = 'GlobalAlignmentKernelKMeans'):
    print(algorithm)
    assert algorithm == 'GlobalAlignmentKernelKMeans' or algorithm == 'TimeSeriesKMeans', \
        "algorithm must be one of \'GlobalAlignmentKernelKMeans\' or \'TimeSeriesKMeans\'"
    seed = 0
    np.random.seed(seed)
    if algorithm == 'TimeSeriesKMeans':
        km = TimeSeriesKMeans(n_clusters=n_clusters, n_init=20, verbose=True, random_state=seed)
    else:
        km = GlobalAlignmentKernelKMeans(n_clusters=n_clusters, sigma=sigma_gak(series), n_init=20, verbose=True, random_state=seed)
    y_pred = km.fit_predict(series)

    return y_pred