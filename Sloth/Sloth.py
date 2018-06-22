import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from scipy import sparse
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from collections import Counter

class Simon:
    def __init__(self):
        # nothing to do

    def VisuallyCompareTwoSeries(self,series,i1,i2):
        fig = plt.figure()
        plt.title("Comparing Time Series")
        ax1 = fig.add_subplot(211)
        ax1.plot(np.arange(series.shape[1]-1), series.values[i1,1:])
        ax1.set_xlabel("time")
        ax1.set_ylabel("i1")
        ax2 = fig.add_subplot(212)
        ax2.plot(np.arange(series.shape[1]-1), series.values[i2,1:])
        ax2.set_xlabel("time")
        ax2.set_ylabel("i2")
        plt.show()

    def GenerateSimilarityMatrix(self,series):
        nrows,ncols = series.shape
        # now, compute the whole matrix of similarities
        SimilarityMatrix = np.zeros((nrows,nrows))
        print("Computing similarity matrix...")
        for j in np.arange(nrows):
            if j%10==0:
                print("Processing matrix row for time series "+str(j))
            try:
                row_series = series.values[j,1:]
                for i in np.arange(nrows):
                    try:
                        column_series = series.values[i,1:]
                        distance,path = fastdtw(row_series,column_series,dist=euclidean)
                        SimilarityMatrix[j,i] = distance
                    except Exception as e:
                        print(e)
                        pass
            except:
                pass
        print("DONE!")
        print("Similarity Matrix:")
        print(SimilarityMatrix)

        return SimilarityMatrix

    def ClusterSimilarityMatrix(self,SimilarityMatrix):
        # perform DBSCAN clustering
        db = DBSCAN(eps=20,min_samples=2,metric='precomputed')
        print("DBSCAN settings:")
        print(db)
        db.fit(SimilarityMatrix)
        labels = db.labels_
        nclusters = len(set(labels))-(1 if -1 in labels else 0)
        print("Number of clusters:",nclusters)

        cnt = Counter()
        for label in list(labels):
            cnt[label] += 1

        print("The cluster frequencies are:")
        print(cnt)
        print("The labels are:")
        print(labels)

        return nclusters, labels, cnt

    def SaveSimilarityMatrix(self,SimilarityMatrix):
        np.save("SimilarityMatrix",SimilarityMatrix)

    def SaveSparseSimilarityMatrix(self,SimilarityMatrix):
        # sometimes the following may make sense - create a sparse representation
        SimilarityMatrixSparse = sparse.csr_matrix(SimilarityMatrix)
        with open("SimilarityMatrixSparse.dat",'wb') as outfile:
            pickle.dump(SimilarityMatrixSparse,outfile,pickle.HIGHEST_PROTOCOL)

    def LoadSimilarityMatrix(self):
        SimilarityMatrix = np.load("SimilarityMatrix.npy")
        return SimilarityMatrix


# try  k-medoids clustering?
# try hierarchical clustering?
# try mds/PCA?