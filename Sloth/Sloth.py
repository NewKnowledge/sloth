import numpy as np
import pandas as pd
#import matplotlib
#matplotlib.use('Agg') # uncomment for docker images
import matplotlib.pyplot as plt
import pickle
from scipy import sparse
import hdbscan
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from collections import Counter

from tslearn.clustering import TimeSeriesKMeans, GlobalAlignmentKernelKMeans
from tslearn.metrics import sigma_gak, cdist_gak
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesScalerMinMax

from statsmodels.tsa.seasonal import seasonal_decompose
from pyramid.arima import auto_arima

class Sloth:
    def __init__(self):
        pass # nothing to do

    def VisuallyCompareTwoSeries(self,series,i1,i2):
        fig = plt.figure()
        plt.title("Comparing Time Series")
        ax1 = fig.add_subplot(211)
        ax1.plot(np.arange(series.shape[1]-1), series.values[i1,1:])
        ax1.set_xlabel("time")
        ax1.set_ylabel(str(i1))
        ax2 = fig.add_subplot(212)
        ax2.plot(np.arange(series.shape[1]-1), series.values[i2,1:])
        ax2.set_xlabel("time")
        ax2.set_ylabel(str(i2))
        plt.show()

    def GenerateSimilarityMatrix(self,series):
        nrows,ncols = series.shape
        # now, compute the whole matrix of similarities
        print("Computing similarity matrix...")
        distances = [[fastdtw(series[j,:], series[i,:],dist=euclidean) for i in range(j, nrows)] for j in np.arange(nrows)]
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

    def ClassifySeriesKNN(self,series,series_train,y_train,n_neighbors):
        knn_clf = KNeighborsTimeSeriesClassifier(n_neighbors=n_neighbors, metric="dtw")
        knn_clf.fit(series_train, y_train)
        predicted_labels = knn_clf.predict(series)

        return predicted_labels
    
    def DecomposeSeriesSeasonal(self,series_time_index,series, *frequency):
        data = pd.DataFrame(series,index = series_time_index,columns=["Series"])

        # use additive model if negative values in time series
        model = 'multiplicative'
        if (min(series) <= 0):
            model = 'additive'
        
        # call seasonal_decompose with optional frequency parameter
        if not frequency:
            if isinstance(series_time_index, pd.DatetimeIndex):
                return seasonal_decompose(data, model=model)
            else:
                return seasonal_decompose(data, model=model, freq=1)
        else:
            return seasonal_decompose(data, model=model, freq=frequency[0])
    
    def ScaleSeriesMeanVariance(self, series):
        return TimeSeriesScalerMeanVariance(mu = 0, std = 1).fit_transform(series)

    def ScaleSeriesMinMax(self, series, min, max):
        return TimeSeriesScalerMinMax(min = min, max = max).fit_transform(series)
        
    def FitSeriesARIMA(self, data, seasonal, *seasonal_differencing):
        # default: annual data
        if not seasonal_differencing:
            stepwise_model = auto_arima(data, start_p=1, start_q=1,
                            max_p=5, max_q=5, m=1,
                            seasonal=seasonal,
                            d=None, D=1, trace=True,
                            error_action='ignore',  
                            suppress_warnings=True, 
                            stepwise=True)
        # specified seasonal differencing parameter
        else:
            stepwise_model = auto_arima(data, start_p=1, start_q=1,
                            max_p=5, max_q=5, m=seasonal_differencing[0],
                            seasonal=seasonal,
                            d=None, D=1, trace=True,
                            error_action='ignore',  
                            suppress_warnings=True, 
                            stepwise=True)
        stepwise_model.fit(data)
        return stepwise_model
    
    def PredictSeriesARIMA(self, arima_classifier, n_periods):
        future_forecast = arima_classifier.predict(n_periods=n_periods)
        return future_forecast

    
