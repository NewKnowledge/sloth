import pandas as pd
import numpy as np
from Sloth import Sloth

import matplotlib
#matplotlib.use('TkAgg')
#matplotlib.use('Agg')
import matplotlib.pyplot as plt


from tslearn.preprocessing import TimeSeriesScalerMeanVariance

Sloth = Sloth()
datapath = 'post_frequency_8.09_8.15.csv'
series = pd.read_csv(datapath,dtype='str',header=0)

#print("DEBUG::post frequency data:")
#print(series)

# scaling can sometimes improve performance
X_train = series.values[:,1:]

#X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)

n_samples = 100
X_train = X_train[:n_samples]
X_train = X_train.astype(np.float)

X_train = X_train.reshape((X_train.shape[0],X_train.shape[1]))

nrows,ncols = X_train.shape

print("DEBUG::shape of final data for clustering:")
print(X_train.shape)

## this is the first clustering method, via dbscan
# some hyper-parameters
eps = 90
min_samples = 2
LOAD = True # Flag for loading similarity matrix from file if it has been computed before
if(LOAD):
    SimilarityMatrix = Sloth.LoadSimilarityMatrix()    
else:
    SimilarityMatrix = Sloth.GenerateSimilarityMatrix(X_train[:,1:])
    Sloth.SaveSimilarityMatrix(SimilarityMatrix)

nclusters, labels, cnt = Sloth.ClusterSimilarityMatrix(SimilarityMatrix,eps,min_samples)

print("The cluster frequencies are:")
print(cnt)
        
## try hierarchical clustering
import hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=2,metric="precomputed",min_samples=min_samples)
labels = clusterer.fit_predict(SimilarityMatrix)
nclusters = len(set(labels))-(1 if -1 in labels else 0)
from collections import Counter
cnt = Counter()
for label in list(labels):
    cnt[label] += 1

clusterer.condensed_tree_.plot()
plt.figure()
clusterer.single_linkage_tree_.plot(cmap='viridis',colorbar=True)




print("The hcluster frequencies are:")
print(cnt)

## this is the second clustering method, using tslearn kmeans
nclusters = 10
labels = Sloth.ClusterSeriesKMeans(X_train,nclusters)
nclusters = len(set(labels))-(1 if -1 in labels else 0)
from collections import Counter
cnt = Counter()
for label in list(labels):
    cnt[label] += 1

print("The k-means frequencies are:")
print(cnt)

series_np = X_train

cnt_nontrivial = {x:cnt[x] for x in cnt if cnt[x]>1 and x!=-1}

plt.figure()
idx = 0
for yi in cnt_nontrivial.keys():
    plt.subplot(len(cnt_nontrivial), 1, 1 + idx)
    for xx in series_np[labels == yi]:
        plt.plot(xx.ravel(), "k-")
    plt.xlim(0, ncols)
    plt.title("Cluster %d: %d series" %(yi,cnt[yi]))
    idx = idx+1

clust = 4
print("DEBUG::anomalous series:")
print(series_np[labels==clust])
print(series.values[:n_samples,0][labels==clust])

print("DEBUG::AniyaHadlee cluster:")
print(labels[series.values[:n_samples,0]=='AniyaHadlee'])
print("DEBUG::BarackDonovan cluster:")
print(labels[series.values[:n_samples,0]=='BarackDonovan'])

print("DEBUG::cluster anomaly series:")
print(series_np[labels==clust])
print(series.values[:n_samples,0][labels==clust])

plt.tight_layout()
plt.show()