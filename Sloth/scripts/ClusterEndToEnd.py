import pandas as pd
from Sloth import Sloth

# Main script follows
LOAD = True # Flag for loading similarity matrix from file if it has been computed before
Sloth = Sloth()


if(LOAD):
    SimilarityMatrix = Sloth.LoadSimilarityMatrix()    
else:
    datapath = '/home/azunre/Documents/NewKnowledge/sloth-dev/data/UCR_TS_Archive_2015/synthetic_control/synthetic_control_TRAIN'
    series = pd.read_csv(datapath,dtype='float',header=None)
    nrows,ncols = series.shape
    SimilarityMatrix = Sloth.GenerateSimilarityMatrix(series)
    Sloth.SaveSimilarityMatrix(SimilarityMatrix)

nclusters, labels, cnt = Sloth.ClusterSimilarityMatrix(SimilarityMatrix)

# compare two time series visually
chosen_cluster = 0
print("The indices of the chosen_cluster are:")
chosen_cluster_indices = np.arange(len(list(labels)))[labels==chosen_cluster]
print(chosen_cluster_indices)
Sloth.VisuallyCompareTwoSeries(series,chosen_cluster_indices[0],chosen_cluster_indices[1])