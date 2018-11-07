from Sloth import Shapelets
from Sloth import Sloth
from tslearn.datasets import CachedDatasets
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
np.random.seed(0)

# constants
epochs = 10000
shapelet_length = 0.1
num_shapelet_lengths = 2
time_series_id = 0
n_neighbors = 5

# load data - Diatom size reduction
train = pd.read_csv('datasets/DiatomSizeReduction_TRAIN.txt', delim_whitespace = True, header=None)
y_train = train[0].values
X_train = []
for i in range(0, train.shape[0]):
    X_train.append(pd.DataFrame(train.loc[i,1:train.shape[1]-1]).values)
X_train = np.asarray(X_train)
test = pd.read_csv('datasets/DiatomSizeReduction_TEST.txt', delim_whitespace = True, header=None)
y_test = test[0].values
X_test = []
for i in range(0, test.shape[0]):
    X_test.append(pd.DataFrame(test.loc[i,1:test.shape[1]-1]).values)
X_test = np.asarray(X_test)

trace_shapelets = Shapelets(X_train, y_train, epochs, shapelet_length, num_shapelet_lengths)
predictions_shapelets = trace_shapelets.PredictClasses(X_test)
print("Accuracy Shapelets = ", accuracy_score(y_test, predictions_shapelets))
Sloth = Sloth()
predictions_knn = Sloth.ClassifySeriesKNN(X_test, X_train, y_train, n_neighbors)
print("Accuracy Shapelets = ", accuracy_score(y_test, predictions_knn))

