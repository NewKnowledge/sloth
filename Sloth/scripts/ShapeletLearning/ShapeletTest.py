from Sloth import Shapelets
from tslearn.datasets import CachedDatasets
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
np.random.seed(0)

# constants
epochs = 10000
shapelet_length = 0.175
num_shapelet_lengths = 2
time_series_id = 0

# load data - Cached datasets
#X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
# load data - tweets
'''
datapath = 'post_frequency_garret_0924.csv'
series = pd.read_csv(datapath, header = 0)
X_train = series.values[:,1:].T
X_train = numpy.concatenate((X_train, numpy.random.rand(6,X_train.shape[1])), axis=0)
y_train = numpy.array([1,1,1,1,1,1,0,0,0,0,0,0])
'''

# load data - Chlorine concentration test

path = 'datasets/66_chlorineConcentration/TRAIN/dataset_TRAIN/tables/learningData.csv'
training = pd.read_csv(path, index_col = 0)
y_train = training['label'].values
training_path = 'datasets/66_chlorineConcentration/TRAIN/dataset_TRAIN/timeseries'
X_train = []
X_train.append(pd.read_csv(training_path + '/' + training['timeseries_file'][0], index_col = 0).values)
for i in range(1, len(training)):
    X_train.append(pd.read_csv(training_path + '/' + training['timeseries_file'][i], index_col = 0).values)
X_train = np.asarray(X_train)

path = 'datasets/66_chlorineConcentration/TEST/dataset_TEST/tables/learningData.csv'
testing = pd.read_csv(path, index_col = 0)
y_test = pd.read_csv('datasets/66_chlorineConcentration/SCORE/targets.csv', index_col = 0).values.T[0]
testing_path = 'datasets/66_chlorineConcentration/TEST/dataset_TEST/timeseries'
X_test = []
X_test.append(pd.read_csv(testing_path + '/' + testing['timeseries_file'][467], index_col = 0).values)
for i in range(468, 467+len(testing)):
    X_test.append(pd.read_csv(testing_path + '/' + testing['timeseries_file'][i], index_col = 0).values)
X_test = np.asarray(X_test)

# load data - Coffee concentration from UCR
'''
train = pd.read_csv('datasets/Coffee_TRAIN.txt', delim_whitespace = True, header=None)
y_train = train[0].values
X_train = []
for i in range(0, train.shape[0]):
    X_train.append(pd.DataFrame(train.loc[i,1:train.shape[1]-1]).values)
X_train = np.asarray(X_train)
test = pd.read_csv('datasets/Coffee_TEST.txt', delim_whitespace = True, header=None)
y_test = test[0].values
X_test = []
for i in range(0, test.shape[0]):
    X_test.append(pd.DataFrame(test.loc[i,1:test.shape[1]-1]).values)
X_test = np.asarray(X_test)
'''

# load data - Diatom size reduction
'''
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
'''
# create shapelets
trace_shapelets = Shapelets(X_train, y_train, epochs, shapelet_length, num_shapelet_lengths)

# test methods
predictions = trace_shapelets.PredictClasses(X_test)
#print(predictions)
#print(y_test)
#print(pd.DataFrame(predictions))
#print(np.unique(predictions))
print("Accuracy = ", accuracy_score(y_test, predictions))
trace_shapelets.VisualizeShapelets()
trace_shapelets.VisualizeShapeletLocations(X_test, time_series_id)