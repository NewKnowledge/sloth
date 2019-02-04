from Sloth import Shapelets
from Sloth import Sloth
from tslearn.datasets import CachedDatasets
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
np.random.seed(0)

# constants
epochs = 10000
shapelet_length = 0.2
num_shapelet_lengths = 3
weight_regularizer = .01
learning_rate = .001
time_series_id = 0
n_neighbors = 5

# Shapelets Accuracy = 0.93 / 0.96 depending on shapelets
# KNN Accuracy = 0.96

'''
# load data - Coffee concentration from UCR
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

# Shapelets Accuracy = 0.87
# KNN Accuracy = 0.71
'''
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
'''

# load data - Chlorine concentration test
path = '/Users/jeffreygleason 1/Desktop/New Knowledge/Code/D3M/datasets/seed_datasets_current/66_chlorineConcentration/TRAIN/dataset_TRAIN/tables/learningData.csv'
training = pd.read_csv(path, index_col = 0)
y_train = training['label'].values
training_path = '/Users/jeffreygleason 1/Desktop/New Knowledge/Code/D3M/datasets/seed_datasets_current/66_chlorineConcentration/TRAIN/dataset_TRAIN/timeseries'
X_train = []
X_train.append(pd.read_csv(training_path + '/' + training['timeseries_file'][0], index_col = 0).values)
for i in range(1, len(training)):
    X_train.append(pd.read_csv(training_path + '/' + training['timeseries_file'][i], index_col = 0).values)
X_train = np.asarray(X_train)

path = '/Users/jeffreygleason 1/Desktop/New Knowledge/Code/D3M/datasets/seed_datasets_current/66_chlorineConcentration/TEST/dataset_TEST/tables/learningData.csv'
testing = pd.read_csv(path, index_col = 0)
y_test = pd.read_csv('/Users/jeffreygleason 1/Desktop/New Knowledge/Code/D3M/datasets/seed_datasets_current/66_chlorineConcentration/SCORE/targets.csv', index_col = 0).values.T[0]
testing_path = '/Users/jeffreygleason 1/Desktop/New Knowledge/Code/D3M/datasets/seed_datasets_current/66_chlorineConcentration/TEST/dataset_TEST/timeseries'
X_test = []
X_test.append(pd.read_csv(testing_path + '/' + testing['timeseries_file'][467], index_col = 0).values)
for i in range(468, 467+len(testing)):
    X_test.append(pd.read_csv(testing_path + '/' + testing['timeseries_file'][i], index_col = 0).values)
X_test = np.asarray(X_test)

trace_shapelets = Shapelets(X_train, y_train, epochs, shapelet_length, num_shapelet_lengths, weight_regularizer, learning_rate)
predictions_shapelets = trace_shapelets.PredictClasses(X_test)
np.savetxt("shapelet_predictions.csv", predictions_shapelets, delimiter="\n")
print("Accuracy Shapelets = ", accuracy_score(y_test, predictions_shapelets))
#Sloth = Sloth()
#predictions_knn = Sloth.ClassifySeriesKNN(X_test, X_train, y_train, n_neighbors)
#print("Accuracy KNN = ", accuracy_score(y_test, predictions_knn))