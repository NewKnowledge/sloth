import numpy
from keras.optimizers import Adagrad
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.shapelets import ShapeletModel, grabocka_params_to_shapelet_size_dict
from tslearn.utils import ts_size

import pandas as pd

def ScaleData(input_data):
    TimeSeriesScalerMinMax().fit_transform(input_data)

# parameters:
# length                : base shapelet length, expressed as fraction of length of time series
# num_shapelet_lengths  : number of different shapelet lengths

# returns:  dictionary of shapelet lengths
def GetShapeletSizes(X_train, y_train, length, num_shapelet_lengths):
    return grabocka_params_to_shapelet_size_dict(n_ts = X_train.shape[0], 
                ts_sz = X_train.shape[1], 
                n_classes = len(set(y_train)), 
                l = length, 
                r = num_shapelet_lengths)

# parameters:
# shapelet_sizes        : dictionary of shapelet lengths
# epochs                : number of training epochs

# returns:  shapelet classifier fit on training data    
def LearnShapelets(X_train, y_train, shapelet_sizes, epochs):
    shapelet_clf = ShapeletModel(n_shapelets_per_size=shapelet_sizes,
                        optimizer=Adagrad(lr=.1),
                        weight_regularizer=.01,
                        max_iter=epochs,
                        verbose_level=0)
    shapelet_clf.fit(X_train, y_train)
    return shapelet_clf


