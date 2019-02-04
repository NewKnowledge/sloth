import numpy
from keras.optimizers import Adagrad
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.shapelets import ShapeletModel, grabocka_params_to_shapelet_size_dict
from tslearn.utils import ts_size

import pandas as pd

# parameters:
    # epochs                : number of training epochs
    # length                : base shapelet length, expressed as fraction of length of time series
    # num_shapelet_lengths  : number of different shapelet lengths
class Shapelets():
    def __init__(self, X_train, y_train, epochs, length, num_shapelet_lengths, weight_regularizer, learning_rate):
        self.shapelet_sizes = grabocka_params_to_shapelet_size_dict(n_ts = X_train.shape[0], 
                    ts_sz = X_train.shape[1], 
                    n_classes = len(set(y_train)), 
                    l = length, 
                    r = num_shapelet_lengths)
        #self.shapelet_sizes = {33: 50, 66: 50, 99:50}
        print(self.shapelet_sizes)
        self.shapelet_clf = ShapeletModel(n_shapelets_per_size=self.shapelet_sizes,
                            optimizer=Adagrad(lr=learning_rate),
                            weight_regularizer=weight_regularizer,
                            max_iter=epochs,
                            verbose_level=0)

        # fit classifier
        self.shapelet_clf.fit(X_train, y_train)

    # parameters:
        # X_test            : test data on which to predict classes

    # returns:   class predictions for test data set
    def PredictClasses(self, X_test):
        return self.shapelet_clf.predict(X_test) 

    # parameters:
    def VisualizeShapelets(self):
        plt.figure()
        for i, sz in enumerate(self.shapelet_sizes.keys()):
            plt.subplot(len(self.shapelet_sizes), 1, i + 1)
            plt.title("%d shapelets of size %d" % (self.shapelet_sizes[sz], sz))
            for shapelet in self.shapelet_clf.shapelets_:
                if ts_size(shapelet) == sz:
                    plt.plot(shapelet.ravel())
            plt.xlim([0, max(self.shapelet_sizes.keys())])
        plt.tight_layout()
        plt.show() 

    # parameters:
        # X_test                : test data set
        # test_series_id        : id of test time series to visualize
    def VisualizeShapeletLocations(self, X_test, test_series_id):
        locations = self.shapelet_clf.locate(X_test)
        plt.figure()
        plt.title("Locations of shapelet matches (%d shapelets extracted) in test series %d" % (sum(self.shapelet_sizes.values()), test_series_id))
        plt.plot(X_test[test_series_id].ravel())
        for idx_shapelet, shapelet in enumerate(self.shapelet_clf.shapelets_):
            t0 = locations[test_series_id, idx_shapelet]
            plt.plot(numpy.arange(t0, t0 + len(shapelet)), shapelet, linewidth=2)
        plt.tight_layout()
        plt.show()

#   test using Trace dataset (Bagnall, Lines, Vickers, Keogh, The UEA & UCR Time Series
#   Classification Repository, www.timeseriesclassification.com
if __name__ == '__main__':

    # constants
    epochs = 200
    shapelet_length = 0.2
    num_shapelet_lengths = 3
    weight_regularizer = .01
    learning_rate = .01
    time_series_id = 0

    # create shapelets
    X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
    trace_shapelets = Shapelets(X_train, y_train, epochs, shapelet_length, num_shapelet_lengths, weight_regularizer, learning_rate)

    # test methods
    predictions = trace_shapelets.PredictClasses(X_test)
    print("Accuracy = ", accuracy_score(y_test, predictions))
    trace_shapelets.VisualizeShapelets()
    trace_shapelets.VisualizeShapeletLocations(X_test, time_series_id)

