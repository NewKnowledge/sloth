import numpy
import pandas as pd
from keras.optimizers import Adagrad, Adam
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.shapelets import ShapeletModel, grabocka_params_to_shapelet_size_dict
from tslearn.utils import ts_size
from tslearn.neighbors import KNeighborsTimeSeriesClassifier

class Shapelets():
    def __init__(self, epochs, length, num_shapelet_lengths, num_shapelets, learning_rate, weight_regularizer):
        '''
            initialize shapelet hyperparameters

            hyperparameters:
                epochs                : number of training epochs
                length                : base shapelet length, expressed as fraction of length of time series
                num_shapelet_lengths  : number of different shapelet lengths
                num_shapelets         : number of unique shapelets to learn at each shapelet length, 
                                        expressed as fraction of length of time series
                learning rate         : learning rate of Keras optimizer
                weight regularizer    : weight regularization used when fitting model
        '''
        self.epochs = epochs
        self.length = length
        self.num_shapelet_lengths = num_shapelet_lengths
        self.num_shapelets = num_shapelets
        self.learning_rate = learning_rate
        self.weight_regularizer = weight_regularizer
        self.shapelet_sizes = None
        self.shapelet_clf = None
        self.encoder = LabelEncoder()

    def clear_session(self):
        try:
            assert(self.shapelet_clf is not None)
        except:
            raise ValueError("Cannot clear session that has not been initialized")
        self.shapelet_clf.clear_session()
        return

    def generate_model(self, series_length, nclasses):
        '''
            Generate structure of model used for Shapelet classifier
        '''
        if self.shapelet_clf is None:
            base_size = int(self.length * series_length)
            self.shapelet_sizes = {}
            for sz_idx in range(self.num_shapelet_lengths):
                shp_sz = base_size * (sz_idx + 1)
                self.shapelet_sizes[shp_sz] = int(self.num_shapelets * series_length)
            self.shapelet_clf = ShapeletModel(n_shapelets_per_size=self.shapelet_sizes,
                            optimizer=Adam(lr = self.learning_rate),
                            weight_regularizer=self.weight_regularizer,
                            max_iter=self.epochs)
        return self.shapelet_clf.generate_model(series_length, nclasses)

    def fit(self, X_train, y_train, source_dir = None, val_split = 0.3):
        '''
            fit shapelet classifier on training data

            parameters:
                X_train                : training time series
                y_train                : training labels
        ''' 
        if self.shapelet_clf is None:
            base_size = int(self.length * X_train.shape[1])
            self.shapelet_sizes = {}
            for sz_idx in range(self.num_shapelet_lengths):
                shp_sz = base_size * (sz_idx + 1)
                self.shapelet_sizes[shp_sz] = int(self.num_shapelets * X_train.shape[1])
            self.shapelet_clf = ShapeletModel(n_shapelets_per_size=self.shapelet_sizes,
                                optimizer=Adam(lr = self.learning_rate),
                                weight_regularizer=self.weight_regularizer,
                                max_iter=self.epochs)
        
        # scale training data to between 0 and 1
        X_train_scaled = self.__ScaleData(X_train)

        # fit classifier
        self.shapelet_clf.fit(X_train_scaled, y_train, source_dir, val_split)

    def __ScaleData(self, input_data):
        ''' 
            scale input data to range [0,1]

            parameters:
                input_data        : input data to rescale
        '''

        return TimeSeriesScalerMinMax().fit_transform(input_data)

    def predict(self, X_test):
        '''
            classifications for time series in test data set

            parameters:
                X_test:     test time series on which to predict classes

            returns: classifications for test data set
        '''
        X_test_scaled = self.__ScaleData(X_test)
        return self.shapelet_clf.predict(X_test_scaled) 

    def predict_proba(self, X_test):
        '''
            class probabilities for time series in test data set

            parameters:
                X_test:     test time series on which to predict classes

            returns: classifications for test data set
        '''
        X_test_scaled = self.__ScaleData(X_test)
        return self.shapelet_clf.predict_proba(X_test_scaled) 

    def encode(self, categories):
        '''
            fit label encoder on input categories. returns transformed categories
        '''
        self.encoder.fit(categories)
        return self.encoder.transform(categories)

    def decode(self, y_probs, p_threshold):
        '''
            decode prediction probabilities y_probs into prediction / confidence give p_threshold
        '''
        prob_max = np.amax(y_probs, axis = 1)
        prediction_indices = prob_max > p_threshold
        y_pred = np.zeros(y_probs.shape[0])
        y_pred[prediction_indices] = np.argmax(y_probs, axis = 1)[prediction_indices]
        y_preds = self.encoder.inverse_transform(y_pred)

        confidence = prob_max

        # reintepret confidence in binary case
        if y_probs.shape[1] == 1:
            confidence = (prob_max - p_threshold) / (y_pred - p_threshold)
        confidence = 0.5 + confidence / 2

        return y_preds, confidence

    def VisualizeShapelets(self):
        '''
            visualize all of shapelets learned by shapelet classifier
        '''
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

    def VisualizeShapeletLocations(self, X_test, test_series_id, series_size, num_bins, density):
        '''
            visualize shapelets superimposed on one of the test series

            parameters:
                X_test:             test data set
                test_series_id:     id of test time series to visualize  
                axis:               axis to duplicate to plot shapelets / ts on different y axes  
        '''
        X_test_scaled = self.__ScaleData(X_test)
        locations = self.shapelet_clf.locate(X_test_scaled)
        time_unit = series_size / num_bins / 60
        fig, ax1 = plt.subplots()
        if time_unit == 1:
            ax1.set_xlabel('Minute of the Hour')
        elif time_unit == 0.5:
            ax1.set_xlabel('Half Minute of the Hour')
        if density:
            ax1.set_ylabel('Email Density')
        else:
            ax1.set_ylabel('Emails per Second')
        plt.title("Locations of shapelet matches (%d shapelets extracted) in test series %d" % (sum(self.shapelet_sizes.values()), test_series_id))
        ax1.plot(X_test[test_series_id].ravel(), linewidth=1)
        ax2 = ax1.twinx()
        ax2.set_ylabel('Shapelet Feature Density')
        for idx_shapelet, shapelet in enumerate(self.shapelet_clf.shapelets_):
            t0 = locations[test_series_id, idx_shapelet]
            
            ax2.plot(numpy.arange(t0, t0 + len(shapelet)), shapelet, linewidth=2)
        fig.tight_layout()
        plt.show()

class Knn():
    def __init__(self, n_neighbors):
        '''
            initialize KNN class with dynamic time warping distance metric

            hyperparameters:
                n_neighbors           : number of neighbors on which to make classification decision
        '''
        self.n_neighbors = n_neighbors
        self.knn_clf = KNeighborsTimeSeriesClassifier(n_neighbors=n_neighbors, metric="dtw")

    def __ScaleData(self, input_data):
        ''' 
            scale input data to range [0,1]

            parameters:
                input_data        : input data to rescale
        '''

        return TimeSeriesScalerMinMax().fit_transform(input_data)

    def fit(self, X_train, y_train):
        '''
            fit KNN classifier on training data

            parameters:
                X_train                : training time series
                y_train                : training labels
        ''' 
        # scale training data to between 0 and 1
        X_train_scaled = self.__ScaleData(X_train)
        self.knn_clf.fit(X_train_scaled, y_train)
    
    def predict(self, X_test):
        '''
            classifications for time series in test data set

            parameters:
                X_test:     test time series on which to predict classes

            returns: classifications for test data set
        '''
        # scale test data to between 0 and 1
        X_test_scaled = self.__ScaleData(X_test)
        return self.knn_clf.predict(X_test_scaled) 

#   test using Trace dataset (Bagnall, Lines, Vickers, Keogh, The UEA & UCR Time Series
#   Classification Repository, www.timeseriesclassification.com
if __name__ == '__main__':

    # constants
    epochs = 200
    shapelet_length = 0.1
    num_shapelet_lengths = 2
    time_series_id = 0
    learning_rate = .01
    weight_regularizer = .01

    # create shapelets
    X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
    trace_shapelets = Shapelets(epochs, shapelet_length, num_shapelet_lengths, learning_rate, weight_regularizer)
    trace_shapelets.fit(X_train, y_train)

    # test methods
    predictions = trace_shapelets.predict(X_test)
    print("Shapelet Accuracy = ", accuracy_score(y_test, predictions))
    trace_shapelets.VisualizeShapelets()
    trace_shapelets.VisualizeShapeletLocations(X_test, time_series_id)

    # test KNN classifier
    knn_clf = Knn(n_neighbors = 3)
    knn_clf.fit(X_train, y_train)
    knn_preds = knn_clf.predict(X_test)
    print("KNN Accuracy = ", accuracy_score(y_test, knn_preds))

