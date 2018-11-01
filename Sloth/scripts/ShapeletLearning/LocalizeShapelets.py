import numpy
from keras.optimizers import Adagrad
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.shapelets import ShapeletModel, grabocka_params_to_shapelet_size_dict
from tslearn.utils import ts_size

import pandas as pd

numpy.random.seed(0)

# input stuff: X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")

datapath = 'post_frequency_garret_0924.csv'
series = pd.read_csv(datapath,header=0)

#X_train = series.values[:,1:].T
headers = list(series)[1:]

# concatenate random vectors
#X_train = numpy.concatenate((X_train,numpy.random.rand(6,X_train.shape[1])),axis=0)
#y_train = numpy.array([1,1,1,1,1,1,0,0,0,0,0,0])
# scaling

X_train = TimeSeriesScalerMinMax().fit_transform(X_train)

X_test = TimeSeriesScalerMinMax().fit_transform(X_test)
shapelet_sizes = grabocka_params_to_shapelet_size_dict(n_ts=X_train.shape[0],
                                                       ts_sz=X_train.shape[1],
                                                       n_classes=len(set(y_train)),
                                                       l=0.1,
                                                       r=2)

shp_clf = ShapeletModel(n_shapelets_per_size=shapelet_sizes,
                        optimizer=Adagrad(lr=.1),
                        weight_regularizer=.01,
                        max_iter=50,
                        verbose_level=0)
shp_clf.fit(X_train, y_train)

# predictions
predicted_labels = shp_clf.predict(X_test)
print("Correct classification rate:", accuracy_score(y_test, predicted_labels))

#X_test = X_train
predicted_locations = shp_clf.locate(X_test)

'''
test_ts_id = 0
plt.figure()
plt.title("Example locations of shapelet matches (%d shapelets extracted)" % sum(shapelet_sizes.values()))
plt.plot(X_test[test_ts_id].ravel())
for idx_shp, shp in enumerate(shp_clf.shapelets_):
    t0 = predicted_locations[test_ts_id, idx_shp]
    plt.plot(numpy.arange(t0, t0 + len(shp)), shp, linewidth=2)
plt.figure()
for i, sz in enumerate(shapelet_sizes.keys()):
   plt.subplot(len(shapelet_sizes), 1, i + 1)
   plt.title(“%d shapelets of size %d” % (shapelet_sizes[sz], sz))
   for shp in shp_clf.shapelets_:
       if ts_size(shp) == sz:
           plt.plot(shp.ravel())
   plt.xlim([0, max(shapelet_sizes.keys()) - 1])plt.tight_layout()
plt.show()
plt.tight_layout()
plt.show()
'''

plt.figure()
for i, sz in enumerate(shapelet_sizes.keys()):
   plt.subplot(len(shapelet_sizes), 1, i + 1)
   plt.title("%d shapelets of size %d" % (shapelet_sizes[sz], sz))
   for shp in shp_clf.shapelets_:
       if ts_size(shp) == sz:
           plt.plot(shp.ravel())
   plt.xlim([0, max(shapelet_sizes.keys()) - 1])
plt.tight_layout()
plt.show()