from Sloth.classify import Shapelets
from tslearn.datasets import CachedDatasets
from sklearn.metrics import accuracy_score

# constants
epochs = 200
shapelet_length = 0.1
num_shapelet_lengths = 2
time_series_id = 0

# create shapelets
X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
trace_shapelets = Shapelets(X_train, y_train, epochs, shapelet_length, num_shapelet_lengths)

# test methods
predictions = trace_shapelets.PredictClasses(X_test)
print("Accuracy = ", accuracy_score(y_test, predictions))
trace_shapelets.VisualizeShapelets()
trace_shapelets.VisualizeShapeletLocations(X_test, time_series_id)