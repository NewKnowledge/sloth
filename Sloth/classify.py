def ClassifySeriesKNN(series,series_train,y_train,n_neighbors):
    knn_clf = KNeighborsTimeSeriesClassifier(n_neighbors=n_neighbors, metric="dtw")
    knn_clf.fit(series_train, y_train)
    predicted_labels = knn_clf.predict(series)

    return predicted_labels