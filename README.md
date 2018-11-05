# Sloth
Sloth - Strategic Labeling Over Time Heuristics - Tools for time series analysis

This library contains two classes, **Sloth** and **Shapelets**. **Sloth** contains tools for time series analysis: generating a similarity matrix, [DBSCAN clustering], [HDBSCAN clustering], KMEANS clustering, KNN classification, [seasonal decomposition], and [ARIMA prediction]. **Shapelets** learns representative [time series subsequences] to use for time series classification. The shapelets that **Shapelets** learns are not unique solutions. 

## Available Functions

## Sloth

#### DecomposeSeriesSeasonal
Inputs = time series index, time series, (optionally) frequency of time series. Outputs = object with observed, trend, seasonal, and residual components. 

#### ScaleSeriesMeanVariance
Inputs = time series. Outputs = time series scaled with mean = 0, variance = 1.

#### ScaleSeriesMinMax
Inputs = time series, min, max. Outputs = time series scaled with min = min, max = max.

#### PredictSeriesArima
Inputs = data frame containing two columns, 1) time series index and 2) time series values, number of periods to predict in the future, whether the time series is seasonal, and (optionally) the period of seasonal differencing. Outputs = time series prediction for the number of periods to predict.

## Shapelets

#### PredictClasses
Inputs = time series. Output = classification of time series as one of time series in training data. 

#### VisualizeShapelets
Inputs = none. Output = graphs of learned shapelets

#### VisualizeShapeletLocations
Inputs = time series, id of time series from training data. Output = graph of learned shapelets superimposed on indexed time series from training data. 

[DBSCAN clustering]: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
[HDBSCAN clustering]: https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html
[seasonal decomposition]: https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.seasonal_decompose.html
[ARIMA prediction]: https://www.alkaline-ml.com/pyramid/modules/generated/pyramid.arima.auto_arima.html
[time series subsequences]: http://fs.ismll.de/publicspace/LearningShapelets/

