from statsmodels.tsa.seasonal import seasonal_decompose
from pyramid.arima import auto_arima

def DecomposeSeriesSeasonal(series_time_index,series, *frequency):
    data = pd.DataFrame(series,index = series_time_index,columns=["Series"])

    # use additive model if negative values in time series
    model = 'multiplicative'
    if (min(series) <= 0):
        model = 'additive'
    
    # call seasonal_decompose with optional frequency parameter
    if not frequency:
        if isinstance(series_time_index, pd.DatetimeIndex):
            return seasonal_decompose(data, model=model)
        else:
            return seasonal_decompose(data, model=model, freq=1)
    else:
        return seasonal_decompose(data, model=model, freq=frequency[0])

def FitSeriesARIMA(data, seasonal, *seasonal_differencing):
    # default: annual data
    if not seasonal_differencing:
        stepwise_model = auto_arima(data, start_p=1, start_q=1,
                        max_p=5, max_q=5, m=1,
                        seasonal=seasonal,
                        d=None, D=1, trace=True,
                        error_action='ignore',  
                        suppress_warnings=True, 
                        stepwise=True)
    # specified seasonal differencing parameter
    else:
        stepwise_model = auto_arima(data, start_p=1, start_q=1,
                        max_p=5, max_q=5, m=seasonal_differencing[0],
                        seasonal=seasonal,
                        d=None, D=1, trace=True,
                        error_action='ignore',  
                        suppress_warnings=True, 
                        stepwise=True)
    stepwise_model.fit(data)
    return stepwise_model

def PredictSeriesARIMA(arima_classifier, n_periods):
    future_forecast = arima_classifier.predict(n_periods=n_periods)
    return future_forecast