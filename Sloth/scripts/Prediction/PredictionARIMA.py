import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

from pyramid.arima import auto_arima

from Sloth import Sloth

#data =pd.read_csv("Electronic_Production.csv",index_col=0)
data = pd.read_csv("PRSA_data_2010.1.1-2014.12.31.csv",index_col=0)

# clean data - set datetime, take temperature at hour 0, set index
data = data.loc[data['hour'] == 0]
data["date"] = pd.to_datetime(data['year'].map(str) + ' ' + data['month'].map(str) + ' ' + data['day'].map(str))
data = data[['TEMP', 'date']]
data = data.set_index('date')
# shift data to positive for multiplicative decomposition
#data['TEMP'] = data['TEMP'] - data['TEMP'].min() + 1
print(data.head())
#data.index = pd.to_datetime(data.index)
#data.columns = ['Energy Production']

plt.figure()
plt.subplot(1, 1, 1)
plt.plot(data.index, data.values, "k-")
plt.xlabel("data point index")
#plt.ylabel("energy production")
plt.ylabel("temperature")
#plt.title("Energy Production Jan 1939--June 2018")
plt.title("Beijing Temperature 2010-2014")

plt.tight_layout()
plt.show()

Sloth = Sloth()
result = seasonal_decompose(data, model = 'additive')
fig = result.plot()
plt.show()

#train = data.loc['1985-01-01':'2016-12-01']
#test = data.loc['2017-01-01':]
train = data.loc['2010-01-01':'2014-10-31']
test = data.loc['2014-11-01':]

print("DEBUG:the size of test is:")
print(test.shape)

#future_forecast = Sloth.PredictSeriesARIMA(train,test.shape[0],False)

data = train
n_periods=test.shape[0]
seasonal=True
stepwise_model = auto_arima(data, start_p=2, start_q=2,
                           max_p=5, max_q=5, m=12,
                           start_P=0, seasonal=seasonal,
                           d=None, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
stepwise_model.fit(data)
future_forecast = stepwise_model.predict(n_periods=n_periods)

print("DEBUG::Future forecast:")
print(future_forecast)

future_forecast = pd.DataFrame(future_forecast,index = test.index,columns=["Prediction"])

plt.subplot(2, 1, 1)
plt.plot(pd.concat([test,future_forecast],axis=1).values)
plt.xlabel("data point index")
#plt.ylabel("energy production")
plt.ylabel("temperature")
plt.title("Future Forecast")

plt.subplot(2, 1, 2)
plt.plot(pd.concat([data,future_forecast],axis=1).values)
plt.xlabel("data point index")
#plt.ylabel("energy production")
plt.ylabel("temperature")

plt.show()