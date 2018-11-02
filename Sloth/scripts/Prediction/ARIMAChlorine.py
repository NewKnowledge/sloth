import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

from pyramid.arima import auto_arima

from Sloth import Sloth

data = pd.read_csv("datasets/0000_train_ts.csv",index_col=0)
#data = data['sunspot.month'].dropna()
#data = data.loc[49:100]
print(data.head())

# clean data - set datetime, take temperature at hour 0, set index
#data = data.loc[data['hour'] == 0]
#data["date"] = pd.to_datetime(data['year'].map(str) + ' ' + data['month'].map(str) + ' ' + data['day'].map(str))
#data = data[['TEMP', 'date']]
#data = data.set_index('date')
# shift data to positive for multiplicative decomposition
#data['TEMP'] = data['TEMP'] - data['TEMP'].min() + 1
#data.index = pd.to_datetime(data.index)
#data.columns = ['Energy Production']

plt.figure()
plt.subplot(1, 1, 1)
plt.plot(data.index, data.values, "k-")
plt.xlabel("data point index")
plt.ylabel("Chlorine concentration")
plt.title("Chlorine concentration")

plt.tight_layout()
plt.show()

Sloth = Sloth()
result = Sloth.DecomposeSeriesSeasonal(data.index, data.values)
fig = result.plot()
plt.show()

train = data.loc[49:2700]
test = data.loc[2701:]

print("DEBUG:the size of test is:")
print(test.shape)

future_forecast = Sloth.PredictSeriesARIMA(train,test.shape[0],True, 144)

print("DEBUG::Future forecast:")
print(future_forecast)

future_forecast = pd.DataFrame(future_forecast,index = test.index, columns=["Prediction"])

plt.subplot(2, 1, 1)
plt.plot(pd.concat([test,future_forecast],axis=1).values)
plt.xlabel("data point index")
plt.ylabel("temperature")
plt.title("Future Forecast")

plt.subplot(2, 1, 2)
plt.plot(pd.concat([data,future_forecast],axis=1).values)
plt.xlabel("data point index")
plt.ylabel("temperature")
plt.show()