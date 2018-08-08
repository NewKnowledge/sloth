import pandas as pd
import matplotlib.pyplot as plt

data =pd.read_csv("Electronic_Production.csv",index_col=0)
print(data.head())

data.index = pd.to_datetime(data.index)

data.columns = ['Energy Production']

#data.iplot(title="Energy Production Jan 1985--Jan 2018")
#print(data.values)

plt.figure()
plt.subplot(1, 1, 1)
plt.plot(data.values, "k-")
plt.xlabel("data point index")
plt.ylabel("energy production")
plt.title("Energy Production Jan 1939--June 2018")

plt.tight_layout()
plt.show()

#from plotly.plotly import plot_mpl
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(data, model="multiplicative")
fig = result.plot()
plt.show()

from pyramid.arima import auto_arima

stepwise_model = auto_arima(data, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

print(stepwise_model.aic())


train = data.loc['1985-01-01':'2016-12-01']
test = data.loc['2017-01-01':]

stepwise_model.fit(train)

future_forecast = stepwise_model.predict(n_periods=18)

print("Future forecast:")
print(future_forecast)

future_forecast = pd.DataFrame(future_forecast,index = test.index,columns=["Prediction"])

plt.subplot(2, 1, 1)
plt.plot(pd.concat([test,future_forecast],axis=1).values)
plt.xlabel("data point index")
plt.ylabel("energy production")
plt.title("Future Forecast")

plt.subplot(2, 1, 2)
plt.plot(pd.concat([data,future_forecast],axis=1).values)
plt.xlabel("data point index")
plt.ylabel("energy production")

plt.show()





