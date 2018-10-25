import pandas as pd
import matplotlib.pyplot as plt

from Sloth import Sloth

data =pd.read_csv("Electronic_Production.csv",index_col=0)
print(data.head())
data.index = pd.to_datetime(data.index)
data.columns = ['Energy Production']

plt.figure()
plt.subplot(1, 1, 1)
plt.plot(data.values, "k-")
plt.xlabel("data point index")
plt.ylabel("energy production")
plt.title("Energy Production Jan 1939--June 2018")

plt.tight_layout()
plt.show()

Sloth = Sloth()
result = Sloth.DecomposeSeriesSeasonal(data.index,data.values)
fig = result.plot()
plt.show()

train = data.loc['1985-01-01':'2016-12-01']
test = data.loc['2017-01-01':]

future_forecast = Sloth.PredictSeriesARIMA(train.index,train.values,18,True)

print("DEBUG::Future forecast:")
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