import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

np.random.seed(42)

n = 200

eps = np.random.normal(0, 1, n)

x = np.zeros(n)

phi = 0.65
theta = 0.35

for t in range(1, n):
    x[t] = phi * x[t-1] + eps[t] + theta * eps[t-1]

series = pd.Series(x, name='Value')

print(series.head())

result = adfuller(series)

print('ADF statistic:', result[0])
print('p-value:', result[1])

plt.figure(figsize=(10, 4))
plt.plot(series)
plt.title('Stationary Time Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.grid(True)
plt.show()

model = ARIMA(series, order=(1, 0, 1))

fitted = model.fit()

print(fitted.summary())

fitted_values = fitted.fittedvalues

forecast = fitted.forecast(steps=20)

plt.figure(figsize=(10, 4))

plt.plot(series, label='Actual')

plt.plot(fitted_values, label='Fitted')

plt.plot(
    range(n, n + 20),
    forecast,
    label='Forecast'
)

plt.axvline(n - 1, linestyle='--')

plt.title('ARMA(1,1): Actual, Fitted and Forecast')
plt.xlabel('Time')
plt.ylabel('Value')

plt.legend()
plt.grid(True)
plt.show()

mae = np.mean(np.abs(series - fitted_values))

print('In-sample MAE:', mae)