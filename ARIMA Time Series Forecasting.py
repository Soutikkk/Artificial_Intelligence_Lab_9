import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Generate sample data
np.random.seed(42)

n = 100
data = np.cumsum(np.random.normal(0, 1, n))

series = pd.Series(data, name="Value")

print("First five values:")
print(series.head())

# Check stationarity
result = adfuller(series)

print("ADF Statistic:", result[0])
print("P-value:", result[1])

# Plot original data
plt.figure(figsize=(10, 4))
plt.plot(series)
plt.title("Original Time Series")
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True)
plt.show()

# Fit ARIMA model
model = ARIMA(series, order=(1, 1, 1))
fitted = model.fit()

print(fitted.summary())

# Forecast next 10 values
forecast = fitted.forecast(steps=10)

print("Future Forecast:")
print(forecast)

# Plot actual data and forecast
plt.figure(figsize=(10, 4))
plt.plot(series, label="Actual")
plt.plot(
    range(n, n + 10),
    forecast,
    label="Forecast",
    color="red"
)

plt.title("ARIMA(1,1,1) Forecast")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()
