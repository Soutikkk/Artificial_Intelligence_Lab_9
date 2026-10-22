
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error

# Generate synthetic MA(1) time series
np.random.seed(42)

n = 200
theta = 0.60

noise = np.random.normal(0, 1, n)
series_values = np.zeros(n)

# MA(1): X(t) = noise(t) + theta * noise(t-1)
series_values[0] = noise[0]

for t in range(1, n):
    series_values[t] = noise[t] + theta * noise[t - 1]

series = pd.Series(series_values, name="MA1_Value")

print("First five observations:")
print(series.head())

# Stationarity test using ADF
result = adfuller(series)

print("\nADF Statistic:", result[0])
print("p-value:", result[1])

if result[1] < 0.05:
    print("The series appears to be stationary.")
else:
    print("The series may be non-stationary.")

# Plot the generated time series
plt.figure(figsize=(10, 4))
plt.plot(series, label="MA(1) Series")
plt.title("MA(1) Time Series")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.show()

# Fit MA(1) model: ARIMA(0, 0, 1)
model = ARIMA(series, order=(0, 0, 1))
fitted = model.fit()

print("\nMA(1) Model Summary:")
print(fitted.summary())

# Calculate in-sample MAE
fitted_values = fitted.fittedvalues

mae = mean_absolute_error(
    series.iloc[1:],
    fitted_values.iloc[1:]
)

print("\nIn-sample MAE:", mae)

# Forecast the next 20 observations
forecast_steps = 20
forecast = fitted.forecast(steps=forecast_steps)

forecast_index = np.arange(n, n + forecast_steps)

print("\nNext 20 Forecasts:")
print(forecast)

# Plot actual, fitted, and forecast values
plt.figure(figsize=(11, 5))

plt.plot(series.index, series, label="Actual Values")
plt.plot(series.index, fitted_values, label="Fitted Values")
plt.plot(
    forecast_index,
    forecast.to_numpy(),
    color="red",
    label="Forecast"
)

plt.axvline(
    x=n - 1,
    color="gray",
    linestyle="--",
    label="Forecast Start"
)

plt.title("MA(1) Forecasting")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.show()
