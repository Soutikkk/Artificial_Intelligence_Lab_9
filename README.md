# ARMA(1,1) Time Series Analysis and Forecasting

This project demonstrates how to generate, analyze, model, and forecast a **stationary time series** using an **ARMA(1,1)** model in Python.

The project uses **NumPy**, **Pandas**, **Matplotlib**, and **Statsmodels** to generate synthetic time-series data, test its stationarity, fit an ARMA model, generate forecasts, and evaluate the model using Mean Absolute Error (MAE).

## 📌 Objective

The main objectives of this project are to:

- Generate a synthetic stationary time series.
- Simulate an **ARMA(1,1)** process.
- Check stationarity using the **Augmented Dickey-Fuller (ADF) Test**.
- Visualize the generated time series.
- Fit an ARMA(1,1) model using `ARIMA(1,0,1)`.
- Compare actual and fitted values.
- Forecast the next **20 time steps**.
- Evaluate model performance using **Mean Absolute Error (MAE)**.

## 🛠️ Technologies and Libraries Used

- Python
- NumPy
- Pandas
- Matplotlib
- Statsmodels

## 📦 Installation

Install the required Python libraries using:

```bash
pip install numpy pandas matplotlib statsmodels
```

## 📥 Importing Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
```

## 📊 Dataset Generation

The program generates **200 observations** of a synthetic ARMA(1,1) time series.

The parameters used are:

```python
phi = 0.65
theta = 0.35
```

The time series is generated using:

```text
X(t) = φX(t-1) + ε(t) + θε(t-1)
```

where:

- `φ = 0.65` is the autoregressive coefficient.
- `θ = 0.35` is the moving-average coefficient.
- `ε(t)` represents random white noise.

A random seed is used to make the generated data reproducible:

```python
np.random.seed(42)
```

## 🔍 Stationarity Test

The **Augmented Dickey-Fuller (ADF) Test** is used to determine whether the generated time series is stationary.

```python
result = adfuller(series)

print('ADF statistic:', result[0])
print('p-value:', result[1])
```

### Interpretation

- **p-value < 0.05** → Reject the null hypothesis → The series is considered stationary.
- **p-value ≥ 0.05** → Fail to reject the null hypothesis → The series may be non-stationary.

## 📈 Time Series Visualization

The generated time series is plotted using Matplotlib.

```python
plt.plot(series)
```

This helps visualize the fluctuations and behavior of the stationary time series over time.

## 🤖 ARMA(1,1) Model

The ARMA(1,1) model is implemented using the `ARIMA` class:

```python
model = ARIMA(series, order=(1, 0, 1))
fitted = model.fit()
```

The order `(1, 0, 1)` represents:

- `p = 1` → One autoregressive (AR) term.
- `d = 0` → No differencing because the series is stationary.
- `q = 1` → One moving-average (MA) term.

Therefore:

```text
ARIMA(1,0,1) = ARMA(1,1)
```

## 🔮 Forecasting

The fitted model predicts the next **20 observations**:

```python
forecast = fitted.forecast(steps=20)
```

The final graph displays:

- **Actual values**
- **Fitted values**
- **Forecasted values**

A vertical dashed line separates the original observations from the forecast period.

## 📏 Model Evaluation

The model is evaluated using **Mean Absolute Error (MAE)**:

```python
mae = np.mean(np.abs(series - fitted_values))

print('In-sample MAE:', mae)
```

MAE measures the average absolute difference between the actual values and the fitted values.

A **lower MAE** generally indicates that the fitted model follows the observed data more closely.

## ▶️ How to Run

Save the Python program, for example, as:

```text
arma_model.py
```

Run it from the terminal:

```bash
python arma_model.py
```

The program will:

1. Generate the ARMA(1,1) time series.
2. Display the first few observations.
3. Perform the ADF stationarity test.
4. Plot the generated time series.
5. Fit the ARMA(1,1) model.
6. Display the model summary.
7. Forecast the next 20 observations.
8. Plot actual, fitted, and forecasted values.
9. Calculate and display the in-sample MAE.

## 📂 Project Structure

```text
ARMA-Time-Series/
│
├── arma_model.py
└── README.md
```

## 📝 Conclusion

This project demonstrates the basic workflow of **time-series modeling using ARMA(1,1)**. A synthetic stationary time series is generated and verified using the Augmented Dickey-Fuller test. An ARMA(1,1) model is then fitted to the data and used to forecast future observations.

The project provides a simple introduction to **stationarity testing, ARMA modeling, forecasting, visualization, and model evaluation** using Python.
