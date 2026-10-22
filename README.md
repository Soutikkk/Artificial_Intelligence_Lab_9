
# Time Series Analysis and Forecasting

## Overview

This project demonstrates three time-series forecasting models using Python:

1. AR(1) - Autoregressive model
2. MA(1) - Moving Average model
3. ARMA(1,1) - Autoregressive Moving Average model

The project uses synthetic time-series data to demonstrate stationarity testing, model fitting, forecasting, visualization, and model evaluation.

## Objectives

- Generate synthetic stationary time-series data.
- Understand AR(1), MA(1), and ARMA(1,1) models.
- Test stationarity using the Augmented Dickey-Fuller (ADF) test.
- Fit time-series models using Statsmodels.
- Visualize actual and fitted observations.
- Forecast the next 20 observations.
- Evaluate fitted values using Mean Absolute Error (MAE).

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Statsmodels
- Scikit-learn

## Installation

Install the required libraries:

```bash
python -m pip install numpy pandas matplotlib statsmodels scikit-learn
```

## Project Structure

```text
ARMA-Time-Series/
│
├── AR1_Time_Series_Forecasting.py
├── MA1_Time_Series_Forecasting.py
├── ARIMA Time Series Forecasting.py
└── README.md
```

The ARIMA file represents the original ARMA(1,1) program.
If your original file has a different name, use that name instead.

## 1. AR(1) Model

The AR(1) model is defined as:

X(t) = phi * X(t-1) + epsilon(t)

Parameters:
- phi = 0.70
- 200 observations
- Random seed = 42

The model is implemented using:

ARIMA(1, 0, 0)

Run the program:

```bash
python AR1_Time_Series_Forecasting.py
```

## 2. MA(1) Model

The MA(1) model is defined as:

X(t) = epsilon(t) + theta * epsilon(t-1)

Parameters:
- theta = 0.60
- 200 observations
- Random seed = 42

The model is implemented using:

ARIMA(0, 0, 1)

Run the program:

```bash
python MA1_Time_Series_Forecasting.py
```

## 3. ARMA(1,1) Model

The ARMA(1,1) model combines autoregressive and moving-average components.

X(t) = phi * X(t-1) + epsilon(t)
       + theta * epsilon(t-1)

The model is implemented using:

ARIMA(1, 0, 1)

Run the original program:

```bash
python "ARIMA Time Series Forecasting.py"
```

Change the filename in the command if necessary.

## Stationarity Testing

The Augmented Dickey-Fuller test checks for evidence of a unit root.

- p-value < 0.05: Reject the unit-root null hypothesis.
- p-value >= 0.05: Fail to reject the unit-root null hypothesis.

The result should be interpreted as statistical evidence, not an absolute guarantee.

## Forecasting

Each program forecasts the next 20 time steps.

The final graph displays:
- Actual values
- Fitted values
- Forecasted values

A vertical dashed line indicates where the forecast period begins.

## Model Evaluation

Mean Absolute Error (MAE) measures the average absolute difference between observed and fitted values.

MAE = mean(abs(actual values - fitted values))

A lower MAE indicates smaller average in-sample errors. However, in-sample MAE alone does not establish future forecasting performance.

## How to Run

1. Open the project folder in VS Code.
2. Open the terminal.
3. Install the dependencies.
4. Run each Python file separately.
5. View the printed model summaries, stationarity results, MAE, and forecast plots.

## Conclusion

This project demonstrates three fundamental time-series models: AR(1), MA(1), and ARMA(1,1).

It provides practical experience with synthetic data generation, stationarity testing, statistical model fitting, forecasting, visualization, and error evaluation using Python.
