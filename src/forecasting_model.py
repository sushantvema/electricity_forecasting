# src/forecasting_model.py
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt


def train_test_split(df, train_size=0.8):
    """Split the data into training and testing sets."""
    split_idx = int(len(df) * train_size)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    return train_df, test_df


def train_arima_model(train_df, order=(5, 1, 0)):
    """Train an ARIMA model."""
    model = ARIMA(train_df["rt_price"], order=order)
    model_fit = model.fit()
    return model_fit


def forecast(model_fit, steps=24):
    """Forecast future values."""
    forecast = model_fit.forecast(steps=steps)
    return forecast


def evaluate_forecast(test_df, forecast, steps=24):
    """Calculate evaluation metrics."""
    y_true = test_df["rt_price"].iloc[:steps].values
    y_pred = forecast.values
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mae = mean_absolute_error(y_true, y_pred)
    return {"RMSE": rmse, "MAPE": mape, "MAE": mae}


def plot_forecast(test_df, forecast, steps=24):
    """Plot the forecast and actual values."""
    plt.plot(
        test_df["datetime"].iloc[:steps],
        test_df["rt_price"].iloc[:steps],
        label="Actual",
    )
    plt.plot(
        test_df["datetime"].iloc[:steps], forecast, label="Forecast", linestyle="--"
    )
    plt.legend()
    plt.xlabel("Datetime")
    plt.ylabel("Price")
    plt.title("Electricity Price Forecast")
    plt.show()
