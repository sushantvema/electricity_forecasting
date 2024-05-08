from argparse import ArgumentParser
from pandas import read_csv, to_datetime, DataFrame, date_range
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import ydata_profiling
from pandas.plotting import autocorrelation_plot
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from datetime import timedelta
import ipdb

FORECAST_HORIZON = 96 * 7  # Just to see performance over a week
NUM_SPLITS = 3  # For sake of computation time


def main():
    df = read_csv("../data/price_temp_data.csv")
    df = df.rename(columns={"Unnamed: 0": "Date"})
    df = df.set_index("Date")
    df.index = to_datetime(df.index)
    # profile = ydata_profiling.ProfileReport(df)
    # profile.to_file("auto_eda.html")

    # Deseasonalize `rt_energy`
    # result = seasonal_decompose(
    #     df["rt_energy"], model="additive", period=96 * 7 * 4
    # )  # Very tenuous whether there is seasonality
    # df["rt_energy"] = df["rt_energy"] - result.seasonal
    # df["target"] = df["rt_energy"]

    # autocorrelation_plot(df["rt_energy"][:36])
    # plt.scatter(df["Date"], df["rt_energy"])
    # diff = df["rt_energy"].diff()
    # plt.plot(diff)
    # plt.plot(df["rt_energy"])
    # plt.scatter(df["dah_energy"], df["rt_energy"].shift(-1))
    # plt.show()

    # Split data into features and target
    X = df.drop("target", axis=1).drop("rt_energy", axis=1)
    y = df["target"]

    rmse_values = []
    percentage_error_values = []
    tscv = TimeSeriesSplit(gap=95, n_splits=NUM_SPLITS)
    for i, (train_index, test_index) in enumerate(tscv.split(X)):
        # Split the data into train and test sets
        train_data = df.iloc[train_index]
        test_data = df.iloc[test_index]
        # Fit the ARIMA model
        arimax_model = ARIMA(
            train_data["target"],
            order=(5, 0, 1),
            exog=train_data[["temp_BAR", "month", "hour", "season"]],
            freq="15min",
        )
        arimax_fit = arimax_model.fit()

        # Generate assumptions for future exogenous variable (just temp_BAR)
        # In this case, let's use forecasted values of the exogenous variables
        exog_variables = ["temp_BAR"]
        exog_variables_arima_models = {}

        for exog_variable in exog_variables:
            result = adfuller(train_data[exog_variable])
            print("ADF Statistic:", result[0])
            print("p-value:", result[1])

            # If the p-value is above a certain threshold (e.g., 0.05), the series is non-stationary
            if result[1] > 0.05:
                # Apply differencing to make it stationary
                train_data[exog_variable] = train_data[exog_variable].diff().dropna()

            # Plot ACF to determine possible values for 'q'
            # plot_acf(train_data[exog_variable], lags=20)  # Examine 20 lags
            # plt.show()
            #
            # Plot PACF to determine possible values for 'p'
            # plot_pacf(train_data[exog_variable], lags=20)  # Examine 20 lags
            # plt.show()

            # Deseasonalize the temperature
            result = seasonal_decompose(
                train_data["temp_BAR"], model="additive", period=96 * 7 * 4
            )  # Monthly periodicity
            train_data["temp_BAR"] = train_data["temp_BAR"] - result.seasonal

            # Add custom lags for temp_BAR based on ACF analysis
            # lags = [1, 2, 3, 4, 5, 95, 96, 97]
            # for lag in lags:
            #     train_data[
            #         f"temp_BAR_lag_{
            #         lag}"
            #     ] = train_data["temp_BAR"].shift(-lag)

            # Fit the ARMA model on data we need to extrude
            arima_model = ARIMA(
                train_data[exog_variable],
                order=(3, 0, 5),
                # exog=train_data[["temp_BAR"] + [f"temp_BAR_lag_{lag}" for lag in lags]],
                freq="15min",
            )
            arima_fit = arima_model.fit()
            exog_variables_arima_models[exog_variable] = arima_fit

        # Create future exogenous DataFrame with the estimated values
        future_exog = DataFrame(
            index=date_range(
                start=train_data.index[-1], periods=FORECAST_HORIZON, freq="15T"
            ),  # Forecasting 96 steps ahead
        )
        future_exog["hour"] = future_exog.index.hour
        future_exog["month"] = future_exog.index.month
        future_exog["season"] = [(month - 1) // 3 + 1 for month in future_exog.month]
        future_exog["temp_BAR"] = exog_variables_arima_models["temp_BAR"].predict(
            start=future_exog.index[0],
            end=future_exog.index[0] + timedelta(minutes=15 * FORECAST_HORIZON),
        )

        # Make a forecast
        forecast = arimax_fit.forecast(
            steps=FORECAST_HORIZON,
            exog=future_exog[["temp_BAR", "hour", "month", "season"]],
        )  # Use estimated exogenous values

        # Calculate the mean squared error (or other metrics)
        targets = test_data["target"][:FORECAST_HORIZON]
        rmse = mean_squared_error(targets, forecast) ** 0.5
        rmse_values.append(rmse)
        percentage_error = abs((forecast - targets) / targets)
        percentage_error_values.append(percentage_error)
        mape = (sum(percentage_error_values) / len(percentage_error_values)) * 100

        # Plot the results for visual inspection (optional)
        plt.figure(figsize=(10, 6))
        plt.plot(train_data.index, train_data["target"], label="Training Data")
        plt.plot(targets.index, targets, label="Actual Test Data")
        plt.plot(targets.index, forecast, label="ARIMAX Forecast")
        plt.title(f"ARIMAX Cross-Validation - Split {len(rmse_values)}")
        plt.legend()
        plt.show()  # Train Model

        # Save the ARIMAX model to a file
        import pickle

        # Define the file name for saving model
        arimax_model_file = "../models/arimax_model.pkl"

        # Save the fitted ARIMAX model
        with open(arimax_model_file, "wb") as f:
            pickle.dump(arimax_fit, f)  # Serialize the ARIMAX model to a file

    # Calculate Metrics
    index = range(NUM_SPLITS)
    plt.plot(index, rmse_values, label="RMSE")
    plt.plot(index, percentage_error_values, label="Percentage Error")
    plt.title("Performance across validation splits.")
    plt.legend()
    plt.show()

    return


if __name__ == "__main__":
    main()
