from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import ydata_profiling
from pandas.plotting import autocorrelation_plot
from sklearn.model_selection import TimeSeriesSplit
import ipdb


def main():
    df = read_csv("../data/price_temp_data.csv")
    df = df.rename(columns={"Unnamed: 0": "Date"})
    df = df.set_index("Date")
    # profile = ydata_profiling.ProfileReport(df)
    # profile.to_file("auto_eda.html")

    # Create a lagged feature for 24-hour ahead prediction
    # 24 hours at 15-minute intervals is 96 data points ahead
    df["target"] = df["rt_energy"].shift(-96)  # Target is 24 hours ahead

    # Drop rows where target is NaN (last 24 hours)
    df = df.dropna()

    ipdb.set_trace()
    # autocorrelation_plot(df["rt_energy"][:36])
    # plt.scatter(df["Date"], df["rt_energy"])
    diff = df["rt_energy"].diff()
    # plt.plot(diff)
    # plt.plot(df["rt_energy"])
    plt.scatter(df["dah_energy"], df["rt_energy"].shift(-1))
    plt.show()
    ipdb.set_trace()

    # Split data into features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    return


if __name__ == "__main__":
    main()
