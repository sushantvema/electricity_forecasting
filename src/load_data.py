# src/load_data.py
import pandas as pd


def load_data(file_path):
    """Load dataset and preprocess."""
    df = pd.read_csv(file_path, parse_dates=["datetime"])
    df = df.dropna()  # Handle missing values
    return df
