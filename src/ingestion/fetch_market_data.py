import yfinance as yf
import pandas as pd
from pathlib import Path

# Where raw data will be saved
DATA_PATH = Path("data/raw/nifty.csv")

def fetch_market_data(symbol="^NSEI", start="2010-01-01"):
    """
    Download market data from Yahoo Finance
    """
    df = yf.download(symbol, start=start)

    # Keep only required columns
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    # Remove missing rows
    df.dropna(inplace=True)

    return df

def save_raw_data(df):
    """
    Save dataframe to CSV
    """
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH)

if __name__ == "__main__":
    print("Downloading market data...")
    df = fetch_market_data()

    save_raw_data(df)

    print("Raw data saved at:", DATA_PATH)