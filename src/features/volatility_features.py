import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path("data/raw/nifty.csv")
PROCESSED_PATH = Path("data/processed/nifty_features.csv")

def add_features(df, window=10):
    """
    Create volatility and technical features
    """
    df = df.copy()

    # Ensure numeric columns
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=numeric_cols, inplace=True)

    # Returns
    df['return'] = df['Close'].pct_change(fill_method=None)
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

    # Range
    df['hl_range'] = (df['High'] - df['Low']) / df['Close']

    # Moving averages
    df['ma_10'] = df['Close'].rolling(10).mean()
    df['ma_20'] = df['Close'].rolling(20).mean()

    # Volatility
    df['volatility'] = df['log_return'].rolling(window).std()

    # --- Normalized volatility target ---
    df["vol_mean_20"] = df["volatility"].rolling(20).mean()
    df["vol_norm"] = df["volatility"] / df["vol_mean_20"]

    # FINAL cleanup (after ALL rolling)
    df.dropna(inplace=True)

    return df

def main():
    print("Loading raw data...")
    df = pd.read_csv(RAW_PATH, index_col=0)

    print("Creating features...")
    df = add_features(df)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH)

    print("Features saved at:", PROCESSED_PATH)
    print("Dataset shape:", df.shape)

if __name__ == "__main__":
    main()