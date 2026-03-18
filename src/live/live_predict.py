import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from src.data.market_data import update_market_data
from src.inference.predictor import VolatilityPredictor

SEQ_LEN = 30
SCALER_PATH = Path("data/processed/feature_scaler.save")

FEATURES = [
    'Open','High','Low','Close','Volume',
    'return','log_return','hl_range','ma_10','ma_20'
]


def add_features(df, window=10):
    df = df.copy()

    df['return'] = df['Close'].pct_change()
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['hl_range'] = (df['High'] - df['Low']) / df['Close']
    df['ma_10'] = df['Close'].rolling(10).mean()
    df['ma_20'] = df['Close'].rolling(20).mean()
    df['volatility'] = df['log_return'].rolling(window).std() * 100

    df.dropna(inplace=True)
    return df


def fetch_latest_data(symbol="^NSEI"):
    df = yf.download(symbol, period="6mo")
    df = df[['Open','High','Low','Close','Volume']]
    df.dropna(inplace=True)
    return df


def build_live_sequence(df):
    """
    Build normalized feature sequence for model input
    """

    if len(df) < SEQ_LEN:
        raise ValueError(
            f"Not enough data after feature engineering. "
            f"Have {len(df)} rows, need {SEQ_LEN}."
        )

    scaler = joblib.load(SCALER_PATH)

    feature_values = df[FEATURES].values

    if feature_values.shape[0] == 0:
        raise ValueError("Feature dataframe empty after processing")

    feature_values = scaler.transform(feature_values)

    seq = feature_values[-SEQ_LEN:]

# ✅ Add batch dimension (ONLY ONCE)
    seq = np.expand_dims(seq, axis=0)  # (1, 30, 10)

    return seq


def main():
    print("Fetching latest market data...")
    df =update_market_data()

    print("Building features...")
    df = add_features(df)

    print("Preparing sequence...")
    seq = build_live_sequence(df)

    print("Loading HyperFusion...")
    model = VolatilityPredictor()

    pred = model.predict(seq)

    print("\n📊 Live HyperFusion Forecast")
    print(f"Volatility (model units): {pred:.6f}")
    print(f"Expected daily move: {pred*100:.2f}%")
    
if __name__ == "__main__":
    main()