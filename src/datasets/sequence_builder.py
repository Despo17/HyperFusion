import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

DATA_PATH = Path("data/processed/nifty_features.csv")

FEATURES = [
    'Open','High','Low','Close','Volume',
    'return','log_return','hl_range','ma_10','ma_20'
]

TARGET = 'volatility'
SEQ_LEN = 30

SCALER_PATH = Path("data/processed/feature_scaler.save")


def build_sequences(df, seq_len=SEQ_LEN):
    X, y = [], []

    scaler = StandardScaler()
    feature_values = df[FEATURES].values
    feature_values = scaler.fit_transform(feature_values)

    # save scaler for inference
    joblib.dump(scaler, SCALER_PATH)

    for i in range(seq_len, len(df)):
        seq = feature_values[i-seq_len:i]
        target = df[TARGET].iloc[i]

        X.append(seq)
        y.append(target)

    return np.array(X), np.array(y)


def main():
    print("Loading feature dataset...")
    df = pd.read_csv(DATA_PATH, index_col=0)

    print("Scaling + building sequences...")
    X, y = build_sequences(df)

    Path("data/processed").mkdir(parents=True, exist_ok=True)

    np.save("data/processed/X.npy", X)
    np.save("data/processed/y.npy", y)

    print("Sequences saved:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)


if __name__ == "__main__":
    main()