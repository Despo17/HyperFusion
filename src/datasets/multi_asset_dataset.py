import pandas as pd
import numpy as np
from pathlib import Path

from src.data.market_data import ASSETS, update_market_data
from src.features.volatility_features import add_features

SEQ_LEN = 30
FEATURES = [
    'Open','High','Low','Close','Volume',
    'return','log_return','hl_range','ma_10','ma_20'
]


def build_multi_asset_sequences():
    X_seq = []
    X_asset = []
    y = []

    for asset_idx, asset in enumerate(ASSETS.keys()):
        df = update_market_data(asset)
        df = add_features(df)

        df.dropna(inplace=True)

        features = df[FEATURES].values
        target = df["vol_norm"].values

        for i in range(SEQ_LEN, len(df)):
            X_seq.append(features[i-SEQ_LEN:i])
            X_asset.append(asset_idx)
            y.append(target[i])

    return np.array(X_seq), np.array(X_asset), np.array(y)