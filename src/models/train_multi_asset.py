from src.datasets.multi_asset_dataset import build_multi_asset_sequences
from src.models.hyperfusion_multi import build_multi_asset_hyperfusion
from src.data.market_data import ASSETS

import numpy as np


def main():

    print("Building multi-asset dataset...")
    X_seq, X_asset, y = build_multi_asset_sequences()

    num_assets = len(ASSETS)

    print("Shapes:")
    print("X_seq:", X_seq.shape)
    print("X_asset:", X_asset.shape)
    print("y:", y.shape)

    model = build_multi_asset_hyperfusion(
        seq_len=X_seq.shape[1],
        num_features=X_seq.shape[2],
        num_assets=num_assets
    )

    model.summary()

    model.fit(
        [X_seq, X_asset],
        y,
        epochs=10,
        batch_size=32,
        validation_split=0.1
    )

    model.save("models/hyperfusion_multi.h5")
    print("Saved multi-asset model")


if __name__ == "__main__":
    main()