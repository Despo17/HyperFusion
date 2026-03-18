import numpy as np
import tensorflow as tf
from src.data.market_data import ASSETS
from src.models.hyperfusion_multi import build_multi_asset_hyperfusion


class MultiAssetPredictor:

    def __init__(self, model_path="models/hyperfusion_multi.h5"):
        # Rebuild architecture
        self.model = build_multi_asset_hyperfusion(
            seq_len=30,
            num_features=10,
            num_assets=len(ASSETS)
        )

        # Load weights only
        self.model.load_weights(model_path)

        self.asset_to_id = {
            asset: i for i, asset in enumerate(ASSETS.keys())
        }

    def predict(self, sequence, asset, vol_mean_20):

        asset_id = self.asset_to_id[asset]

        seq = np.array(sequence)

        if seq.ndim == 2:
            seq = np.expand_dims(seq, axis=0)

        asset_arr = np.array([[asset_id]])

    # ✅ FIX: pass as list, not dict
        vol_norm = self.model.predict(
            [seq, asset_arr],
            verbose=0
        )[0][0]

        vol = vol_norm * vol_mean_20

        return float(vol)