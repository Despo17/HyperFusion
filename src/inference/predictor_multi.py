import numpy as np
import tensorflow as tf
import os

from src.data.market_data import ASSETS
from src.models.hyperfusion_multi import build_multi_asset_hyperfusion


class MultiAssetPredictor:

    def __init__(self, model_path=None):

        # ==============================
        # ✅ FIX 1: ABSOLUTE PATH
        # ==============================
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

        if model_path is None:
            model_path = os.path.join(BASE_DIR, "models", "hyperfusion_multi.h5")

        # ==============================
        # ✅ FIX 2: LOAD MODEL SAFELY
        # ==============================
        try:
            # Try loading FULL model (best case)
            self.model = tf.keras.models.load_model(model_path)
        except:
            # Fallback → rebuild architecture + load weights
            self.model = build_multi_asset_hyperfusion(
                seq_len=30,
                num_features=10,
                num_assets=len(ASSETS)
            )
            self.model.load_weights(model_path)

        # ==============================
        # ASSET MAPPING
        # ==============================
        self.asset_to_id = {
            asset: i for i, asset in enumerate(ASSETS.keys())
        }

    def predict(self, sequence, asset, vol_mean_20):

        asset_id = self.asset_to_id[asset]

        seq = np.array(sequence)

        # Ensure correct shape
        if seq.ndim == 2:
            seq = np.expand_dims(seq, axis=0)

        asset_arr = np.array([[asset_id]])

        # ==============================
        # MODEL PREDICTION
        # ==============================
        vol_norm = self.model.predict(
            [seq, asset_arr],
            verbose=0
        )[0][0]

        # Rescale output
        vol = vol_norm * vol_mean_20

        return float(vol)