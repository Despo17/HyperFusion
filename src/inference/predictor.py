import numpy as np
import tensorflow as tf
from pathlib import Path

MODEL_PATH = Path("saved_models/hyperfusionnet.keras")


class VolatilityPredictor:
    """
    Loads trained HyperFusionNet and predicts volatility
    """

    def __init__(self, model_path=MODEL_PATH):
        self.model = tf.keras.models.load_model(model_path,compile=False)

    def predict(self, sequence):
        """
        sequence shape: (seq_len, features)
        """
        sequence = np.expand_dims(sequence, axis=0)
        pred = self.model.predict(sequence, verbose=0)
        return float(pred[0][0])