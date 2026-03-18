import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
import numpy as np
from src.inference.predictor import VolatilityPredictor

# Load last sequence from dataset
X = np.load("data/processed/X.npy")

last_sequence = X[-1]

model = VolatilityPredictor()

pred = model.predict(last_sequence)

print("Predicted next-day volatility:", f"{pred:.10f}")