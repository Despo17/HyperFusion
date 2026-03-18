import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from src.inference.predictor import VolatilityPredictor

app = FastAPI(title="HyperFusion Volatility API")

model = VolatilityPredictor()


# ✅ request schema
class SequenceRequest(BaseModel):
    sequence: list


@app.get("/")
def home():
    return {"status": "HyperFusion API running"}


@app.post("/predict")
def predict(req: SequenceRequest):
    seq = np.array(req.sequence)
    pred = model.predict(seq)
    return {"predicted_volatility": float(pred)}