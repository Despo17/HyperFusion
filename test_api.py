import numpy as np
import requests

# Load last sequence
X = np.load("data/processed/X.npy")
last_seq = X[-1].tolist()

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"sequence": last_seq}
)

print("API response:", response.json())