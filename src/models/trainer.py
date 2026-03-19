import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.models.hyperfusionnet import build_hyperfusionnet

X_PATH = Path("data/processed/X.npy")
Y_PATH = Path("data/processed/y.npy")
MODEL_PATH = Path("saved_models/hyperfusionnet.keras")


def main():
    print("Loading sequences...")
    X = np.load(X_PATH)
    y = np.load(Y_PATH)

    seq_len = X.shape[1]
    n_features = X.shape[2]

    print("Dataset:", X.shape, y.shape)

    # Time-series split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print("Train:", X_train.shape)
    print("Test:", X_test.shape)

    print("Building HyperFusionNet...")
    model = build_hyperfusionnet(seq_len, n_features)
    model.summary()

    print("Training...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=32
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)

    print("Model saved at:", MODEL_PATH)


if __name__ == "__main__":
    main()