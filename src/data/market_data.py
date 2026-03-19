import pandas as pd
import yfinance as yf
from pathlib import Path

# ==============================
# Configuration
# ==============================

DATA_DIR = Path("data")

ASSETS = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "RELIANCE": "RELIANCE.NS",
    "BTC": "BTC-USD",
}

# ==============================
# Helpers
# ==============================

def _standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure dataframe has flat OHLCV columns.
    Handles yfinance multi-index output.
    """
    if df is None or df.empty:
        return df

    # Flatten MultiIndex (yfinance issue)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep only OHLCV
    required = ["Open", "High", "Low", "Close", "Volume"]
    df = df[required]

    # Drop missing rows
    df.dropna(inplace=True)

    return df


# ==============================
# Main loader
# ==============================

def update_market_data(asset: str) -> pd.DataFrame:
    """
    Load + incrementally update market data for an asset.
    """

    if asset not in ASSETS:
        raise ValueError(f"Unknown asset: {asset}")

    symbol = ASSETS[asset]
    file_path = DATA_DIR / f"{asset}.csv"

    # ---------- First download ----------
    if not file_path.exists():
        df = yf.download(symbol, start="2010-01-01", progress=False)
        df = _standardize_ohlcv(df)

        file_path.parent.mkdir(exist_ok=True)
        df.to_csv(file_path)

        return df

    # ---------- Load existing (SAFE VERSION) ----------
    local_df = pd.read_csv(file_path)

    # =========================
    # ✅ FIX: HANDLE DATE COLUMN SAFELY
    # =========================

    # Find date column
    date_col = None
    for col in local_df.columns:
        if "date" in col.lower():
            date_col = col
            break

    # If not found, assume first column is date
    if date_col is None:
        date_col = local_df.columns[0]

    # Convert safely
    local_df[date_col] = pd.to_datetime(local_df[date_col], errors='coerce')

    # Drop invalid dates
    local_df = local_df.dropna(subset=[date_col])

    # Set index
    local_df.set_index(date_col, inplace=True)

    # Sort index
    local_df.sort_index(inplace=True)

    # Ensure OHLCV structure
    local_df = _standardize_ohlcv(local_df)

    if local_df.empty:
        # fallback: re-download if corrupted
        df = yf.download(symbol, start="2010-01-01", progress=False)
        df = _standardize_ohlcv(df)
        df.to_csv(file_path)
        return df

    last_date = local_df.index[-1]

    # ---------- Download new data ----------
    new_df = yf.download(
        symbol,
        start=last_date.strftime("%Y-%m-%d"),
        progress=False
    )

    if new_df is None or new_df.empty:
        return local_df

    new_df = _standardize_ohlcv(new_df)

    # ---------- Merge ----------
    combined = pd.concat([local_df, new_df])
    combined = combined[~combined.index.duplicated(keep="last")]

    # Save updated data
    combined.to_csv(file_path)

    return combined
