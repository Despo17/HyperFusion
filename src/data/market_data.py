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
    Handles yfinance multi-index output permanently.
    """
    if df is None or df.empty:
        return df

    # Flatten multi-index columns from yfinance
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
    Data stored locally per asset as CSV.
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

    # ---------- Load existing ----------
    local_df = pd.read_csv(file_path, index_col=0, parse_dates=True)

    # Safety: ensure datetime index
    local_df.index = pd.to_datetime(local_df.index)

    last_date = local_df.index[-1]

    # ---------- Download new data ----------
    new_df = yf.download(symbol, start=last_date, progress=False)

    if new_df is None or new_df.empty:
        return local_df

    new_df = _standardize_ohlcv(new_df)

    # ---------- Merge ----------
    combined = pd.concat([local_df, new_df])
    combined = combined[~combined.index.duplicated(keep="last")]

    combined.to_csv(file_path)

    return combined