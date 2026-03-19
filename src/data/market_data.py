import yfinance as yf
import pandas as pd
import time


def update_market_data(asset):
    try:
        # Fetch data
        df = yf.download(asset, period="6mo", interval="1d", progress=False)

        # Retry once if rate limited / empty
        if df is None or df.empty:
            time.sleep(2)
            df = yf.download(asset, period="6mo", interval="1d", progress=False)

        # Final check
        if df is None or df.empty:
            print("❌ Failed to fetch data from Yahoo Finance")
            return pd.DataFrame()

        # Reset index
        df.reset_index(inplace=True)

        # Safe datetime conversion
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])

        # Sort properly
        df = df.sort_values(by="Date")

        return df

    except Exception as e:
        print("❌ Data fetch error:", e)
        return pd.DataFrame()