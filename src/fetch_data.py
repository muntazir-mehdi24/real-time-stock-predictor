"""Fetch historical stock data and save it to data/raw_data."""

import yfinance as yf
import pandas as pd
import os
import argparse



def fetch_data(symbol, period='2y'):
    """Fetch stock history from Yahoo Finance for the given symbol."""

    # Yahoo Finance expects the NSE suffix for Indian stocks.
    ticker = yf.Ticker(symbol + ".NS")
    data = ticker.history(period=period)
    df = pd.DataFrame(data)

    if df.empty:
        print(f"No data found for symbol: {symbol}")
        return None
    else:
        print(f"Data for {symbol} fetched successfully.")
        return df

def save_data(df, symbol):
    """Persist the fetched dataframe to data/raw_data as CSV."""
    # Ensure target directories exist before writing.
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('data/raw_data'):
        os.makedirs('data/raw_data')
    filepath = f"data/raw_data/{symbol}.csv"
    df.to_csv(filepath)
    print(f"Data for {symbol} saved to {filepath}")

if __name__ == "__main__":
    # Parse CLI arguments when executed directly.
    parser = argparse.ArgumentParser(description='Fetch stock data and save to CSV.')
    parser.add_argument('symbol', type = str, help = 'Stock symbol to fetch data for (e.g., TCS)')
    args = parser.parse_args()
    symbol = args.symbol
    df = fetch_data(symbol)
    if df is not None:
        save_data(df, symbol)
