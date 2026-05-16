"""Preprocess raw stock data and persist scaled output for modeling."""

import pandas as pd
import os
import argparse
from sklearn.preprocessing import MinMaxScaler
import joblib


def preprocess_data(raw_filepath):
    """Load raw CSV, clean it, and return a dataframe indexed by date."""
    df = pd.read_csv(raw_filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.drop(columns=['Dividends', 'Stock Splits'])

    # Handle missing values by dropping rows with gaps.
    print(f"Missing values before preprocessing: {df.isnull().sum().sum()}")
    df = df.dropna()
    print(f"Missing values after preprocessing: {df.isnull().sum().sum()}")
    return df

def scale_data(df):
    """Scale numerical columns to [0, 1] and persist the scaler."""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    # Save the scaler for later use in prediction and inverse transforms.
    joblib.dump(scaler, f"data/preprocessed_data/{symbol}_scaler.pkl")
    print(f"Scaler saved to data/preprocessed_data/{symbol}_scaler.pkl")
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
    return scaled_df, scaler

def save_preprocessed_data(df, symbol):
    """Save the scaled dataframe to data/preprocessed_data as CSV."""
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('data/preprocessed_data'):
        os.makedirs('data/preprocessed_data')

    filepath = f"data/preprocessed_data/{symbol}_preprocessed.csv"
    df.to_csv(filepath)
    print(f"Preprocessed data for {symbol} saved to {filepath}")

if __name__ == "__main__":
    # Parse CLI input and perform preprocessing pipeline.
    parser = argparse.ArgumentParser(description='Preprocess stock data.')
    parser.add_argument('symbol', type=str, help='Stock symbol to preprocess data for (e.g., TCS)')
    args = parser.parse_args()
    symbol = args.symbol
    raw_filepath = f"data/raw_data/{symbol}.csv"
    if os.path.exists(raw_filepath):
        df = preprocess_data(raw_filepath)
        scaled_df, scaler = scale_data(df)
        save_preprocessed_data(scaled_df, symbol)
        
    else:
        print(f"Raw data file not found for symbol: {symbol}. Please run fetch_data.py first.")