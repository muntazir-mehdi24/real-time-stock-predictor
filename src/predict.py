"""Load a trained model and generate a next-day price prediction."""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import argparse
import joblib
import os



def load_model(symbol):
    """Load the saved Keras model for the requested symbol."""
    model = keras.models.load_model(f"models/{symbol}_best_model.keras")
    print(f"Model for {symbol} loaded successfully.")
    return model

def predict(model, last_seq):
    """Run model inference on the latest input sequence."""
    predictions = model.predict(last_seq)
    return predictions

def reverse_scaling(predictions, scaler):
    """Convert scaled predictions back to real price units."""
    # Build a dummy matrix to match the scaler's expected feature count.
    dummy = np.zeros((len(predictions), 5))
    dummy[:, 3] = predictions[:, 0]
    reversed_predictions = scaler.inverse_transform(dummy)[:, 3]
    return reversed_predictions


if __name__ == "__main__":
    # Parse the symbol from CLI input.
    argparser = argparse.ArgumentParser(description='Predict stock prices using trained LSTM model.')
    argparser.add_argument('symbol', type=str, help='Stock symbol to predict for (e.g., TCS)')
    args = argparser.parse_args()
    symbol = args.symbol
    # Load the scaler and latest preprocessed data.
    scaler = joblib.load(f"data/preprocessed_data/{symbol}_scaler.pkl")
    df = pd.read_csv(f"data/preprocessed_data/{symbol}_preprocessed.csv", index_col='Date')
    data = df.to_numpy()
    # Use the last 60 timesteps as the prediction window.
    last_seq = data[-60:]
    last_seq = last_seq.reshape(1, 60, 5)
    model = load_model(symbol)
    predictions = predict(model, last_seq)
    reversed_predictions = reverse_scaling(predictions, scaler)
    print(f"Predicted closing price for {symbol} tomorrow: INR {reversed_predictions[0]:.2f}")

