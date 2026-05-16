"""Train an LSTM model for next-day stock price prediction."""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import argparse
import os


def create_sequences(data, seq_len):
    """Convert a 2D array into LSTM sequences and next-step targets."""
    x = []
    y = []
    for i in range(len(data) - seq_len):
        # Each input sequence is seq_len rows; target is the next Close value.
        x.append(data[i:i + seq_len])
        y.append(data[i + seq_len, 3])
    return np.array(x), np.array(y)

def build_model(input_shape):
    """Create and compile the LSTM architecture used for prediction."""
    model = Sequential()
    # Two stacked LSTM layers with dropout for regularization.
    model.add(LSTM(60, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(60, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    print(model.summary())
    return model

def train_model(model, x_train, y_train, x_test, y_test):
    """Fit the model and save the best checkpoint to disk."""
    # Stop early if validation loss stops improving to prevent overfitting.
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    os.makedirs('models', exist_ok=True)
    # Save the best model weights for the requested symbol.
    model_checkpoint = ModelCheckpoint(
        f"models/{symbol}_best_model.keras",
        monitor='val_loss',
        save_best_only=True
    )
    history = model.fit(
        x_train,
        y_train,
        epochs=100,
        batch_size=32,
        validation_data=(x_test, y_test),
        callbacks=[early_stopping, model_checkpoint]
    )
    return history

if __name__ == "__main__":
    # Read the symbol from CLI and load its preprocessed data.
    argparser = argparse.ArgumentParser(description='Train LSTM model for stock price prediction.')
    argparser.add_argument('symbol', type=str, help='Stock symbol to train model for (e.g., TCS)')
    args = argparser.parse_args()
    symbol = args.symbol
    df = pd.read_csv(f"data/preprocessed_data/{symbol}_preprocessed.csv", index_col='Date')
    data = df.to_numpy()
    seq_len = 60
    x, y = create_sequences(data, seq_len)

    # Split into training and test sets without shuffling to keep time order.
    split = int(0.8 * len(x))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]
    model = build_model((x_train.shape[1], x_train.shape[2]))
    history = train_model(model, x_train, y_train, x_test, y_test)
    print("Model module loaded successfully.")
    print(f"Training model for stock symbol: {symbol}")
    print(f"Model saved to models/{symbol}_best_model.keras")
