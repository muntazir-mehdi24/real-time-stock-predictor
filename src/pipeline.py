"""Run the full data->train->predict pipeline for one stock symbol."""

import subprocess
import argparse
import os
import sys

parser = argparse.ArgumentParser(description='Run the entire stock prediction pipeline for a given symbol.')
parser.add_argument('symbol', type=str, help='Stock symbol to run the pipeline for (e.g., TCS)')
args = parser.parse_args()
symbol = args.symbol


# Step 1: Fetch raw data from the data provider.
subprocess.run([sys.executable, "src/fetch_data.py", symbol], check=True)
# Step 2: Clean and scale the raw data.
subprocess.run([sys.executable, "src/preprocess.py", symbol], check=True)
# Step 3: Train the model if it does not already exist.
if not os.path.exists(f"models/{symbol}_best_model.keras"):
    subprocess.run([sys.executable, "src/model.py", symbol], check=True)
else:
    print(f"Model for {symbol} already exists. Skipping training.")
# Step 4: Generate a prediction using the latest data.
subprocess.run([sys.executable, "src/predict.py", symbol], check=True)