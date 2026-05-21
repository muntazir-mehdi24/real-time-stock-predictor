"""FastAPI app that serves the dashboard and prediction endpoints."""

from fastapi import FastAPI, HTTPException
import subprocess
import sys

from fastapi.staticfiles import StaticFiles
from src.predict import load_model, predict, reverse_scaling
import pandas as pd
import joblib
import numpy as np
import os
from fastapi.responses import FileResponse

app = FastAPI()


# Serve static assets (HTML/CSS/JS) from the static directory.
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/dashboard")
async def dashboard():
    # Return the single-page app UI.
    return FileResponse("static/index.html")

@app.get("/")
async def root():
    # Lightweight health check / welcome message.
    return {"message": "Welcome to the Real-Time Stock Prediction API!"}

@app.get("/predict")
async def predict_stock(symbol: str):

    # check if symbol is valid string (e.g., non-empty, non-numeric using isalpha())
    if not symbol or not symbol.isalpha():
        raise HTTPException(status_code=400, detail="Invalid stock symbol")

    # Run the pipeline steps to make sure we have fresh data and a model.
    subprocess.run([sys.executable, "src/fetch_data.py", symbol], check=True)
    subprocess.run([sys.executable, "src/preprocess.py", symbol], check=True)
    if not os.path.exists(f"models/{symbol}_best_model.keras"):
        # Train a model on demand when one is not yet saved.
        subprocess.run([sys.executable, "src/model.py", symbol], check=True)

    try:
        # Load the scaler and preprocessed data used for sequence construction.
        scaler = joblib.load(f"data/preprocessed_data/{symbol}_scaler.pkl")
        df = pd.read_csv(f"data/preprocessed_data/{symbol}_preprocessed.csv", index_col='Date')
        data = df.to_numpy()
        last_seq = data[-60:]
        last_seq = last_seq.reshape(1, 60, 5)

        # Load the model and make predictions for the next closing price.
        model = load_model(symbol)
        predictions = predict(model, last_seq)
        reversed_predictions = reverse_scaling(predictions, scaler)
        predicted_price = reversed_predictions[0]
        
        return {"predicted_price": f"INR {predicted_price:.2f}"}
    
    except Exception as e:
        # Surface any pipeline errors as API errors for the client.
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/history")
async def get_history(symbol: str):
    # check if symbol is valid string (e.g., non-empty, non-numeric using isalpha())
    if not symbol or not symbol.isalpha():
        raise HTTPException(status_code=400, detail="Invalid stock symbol")
    # Build a response of the last 60 real closing prices for charting.
    scaler = joblib.load(f"data/preprocessed_data/{symbol}_scaler.pkl")
    df = pd.read_csv(f"data/preprocessed_data/{symbol}_preprocessed.csv", index_col='Date')
    data = df.to_numpy()
    last_seq = data[-60:]
    dummy = np.zeros((60, 5))
    # Put Close column in the dummy matrix so the scaler can invert it.
    dummy[:, 3] = last_seq[:, 3]
    # Inverse-transform and extract only the Close column.
    real_closes = scaler.inverse_transform(dummy)[:, 3]
    dates = df.index[-60:].tolist()


    return {"closes": real_closes.tolist(), "dates": dates}
