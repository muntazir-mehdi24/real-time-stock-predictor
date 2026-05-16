# NSE Real-Time Stock Predictor

A full-stack machine learning web application that predicts next-day closing prices for Indian NSE stocks using LSTM neural networks. Built with a FastAPI backend, Chart.js dashboard, and a complete automated data pipeline.

![Dashboard](static/index.html)

---

## What It Does

Given any NSE stock symbol (e.g. TCS, RELIANCE, WIPRO), the application:

1. Fetches the latest 2 years of historical price data from Yahoo Finance
2. Preprocesses and scales the data
3. Trains an LSTM model (only on first request)
4. Predicts the next day's closing price
5. Displays the result on a live web dashboard with a 60-day price chart

---

## How It Works

```
User enters symbol
       ↓
fetch_data.py     → Fetches live NSE data via yfinance
       ↓
preprocess.py     → Cleans data, handles missing values, scales to [0,1]
       ↓
model.py          → Trains LSTM (skipped if model already exists)
       ↓
predict.py        → Loads model, predicts, inverse-scales to real rupee price
       ↓
app.py            → Serves prediction via FastAPI REST endpoint
       ↓
index.html        → Displays result on web dashboard with Chart.js
```

The model is trained once per stock and reused on subsequent requests. Fresh market data is fetched on every prediction request to ensure the latest prices are used.

---

## Project Structure

```
real_time_stock_predictor/
├── src/
│   ├── fetch_data.py        # Fetches raw NSE stock data from yfinance
│   ├── preprocess.py        # Cleans, scales data, saves scaler
│   ├── model.py             # Builds and trains LSTM model
│   ├── predict.py           # Loads model and makes predictions
│   ├── pipeline.py          # Runs full pipeline from CLI
│   └── app.py               # FastAPI server with REST endpoints
├── static/
│   └── index.html           # Web dashboard (Chart.js frontend)
├── data/
│   ├── raw_data/            # Raw CSV files from yfinance
│   └── preprocessed_data/   # Scaled CSVs and scaler .pkl files
├── models/                  # Saved .keras model files
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.11
- Miniconda or Anaconda
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/real_time_stock_predictor.git
cd real_time_stock_predictor

# Create and activate conda environment
conda create -n real_time_stock_predictor python=3.11
conda activate real_time_stock_predictor

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Web Dashboard

```bash
# Start the FastAPI server
uvicorn src.app:app --reload

# Open in browser
http://127.0.0.1:8000/dashboard
```

Enter any NSE stock symbol and click **Predict**. The app will automatically fetch data, train a model if needed, and display the prediction with a 60-day price chart.

### Command Line Pipeline

```bash
# Run the full pipeline for any stock
python src/pipeline.py TCS
python src/pipeline.py RELIANCE
python src/pipeline.py HDFCBANK
```

### Individual Scripts

```bash
# Fetch raw data
python src/fetch_data.py TCS

# Preprocess data
python src/preprocess.py TCS

# Train model
python src/model.py TCS

# Predict
python src/predict.py TCS
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API health check |
| `/dashboard` | GET | Web dashboard |
| `/predict?symbol=TCS` | GET | Predict next day closing price |
| `/history?symbol=TCS` | GET | Get last 60 days closing prices |
| `/docs` | GET | Interactive API documentation (Swagger UI) |

---

## Model Architecture

- **Type:** LSTM (Long Short-Term Memory)
- **Input:** 60-day sequences of Open, High, Low, Close, Volume
- **Architecture:** LSTM(60) → Dropout(0.2) → LSTM(60) → Dropout(0.2) → Dense(1)
- **Loss:** Mean Squared Error
- **Optimizer:** Adam
- **Training:** Early stopping with patience=10, ModelCheckpoint saves best weights

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data | yfinance, pandas, numpy |
| ML | TensorFlow, Keras, scikit-learn |
| Backend | FastAPI, uvicorn |
| Frontend | HTML, CSS, JavaScript, Chart.js |
| Environment | Miniconda, pip |

---

## Limitations

- Predictions are based on historical price patterns only — they do not account for news, earnings, or market sentiment
- Model is retrained only on first request; to retrain with fresh data, delete the model file from `models/`
- TensorFlow GPU support requires WSL2 on Windows (currently runs on CPU)

---

## Author

**Muntaha** — Engineering Student  
Built as part of a structured ML engineering learning roadmap.

---

## License

MIT License
