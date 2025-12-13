# main.py
# for request parameters and response models
from fastapi import FastAPI, HTTPException, Query 

# CORS middleware
from fastapi.middleware.cors import CORSMiddleware

# data models
from pydantic import BaseModel

from typing import List
import datetime
import requests
import pandas as pd
import numpy as np

# environment
import os

# ML model
from sklearn.ensemble import RandomForestRegressor

# model persistence
from joblib import dump, load

# async HTTP client
import httpx

# import price endpoint with async cached function
from price_endpoint import get_price_async

# Config
BILLWISE_BASE = os.environ.get("BILLWISE_BASE", "http://localhost:8080")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# FastAPI app
app = FastAPI(title="BillWise Forecast Service (Real Model)")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response Models ------------------------------------------------

# Daily Prediction Model 
class DailyPrediction(BaseModel):
    date: datetime.date
    predictedUnits: int
    predictedRevenue: float

# Overall Prediction Response Model
class PredictionResponse(BaseModel):
    productId: int
    generatedAt: datetime.date
    forecastingWindow: str
    predictedTotalRevenue: float
    dailyPredictions: List[DailyPrediction]

# Helper utilities ------------------------------------------------

# Fetch monthly sales data from BillWise
def fetch_monthly_sales(product_id: int):
    """
    Pull monthly aggregated sales from BillWise:
    GET /api/ai/monthly-sales/{productId}
    expected format: [{ "date": "YYYY-MM", "unitsSold": 120 }, ...]
    """
    # make request
    url = f"{BILLWISE_BASE}/api/ai/monthly-sales/{product_id}"

    # Use requests for simplicity here
    resp = requests.get(url, timeout=10)
    if resp.status_code == 204:
        return []
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Failed to fetch sales from BillWise: {resp.status_code}")
    return resp.json()

# Prepare DataFrame from fetched data --------------------------------

# Convert list of {"date":"YYYY-MM","unitsSold":N} -> pandas Series indexed by Period (monthly).
def prepare_dataframe(monthly_points):
    """
    Convert list of {"date":"YYYY-MM","unitsSold":N} -> pandas Series indexed by Period (monthly).
    """
    if not monthly_points:
        return pd.Series(dtype=float)

    # convert to DataFrame
    rows = []
    for r in monthly_points:
        # accept "YYYY-MM" or "YYYY-MM-DD"
        d = r.get("date")
        # normalize to first day of month
        try:
            if len(d) == 7:
                dt = pd.to_datetime(d + "-01")
            else:
                dt = pd.to_datetime(d)
                dt = dt.replace(day=1)
        except Exception:
            # fallback
            dt = pd.to_datetime(d, errors='coerce')
            if pd.isna(dt):
                continue
            dt = dt.replace(day=1)
        rows.append((dt, int(r.get("unitsSold", 0))))
    if not rows:
        return pd.Series(dtype=float)
    
    # create DataFrame
    df = pd.DataFrame(rows, columns=["ds", "y"])

    # sort by date and drop duplicates (keep last)
    df = df.sort_values("ds").drop_duplicates("ds", keep="last")

    # set index
    df.set_index("ds", inplace=True)

    # reindex monthly continuous range
    idx = pd.date_range(df.index.min(), df.index.max(), freq='MS')

    # fill missing months with 0
    df = df.reindex(idx, fill_value=0)

    # return as Series
    series = df["y"]
    series.index.name = "ds"
    return series

# Create lag features for supervised learning
def create_lag_features(series: pd.Series, n_lags=12):
    """
    Create supervised dataset from monthly series using lags.
    """
    # Build DataFrame with lag features
    df = pd.DataFrame({"y": series})
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df = df.dropna()
    return df

# Model training, loading, and forecasting ------------------------

# Train model for a given product_id
# Fetch historical sales
    # 1) load history
    # Fetch monthly sales data
    # 2) prepare DataFrame
    # 3) train model
    # 4) save model with metadata
    # 5) return model path
    # If very short series, save simple mean predictor

def train_model_for_product(product_id: int):
    """
    Train model and save to disk. Returns model path.
    """
    monthly = fetch_monthly_sales(product_id)
    series = prepare_dataframe(monthly)

    # Check if we have enough data to train a model (at least some data)
    if series.empty:
        raise HTTPException(status_code=404, detail="No historical sales data available for this product.")

    # If very short series (<6 months), we use a naive model (mean) stored as special file
    if len(series) < 6:
        # Save simple mean predictor as dict -> joblib
        mean_val = float(series.mean())
        # Save with metadata (last index) 
        model_path = os.path.join(MODELS_DIR, f"{product_id}.mean.joblib")
        dump({"type": "mean", "mean": mean_val, "last_index": series.index.max().to_pydatetime()}, model_path)
        return model_path

    # Create lag features
    n_lags = min(12, max(3, len(series) // 2))
    df_supervised = create_lag_features(series, n_lags=n_lags)
    # features and target
    X = df_supervised.drop(columns=["y"]).values
    y = df_supervised["y"].values

    # Train a RandomForestRegressor (robust with small data)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)

    # Save with metadata (n_lags & last index)
    model_path = os.path.join(MODELS_DIR, f"{product_id}.pkl")
    dump({"type": "rf", "model": model, "n_lags": n_lags, "last_index": series.index.max().to_pydatetime()}, model_path)
    return model_path

# Load model from disk
def load_model(product_id: int):
    """
    Load model file if exists. Return dict with metadata.
    """

    # Check for both RandomForest and mean predictor
    p1 = os.path.join(MODELS_DIR, f"{product_id}.pkl")
    p2 = os.path.join(MODELS_DIR, f"{product_id}.mean.joblib")

    # Try RandomForest first
    if os.path.exists(p1):
        data = load(p1)
        return data
    if os.path.exists(p2):
        data = load(p2)
        return data
    return None

# Forecast using loaded model or train-on-the-fly if needed
def forecast_using_model(product_id: int, months: int = 3):
    """
    Predict next `months` months of demand (units) using saved model or train-on-the-fly.
    Returns list of (date, units)
    """
    # 1) load history
    monthly = fetch_monthly_sales(product_id)
    series = prepare_dataframe(monthly)
    if series.empty:
        raise HTTPException(status_code=404, detail="No historical sales data available for forecasting.")

    model_data = load_model(product_id)
    if model_data is None:
        # train on the fly
        train_model_for_product(product_id)
        model_data = load_model(product_id)
        if model_data is None:
            raise HTTPException(status_code=500, detail="Failed to create/load model.")

    # If simple mean predictor
    if model_data.get("type") == "mean":
        mean_val = float(model_data["mean"])
        last_dt = pd.to_datetime(model_data.get("last_index"))
        preds = []
        for i in range(1, months + 1):
            dt = (last_dt + pd.DateOffset(months=i)).date()
            preds.append((dt, int(round(mean_val))))
        return preds

    # RandomForest path
    model = model_data["model"]
    n_lags = int(model_data["n_lags"])
    # Build last n_lags values as feature vector
    last_index = pd.to_datetime(model_data["last_index"])
    # Ensure series has continuous months up to last_index (it should)
    series = series.asfreq('MS').fillna(0)
    vals = list(series.values)
    preds = []
    current = vals.copy()
    for i in range(1, months + 1):
        # Take last n_lags from current
        if len(current) < n_lags:
            input_vec = [0] * n_lags
            start = n_lags - len(current)
            input_vec[start:] = current
        else:
            input_vec = current[-n_lags:]
        X_pred = np.array(input_vec).reshape(1, -1)
        yhat = model.predict(X_pred)[0]
        yhat = max(0, float(round(yhat)))
        next_dt = (last_index + pd.DateOffset(months=i)).date()
        preds.append((next_dt, int(yhat)))
        # append predicted value to current for multi-step forecasting
        current.append(yhat)

    return preds

# API endpoints --------------------------------------------------

# Train model endpoint 
# POST /train/{product_id}
@app.post("/train/{product_id}")
def train(product_id: int):
    """
    Train (or re-train) model for a given product.
    """
    path = train_model_for_product(product_id)
    return {"status": "trained", "model_path": path}

# Price endpoint
# GET /price?product_id=123
@app.get("/price")
async def get_price(product_id: int = Query(...)):
    """
    Return explicit unit price for a product by querying BillWise product endpoint.
    Response: { "productId": 123, "unitPrice": 12.34, "source": "billwise" }
    Uses async httpx client and TTL cache for performance.
    """
    return await get_price_async(product_id)

# Forecast endpoint
# GET /forecast?product_id=123&months=3
@app.get("/forecast", response_model=PredictionResponse)
def forecast(product_id: int, months: int = Query(3, ge=1, le=24)):
    """
    Forecast next `months` months for product_id.
    Returns PredictionResponse with 'dailyPredictions' used for monthly entries.
    Fetches unit price from BillWise to compute consistent revenue estimates.
    """
    preds = forecast_using_model(product_id, months=months)

    # Fetch product price for revenue estimate (async, with caching)
    unit_price = None
    try:
        # We need to run the async function in sync context
        # Use httpx sync client for simplicity, fallback to requests
        import httpx
        BILLWISE_API_KEY = os.environ.get("BILLWISE_API_KEY", None)
        headers = {}
        if BILLWISE_API_KEY:
            headers["Authorization"] = f"Bearer {BILLWISE_API_KEY}"
        
        # Fetch product details
        # Use httpx sync client
        # Note: in production, consider using async FastAPI endpoint for better performance
        with httpx.Client(timeout=5.0) as client:
            p = client.get(f"{BILLWISE_BASE}/api/products/{product_id}", headers=headers)
            if p.status_code == 200:
                j = p.json()
                # Try common price fields
                for key in ("sellingPricePerBaseUnit", "selling_price_per_base_unit", "sellingPrice", "price", "unitPrice"):
                    if key in j and j[key] is not None:
                        try:
                            unit_price = float(j[key])
                            break
                        except (ValueError, TypeError):
                            continue
    except Exception:
        # Gracefully degrade: proceed with unit_price = None
        unit_price = None

    # Build daily predictions from monthly preds (for compatibility)
    today = datetime.date.today()
    daily_preds = []
    total_revenue = 0.0

    # Expand monthly preds into daily entries (same value for each day in month)
    for dt, units in preds:
        revenue = units * (unit_price if unit_price is not None else 0.0)
        total_revenue += revenue
        daily_preds.append({"date": dt, "predictedUnits": int(units), "predictedRevenue": float(revenue)})

    return PredictionResponse(
        productId=product_id,
        generatedAt=today,
        forecastingWindow=f"{months} Months",
        predictedTotalRevenue=float(total_revenue),
        dailyPredictions=daily_preds
    )

# Train all products endpoint
# POST /train-all
@app.post("/train-all")
def train_all_products():
    """
    Fetch all products from BillWise and train ML models for each.
    """
    url = f"{BILLWISE_BASE}/api/products?page=0&size=500"
    resp = requests.get(url, timeout=10)

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch products from BillWise: {resp.status_code}"
        )

    data = resp.json()

    # Extract list of products inside "content"
    products = data.get("content", [])
    if not products:
        raise HTTPException(status_code=404, detail="No products found in BillWise")

    trained = []

    for p in products:
        product_id = p.get("id")
        if product_id is None:
            continue

        try:
            model_path = train_model_for_product(product_id)
            trained.append({
                "product_id": product_id,
                "model_path": model_path,
                "status": "success"
            })
        except Exception as e:
            trained.append({
                "product_id": product_id,
                "status": "failed",
                "error": str(e)
            })

    return {
        "total_products": len(products),
        "trained": trained
    }



