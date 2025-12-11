# main.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import datetime
import requests
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load

# Config
BILLWISE_BASE = os.environ.get("BILLWISE_BASE", "http://localhost:8080")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

app = FastAPI(title="BillWise Forecast Service (Real Model)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response Models ------------------------

class DailyPrediction(BaseModel):
    date: datetime.date
    predictedUnits: int
    predictedRevenue: float

class PredictionResponse(BaseModel):
    productId: int
    generatedAt: datetime.date
    forecastingWindow: str
    predictedTotalRevenue: float
    dailyPredictions: List[DailyPrediction]

# Helper utilities ------------------------------------------------

def fetch_monthly_sales(product_id: int):
    """
    Pull monthly aggregated sales from BillWise:
    GET /api/ai/monthly-sales/{productId}
    expected format: [{ "date": "YYYY-MM", "unitsSold": 120 }, ...]
    """
    url = f"{BILLWISE_BASE}/api/ai/monthly-sales/{product_id}"
    resp = requests.get(url, timeout=10)
    if resp.status_code == 204:
        return []
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Failed to fetch sales from BillWise: {resp.status_code}")
    return resp.json()

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
            dt = pd.to_datetime(d, errors='coerce')
            if pd.isna(dt):
                continue
            dt = dt.replace(day=1)
        rows.append((dt, int(r.get("unitsSold", 0))))
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows, columns=["ds", "y"])
    df = df.sort_values("ds").drop_duplicates("ds", keep="last")
    df.set_index("ds", inplace=True)
    # reindex monthly continuous range
    idx = pd.date_range(df.index.min(), df.index.max(), freq='MS')
    df = df.reindex(idx, fill_value=0)
    series = df["y"]
    series.index.name = "ds"
    return series

def create_lag_features(series: pd.Series, n_lags=12):
    """
    Create supervised dataset from monthly series using lags.
    """
    df = pd.DataFrame({"y": series})
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df = df.dropna()
    return df

def train_model_for_product(product_id: int):
    """
    Train model and save to disk. Returns model path.
    """
    monthly = fetch_monthly_sales(product_id)
    series = prepare_dataframe(monthly)

    if series.empty:
        raise HTTPException(status_code=404, detail="No historical sales data available for this product.")

    # If very short series (<6 months), we use a naive model (mean) stored as special file
    if len(series) < 6:
        # Save simple mean predictor as dict -> joblib
        mean_val = float(series.mean())
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

def load_model(product_id: int):
    """
    Load model file if exists. Return dict with metadata.
    """
    p1 = os.path.join(MODELS_DIR, f"{product_id}.pkl")
    p2 = os.path.join(MODELS_DIR, f"{product_id}.mean.joblib")
    if os.path.exists(p1):
        data = load(p1)
        return data
    if os.path.exists(p2):
        data = load(p2)
        return data
    return None

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

@app.post("/train/{product_id}")
def train(product_id: int):
    """
    Train (or re-train) model for a given product.
    """
    path = train_model_for_product(product_id)
    return {"status": "trained", "model_path": path}

@app.get("/forecast", response_model=PredictionResponse)
def forecast(product_id: int, months: int = Query(3, ge=1, le=24)):
    """
    Forecast next `months` months for product_id.
    Returns PredictionResponse with 'dailyPredictions' used for monthly entries.
    """
    preds = forecast_using_model(product_id, months=months)

    # Try to fetch product price for revenue estimate (best-effort)
    unit_price = None
    try:
        p = requests.get(f"{BILLWISE_BASE}/api/products/{product_id}", timeout=5)
        if p.status_code == 200:
            j = p.json()
            # try common price fields
            unit_price = j.get("sellingPricePerBaseUnit") or j.get("selling_price_per_base_unit") or j.get("sellingPrice") or j.get("price")
            if unit_price is not None:
                unit_price = float(unit_price)
    except Exception:
        unit_price = None

    today = datetime.date.today()
    daily_preds = []
    total_revenue = 0.0
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



