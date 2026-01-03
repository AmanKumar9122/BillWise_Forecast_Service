✨ Features
📈 Demand forecasting using Random Forest Regressor
🧠 Lag-based time-series feature engineering
🔁 Automatic fallback for sparse data
💾 Forecast persistence into BillWise backend
📊 Accuracy evaluation using MAE & RMSE
🧮 Revenue estimation using real product prices
🚀 FastAPI-based REST service
⚡ Async price lookup with TTL caching
🔗 Seamless integration with Spring Boot backend

🏗️ Architecture Overview
BillWise Backend (Spring Boot)
        ↑        ↓
 REST APIs (JSON)
        ↑        ↓
BillWise ML Service (FastAPI + Python)

📁 Project Structure
billwise-forecast-service/
├── main.py                 # FastAPI app + ML logic
├── price_endpoint.py       # Async product price lookup
├── test_endpoints.py       # Quick API test script
├── requirements.txt        # Python dependencies
├── models/                 # Saved ML models
└── README.md               # This file

🧰 Prerequisites
Make sure the following are installed:
Python 3.9+
Git
BillWise Backend running (Spring Boot)

Check Python version:
python --version

📥 Clone the Repository
git clone https://github.com/<your-username>/billwise-forecast-service.git
cd billwise-forecast-service

🧪 Set Up Virtual Environment
Create virtual environment
python -m venv venv

Activate virtual environment
Windows (PowerShell):
.\venv\Scripts\Activate.ps1

Windows (CMD):
venv\Scripts\activate

Linux / macOS:
source venv/bin/activate

You should see:
(venv)

📦 Install Dependencies
pip install -r requirements.txt

⚙️ Environment Variables
The ML service needs to know where the BillWise backend is running.

Required
BILLWISE_BASE=http://localhost:8080

Optional
BILLWISE_API_KEY=your_api_token_here
PRICE_CACHE_TTL_SECONDS=300

Set variables
Windows (PowerShell):
$env:BILLWISE_BASE="http://localhost:8080"

Linux / macOS:
export BILLWISE_BASE=http://localhost:8080

▶️ Run the ML Service
uvicorn main:app --reload --port 5000

Expected output:
Uvicorn running on http://127.0.0.1:5000
Application startup complete.

📖 API Documentation
Open in browser:
http://127.0.0.1:5000/docs
Swagger UI will show all available endpoints.

🔌 Available Endpoints
🔹 Train Model
POST /train/{product_id}

🔹 Train All Products
POST /train-all

🔹 Forecast Demand
GET /forecast?product_id=1&months=3

🔹 Get Product Price
GET /price?product_id=1

📊 Evaluation Metrics:
Uses temporal 80/20 train-test split

Computes:
Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
Metrics are persisted into BillWise backend
