## 📥 Clone the Repository

```bash
git clone https://github.com/<your-username>/billwise-forecast-service.git
cd billwise-forecast-service
```

---

## 🧪 Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

**Windows (PowerShell):**

```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**

```cmd
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

You should see:

```
(venv)
```

---

## 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ⚙️ Environment Variables

The ML service communicates with the BillWise backend.

### Required

```bash
BILLWISE_BASE=http://localhost:8080
```

### Optional

```bash
BILLWISE_API_KEY=your_api_token
PRICE_CACHE_TTL_SECONDS=300
```

#### Set Environment Variables

**Windows (PowerShell):**

```powershell
$env:BILLWISE_BASE="http://localhost:8080"
```

**Linux / macOS:**

```bash
export BILLWISE_BASE=http://localhost:8080
```

---

## ▶️ Run the ML Service

```bash
uvicorn main:app --reload --port 5000
```

Expected output:

```
Uvicorn running on http://127.0.0.1:5000
Application startup complete.
```

---

## 📖 API Documentation (Swagger)

Open in browser:

```
http://127.0.0.1:5000/docs
```

---

## 🔌 Available Endpoints

### Train model for a product

```
POST /train/{product_id}
```

### Train models for all products

```
POST /train-all
```

### Forecast demand

```
GET /forecast?product_id=1&months=3
```

### Fetch product price

```
GET /price?product_id=1
```

---

## 📊 Forecast Evaluation

* Uses **time-aware 80/20 train–test split**
* Evaluation metrics:

  * Mean Absolute Error (MAE)
  * Root Mean Squared Error (RMSE)
* Metrics are persisted in the BillWise backend

