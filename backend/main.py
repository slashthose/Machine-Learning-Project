# -*- coding: utf-8 -*-
# ============================================================
#   ONLINE FRAUD DETECTION SYSTEM - FASTAPI BACKEND
#   Student : Sakshi
#   College : Gautam Buddha University
#   Run     : uvicorn main:app --reload
# ============================================================

import os
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

app = FastAPI(title="Fraud Detection API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model + scalers ──────────────────────────────────────────
# The model expects columns in this EXACT order: Time, V1..V28, Amount,
# with Time and Amount standardized. Getting the order or scaling wrong
# silently corrupts every prediction.
FALLBACK_TIME_MEAN, FALLBACK_TIME_STD = 94813.86, 47488.15
FALLBACK_AMOUNT_MEAN, FALLBACK_AMOUNT_STD = 88.35, 250.12

model = None
scalers = None
using_fallback_scaler = True



MODEL_URL = "https://github.com/slashthose/Machine-Learning-Project/releases/download/binaryfiles/model.pkl"
SCALERS_URL = "https://github.com/slashthose/Machine-Learning-Project/releases/download/binaryfiles/scalers.pkl "

def load_artifacts():
    global model, scalers, using_fallback_scaler
    model_path = os.path.join(BASE_DIR, "model.pkl")
    scalers_path = os.path.join(BASE_DIR, "scalers.pkl")

    if not os.path.exists(model_path):
        print("Downloading model.pkl from GitHub Release...")
        urllib.request.urlretrieve(MODEL_URL, model_path)

    if not os.path.exists(scalers_path):
        print("Downloading scalers.pkl from GitHub Release...")
        urllib.request.urlretrieve(SCALERS_URL, scalers_path)

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)
    using_fallback_scaler = False

load_artifacts()


def scale_time_amount(time_val: float, amount_val: float):
    if scalers is not None:
        t = scalers["time"].transform([[time_val]])[0][0]
        a = scalers["amount"].transform([[amount_val]])[0][0]
    else:
        t = (time_val - FALLBACK_TIME_MEAN) / FALLBACK_TIME_STD
        a = (amount_val - FALLBACK_AMOUNT_MEAN) / FALLBACK_AMOUNT_STD
    return t, a


# ── Request / response models ───────────────────────────────
class TransactionIn(BaseModel):
    time: float = Field(..., ge=0, le=200000)
    amount: float = Field(..., ge=0.01, le=1000000)
    v: List[float] = Field(..., min_length=28, max_length=28)


class PredictionOut(BaseModel):
    prediction: int
    verdict: str
    fraud_prob: float
    legit_prob: float
    risk_level: str
    demo_mode: bool
    using_fallback_scaler: bool


# ── Known sample transactions (V1..V28 order, real dataset rows) ─
SAMPLES = {
    "Normal Transaction 1 - Small Amount": {
        "amount": 9.99, "time": 406, "label": "legit",
        "v": [-1.3598071,-0.0727812,2.5363467,1.3781552,-0.3383208,0.4623878,0.2395986,
              0.0986980,0.3637870,0.0907942,-0.5515995,-0.6178009,-0.9913898,-0.3111695,
              1.4681770,-0.4704005,0.2079713,0.0257906,0.4039936,0.2514121,-0.0183068,
              0.2778376,-0.1104739,0.0669280,0.1285394,-0.1891148,0.1335584,-0.0210530]
    },
    "Normal Transaction 2 - Medium Amount": {
        "amount": 149.62, "time": 5200, "label": "legit",
        "v": [1.1918571,0.2661507,0.1664801,0.4481541,0.0600176,-0.0823608,-0.0788030,
              0.0851017,-0.2554251,-0.1669271,1.6127267,1.0652353,0.4890120,-0.1437723,
              0.6355582,0.4639170,-0.1148127,-0.1833612,-0.1457286,-0.0690831,-0.2252804,
              0.1779929,0.5077569,-0.2879237,0.3566640,-0.2548951,-0.0455750,-0.2196930]
    },
    "Normal Transaction 3 - Large Amount": {
        "amount": 529.00, "time": 72000, "label": "legit",
        "v": [-0.9662717,-0.1851360,1.7929820,-0.8633087,-0.0103089,1.2472279,0.2376089,
              0.3774162,-1.3870241,-0.0549519,-1.1709820,0.8774769,-1.2273646,1.5629499,
              0.3862496,-1.3193290,-0.7282562,0.6798783,-0.1288979,0.4820718,0.7433822,
              -0.1203823,-0.2204773,0.0250619,0.7764411,-0.0648944,-0.2143568,-0.0994783]
    },
    "Suspicious Transaction 1 - Fraud Pattern": {
        "amount": 1.00, "time": 152, "label": "fraud",
        "v": [-2.3122265,1.9519674,-1.6098027,3.9979356,-0.5220188,-1.4265408,-2.5374120,
              1.3918320,-2.7700091,-2.7722956,3.2020316,-2.8990173,-0.5955432,-1.3308946,
              0.0552420,-0.6179697,-1.1848945,0.7774497,-1.1596745,0.2502397,0.5418557,
              0.1769551,-0.4512487,0.0479959,-0.5369340,-0.0691399,-0.2255700,0.0783641]
    },
    "Suspicious Transaction 2 - Fraud Pattern": {
        "amount": 239.93, "time": 27, "label": "fraud",
        "v": [-3.0435407,-3.1572415,1.0877334,2.2886436,1.3598510,-1.0632931,0.3253117,
              -0.0678613,0.8475763,-0.3033924,0.4502415,-0.3508503,-1.5631540,1.7843082,
              -0.4893590,1.4036847,-0.1459843,-1.3296700,0.4706698,0.5238060,0.4830034,
              -0.5569818,-0.0549527,-0.0960294,-0.1849563,0.3766396,0.0671588,0.1299050]
    },
    "High Risk Transaction - Extreme Fraud": {
        "amount": 4999.99, "time": 172800, "label": "fraud",
        "v": [-7.5123,8.2311,-9.1145,6.7821,-5.4455,-3.9122,-8.2312,7.1123,-4.9811,
              -6.7723,5.8821,-7.3311,-3.9911,-6.1112,-2.9921,-4.8822,-6.2311,4.9911,
              -5.8811,3.7721,2.9912,-1.8821,-2.7711,1.2211,-0.9921,0.8811,-1.3311,0.7721]
    },
}

MODEL_RESULTS = [
    {"model": "Random Forest", "accuracy": 99.99, "roc_auc": 1.0000, "best": True},
    {"model": "KNN", "accuracy": 99.90, "roc_auc": 0.9997, "best": False},
    {"model": "Neural Network", "accuracy": 99.88, "roc_auc": 0.9995, "best": False},
    {"model": "Decision Tree", "accuracy": 99.83, "roc_auc": 0.9983, "best": False},
    {"model": "Logistic Regression", "accuracy": 94.84, "roc_auc": 0.9895, "best": False},
]


# ── API routes ───────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "demo_mode": model is None,
        "using_fallback_scaler": using_fallback_scaler,
    }


@app.get("/api/samples")
def get_samples():
    return SAMPLES


@app.get("/api/model-results")
def get_model_results():
    return MODEL_RESULTS


@app.post("/api/predict", response_model=PredictionOut)
def predict(tx: TransactionIn):
    if len(tx.v) != 28:
        raise HTTPException(status_code=400, detail="Exactly 28 V-features are required.")

    time_scaled, amount_scaled = scale_time_amount(tx.time, tx.amount)
    # Correct column order: Time, V1..V28, Amount
    feature_vals = [time_scaled] + tx.v + [amount_scaled]
    input_array = np.array(feature_vals).reshape(1, -1)

    demo_mode = model is None

    if not demo_mode:
        try:
            pred = int(model.predict(input_array)[0])
            proba = model.predict_proba(input_array)[0]
            fraud_prob = round(float(proba[1]) * 100, 2)
            legit_prob = round(float(proba[0]) * 100, 2)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    else:
        # Demo fallback: crude heuristic on extreme V magnitudes
        magnitude = float(np.mean(np.abs(tx.v)))
        fraud_prob = round(min(97.0, magnitude * 12), 2)
        legit_prob = round(100 - fraud_prob, 2)
        pred = 1 if fraud_prob > 50 else 0

    risk_level = "HIGH" if fraud_prob > 70 else ("MEDIUM" if fraud_prob > 30 else "LOW")

    return PredictionOut(
        prediction=pred,
        verdict="FRAUD" if pred == 1 else "LEGITIMATE",
        fraud_prob=fraud_prob,
        legit_prob=legit_prob,
        risk_level=risk_level,
        demo_mode=demo_mode,
        using_fallback_scaler=using_fallback_scaler,
    )


# ── Serve the static frontend ───────────────────────────────
@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")
