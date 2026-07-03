<div align="center">

# 🛡️ Fraud Monitor — FinWise Secure

**Real-time credit card fraud detection powered by machine learning.**

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-F7931E?logo=scikit-learn&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-Frontend-F7DF1E?logo=javascript&logoColor=black)
![Render](https://img.shields.io/badge/Backend-Render-46E3B7?logo=render&logoColor=white)
![Vercel](https://img.shields.io/badge/Frontend-Vercel-000000?logo=vercel&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

[Live Demo](https://machine-learning-project-topaz.vercel.app/) · [API](https://machine-learning-project-u3mm.onrender.com/) · [Report Bug](https://github.com/slashthose/Machine-Learning-Project/issues) · [Request Feature](https://github.com/slashthose/Machine-Learning-Project/issues)

</div>

---

## 📑 Table of Contents

- [About](#-about)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Folder Structure](#-folder-structure)
- [Installation](#-installation)
- [Environment Variables](#-environment-variables)
- [Usage](#-usage)
- [Screenshots](#-screenshots)
- [API Documentation](#-api-documentation)
- [Machine Learning](#-machine-learning)
- [Performance](#-performance)
- [Challenges Faced](#-challenges-faced)
- [Future Improvements](#-future-improvements)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 📌 About

**Fraud Monitor** is a real-time credit card fraud detection system that classifies transactions as **Legitimate** or **Fraudulent** using a machine learning model trained on anonymized, PCA-transformed transaction data.

**Problem it solves:** Credit card fraud costs billions annually, and manual transaction review doesn't scale. This project demonstrates an end-to-end ML pipeline — from a trained classification model to a live, interactive risk-scoring dashboard — that can flag suspicious transactions instantly based on transaction amount, timing, and behavioral features.

**Who it's for:**
- Recruiters and engineers reviewing an end-to-end applied ML project
- Developers learning how to take a trained model from notebook → production API → deployed frontend
- Anyone exploring fraud detection techniques on the classic anonymized credit card dataset

**Project type:** Full-Stack Machine Learning Application
**Status:** 🚧 In Development

---

## ✨ Features

- 🔍 **Real-time transaction analysis** — submit a transaction and get an instant classification
- ⚡ **Quick-load sample transactions** — test the model against known transaction profiles without manual entry
- 📊 **Risk scoring readout** — Fraud Risk % and Legitimate Score % displayed as visual meters
- 🚦 **Risk level tagging** — transactions are labeled Low / Medium / High risk
- 🧠 **ML-backed classification** — Random Forest model trained on PCA-transformed features (V1–V28), Amount, and Time
- 🌐 **REST API backend** — FastAPI service exposing a prediction endpoint, decoupled from the frontend
- 🖥️ **Custom dashboard UI** — dark-themed, purpose-built HTML/CSS/JS interface (no framework overhead)
- 🛟 **Graceful degradation** — if the trained model isn't available, the API falls back to a heuristic estimate rather than crashing
- ☁️ **Split cloud deployment** — frontend on Vercel, backend on Render, model artifacts served via GitHub Releases

---

## 🛠️ Tech Stack

**Frontend:**
- HTML5, CSS3, Vanilla JavaScript (custom dashboard, no framework)

**Backend:**
- Python 3.13
- FastAPI
- Uvicorn (ASGI server)

**Machine Learning:**
- scikit-learn (Random Forest Classifier)
- pandas, NumPy
- pickle (model serialization)

**Deployment:**
- Backend → [Render](https://render.com)
- Frontend → [Vercel](https://vercel.com)
- Model artifacts (`model.pkl`, `scalers.pkl`) → hosted as GitHub Release assets, downloaded at server startup

**Dataset:**
- Kaggle — Credit Card Fraud Detection dataset (anonymized, PCA-transformed features V1–V28)

---

## 📂 Folder Structure

```
Machine-Learning-Project/
├── backend/
│   ├── main.py              # FastAPI app entry point, model loading, prediction endpoint
│   ├── fraud_detection.py   # Model training pipeline (Random Forest + scalers)
│   ├── requirements.txt     # Backend Python dependencies
│   └── (model.pkl, scalers.pkl generated locally / downloaded at runtime — not committed)
├── frontend/
│   ├── index.html           # Dashboard UI
│   ├── style.css
│   └── script.js            # Calls backend API, renders analysis readout
├── README.md
└── .gitattributes           # Ensures binary files (e.g. .pkl) aren't corrupted by line-ending conversion
```

> **Note:** `model.pkl`, `scalers.pkl`, and the training dataset (`creditcard.csv`) are intentionally **not committed to Git** due to file size (100MB+ limits on standard GitHub repos). See [Machine Learning](#-machine-learning) below for how they're provisioned.

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- pip
- Git

### 1. Clone the repository
```bash
git clone https://github.com/slashthose/Machine-Learning-Project.git
cd Machine-Learning-Project
```

### 2. Set up a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install backend dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 4. Obtain the trained model
The model isn't stored in this repo. Either:
- **Download it automatically** — the backend fetches `model.pkl` and `scalers.pkl` from the project's GitHub Release the first time it starts (see `load_artifacts()` in `main.py`), or
- **Train it yourself** — download the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud), place `creditcard.csv` in `backend/`, and run:
```bash
python fraud_detection.py
```

### 5. Run the backend
```bash
uvicorn main:app --reload --port 8000
```
API will be available at `http://localhost:8000`.

### 6. Run the frontend
Open `frontend/index.html` directly in a browser, or serve it locally:
```bash
cd frontend
python -m http.server 5500
```
Then visit `http://localhost:5500`. Update the API base URL in `script.js` to point to your local backend (`http://localhost:8000`) if testing locally.

---

## 🔐 Environment Variables

Create a `.env` file inside `backend/` (not committed to Git):

```env
MODEL_URL=https://github.com/slashthose/Machine-Learning-Project/releases/download/v1.0-model/model.pkl
SCALERS_URL=https://github.com/slashthose/Machine-Learning-Project/releases/download/v1.0-model/scalers.pkl
MODEL_PATH=model.pkl
SCALERS_PATH=scalers.pkl
PORT=8000
```

> No API keys or secrets are currently required — this project doesn't use external paid APIs or authentication.

---

## 🚀 Usage

1. Visit the **[live dashboard](https://machine-learning-project-topaz.vercel.app/)**.
2. Either:
   - Select a **known sample transaction** from the dropdown, or
   - Manually enter an **Amount** and **Time (seconds since first transaction)**.
3. Click **Analyze Transaction**.
4. The **Analysis Readout** panel returns:
   - Classification: `LEGITIMATE` or `FRAUDULENT`
   - Risk level: `LOW RISK` / `MEDIUM RISK` / `HIGH RISK`
   - Fraud Risk % and Legitimate Score % (visual meters)
   - A recommended action (e.g., *"Approve and process the transaction"* or *"Flag for manual review"*)

---

## 🖼️ Screenshots

> Add screenshots to a `/screenshots` folder and update the paths below.

| Dashboard | Analysis Readout |
|---|---|
| ![Dashboard](./screenshots/dashboard.png) | ![Analysis Readout](./screenshots/analysis-readout.png) |

---

## 📡 API Documentation

### `POST /predict`

Analyzes a transaction and returns a fraud classification.

**Request Body:**
```json
{
  "amount": 150.00,
  "time": 50000,
  "features": [0.0, 0.0, "...", 0.0]
}
```
*(`features` = V1–V28 PCA-transformed values; defaults to zeroed values for manual entries not sourced from a known sample)*

**Response:**
```json
{
  "prediction": "legitimate",
  "fraud_risk_percent": 0,
  "legitimate_score_percent": 100,
  "risk_level": "low",
  "recommendation": "Approve and process the transaction.",
  "using_fallback_model": false
}
```

**Status Codes:**
| Code | Meaning |
|---|---|
| `200` | Prediction successful |
| `422` | Invalid request payload |
| `500` | Internal server / model loading error |

> Exact field names may differ slightly depending on your current `main.py` implementation — update this section to match your actual schema before publishing.

---

## 🤖 Machine Learning

- **Model:** Random Forest Classifier (scikit-learn), selected as the best-performing model after evaluation against alternative classifiers
- **Dataset:** [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) — anonymized transactions with PCA-transformed features `V1`–`V28`, plus `Amount` and `Time`
- **Preprocessing:** `Amount` and `Time` are scaled separately using dedicated scalers (`scaler_time`, `scaler_amount`) before being passed to the model
- **Training:** Handled in `fraud_detection.py` — includes data loading, scaling, train/test split, model fitting, and evaluation
- **Inference:** The FastAPI backend loads the serialized model and scalers via `pickle` and returns a classification + probability score per request
- **Evaluation Metric:** ~99.99% accuracy on the held-out test set (severely imbalanced dataset — precision/recall/F1 and confusion matrix are more meaningful than raw accuracy given the extreme class imbalance in fraud data)
- **Libraries:** scikit-learn, pandas, NumPy, pickle

**Model & Scaler Provisioning:**
Because the trained model (400–600MB) and dataset exceed GitHub's standard file size limits, they are **not committed to this repository**. Instead:
- `model.pkl` and `scalers.pkl` are published as assets on a [GitHub Release](https://github.com/slashthose/Machine-Learning-Project/releases)
- On startup, `main.py`'s `load_artifacts()` downloads them automatically if not already present locally
- If the files fail to download, the API falls back to a simple heuristic estimate rather than crashing, and flags the response accordingly

---

## ⚡ Performance

- **Inference time:** Sub-second per transaction (single Random Forest prediction, no batch overhead)
- **Scalability:** Stateless FastAPI service — horizontally scalable behind a load balancer if needed
- **Cold starts:** On Render's free tier, the first request after inactivity may be slower due to model download + service spin-up
- **Optimization opportunities:** Model size (400–600MB) is large for a Random Forest and is a candidate for compression via reduced `n_estimators`/`max_depth`, which would also speed up cold-start downloads

---

## 🧗 Challenges Faced

- **Binary file corruption via Git line-ending conversion** — the model `.pkl` file was corrupted in transit due to Git's `autocrlf` normalizing line endings in a binary file, causing `_pickle.UnpicklingError` on deploy
- **GitHub's 100MB file size limit** — the trained model (400–600MB) couldn't be committed directly, requiring a rethink of the deployment/model-loading architecture
- **Decoupling training from serving** — ensuring the API can start reliably even when the model artifact isn't bundled with the code, without silently masking failures
- **Balancing graceful degradation with correctness** — the app intentionally falls back to a heuristic rather than crashing when the model is unavailable, which improves uptime but requires clear signaling to the frontend/user that predictions aren't from the real model

---

## 🔮 Future Improvements

1. Add authentication for the API (JWT-based)
2. Add rate limiting to the `/predict` endpoint
3. Log all predictions to a database for audit/history tracking
4. Add a batch-prediction endpoint for CSV uploads
5. Build a model retraining pipeline triggered on new labeled data
6. Add SHAP/feature-importance explainability to the analysis readout
7. Compress the model (fewer estimators / max depth tuning) to speed up cold starts
8. Add automated CI/CD (GitHub Actions) for testing before deploy
9. Add unit tests for the FastAPI endpoints
10. Add integration tests covering the fallback heuristic path
11. Add a Dockerfile for consistent local/prod parity
12. Add real-time transaction streaming (WebSocket) demo mode
13. Add user accounts to save transaction history
14. Add multi-model comparison (e.g., XGBoost vs Random Forest vs Neural Net) in the readout
15. Add dark/light theme toggle to the dashboard
16. Add mobile-responsive layout improvements
17. Add model versioning so old model artifacts can be rolled back to

---

## 🧪 Testing

- **Manual testing:** Verified via the live dashboard using both known sample transactions and manual input across varying amount/time ranges
- **Edge cases to cover:** zero/negative amounts, extreme time values, missing PCA features, malformed request payloads
- **Unit testing:** Not yet implemented — recommended: `pytest` + `httpx.AsyncClient` for FastAPI endpoint testing
- **Security testing:** Not yet implemented — recommended: basic input validation fuzzing and dependency vulnerability scanning (`pip-audit`)
- **Performance testing:** Not yet implemented — recommended: load testing the `/predict` endpoint with `locust` or `k6`

---

## ☁️ Deployment

| Component | Platform | Notes |
|---|---|---|
| Backend (FastAPI) | [Render](https://render.com) | Auto-deploys from `main` branch; downloads model artifacts from GitHub Releases at startup |
| Frontend (Dashboard) | [Vercel](https://vercel.com) | Static site deploy, auto-deploys from `main` branch |
| Model Artifacts | GitHub Releases | Hosted as binary release assets to bypass Git's 100MB file limit |

**To deploy your own copy:**
1. Fork this repo
2. Create a GitHub Release and attach your own `model.pkl` / `scalers.pkl`
3. Connect the `backend/` folder as a new Web Service on Render, set the start command to `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Connect the `frontend/` folder as a new Vercel project
5. Update `MODEL_URL` / `SCALERS_URL` in your backend environment variables to point to your release assets

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to your branch: `git push origin feature/your-feature-name`
5. Open a Pull Request describing your changes

Please open an issue first for significant changes so we can discuss the approach.

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

## 🙏 Acknowledgements

- [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- [scikit-learn](https://scikit-learn.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Render](https://render.com) and [Vercel](https://vercel.com) for free-tier hosting

---

<div align="center">

**Built by [Sakshi](https://github.com/slashthose)**

⭐ Star this repo if you found it useful!

</div>
