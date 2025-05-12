# ✈️ Flight Delay Prediction API – Challenge Documentation

## 👨‍💻 Author

- **Name:** Santiago Cartagena Agudelo
- **Email:** santwhm@gmail.com
- **GitHub:** [https://github.com/Scarta19/scartagena-challenge-latam-mle](https://github.com/Scarta19/scartagena-challenge-latam-mle)
- **API URL:** [https://ml-api-797695519065.us-central1.run.app](https://ml-api-797695519065.us-central1.run.app)

---

## 🧩 Project Overview

This project delivers a Flight Delay Prediction API built with FastAPI and deployed on Google Cloud Run. The full ML lifecycle has been automated: model training, validation, unit testing, load testing, and continuous delivery. This as a response for LATAM Software Engineer (ML & LLMs) Interview Challenge.

---

## ⚙️ Tech Stack

- **Language:** Python 3.11
- **Web Framework:** FastAPI + Gunicorn + UvicornWorker
- **Deployment:** Docker + Cloud Run (GCP)
- **CI/CD:** GitHub Actions (CI for testing, CD for production deployment)
- **Testing:** Pytest (unit), Locust (stress/load)
- **Model Serialization:** `joblib`

---

## 🧠 Model Details

The model is trained on the following features:
- `OPERA`: Airline operator (validated using a whitelist)
- `TIPOVUELO`: Flight type (Domestic or International)
- `MES`: Month of the year

The model encapsulates its own preprocessing pipeline via the `model.preprocess()` method, allowing seamless predictions.

---

## 🔬 Testing Strategy

### Unit Tests (Pytest)
- Coverage includes `/health` and `/predict` endpoints
- Edge cases and input validation were rigorously tested
- **Test Coverage:** 42%

### Load Testing (Locust)
- Simulated 100 concurrent users for 60 seconds
- Results:
  - **Total requests:** 6110
  - **Avg response time:** 294ms
  - **Failure rate:** 0.00%
  - **95th percentile latency:** 680ms

---

## 🛠️ Project Structure

## 🛠️ Project Structure

```plaintext
challenge_MLE/
├── challenge/              # Source code: FastAPI app and model logic
├── tests/                  # Unit and stress test modules
├── Dockerfile              # Optimized production-ready container
├── Makefile                # CLI automation: install, test, stress-test
├── .github/
│   └── workflows/          # CI (ci.yml) and CD (cd.yml) pipelines
├── model.pkl               # Trained and serialized model
├── requirements.txt        # Runtime dependencies
├── requirements-dev.txt    # Dev tools and linters
├── requirements-test.txt   # Test-only dependencies
└── docs/
    └── challenge.md        # Documentation (this file)



---

## 🚀 Production Deployment

- **Platform:** Google Cloud Run
- **Region:** `us-central1`
- **Public URL:**
  [https://ml-api-797695519065.us-central1.run.app](https://ml-api-797695519065.us-central1.run.app)

---

## 🔁 CI/CD Pipelines

- **CI (ci.yml):** Triggered on every push to `develop`. Runs all tests and generates coverage reports.
- **CD (cd.yml):** Automatically deploys the API to Cloud Run upon merge to `main`.

Environment secret used:
- `GCP_SA_KEY`: Encrypted Google Cloud service account credentials for authentication and deployment.
