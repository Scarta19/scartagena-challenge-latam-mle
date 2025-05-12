from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel

app = FastAPI()
model = None


class FlightItem(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int


class FlightRequest(BaseModel):
    flights: List[FlightItem]


@app.on_event("startup")
def load_model():
    global model
    try:
        print("Loading model...")
        model_path = os.path.join(os.getcwd(), "model.pkl")
        model = joblib.load(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")



@app.get("/health", status_code=200)
def get_health() -> dict:
    return {"status": "OK"}


@app.post("/predict")
def predict(flight_request: FlightRequest):
    try:
        df = pd.DataFrame([f.dict() for f in flight_request.flights])

        required_cols = {"OPERA", "TIPOVUELO", "MES"}
        if not required_cols.issubset(df.columns):
            raise ValueError("Missing required columns")

        valid_operas = {
            "American Airlines",
            "Air Canada",
            "Air France",
            "Aeromexico",
            "Aerolineas Argentinas",
            "Austral",
            "Avianca",
            "Alitalia",
            "British Airways",
            "Copa Air",
            "Delta Air",
            "Gol Trans",
            "Iberia",
            "K.L.M.",
            "Qantas Airways",
            "United Airlines",
            "Grupo LATAM",
            "Sky Airline",
            "Latin American Wings",
            "Plus Ultra Lineas Aereas",
            "JetSmart SPA",
            "Oceanair Linhas Aereas",
            "Lacsa",
        }

        if not df["OPERA"].isin(valid_operas).all():
            raise ValueError("Invalid OPERA values")

        if not df["TIPOVUELO"].isin(["I", "N"]).all():
            raise ValueError("Invalid TIPOVUELO values")

        if not df["MES"].between(1, 12).all():
            raise ValueError("Invalid MES values")

        X = model.preprocess(df)
        preds = model.predict(X)
        return {"predict": preds}

    except Exception:
        raise HTTPException(status_code=400, detail="Invalid input data")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("challenge.api:app", host="0.0.0.0", port=8080)
