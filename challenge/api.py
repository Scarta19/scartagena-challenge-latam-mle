# api.py - FastAPI application to serve DelayModel predictions
from typing import List

import pandas as pd
from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel

from challenge.model import DelayModel

# Initialize FastAPI app
app = FastAPI()

# Instantiate and train the DelayModel with dummy data
model = DelayModel()
dummy_data = pd.DataFrame(
    [
        {
            "OPERA": "Grupo LATAM",
            "TIPOVUELO": "I",
            "MES": 1,
            "Fecha-I": "2023-01-01T12:00:00",
            "Fecha-O": "2023-01-01T12:16:00",
        },
        {
            "OPERA": "Sky Airline",
            "TIPOVUELO": "N",
            "MES": 4,
            "Fecha-I": "2023-04-01T12:00:00",
            "Fecha-O": "2023-04-01T12:10:00",
        },
    ]
)
X_train, y_train = model.preprocess(dummy_data, target_column="delay")
model.fit(X_train, y_train)


# Input schema for a single flight
class FlightItem(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int


# Input schema for a request payload with multiple flights
class FlightRequest(BaseModel):
    flights: List[FlightItem]


@app.get("/health", status_code=200)
def get_health() -> dict:
    """Basic health check endpoint."""
    return {"status": "OK"}


@app.post("/predict", status_code=200)
def predict(flight_request: FlightRequest) -> dict:
    """
    Predict delays for incoming flights.
    If required columns are missing or malformed, return 400.
    """
    try:
        # Convert validated input to DataFrame
        df = pd.DataFrame([flight.dict() for flight in flight_request.flights])

        # Validate required features
        required = {"OPERA", "TIPOVUELO", "MES"}
        if not required.issubset(df.columns):
            raise ValueError("Missing required columns")

        # Preprocess and predict
        X = model.preprocess(df)
        preds = model.predict(X)

        return {"predict": preds}
    except Exception:
        raise HTTPException(
            status_code=400, detail="Invalid input data or missing columns"
        )
