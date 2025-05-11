# api.py
from typing import List

import pandas as pd
from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel

from challenge.model import DelayModel

app = FastAPI()
model = DelayModel()

# Dummy training data for initial model setup
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
X, y = model.preprocess(dummy_data, target_column="delay")
model.fit(X, y)


class FlightItem(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int


class FlightRequest(BaseModel):
    flights: List[FlightItem]


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


@app.post("/predict")
def predict(flight_request: FlightRequest):
    try:
        df = pd.DataFrame([f.dict() for f in flight_request.flights])

        required_cols = {"OPERA", "TIPOVUELO", "MES"}
        if not required_cols.issubset(df.columns):
            raise ValueError("Missing required columns")

        # Validate values for each required feature
        valid_operas = {
            "Grupo LATAM",
            "Sky Airline",
            "Latin American Wings",
            "Copa Air",
        }
        valid_tipos = {"I", "N"}
        valid_meses = set(range(1, 13))

        if (
            not df["OPERA"].isin(valid_operas).all()
            or not df["TIPOVUELO"].isin(valid_tipos).all()
            or not df["MES"].isin(valid_meses).all()
        ):
            raise ValueError("Invalid column values")

        X = model.preprocess(df)
        preds = model.predict(X)
        return {"predict": preds}

    except Exception:
        raise HTTPException(status_code=400, detail="Invalid input data")
