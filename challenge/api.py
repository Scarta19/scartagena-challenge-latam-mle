import fastapi
import pandas as pd
from fastapi import Request

from challenge.model import DelayModel

app = fastapi.FastAPI()

# Cargamos el modelo una vez
model = DelayModel()


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


@app.post("/predict", status_code=200)
async def post_predict(request: Request) -> dict:
    body = await request.json()
    df = pd.DataFrame(body)

    # Preprocesar y predecir
    features = model.preprocess(df)
    predictions = model.predict(features)

    return {"predictions": predictions}
