# train.py
import joblib
import pandas as pd

from challenge.model import DelayModel

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

model = DelayModel()
X, y = model.preprocess(dummy_data, target_column="delay")
model.fit(X, y)

joblib.dump(model, "model.pkl")
