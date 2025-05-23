# model.py - Updated and linted version with safe preprocessing
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


class DelayModel:
    def __init__(self):
        self._model = None
        self._features = [
            "OPERA_Latin American Wings",
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air",
        ]

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        df = data.copy()

        # If delay column is missing, compute it from timestamps
        if "delay" not in df.columns:
            if "Fecha-O" in df.columns and "Fecha-I" in df.columns:
                df["Fecha-O"] = pd.to_datetime(df["Fecha-O"])
                df["Fecha-I"] = pd.to_datetime(df["Fecha-I"])
                df["min_diff"] = (df["Fecha-O"] - df["Fecha-I"]).dt.total_seconds() / 60
                df["delay"] = np.where(df["min_diff"] > 15, 1, 0)

        # Use encoded features if already available
        if set(self._features).issubset(df.columns):
            X = df[self._features].copy()
        else:
            # Encode raw features
            df_encoded = pd.get_dummies(
                df[["OPERA", "TIPOVUELO", "MES"]],
                columns=["OPERA", "TIPOVUELO", "MES"],
            )
            for feature in self._features:
                if feature not in df_encoded.columns:
                    df_encoded[feature] = 0
            X = df_encoded[self._features]

        if target_column:
            y = df[[target_column]]
            return X, y

        return X

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame,
    ) -> None:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[("num", numeric_transformer, features.columns.tolist())],
            remainder="passthrough",
        )

        model = XGBClassifier(random_state=42, n_jobs=-1)

        self._model = ImbPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("oversample", RandomOverSampler(random_state=42)),
                ("classifier", model),
            ]
        )

        self._model.fit(features, target)

    def predict(self, features: pd.DataFrame) -> List[int]:
        if self._model is None:
            raise ValueError(
                "Model has not been trained. Call `fit()` before `predict()`."
            )
        return self._model.predict(features).tolist()
