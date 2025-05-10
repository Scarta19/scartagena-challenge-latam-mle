from typing import List, Tuple, Union
import pandas as pd
import numpy as np

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


class DelayModel:
    def __init__(self):
        self._model = None  
        self._pipeline_features = ["OPERA", "TIPOVUELO", "MES"]
        self._trained_columns = None

    def preprocess(
        self, data: pd.DataFrame, target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.Series], pd.DataFrame]:
        """
        Compute delay column and extract relevant raw features.

        Returns:
            Tuple of raw features and target if target_column is provided,
            otherwise just the raw features.
        """
        df = data.copy()

        if "delay" not in df.columns:
            df["Fecha-O"] = pd.to_datetime(df["Fecha-O"])
            df["Fecha-I"] = pd.to_datetime(df["Fecha-I"])
            df["min_diff"] = (df["Fecha-O"] - df["Fecha-I"]).dt.total_seconds() / 60
            df["delay"] = np.where(df["min_diff"] > 15, 1, 0)

        features = df[self._pipeline_features]

        if target_column:
            return features, df[target_column]
        return features

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        """
        Train the model pipeline on features and target.
        """
        categorical_features = features.select_dtypes(include=["object", "category"]).columns.tolist()
        numeric_features = features.select_dtypes(include=["int64", "float64"]).columns.tolist()

        # Define transformers
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        preprocessor = ColumnTransformer(transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numeric_transformer, numeric_features)
        ])

        classifier = XGBClassifier(random_state=42, n_jobs=-1)

        self._model = ImbPipeline(steps=[
            ("preprocessor", preprocessor),
            ("oversample", RandomOverSampler(random_state=42)),
            ("classifier", classifier)
        ])

        self._model.fit(features, target)

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays using the trained pipeline.
        """
        if self._model is None:
            raise ValueError("Model has not been trained. Call `fit()` before `predict()`.")
        return self._model.predict(features).tolist()
