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
        self._model = None  # Model should be saved in this attribute.
        self._features = [
            "OPERA_Aerolineas Argentinas",
            "MES_7",
            "MES_11",
            "OPERA_LATAM",
            "OPERA_SKY Airline",
            "MES_12",
            "TIPOVUELO_I",
            "OPERA_JetSMART SPA",
            "MES_10",
            "MES_1",
        ]

    def preprocess(
        self, data: pd.DataFrame, target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or prediction.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        df = data.copy()

        # Delay computation based on time difference
        if "delay" not in df.columns:
            df["Fecha-O"] = pd.to_datetime(df["Fecha-O"])
            df["Fecha-I"] = pd.to_datetime(df["Fecha-I"])
            df["min_diff"] = (df["Fecha-O"] - df["Fecha-I"]).dt.total_seconds() / 60
            df["delay"] = np.where(df["min_diff"] > 15, 1, 0)

        # One-hot encoding to generate expected features
        df_encoded = pd.get_dummies(df[["OPERA", "TIPOVUELO", "MES"]], columns=["OPERA", "TIPOVUELO", "MES"])

        # Fill missing expected features with 0
        for feature in self._features:
            if feature not in df_encoded.columns:
                df_encoded[feature] = 0

        X = df_encoded[self._features]

        if target_column:
            y = df[[target_column]]
            return X, y
        return X

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        preprocessor = ColumnTransformer(
            transformers=[("num", numeric_transformer, features.columns.tolist())],
            remainder="passthrough"
        )

        model = XGBClassifier(random_state=42, n_jobs=-1)

        self._model = ImbPipeline(steps=[
            ("preprocessor", preprocessor),
            ("oversample", RandomOverSampler(random_state=42)),
            ("classifier", model)
        ])

        self._model.fit(features, target)

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            List[int]: predicted targets.
        """
        return self._model.predict(features).tolist()
