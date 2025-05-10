from typing import List
from typing import Tuple
from typing import Union

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
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
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        df = data.copy()
        if target_column:
            X = df[self._features]
            y = df[target_column]
            return X, y
        return df[self._features]

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        categorical_cols = features.select_dtypes(include="object").columns.tolist()
        numeric_cols = features.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols),
            ]
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
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            List[int]: predicted targets.
        """
        return self._model.predict(features).tolist()
