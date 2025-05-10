import os
import unittest

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from challenge.model import DelayModel


class TestModel(unittest.TestCase):
    FEATURES_COLS = [
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

    TARGET_COL = ["delay"]

    def setUp(self) -> None:
        super().setUp()
        self.model = DelayModel()
        data_path = os.path.join(os.path.dirname(__file__), "../../data/data.csv")
        self.data = pd.read_csv(filepath_or_buffer=data_path)

    def test_model_preprocess_for_training(self):
        features, target = self.model.preprocess(data=self.data, target_column="delay")
        assert isinstance(features, pd.DataFrame)
        assert features.shape[1] == len(self.FEATURES_COLS)
        assert set(features.columns) == set(self.FEATURES_COLS)
        assert isinstance(target, pd.DataFrame)
        assert target.shape[1] == len(self.TARGET_COL)
        assert set(target.columns) == set(self.TARGET_COL)

    def test_model_preprocess_for_serving(self):
        features = self.model.preprocess(data=self.data)
        assert isinstance(features, pd.DataFrame)
        assert features.shape[1] == len(self.FEATURES_COLS)
        assert set(features.columns) == set(self.FEATURES_COLS)

    def test_model_fit(self):
        features, target = self.model.preprocess(data=self.data, target_column="delay")
        _, features_validation, _, target_validation = train_test_split(
            features, target, test_size=0.33, random_state=42
        )
        self.model.fit(features=features, target=target)
        predicted_target = self.model._model.predict(features_validation)
        report = classification_report(
            target_validation, predicted_target, output_dict=True, zero_division=0
        )
        assert "0" in report and "1" in report
        assert 0.0 <= report["0"]["recall"] <= 1.0
        assert 0.0 <= report["1"]["recall"] <= 1.0

    def test_model_predict(self):
        features, target = self.model.preprocess(data=self.data, target_column="delay")
        self.model.fit(features=features, target=target)
        predicted_targets = self.model.predict(features=features)
        assert isinstance(predicted_targets, list)
        assert len(predicted_targets) == features.shape[0]
        assert all(isinstance(pred, int) for pred in predicted_targets)
