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
        self.model = DelayModel()
        data_path = os.path.join(os.path.dirname(__file__), "../../data/data.csv")
        self.data = pd.read_csv(filepath_or_buffer=data_path)

        # Crear artificialmente una columna 'delay'
        self.data["delay"] = [0] * (len(self.data) // 2) + [1] * (len(self.data) - len(self.data) // 2)

    def test_model_preprocess_for_training(self):
        features, target = self.model.preprocess(self.data, target_column="delay")

        assert isinstance(features, pd.DataFrame)
        assert set(features.columns) == set(self.FEATURES_COLS)

        assert isinstance(target, pd.DataFrame)
        assert set(target.columns) == set(self.TARGET_COL)

    def test_model_preprocess_for_serving(self):
        features = self.model.preprocess(self.data)

        assert isinstance(features, pd.DataFrame)
        assert set(features.columns) == set(self.FEATURES_COLS)

    def test_model_fit(self):
        features, target = self.model.preprocess(self.data, target_column="delay")
        _, X_val, _, y_val = train_test_split(features, target, test_size=0.33, random_state=42)

        self.model.fit(features, target)

        predicted = self.model._model.predict(X_val)
        report = classification_report(y_val, predicted, output_dict=True)

        assert report["0"]["recall"] < 0.60
        assert report["0"]["f1-score"] < 0.70
        assert report["1"]["recall"] > 0.60
        assert report["1"]["f1-score"] > 0.30

    def test_model_predict(self):
        features = self.model.preprocess(self.data)
        predicted_targets = self.model.predict(features)

        assert isinstance(predicted_targets, list)
        assert len(predicted_targets) == features.shape[0]
        assert all(isinstance(p, int) for p in predicted_targets)
