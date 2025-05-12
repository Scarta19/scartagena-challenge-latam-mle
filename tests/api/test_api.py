# tests/api/test_api.py
import unittest
from fastapi.testclient import TestClient
from challenge.api import app


class TestBatchPipeline(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_should_get_predict(self):
        data = {
            "flights": [{"OPERA": "Aerolineas Argentinas", "TIPOVUELO": "N", "MES": 3}]
        }
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 200)
        json_response = response.json()
        self.assertIn("predict", json_response)
        self.assertIsInstance(json_response["predict"], list)
        self.assertEqual(len(json_response["predict"]), 1)
        self.assertIn(json_response["predict"][0], [0, 1])

    def test_should_failed_unkown_column_1(self):
        data = {
            "flights": [{"OPERA": "Aerolineas Argentinas", "TIPOVUELO": "N", "MES": 13}]
        }
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 400)

    def test_should_failed_unkown_column_2(self):
        data = {
            "flights": [{"OPERA": "Aerolineas Argentinas", "TIPOVUELO": "O", "MES": 13}]
        }
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 400)

    def test_should_failed_unkown_column_3(self):
        data = {"flights": [{"OPERA": "Argentinas", "TIPOVUELO": "O", "MES": 13}]}
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 400)
