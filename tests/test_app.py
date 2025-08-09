import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_predict():
    payload = {
      "MedInc": 8.3,
      "HouseAge": 41,
      "AveRooms": 6,
      "AveBedrms": 1.1,
      "Population": 500,
      "AveOccup": 3.5,
      "Latitude": 37.88,
      "Longitude": -122.23
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert 'prediction' in data
    assert isinstance(data['prediction'], float)