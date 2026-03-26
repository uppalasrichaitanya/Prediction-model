import pytest
from fastapi.testclient import TestClient
import os
from api.main import app
from api.config import MODEL_PATH

client = TestClient(app)

def test_health_no_model():
    if not os.path.exists(MODEL_PATH):
        response = client.get("/health")
        assert response.status_code == 503

@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Model not trained yet")
def test_health_with_model():
    response = client.get("/health")
    assert response.status_code == 200

@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Model not trained yet")
def test_predict_endpoint_with_model():
    payload = {
        "batting_team": "Chennai Super Kings",
        "bowling_team": "Mumbai Indians",
        "city": "Chennai",
        "runs_left": 67,
        "balls_left": 42,
        "wickets_remaining": 6,
        "crr": 8.5,
        "rrr": 9.57
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "batting_team_win_prob" in data
