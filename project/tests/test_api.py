# tests/test_api.py

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from src.api.app import app


@pytest.fixture(scope="module")
def client():
    """Клиент с поддержкой lifespan (загрузка модели при старте)"""
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict(client):
    campaign = {
        "platform": "Facebook",
        "campaign_objective": "Conversions",
        "device_type": "Desktop",
        "operating_system": "Windows",
        "ad_placement": "feed",
        "day_of_week": "Monday",
        "ad_spend": 5000.0,
        "start_date": "2025-01-15",
    }
    response = client.post("/predict", json=campaign)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_ROAS" in data
    assert data["predicted_ROAS"] > 0
    assert data["status"] == "success"


def test_recommend(client):
    campaign = {
        "campaign_objective": "Conversions",
        "device_type": "Desktop",
        "operating_system": "Windows",
        "ad_placement": "feed",
        "day_of_week": "Monday",
        "ad_spend": 5000.0,
        "start_date": "2025-01-15",
    }
    response = client.post("/recommend", json=campaign)
    assert response.status_code == 200
    data = response.json()
    assert "best_channel" in data
    assert "best_ROAS" in data
    assert len(data["all_results"]) == 6