# tests/test_predictor.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.predictor import ROASPredictor


def test_predictor_loads():
    """Проверка загрузки модели"""
    predictor = ROASPredictor()
    assert predictor.model is not None
    assert predictor.processor is not None


def test_predict_single():
    """Проверка предсказания для одной кампании"""
    predictor = ROASPredictor()

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

    result = predictor.predict_single(campaign)

    assert "predicted_ROAS" in result
    assert result["status"] == "success"
    assert result["predicted_ROAS"] > 0


def test_recommend_channel():
    """Проверка рекомендации канала — возвращает 6 каналов"""
    predictor = ROASPredictor()

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

    result = predictor.recommend_channel(campaign)

    assert "best_channel" in result
    assert "best_ROAS" in result
    assert len(result["all_results"]) == 6
    # лучший канал — первый в списке all_results
    assert result["best_channel"] == result["all_results"][0]["platform"]