# src/api/app.py

import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH (ДО всех импортов!)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, HTTPException
from src.models.predictor import ROASPredictor
from src.api.schemas import CampaignRequest, PredictionResponse, ChannelRecommendation

app = FastAPI(
    title="ROAS Prediction Service",
    description="Сервис предсказания эффективности рекламных кампаний",
    version="1.0.0"
)

# Глобальный объект предиктора
predictor = None
#автоматически запускается обучение если модели нет
@app.on_event("startup")
def load_model():
    global predictor
    model_path = Path(__file__).parent.parent.parent / 'artifacts' / 'models' / 'best_model_service.pkl'
    
    if not model_path.exists():
        print("Модель не найдена. Запуск обучения...")
        from src.train import train_model
        train_model()
    
    predictor = ROASPredictor()
    print("✅ Сервис готов")


@app.get("/")
def root():
    return {
        "service": "ROAS Prediction API",
        "version": "1.0.0",
        "status": "running"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(campaign: CampaignRequest):
    """Предсказать ROAS для одной кампании"""
    if predictor is None:
        raise HTTPException(500, "Model not loaded")
    
    result = predictor.predict_single(campaign.model_dump())
    return result


@app.post("/recommend", response_model=ChannelRecommendation)
def recommend(campaign: CampaignRequest, channels: str = None):
    """Рекомендовать лучший канал"""
    if predictor is None:
        raise HTTPException(500, "Model not loaded")
    
    channel_list = channels.split(',') if channels else None
    result = predictor.recommend_channel(campaign.model_dump(), channel_list)
    return result


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": predictor is not None}