# src/api/app.py

import sys
import logging
from contextlib import asynccontextmanager
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH (ДО всех импортов!)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, HTTPException, Request
from src.models.predictor import ROASPredictor
from src.api.schemas import CampaignRequest, RecommendRequest, PredictionResponse, ChannelRecommendation

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("roas_service")

# Глобальный объект предиктора
predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загрузка модели при старте, освобождение ресурсов при завершении"""
    global predictor
    model_path = Path(__file__).parent.parent.parent / 'artifacts' / 'models' / 'best_model_service.pkl'

    if not model_path.exists():
        logger.warning("Модель не найдена по пути %s. Запуск обучения...", model_path)
        from src.train import train_model
        train_model()

    try:
        predictor = ROASPredictor()
        logger.info("✅ Модель успешно загружена. Сервис готов.")
        logger.info("Сервис доступен по адресу: http://localhost:8000/docs")
    except Exception as e:
        logger.error("❌ Ошибка загрузки модели: %s", e)
        raise

    yield  # приложение работает

    logger.info("Сервис завершает работу.")


app = FastAPI(
    title="ROAS Prediction Service",
    description="Сервис предсказания эффективности рекламных кампаний",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "service": "ROAS Prediction API",
        "version": "1.0.0",
        "status": "running",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(campaign: CampaignRequest, request: Request):
    """Предсказать ROAS для одной кампании"""
    logger.info("POST /predict | client=%s | platform=%s | ad_spend=%.2f",
                request.client.host, campaign.platform, campaign.ad_spend)

    if predictor is None:
        logger.error("Модель не загружена при вызове /predict")
        raise HTTPException(500, "Model not loaded")

    try:
        result = predictor.predict_single(campaign.model_dump())
        logger.info("POST /predict | predicted_ROAS=%.4f", result["predicted_ROAS"])
        return result
    except Exception as e:
        logger.exception("Ошибка при предсказании: %s", e)
        raise HTTPException(500, f"Prediction error: {e}")


@app.post("/recommend", response_model=ChannelRecommendation)
def recommend(campaign: RecommendRequest, request: Request, channels: str = None):
    """Рекомендовать лучший канал"""
    logger.info("POST /recommend | client=%s | ad_spend=%.2f",
                request.client.host, campaign.ad_spend)

    if predictor is None:
        logger.error("Модель не загружена при вызове /recommend")
        raise HTTPException(500, "Model not loaded")

    try:
        channel_list = channels.split(',') if channels else None
        result = predictor.recommend_channel(campaign.model_dump(), channel_list)
        logger.info("POST /recommend | best_channel=%s | best_ROAS=%.4f",
                    result["best_channel"], result["best_ROAS"])
        return result
    except Exception as e:
        logger.exception("Ошибка при рекомендации канала: %s", e)
        raise HTTPException(500, f"Recommendation error: {e}")


@app.get("/health")
def health():
    status = "healthy" if predictor is not None else "degraded"
    logger.debug("GET /health | status=%s", status)
    return {"status": status, "model_loaded": predictor is not None}
