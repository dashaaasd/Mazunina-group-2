# src/api/schemas.py

from pydantic import BaseModel, Field
from typing import Optional


class CampaignRequest(BaseModel):
    """Входные данные кампании"""
    
    # Платформа (только для /predict)
    platform: str = Field(..., example="Facebook")
    
    # ... все остальные поля без изменений


class PredictionResponse(BaseModel):
    predicted_ROAS: float
    status: str


class RecommendRequest(BaseModel):
    """Входные данные для /recommend (без platform)"""
    
    campaign_objective: str = Field(..., example="Conversions")
    device_type: str = Field(..., example="Desktop")
    operating_system: str = Field(..., example="Windows")
    ad_placement: str = Field(..., example="feed")
    day_of_week: str = Field(..., example="Monday")
    creative_format: Optional[str] = Field(None, example="Video")
    creative_size: Optional[str] = Field(None, example="1920x1080")
    ad_copy_length: Optional[str] = Field(None, example="Short")
    has_call_to_action: Optional[bool] = Field(None, example=False)
    creative_emotion: Optional[str] = Field(None, example="Joy")
    creative_age_days: Optional[int] = Field(None, example=30)
    target_audience_age: Optional[str] = Field(None, example="25-34")
    target_audience_gender: Optional[str] = Field(None, example="Female")
    audience_interest_category: Optional[str] = Field(None, example="Shoppers")
    income_bracket: Optional[str] = Field(None, example="$50K-$100K")
    purchase_intent_score: Optional[str] = Field(None, example="Medium")
    retargeting_flag: Optional[bool] = Field(None, example=False)
    ad_spend: float = Field(..., example=5000.0)
    clicks: Optional[int] = Field(None, example=150)
    impressions: Optional[int] = Field(None, example=10000)
    conversions: Optional[int] = Field(None, example=50)
    CTR: Optional[float] = Field(None, example=1.5)
    CPC: Optional[float] = Field(None, example=2.5)
    conversion_rate: Optional[float] = Field(None, example=3.5)
    CPA: Optional[float] = Field(None, example=100.0)
    quality_score: Optional[float] = Field(None, example=7.0)
    actual_cpc: Optional[float] = Field(None, example=0.5)
    bounce_rate: Optional[float] = Field(None, example=35.0)
    avg_session_duration_seconds: Optional[float] = Field(None, example=120.0)
    pages_per_session: Optional[float] = Field(None, example=3.0)
    industry_vertical: Optional[str] = Field(None, example="E-commerce")
    budget_tier: Optional[str] = Field(None, example="Medium")
    start_date: str = Field(..., example="2025-01-15")
    quarter: Optional[int] = Field(None, example=1)
    hour_of_day: Optional[int] = Field(None, example=14)
    campaign_day: Optional[int] = Field(None, example=7)
    campaign_id: Optional[str] = None
    revenue: Optional[float] = None
    profit: Optional[float] = None


class PredictionResponse(BaseModel):
    predicted_ROAS: float
    status: str


class ChannelRecommendation(BaseModel):
    best_channel: str
    best_ROAS: float
    all_results: list