# src/api/schemas.py

from pydantic import BaseModel, Field
from typing import Optional


class CampaignRequest(BaseModel):
    """Входные данные кампании"""
    

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "platform": "Facebook",
                "campaign_objective": "Conversions",
                "device_type": "Desktop",
                "operating_system": "Windows",
                "ad_placement": "Feed",
                "day_of_week": "Monday",
                "ad_spend": 5000.0,
                "start_date": "2025-01-15",
                "creative_format": "Video",
                "ad_copy_length": "Medium",
                "has_call_to_action": True,
                "creative_emotion": "Trust",
                "target_audience_age": "25-34",
                "target_audience_gender": "All",
                "income_bracket": "$50K-$100K",
                "purchase_intent_score": "High",
                "retargeting_flag": False,
                "industry_vertical": "E-commerce",
                "budget_tier": "Medium",
                "quality_score": 7.5,
                "bounce_rate": 35.0,
                "clicks": 250,
                "impressions": 15000,
                "conversions": 18
            }]
        }
    }

    # Платформа (только для /predict)
    platform: str = Field(...)
    campaign_objective: str = Field(...)
    device_type: str = Field(...)
    operating_system: str = Field(...)
    ad_placement: str = Field(...)
    day_of_week: str = Field(...)
    creative_format: Optional[str] = Field(None)
    creative_size: Optional[str] = Field(None)
    ad_copy_length: Optional[str] = Field(None)
    has_call_to_action: Optional[bool] = Field(None)
    creative_emotion: Optional[str] = Field(None)
    creative_age_days: Optional[int] = Field(None)
    target_audience_age: Optional[str] = Field(None)
    target_audience_gender: Optional[str] = Field(None)
    audience_interest_category: Optional[str] = Field(None)
    income_bracket: Optional[str] = Field(None)
    purchase_intent_score: Optional[str] = Field(None)
    retargeting_flag: Optional[bool] = Field(None)
    ad_spend: float = Field(...)
    clicks: Optional[int] = Field(None)
    impressions: Optional[int] = Field(None)
    conversions: Optional[int] = Field(None)
    CTR: Optional[float] = Field(None)
    CPC: Optional[float] = Field(None)
    conversion_rate: Optional[float] = Field(None)
    CPA: Optional[float] = Field(None)
    quality_score: Optional[float] = Field(None)
    actual_cpc: Optional[float] = Field(None)
    bounce_rate: Optional[float] = Field(None)
    avg_session_duration_seconds: Optional[float] = Field(None)
    pages_per_session: Optional[float] = Field(None)
    industry_vertical: Optional[str] = Field(None)
    budget_tier: Optional[str] = Field(None)
    start_date: str = Field(...)
    quarter: Optional[int] = Field(None)
    hour_of_day: Optional[int] = Field(None)
    campaign_day: Optional[int] = Field(None)
    campaign_id: Optional[str] = None
    revenue: Optional[float] = None
    profit: Optional[float] = None



class RecommendRequest(BaseModel):
    """Входные данные для /recommend (без platform)"""
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "campaign_objective": "Conversions",
                "device_type": "Desktop",
                "operating_system": "Windows",
                "ad_placement": "Feed",
                "day_of_week": "Monday",
                "ad_spend": 5000.0,
                "start_date": "2025-01-15",
                "creative_format": "Video",
                "ad_copy_length": "Medium",
                "has_call_to_action": True,
                "creative_emotion": "Trust",
                "target_audience_age": "25-34",
                "target_audience_gender": "All",
                "income_bracket": "$50K-$100K",
                "purchase_intent_score": "High",
                "retargeting_flag": False,
                "industry_vertical": "E-commerce",
                "budget_tier": "Medium",
                "quality_score": 7.5,
                "bounce_rate": 35.0,
                "clicks": 250,
                "impressions": 15000,
                "conversions": 18
            }]
        }
    }

    
    campaign_objective: str = Field(...)
    device_type: str = Field(...)
    operating_system: str = Field(...)
    ad_placement: str = Field(...)
    day_of_week: str = Field(...)
    creative_format: Optional[str] = Field(None)
    creative_size: Optional[str] = Field(None)
    ad_copy_length: Optional[str] = Field(None)
    has_call_to_action: Optional[bool] = Field(None)
    creative_emotion: Optional[str] = Field(None)
    creative_age_days: Optional[int] = Field(None)
    target_audience_age: Optional[str] = Field(None)
    target_audience_gender: Optional[str] = Field(None)
    audience_interest_category: Optional[str] = Field(None)
    income_bracket: Optional[str] = Field(None)
    purchase_intent_score: Optional[str] = Field(None)
    retargeting_flag: Optional[bool] = Field(None)
    ad_spend: float = Field(...)
    clicks: Optional[int] = Field(None)
    impressions: Optional[int] = Field(None)
    conversions: Optional[int] = Field(None)
    CTR: Optional[float] = Field(None)
    CPC: Optional[float] = Field(None)
    conversion_rate: Optional[float] = Field(None)
    CPA: Optional[float] = Field(None)
    quality_score: Optional[float] = Field(None)
    actual_cpc: Optional[float] = Field(None)
    bounce_rate: Optional[float] = Field(None)
    avg_session_duration_seconds: Optional[float] = Field(None)
    pages_per_session: Optional[float] = Field(None)
    industry_vertical: Optional[str] = Field(None)
    budget_tier: Optional[str] = Field(None)
    start_date: str = Field(...)
    quarter: Optional[int] = Field(None)
    hour_of_day: Optional[int] = Field(None)
    campaign_day: Optional[int] = Field(None)
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