# src/models/predictor.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import pandas as pd
from data.preprocessor import CampaignDataProcessor


class ROASPredictor:
    """Загрузка модели и предсказание ROAS"""
    
    def __init__(self, model_path: str = None, preprocessor_path: str = None):
        root = Path(__file__).parent.parent.parent / 'artifacts' / 'models'
        
        model_path = model_path or str(root / 'best_model_service.pkl')
        preprocessor_path = preprocessor_path or str(root / 'preprocessor.pkl')
        
        self.model = joblib.load(model_path)
        self.processor = CampaignDataProcessor()
        self.processor.load_preprocessor(preprocessor_path)
    
    def predict_single(self, campaign: dict) -> dict:
        df = pd.DataFrame([campaign])
        X = self.processor.prepare_features(df, fit=False)
        feature_cols = [c for c in self.processor.feature_columns if c in X.columns]
        X = X[feature_cols]
        prediction = float(self.model.predict(X)[0])
        
        return {
            'predicted_ROAS': round(prediction, 2),
            'status': 'success'
        }
    
    def predict_batch(self, campaigns: list) -> list:
        df = pd.DataFrame(campaigns)
        X = self.processor.prepare_features(df, fit=False)
        feature_cols = [c for c in self.processor.feature_columns if c in X.columns]
        X = X[feature_cols]
        predictions = self.model.predict(X)
        
        return [{'predicted_ROAS': round(float(p), 2)} for p in predictions]
    
    def recommend_channel(self, campaign: dict, channels: list = None) -> dict:
        if channels is None:
            channels = ['LinkedIn', 'Google Ads', 'Facebook', 'Instagram', 'Twitter', 'TikTok']
        
        results = []
        for channel in channels:
            campaign_copy = campaign.copy()
            campaign_copy['platform'] = channel
            try:
                prediction = self.predict_single(campaign_copy)
                results.append({
                    'platform': channel,
                    'predicted_ROAS': prediction['predicted_ROAS']
                })
            except Exception as e:
                results.append({
                    'platform': channel,
                    'predicted_ROAS': None,
                    'error': str(e)
                })
        
        results = sorted(results, key=lambda x: x['predicted_ROAS'] if x['predicted_ROAS'] else 0, reverse=True)
        
        return {
            'best_channel': results[0]['platform'],
            'best_ROAS': results[0]['predicted_ROAS'],
            'all_results': results
        }