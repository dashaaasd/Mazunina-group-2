# src/data/preprocessor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

class CampaignDataProcessor:
    """
    Препроцессор для датасета.
    Целевая переменная: ROAS (Return on Ad Spend).
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.feature_medians = {}
        
        # Категориальные признаки
        self.categorical_columns = [
            'platform',
            'campaign_objective',
            'device_type',
            'operating_system',
            'ad_placement',
            'day_of_week'
        ]
        
        # Признаки, которые НЕ используются
        self.drop_columns = [
            'campaign_id',
            'start_date',   # будет разобран на временные признаки
            'profit',       # альтернативная цель (утечка)
            'revenue'       # утечка — ROAS = revenue / ad_spend
        ]
        
    def extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Извлечение временных признаков из start_date"""
        df = df.copy()
        
        if 'start_date' in df.columns:
            df['start_date'] = pd.to_datetime(df['start_date'])
            
            df['year'] = df['start_date'].dt.year
            df['month'] = df['start_date'].dt.month
            df['week_of_year'] = df['start_date'].dt.isocalendar().week.astype(int)
            df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Очистка данных"""
        df = df.copy()
        
        # Извлекаем временные признаки ДО удаления start_date
        df = self.extract_time_features(df)
        
        # Удаляем ненужные колонки
        drop_cols = [c for c in self.drop_columns if c in df.columns]
        df = df.drop(columns=drop_cols, errors='ignore')
        
        # Обработка бесконечностей
        df = df.replace([np.inf, -np.inf], np.nan)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Подготовка признаков для модели"""
        
        # 1. Очистка + временные признаки
        df = self.clean_data(df)
        
        # 2. Кодирование категориальных признаков
        for col in self.categorical_columns:
            if col not in df.columns:
                continue
                
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(
                    df[col].astype(str)
                )
            else:
                le = self.label_encoders.get(col)
                if le:
                    df[f'{col}_encoded'] = df[col].astype(str).map(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
        
        # 3. Числовые признаки
        numeric_features = [
            # Основные метрики
            'clicks', 'impressions', 'conversions', 'ad_spend',
            'quality_score', 'actual_cpc', 'CTR',
            'bounce_rate', 'avg_session_duration_seconds', 'pages_per_session',
            # Исходные временные
            'quarter', 'hour_of_day', 'campaign_day',
            # Созданные временные
            'year', 'month', 'week_of_year', 'is_weekend'
        ]
        
        # Оставляем только существующие
        numeric_features = [c for c in numeric_features if c in df.columns]
        
        # 4. Добавляем закодированные категориальные
        encoded_features = [
            f'{col}_encoded' for col in self.categorical_columns 
            if f'{col}_encoded' in df.columns
        ]
        self.feature_columns = numeric_features + encoded_features
        
        # 5. Заполнение пропусков
        for col in self.feature_columns:
            if col in df.columns and df[col].isnull().any():
                if fit:
                    self.feature_medians[col] = df[col].median()
                fill_value = self.feature_medians.get(col, df[col].median())
                df[col] = df[col].fillna(fill_value if pd.notna(fill_value) else 0)
        
        # 6. Удаление строк с NaN в целевой
        if fit and 'ROAS' in df.columns:
            df = df.dropna(subset=['ROAS'])
        
        # 7. Масштабирование
        if fit:
            df[numeric_features] = self.scaler.fit_transform(df[numeric_features])
        else:
            df[numeric_features] = self.scaler.transform(df[numeric_features])
        
        return df
    
    def save_preprocessor(self, path: str = 'artifacts/preprocessor.pkl'):
        """Сохранение"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'feature_medians': self.feature_medians,
            'categorical_columns': self.categorical_columns
        }, path)
        print(f"Preprocessor saved to {path}")
    
    def load_preprocessor(self, path: str = 'artifacts/preprocessor.pkl'):
        """Загрузка"""
        data = joblib.load(path)
        self.label_encoders = data['label_encoders']
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        self.feature_medians = data.get('feature_medians', {})
        self.categorical_columns = data.get('categorical_columns', [])
        print(f"Preprocessor loaded from {path}")