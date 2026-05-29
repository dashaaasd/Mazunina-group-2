# tests/test_preprocessor.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.data.preprocessor import CampaignDataProcessor

DATA_PATH = Path(__file__).parent.parent / 'data' / 'tech_advertising_campaigns_dataset.csv'


def test_preprocessor_fit():
    """Проверка, что препроцессор обучается без ошибок"""
    df = pd.read_csv(DATA_PATH)
    processor = CampaignDataProcessor()
    df_processed = processor.prepare_features(df, fit=True)

    assert df_processed.shape[0] == len(df)
    assert 'ROAS' in df_processed.columns
    assert len(processor.feature_columns) > 20


def test_preprocessor_no_leaks():
    """Проверка отсутствия утечек в признаках"""
    df = pd.read_csv(DATA_PATH)
    processor = CampaignDataProcessor()
    processor.prepare_features(df, fit=True)

    leaks = ['revenue', 'profit', 'campaign_id']
    for leak in leaks:
        assert leak not in processor.feature_columns, f"Утечка: {leak}"


def test_preprocessor_transform():
    """Проверка transform после fit — нет NaN в признаках"""
    df = pd.read_csv(DATA_PATH)
    processor = CampaignDataProcessor()
    processor.prepare_features(df.head(100), fit=True)

    # Transform на новых данных
    df_new = df.tail(50).copy()
    df_transformed = processor.prepare_features(df_new, fit=False)

    assert df_transformed.shape[0] == 50

    # Проверяем только признаки модели, не целевую переменную
    feature_cols = [c for c in processor.feature_columns if c in df_transformed.columns]
    assert not df_transformed[feature_cols].isnull().any().any(), \
        "Найдены NaN в признаках при transform"