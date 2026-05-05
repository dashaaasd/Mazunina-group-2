# src/train.py
"""
Точка входа для обучения модели.
Запуск: python -m src.train
"""

import pandas as pd
import numpy as np
import joblib
import yaml
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from data.preprocessor import CampaignDataProcessor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score


def train_model():
    """Обучение финальной модели CatBoost с параметрами из конфига"""
    print("ОБУЧЕНИЕ МОДЕЛИ ROAS PREDICTION")
    
    # Загрузка конфига
    config_path = Path(__file__).parent.parent / 'configs' / 'model_config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    params = config['model']
    print(f"Конфигурация: {params['name']}, depth={params['depth']}, lr={params['learning_rate']}")
    
    # Загрузка данных
    data_path = Path(__file__).parent.parent / 'data' / config['data']['file']
    df = pd.read_csv(data_path)
    print(f"Загружено: {df.shape[0]} записей, {df.shape[1]} признаков")
    
    # Предобработка
    df['start_date'] = pd.to_datetime(df['start_date'])
    df = df.sort_values('start_date')
    
    processor = CampaignDataProcessor()
    df_processed = processor.prepare_features(df, fit=True)
    print(f"После предобработки: {df_processed.shape[1]} фичей")
    
    # Временной сплит
    X = df_processed[processor.feature_columns]
    y = df_processed['ROAS']
    
    n = len(df_processed)
    train_size = int(n * config['data']['split_ratio'][0])
    val_size = int(n * config['data']['split_ratio'][1])
    
    X_train = X.iloc[:train_size]
    X_val = X.iloc[train_size:train_size + val_size]
    y_train = y.iloc[:train_size]
    y_val = y.iloc[train_size:train_size + val_size]
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Категориальные индексы для CatBoost
    cat_features_indices = [i for i, col in enumerate(processor.feature_columns) 
                            if '_encoded' in col]
    
    # Обучение модели
    model = CatBoostRegressor(
        iterations=params['iterations'],
        learning_rate=params['learning_rate'],
        depth=params['depth'],
        loss_function=params['loss_function'],
        eval_metric=params['eval_metric'],
        random_seed=params['random_seed'],
        verbose=params['verbose'],
        early_stopping_rounds=params['early_stopping_rounds']
    )
    
    model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_features_indices)
    
    # Сохранение артефактов
    artifacts_path = Path(__file__).parent.parent / config['paths']['artifacts_dir']
    artifacts_path.mkdir(exist_ok=True)
    
    model_path = artifacts_path / config['paths']['model_name']
    preprocessor_path = artifacts_path / config['paths']['preprocessor_name']
    
    joblib.dump(model, model_path)
    processor.save_preprocessor(str(preprocessor_path))
    
    # Метрики на валидации
    y_pred = model.predict(X_val)
    val_r2 = r2_score(y_val, y_pred)
    
    print(f"\n{'='*50}")
    print(f"Val R²: {val_r2:.4f}")
    print(f"Модель: {model_path}")
    print(f"Препроцессор: {preprocessor_path}")
    
    return model


if __name__ == '__main__':
    train_model()