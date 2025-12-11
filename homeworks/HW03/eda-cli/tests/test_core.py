from __future__ import annotations

import pandas as pd
import numpy as np

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)
#"нормальный датасет"
def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )
#датасет "с ошибками"
def _not_sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 2, 3, 1, 5, 6, 7, 8, 9, 10],
            "status": ["active", "active", "active", "active", "active", 
                      "active", "active", "active", "active", "active"],
            "failed_attempts": [0, 0, 0, 0, 0, 0, 0, 2, 3, 1],
            "ailed_attempts": [0, 0, 0, 0, 0, 0, 0, 2, 3, 1],
            "led_attempts": [0, 0, 0, 0, 0, 0, 0, 2, 3, 1],
            "ed_attempts": [0, 0, 0, 0, 0, 0, 0, 2, 3, 1],
            "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
            "transaction_id": [1001, 1002, 1003, 1004, 1005, 
                             1006, 1007, 1008, 1009, 1010],
            "city": ["Moscow", "SPB", "Moscow", "Kazan", "SPB",
                    "Moscow", "SPB", "Kazan", "Moscow", "SPB"],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2

# Новые тесты для эвристик качества данных
def test_has_constant_columns():
    """Тестирование обнаружения константных колонок"""
    df = _not_sample_df()
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    # Проверяем что флаг обнаружения константных колонок True
    assert flags["has_constant_columns"] == True
    # Проверяем что качество ниже из-за константной колонки
    assert flags["quality_score"] < 1.0


def test_has_suspicious_id_duplicates():
    """Тестирование обнаружения дубликатов в ID-колонках"""
    df = _not_sample_df()
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    # Проверяем что флаг обнаружения дубликатов ID True
    assert flags["has_suspicious_id_duplicates"] == True
    # Проверяем что качество ниже из-за дубликатов
    assert flags["quality_score"] < 0.9
    

def test_integration_all_problems():
    """Интеграционный тест: все проблемы одновременно"""
    df=_not_sample_df()
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    # Проверяем все флаги проблем
    assert flags["has_constant_columns"] == True
    assert flags["has_suspicious_id_duplicates"] == True

    # Проверяем что качество низкое из-за множества проблем
    assert flags["quality_score"] < 0.7
