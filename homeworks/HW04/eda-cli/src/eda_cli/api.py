from __future__ import annotations

from time import perf_counter
import json
import logging
from datetime import datetime
from uuid import uuid4
from typing import Dict, Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from contextvars import ContextVar

from .core import compute_quality_flags, missing_table, summarize_dataset

# ---------- Настройка логирования ----------

# Создаем логгер для API
api_logger = logging.getLogger("api_logger")
api_logger.setLevel(logging.INFO)

# Форматтер для JSON логов
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            **getattr(record, "extra", {})
        }
        return json.dumps(log_record, ensure_ascii=False)

# Хэндлер для вывода в stdout
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
api_logger.addHandler(handler)

# Хэндлер для записи в файл (опционально)
file_handler = logging.FileHandler("logs/api.log")
file_handler.setFormatter(JsonFormatter())
api_logger.addHandler(file_handler)

# Контекстная переменная для request_id
request_id_var: ContextVar[str] = ContextVar("request_id", default="")

app = FastAPI(
    title="AIE Dataset Quality API",
    version="0.2.0",
    description=(
        "HTTP-сервис-заглушка для оценки готовности датасета к обучению модели. "
        "Использует простые эвристики качества данных вместо настоящей ML-модели."
    ),
    docs_url="/docs",
    redoc_url=None,
)

# Middleware для добавления request_id и логирования запросов
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid4())
    request_id_var.set(request_id)
    
    start_time = perf_counter()
    
    # Пропускаем запрос дальше
    response = await call_next(request)
    
    # Вычисляем время выполнения
    process_time = (perf_counter() - start_time) * 1000.0
    
    # Логируем базовую информацию о запросе
    log_data = {
        "request_id": request_id,
        "endpoint": request.url.path,
        "method": request.method,
        "status_code": response.status_code,
        "latency_ms": round(process_time, 2),
        "client_host": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent"),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    
    api_logger.info(
        "Request processed",
        extra={"type": "request_summary", **log_data}
    )
    
    # Добавляем request_id в заголовки ответа
    response.headers["X-Request-ID"] = request_id
    
    return response

# ---------- Вспомогательные функции для логирования ----------

def log_quality_request(
    endpoint: str,
    n_rows: int,
    n_cols: int,
    score: float,
    ok_for_model: bool,
    latency_ms: float,
    extra_fields: Dict[str, Any] = None
):
    """Логирует информацию о запросе оценки качества."""
    log_data = {
        "type": "quality_assessment",
        "endpoint": endpoint,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "quality_score": round(score, 4),
        "ok_for_model": ok_for_model,
        "latency_ms": round(latency_ms, 2),
        "request_id": request_id_var.get(),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    
    if extra_fields:
        log_data.update(extra_fields)
    
    api_logger.info(
        f"Quality assessment completed for {n_rows} rows, {n_cols} cols",
        extra=log_data
    )

def log_file_processing(
    endpoint: str,
    filename: str,
    n_rows: int,
    n_cols: int,
    latency_ms: float,
    extra_fields: Dict[str, Any] = None
):
    """Логирует информацию об обработке файла."""
    log_data = {
        "type": "file_processing",
        "endpoint": endpoint,
        "filename": filename,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "latency_ms": round(latency_ms, 2),
        "request_id": request_id_var.get(),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    
    if extra_fields:
        log_data.update(extra_fields)
    
    api_logger.info(
        f"File {filename} processed successfully",
        extra=log_data
    )

def log_error(
    endpoint: str,
    error_message: str,
    status_code: int = 500,
    extra_fields: Dict[str, Any] = None
):
    """Логирует информацию об ошибке."""
    log_data = {
        "type": "error",
        "endpoint": endpoint,
        "error": error_message,
        "status_code": status_code,
        "request_id": request_id_var.get(),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    
    if extra_fields:
        log_data.update(extra_fields)
    
    api_logger.error(
        f"Error in {endpoint}: {error_message}",
        extra=log_data
    )


# ---------- Модели запросов/ответов ----------


class QualityRequest(BaseModel):
    """Агрегированные признаки датасета – 'фичи' для заглушки модели."""

    n_rows: int = Field(..., ge=0, description="Число строк в датасете")
    n_cols: int = Field(..., ge=0, description="Число колонок")
    max_missing_share: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Максимальная доля пропусков среди всех колонок (0..1)",
    )
    numeric_cols: int = Field(
        ...,
        ge=0,
        description="Количество числовых колонок",
    )
    categorical_cols: int = Field(
        ...,
        ge=0,
        description="Количество категориальных колонок",
    )


class QualityResponse(BaseModel):
    """Ответ заглушки модели качества датасета."""

    ok_for_model: bool = Field(
        ...,
        description="True, если датасет считается достаточно качественным для обучения модели",
    )
    quality_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Интегральная оценка качества данных (0..1)",
    )
    message: str = Field(
        ...,
        description="Человекочитаемое пояснение решения",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Время обработки запроса на сервере, миллисекунды",
    )
    flags: dict[str, bool] | None = Field(
        default=None,
        description="Булевы флаги с подробностями (например, too_few_rows, too_many_missing)",
    )
    dataset_shape: dict[str, int] | None = Field(
        default=None,
        description="Размеры датасета: {'n_rows': ..., 'n_cols': ...}, если известны",
    )

class QualityFlagsResponse(BaseModel):
    """Ответ с полным набором флагов качества."""
    flags: dict[str, bool] = Field(
        ...,
        description="Полный набор булевых флагов качества данных"
    )
    dataset_shape: dict[str, int] = Field(
        ...,
        description="Размеры датасета: {'n_rows': ..., 'n_cols': ...}"
    )
    filename: str = Field(
        ...,
        description="Имя обработанного файла"
    )
    request_id: str = Field(
        ...,
        description="Идентификатор запроса"
    )

# ---------- Системный эндпоинт ----------


@app.get("/health", tags=["system"])
def health(request: Request) -> dict[str, str]:
    """Простейший health-check сервиса."""
    response = {
        "status": "ok",
        "service": "dataset-quality",
        "version": "0.2.0",
        "request_id": request_id_var.get()
    }
    
    # Логируем health-check
    api_logger.info(
        "Health check",
        extra={
            "type": "health_check",
            "endpoint": "/health",
            "request_id": request_id_var.get(),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    )
    
    return response


# ---------- Заглушка /quality по агрегированным признакам ----------


@app.post("/quality", response_model=QualityResponse, tags=["quality"])
def quality(req: QualityRequest) -> QualityResponse:
    """
    Эндпоинт-заглушка, который принимает агрегированные признаки датасета
    и возвращает эвристическую оценку качества.
    """

    start = perf_counter()

    # Базовый скор от 0 до 1
    score = 1.0

    # Чем больше пропусков, тем хуже
    score -= req.max_missing_share

    # Штраф за слишком маленький датасет
    if req.n_rows < 1000:
        score -= 0.2

    # Штраф за слишком широкий датасет
    if req.n_cols > 100:
        score -= 0.1

    # Штрафы за перекос по типам признаков (если есть числовые и категориальные)
    if req.numeric_cols == 0 and req.categorical_cols > 0:
        score -= 0.1
    if req.categorical_cols == 0 and req.numeric_cols > 0:
        score -= 0.05

    # Нормируем скор в диапазон [0, 1]
    score = max(0.0, min(1.0, score))

    # Простое решение "ок / не ок"
    ok_for_model = score >= 0.7
    if ok_for_model:
        message = "Данных достаточно, модель можно обучать (по текущим эвристикам)."
    else:
        message = "Качество данных недостаточно, требуется доработка (по текущим эвристикам)."

    latency_ms = (perf_counter() - start) * 1000.0

    # Флаги, которые могут быть полезны для последующего логирования/аналитики
    flags = {
        "too_few_rows": req.n_rows < 1000,
        "too_many_columns": req.n_cols > 100,
        "too_many_missing": req.max_missing_share > 0.5,
        "no_numeric_columns": req.numeric_cols == 0,
        "no_categorical_columns": req.categorical_cols == 0,
    }

    # Примитивный лог — на семинаре можно обсудить, как это превратить в нормальный logger
    print(
        f"[quality] n_rows={req.n_rows} n_cols={req.n_cols} "
        f"max_missing_share={req.max_missing_share:.3f} "
        f"score={score:.3f} latency_ms={latency_ms:.1f} ms"
    )

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags,
        dataset_shape={"n_rows": req.n_rows, "n_cols": req.n_cols},
    )


# ---------- /quality-from-csv: реальный CSV через нашу EDA-логику ----------


@app.post(
    "/quality-from-csv",
    response_model=QualityResponse,
    tags=["quality"],
    summary="Оценка качества по CSV-файлу с использованием EDA-ядра",
)
async def quality_from_csv(file: UploadFile = File(...)) -> QualityResponse:
    """
    Эндпоинт, который принимает CSV-файл, запускает EDA-ядро
    (summarize_dataset + missing_table + compute_quality_flags)
    и возвращает оценку качества данных.

    Именно это по сути связывает S03 (CLI EDA) и S04 (HTTP-сервис).
    """

    start = perf_counter()

    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream"):
        # content_type от браузера может быть разным, поэтому проверка мягкая
        # но для демонстрации оставим простую ветку 400
        raise HTTPException(status_code=400, detail="Ожидается CSV-файл (content-type text/csv).")

    try:
        # FastAPI даёт file.file как file-like объект, который можно читать pandas'ом
        df = pd.read_csv(file.file)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV-файл не содержит данных (пустой DataFrame).")

    # Используем EDA-ядро из S03
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags_all = compute_quality_flags(summary, missing_df)

    # Ожидаем, что compute_quality_flags вернёт quality_score в [0,1]
    score = float(flags_all.get("quality_score", 0.0))
    score = max(0.0, min(1.0, score))
    ok_for_model = score >= 0.7

    if ok_for_model:
        message = "CSV выглядит достаточно качественным для обучения модели (по текущим эвристикам)."
    else:
        message = "CSV требует доработки перед обучением модели (по текущим эвристикам)."

    latency_ms = (perf_counter() - start) * 1000.0

    # Оставляем только булевы флаги для компактности
    flags_bool: dict[str, bool] = {
        key: bool(value)
        for key, value in flags_all.items()
        if isinstance(value, bool)
    }

    # Размеры датасета берём из summary (если там есть поля n_rows/n_cols),
    # иначе — напрямую из DataFrame.
    try:
        n_rows = int(getattr(summary, "n_rows"))
        n_cols = int(getattr(summary, "n_cols"))
    except AttributeError:
        n_rows = int(df.shape[0])
        n_cols = int(df.shape[1])

    print(
        f"[quality-from-csv] filename={file.filename!r} "
        f"n_rows={n_rows} n_cols={n_cols} score={score:.3f} "
        f"latency_ms={latency_ms:.1f} ms"
    )

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags_bool,
        dataset_shape={"n_rows": n_rows, "n_cols": n_cols},
    )

# ---------- Новый эндпоинт для получения всех флагов качества ----------

@app.post(
    "/quality-flags-from-csv",
    response_model=QualityFlagsResponse,
    tags=["quality"],
    summary="Полный набор флагов качества данных из CSV-файла",
)
async def quality_flags_from_csv(file: UploadFile = File(...)) -> QualityFlagsResponse:
    """
    Эндпоинт, который принимает CSV-файл, запускает EDA-ядро
    (summarize_dataset + missing_table + compute_quality_flags)
    и возвращает полный набор флагов качества данных.

    В отличие от /quality-from-csv, возвращает только флаги без оценки модели.
    """

    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Ожидается CSV-файл (content-type text/csv).")

    try:
        df = pd.read_csv(file.file)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV-файл не содержит данных (пустой DataFrame).")

    # Используем EDA-ядро
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags_all = compute_quality_flags(summary, missing_df)

    # Извлекаем только булевы флаги
    flags_bool: dict[str, bool] = {}
    for key, value in flags_all.items():
        # Преобразуем в bool, если значение уже булево или может быть преобразовано
        if isinstance(value, bool):
            flags_bool[key] = value
        elif isinstance(value, (int, float)):
            # Числовые флаги, где 0/1 или 0.0/1.0
            flags_bool[key] = bool(value)
        elif isinstance(value, str):
            # Строковые флаги типа "True"/"False"
            if value.lower() in ('true', '1', 'yes'):
                flags_bool[key] = True
            elif value.lower() in ('false', '0', 'no'):
                flags_bool[key] = False
        # Игнорируем другие типы (например, числовые значения, которые не являются флагами)
    

    # Добавляем дополнительную информацию
    return QualityFlagsResponse(
        flags=flags_bool,
        dataset_shape={"n_rows": df.shape[0], "n_cols": df.shape[1]},
        filename=file.filename,
        request_id=request_id_var.get()
    )
# ---------- Эндпоинт для просмотра логов (опционально) ----------

@app.get("/logs/preview", tags=["system"])
def get_logs_preview(
    lines: int = Query(default=10, ge=1, le=100, description="Количество строк логов"),
    level: str = Query(default="INFO", regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
) -> dict:
    """
    Просмотр последних строк логов.
    Внимание: в production это должно быть защищено аутентификацией!
    """
    try:
        with open("logs/api.log", "r") as f:
            all_lines = f.readlines()
        
        # Берем последние N строк
        recent_lines = all_lines[-lines:] if lines <= len(all_lines) else all_lines
        
        # Парсим JSON логи
        parsed_logs = []
        for line in recent_lines:
            try:
                log_entry = json.loads(line.strip())
                if level == "ALL" or log_entry.get("level") == level:
                    parsed_logs.append(log_entry)
            except json.JSONDecodeError:
                continue
        
        return {
            "total_lines": len(all_lines),
            "returned_lines": len(parsed_logs),
            "logs": parsed_logs,
            "request_id": request_id_var.get()
        }
        
    except FileNotFoundError:
        return {
            "error": "Файл логов не найден",
            "hint": "Убедитесь, что файл logs/api.log существует",
            "request_id": request_id_var.get()
        }


# ---------- Эндпоинт для получения статистики логов ----------

@app.get("/logs/stats", tags=["system"])
def get_logs_stats() -> dict:
    """
    Получение статистики по логам.
    """
    try:
        with open("logs/api.log", "r") as f:
            all_lines = f.readlines()
        
        # Простая статистика
        endpoint_counts = {}
        error_count = 0
        total_latency = 0
        requests_count = 0
        
        for line in all_lines:
            try:
                log_entry = json.loads(line.strip())
                if log_entry.get("type") == "request_summary":
                    endpoint = log_entry.get("endpoint", "unknown")
                    endpoint_counts[endpoint] = endpoint_counts.get(endpoint, 0) + 1
                    total_latency += log_entry.get("latency_ms", 0)
                    requests_count += 1
                
                if log_entry.get("level") == "ERROR":
                    error_count += 1
                    
            except json.JSONDecodeError:
                continue
        
        avg_latency = total_latency / requests_count if requests_count > 0 else 0
        
        return {
            "total_entries": len(all_lines),
            "total_requests": requests_count,
            "error_count": error_count,
            "avg_latency_ms": round(avg_latency, 2),
            "endpoint_distribution": endpoint_counts,
            "request_id": request_id_var.get()
        }
        
    except FileNotFoundError:
        return {
            "error": "Файл логов не найден",
            "request_id": request_id_var.get()
        }