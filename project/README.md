# Итоговый проект по курсу «Инженерия Искусственного Интеллекта»

В этой папке находится итоговый мини-проект по курсу.  
Проект демонстрирует применение методов и инструментов инженерии ИИ: работу с данными, модели, пайплайны, сервис, эксперименты и воспроизводимость.

---

## 1. Паспорт проекта

- **Название проекта:** Сервис предсказания эффективности рекламных кампаний
- **Автор:** Мазунина Дарья Андреевна
- **Группа:** БСБО-51-24
- **Контакт:** @dm_dmaz

**Краткое описание:**

Проект представляет собой ML-сервис для прогнозирования ROAS (Return on Ad Spend) рекламных кампаний и рекомендации наиболее эффективных рекламных платформ. На основе исторических данных о 10 000 кампаний модель CatBoost предсказывает окупаемость вложений и ранжирует каналы по эффективности. Результат — REST API на FastAPI с эндпоинтами `/predict` и `/recommend`, документация Swagger, контейнеризация Docker.

---

## 2. Структура проекта

```
project/
├── requirements.txt          # зависимости проекта
├── report.md                 # отчёт по проекту
├── self-checklist.md         # чеклист самопроверки
├── Dockerfile                # контейнеризация
├── .dockerignore
├── notebooks/
│   ├── 01_eda_v2.ipynb                 # разведочный анализ данных
│   ├── 02_baseline_models.ipynb        # сравнение 7 базовых моделей
│   └── 03_hyperparameter_tuning.ipynb  # подбор гиперпараметров (MLflow)
├── src/
│   ├── train.py                        # обучение модели
│   ├── data/
│   │   └── preprocessor.py             # предобработка данных
│   ├── models/
│   │   └── predictor.py                # загрузка модели + predict/recommend
│   └── api/
│       ├── app.py                       # FastAPI сервер
│       └── schemas.py                   # Pydantic-схемы запросов/ответов
├── data/
│   └── tech_advertising_campaigns_dataset.csv
├── configs/
│   ├── model_config.yaml               # гиперпараметры модели и пути
│   └── .env.example                    # шаблон переменных окружения
├── tests/
│   ├── test_preprocessor.py
│   ├── test_predictor.py
│   └── test_api.py
└── artifacts/
    ├── models/                         # обученные модели (.pkl)
    └── final_metrics.json              # финальные метрики
```

---

## 3. Требования и установка

### 3.1. Требования

- Python `>= 3.10`
- Docker (опционально, для контейнеризации)

### 3.2. Установка окружения

```bash
# Перейти в папку проекта
cd project

# Создать виртуальное окружение
python -m venv .venv

# Активировать окружение:
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# Установить зависимости
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Как запустить проект

### 4.1. Обучение модели

```bash
cd project
source .venv/bin/activate
python -m src.train
```

Модель обучится и сохранится в `artifacts/models/best_model_service.pkl`.  
Препроцессор — в `artifacts/models/preprocessor.pkl`.

> При первом запуске сервиса обучение запускается автоматически, если артефакты не найдены.

### 4.2. Запуск сервиса

```bash
cd project
source .venv/bin/activate
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Порт: **8000**

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/health` | Проверка работоспособности сервиса |
| POST | `/predict` | Прогноз ROAS для кампании на заданной платформе |
| POST | `/recommend` | Рекомендация лучшего канала из 6 платформ |

Swagger UI: http://localhost:8000/docs

### 4.3. Запуск через Docker

```bash
cd project
docker build -t roas-service .
docker run -p 8000:8000 roas-service
```

Браузер: http://localhost:8000/docs

### 4.4. Тестирование API (PowerShell)

```powershell
# Health
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"

# Predict — прогноз ROAS для Facebook
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" `
  -Method Post -ContentType "application/json" `
  -Body '{"platform":"Facebook","campaign_objective":"Conversions","device_type":"Desktop","operating_system":"Windows","ad_placement":"feed","day_of_week":"Monday","ad_spend":5000,"start_date":"2025-01-15"}'

# Recommend — подобрать лучший канал
Invoke-RestMethod -Uri "http://127.0.0.1:8000/recommend" `
  -Method Post -ContentType "application/json" `
  -Body '{"campaign_objective":"Conversions","device_type":"Desktop","operating_system":"Windows","ad_placement":"feed","day_of_week":"Monday","ad_spend":5000,"start_date":"2025-01-15"}'
```

---

## 5. Данные

- **Источник:** Digital Advertising Campaign Performance Dataset ([Kaggle](https://www.kaggle.com/datasets/juniornsa/digital-advertising-campaign-performance-dataset))
- **Тип:** синтетические данные с реалистичными бизнес-правилами
- **Файл:** `data/tech_advertising_campaigns_dataset.csv`
- **Объём:** 10 000 записей, 41 признак, период 2024-01-01 — 2026-01-31
- **Целевая переменная:** ROAS (Return on Ad Spend = revenue / ad_spend)

Подробное описание признаков — в `data/data_description.md`.

---

## 6. Тесты

```bash
cd project
pytest tests -v
```

Что проверяется:

- `test_preprocessor.py` — препроцессор корректно трансформирует данные, нет утечек признаков, `transform` работает без `fit`
- `test_predictor.py` — модель загружается, `predict_single` возвращает ROAS > 0, `recommend_channel` возвращает ровно 6 каналов с корректной структурой
- `test_api.py` — `/health` отвечает 200, `/predict` возвращает `predicted_ROAS`, `/recommend` возвращает `best_channel` и `all_results`

---

## 7. Демонстрация на защите

**Запуск через Docker:**
```bash
docker build -t roas-service .
docker run -p 8000:8000 roas-service
```

При первом запуске контейнер автоматически обучит модель (~25 секунд), затем сервис готов.

**Сценарии:**

1. **Структура проекта** — обзор `src/`, `notebooks/`, `configs/`, `artifacts/`
2. **Swagger UI** (http://localhost:8000/docs) — демонстрация `/predict` и `/recommend` с живыми запросами
3. **Логи в терминале** — показать, что каждый запрос логируется с IP, параметрами и результатом
4. **EDA-ноутбук** — ключевые графики: распределение ROAS, сезонность, эффективность платформ
5. **Сравнение моделей** — таблица 7 baseline-моделей, обоснование выбора CatBoost (R² = 0.882)
6. **Feature Importance** — CPA (33%), conversion_rate (22.5%), income_bracket (11.5%)
7. **`configs/model_config.yaml`** — показать, что гиперпараметры вынесены из кода

---

## 8. Ограничения и дальнейшая работа

**Ограничения:**

- Данные синтетические — перед продакшном необходима валидация на реальных кампаниях
- Высокий MAPE (37.4%) на выбросах — кампании с экстремальным ROAS предсказываются хуже
- Только 6 рекламных платформ — расширение требует дообучения

**Что можно улучшить:**

- Логарифмирование ROAS перед обучением для снижения MAPE
- Кэширование предсказаний для частых запросов
- Автоматическое переобучение по расписанию при деградации метрик

---

## 9. Оценка проекта

Итоговая оценка за проект выставляется по пятибалльной шкале (2–5).

Ориентиры для оценки:

- **2** — проект не принят: не выполняются минимальные требования, грубые нарушения, плагиат.
- **3** — базовый уровень: минимальный функционал есть, по чеклисту выполнено менее 5 пунктов.
- **4** — хороший проект: сервис запускается, есть данные, EDA и эксперименты, выполнено не менее 5 пунктов чеклиста.
- **5** — сильный проект: аккуратный пайплайн, осмысленные эксперименты, наблюдаемость, документация позволяет воспроизвести решение, выполнено не менее 9 пунктов чеклиста.

Чеклист самопроверки: `self-checklist.md`.
