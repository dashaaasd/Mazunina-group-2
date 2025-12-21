# S03 – eda_cli: мини-EDA для CSV

Небольшое CLI-приложение для базового анализа CSV-файлов.
Используется в рамках Семинара 03 курса «Инженерия ИИ».

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему

## Инициализация проекта

В корне проекта (S03):

```bash
uv sync
```

Эта команда:

- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml`;
- установит сам проект `eda-cli` в окружение.

## Запуск CLI

### Краткий обзор

```bash
uv run eda-cli overview data/example.csv
```

Параметры:

- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`).

### Полный EDA-отчёт

```bash
uv run eda-cli report data/example.csv --out-dir reports
```

Основные параметры:

- `--out-dir` – каталог для отчёта (по умолчанию reports);
- `--sep` – разделитель в CSV (по умолчанию ,);
- `--encoding` – кодировка файла (по умолчанию utf-8).
Новые параметры:
- `--max-hist-columns` – сколько числовых колонок включать в гистограммы (по умолчанию: 6);
- `--title` – заголовок отчёта (по умолчанию: "EDA-отчёт");
- `--min-missing-share` – порог доли пропусков, выше которого колонка считается проблемной (по умолчанию: 0.3).

В результате в каталоге `reports/` появятся:

- `report.md` – основной отчёт в Markdown;
- `summary.csv` – таблица по колонкам;
- `missing.csv` – пропуски по колонкам;
- `correlation.csv` – корреляционная матрица (если есть числовые признаки);
- `top_categories/*.csv` – top-k категорий по строковым признакам;
- `hist_*.png` – гистограммы числовых колонок;
- `missing_matrix.png` – визуализация пропусков;
- `correlation_heatmap.png` – тепловая карта корреляций.

```bash
uv run eda-cli head data/example --n
```

-`--n`–количество строк (по умолчанию: 5);
-`--sep`–разделитель (по умолчанию ,);
-`--encoding`–кодировка (по умолчанию utf-8).

## Тесты

```bash
uv run pytest -q
```

# HTTP API сервис (HW04)

Перейдите в директорию HW04 (если структура такая)

```bash
cd homeworks/HW04/eda-cli
```

## Установите зависимости

```bash
uv sync
```

## Запустите сервер

```bash
uv run uvicorn eda_cli.api:app --reload --port 8000
```

## Доступные эндпоинты

Системные(GET):

- `/health` – проверка работоспособности сервиса;
- `/endpoints` – список всех доступных эндпоинтов.

Анализ качества(POST):

- `/quality` – оценка качества по агрегированным признакам;
- `/quality-from-csv` – оценка качества по CSV файлу;
- `/quality-flags-from-csv` – полный набор флагов качества (HW04).

Логирование (Вариант D)(GET):

- `/logs/preview` – просмотр последних логов;
- `/logs/stats` – статистика по логам.

Использование API
Через Swagger UI: http://localhost:8000/docs

Через curl:

bash
## Проверка здоровья

curl http://localhost:8000/health

## Оценка качества CSV

curl -X POST "http://localhost:8000/quality-from-csv" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/example.csv"

## Первые 3 строки
curl -X POST "http://localhost:8000/head?n=3" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/example.csv"