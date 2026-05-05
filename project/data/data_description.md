# Описание данных проекта

## Источник данных

**Датасет:** Digital Advertising Campaign Performance Dataset

**Ссылка:** [\[Kaggle\]](https://www.kaggle.com/datasets/juniornsa/digital-advertising-campaign-performance-dataset)

**Тип данных:** синтетические (сгенерированы на основе реалистичных бизнес-правил)

**Файл:** `tech_advertising_campaigns_dataset.csv`

## Описание датасета

| Характеристика | Значение |
|----------------|----------|
| Количество записей | 10 000 |
| Количество признаков | 41 |
| Временной диапазон | 2024-01-01 — 2026-01-31 |
| Целевая переменная | ROAS (Return on Ad Spend) |
| Размер файла | ~2 МБ |

## Структура данных

## Словарь переменных

### Идентификация кампании (2)

| Переменная | Тип | Описание | Значения |
|------------|-----|----------|----------|
| campaign_id | Строка | Уникальный идентификатор | CAMP_00001 – CAMP_10000 |
| campaign_objective | Категория | Цель кампании | Brand Awareness, Lead Generation, Conversions, App Installs, Engagement |

### Платформа и размещение (4)

| Переменная | Тип | Описание | Значения |
|------------|-----|----------|----------|
| platform | Категория | Рекламная платформа | Google Ads, Facebook, LinkedIn, TikTok, Twitter, Instagram |
| ad_placement | Категория | Место показа объявления | Feed, Stories, Search, Display Network, In-Stream Video, Sidebar |
| device_type | Категория | Тип устройства | Desktop, Mobile, Tablet |
| operating_system | Категория | ОС устройства | iOS, Android, Windows, macOS, Other |

### Креативные атрибуты (6)

| Переменная | Тип | Описание | Значения/Диапазон |
|------------|-----|----------|-------------------|
| creative_format | Категория | Формат объявления | Video, Image, Carousel, Text, Interactive, Story |
| creative_size | Категория | Размеры (пиксели) | 1920x1080, 1080x1080, 300x250, 728x90, 320x50, 1200x628 |
| ad_copy_length | Категория | Длина текста | Short, Medium, Long |
| has_call_to_action | Булева | Наличие CTA-кнопки | True, False |
| creative_emotion | Категория | Эмоциональный тон | Fear, Joy, Urgency, Trust, Curiosity, Neutral |
| creative_age_days | Числовая | Дней с запуска креатива | 1–90 |

### Таргетинг аудитории (6)

| Переменная | Тип | Описание | Значения |
|------------|-----|----------|----------|
| target_audience_age | Категория | Возрастная группа | 18-24, 25-34, 35-44, 45-54, 55-64, 65+ |
| target_audience_gender | Категория | Пол | Male, Female, All |
| audience_interest_category | Категория | Интересы | Tech Enthusiasts, Business Professionals, Gamers, Students, Shoppers, Health & Fitness |
| income_bracket | Категория | Доход домохозяйства | <$50K, $50K-$100K, $100K-$200K, >$200K |
| purchase_intent_score | Категория | Сигнал покупательского намерения | Low, Medium, High |
| retargeting_flag | Булева | Ретаргетинг | True, False |

### Временные (5)

| Переменная | Тип | Описание | Диапазон |
|------------|-----|----------|----------|
| start_date | Дата | Дата запуска | 2024-01-01 – 2026-01-31 |
| quarter | Числовая | Календарный квартал | 1–4 |
| day_of_week | Категория | День недели | Monday–Sunday |
| hour_of_day | Числовая | Час показа (24-часовой) | 0–23 |
| campaign_day | Числовая | День жизненного цикла кампании | 1–90 |

### Аукцион и качество (2)

| Переменная | Тип | Описание | Диапазон |
|------------|-----|----------|----------|
| quality_score | Числовая | Оценка качества платформой | 1–10 |
| actual_cpc | Числовая | Фактическая цена клика ($) | $0.25 – $17.00 |

### Метрики эффективности (5)

| Переменная | Тип | Описание | Ограничения |
|------------|-----|----------|-------------|
| impressions | Числовая | Количество показов | 5 000 – 500 000 |
| clicks | Числовая | Количество кликов | ≥ 10, ≤ impressions |
| conversions | Числовая | Завершённые действия | ≥ 0, ≤ clicks |
| ad_spend | Числовая | Расходы на рекламу ($) | clicks × actual_cpc|
| revenue | Числовая | Полученная выручка ($) | ≥ 0 |

### Качество вовлечённости (3)

| Переменная | Тип | Описание | Диапазон |
|------------|-----|----------|----------|
| bounce_rate | Числовая | % немедленных уходов | 10.0 – 90.0 |
| avg_session_duration_seconds | Числовая | Среднее время на сайте (сек) | 10 – 600 |
| pages_per_session | Числовая | Среднее количество страниц | 1.0 – 10.0 |

### Отраслевой контекст (2)

| Переменная | Тип | Описание | Значения |
|------------|-----|----------|----------|
| industry_vertical | Категория | Сектор бизнеса | SaaS, E-commerce, Healthcare, Finance, Education, Gaming |
| budget_tier | Категория | Классификация бюджета | Low, Medium, High |

### Расчётные метрики (6)

| Переменная | Формула | Описание |
|------------|--------|----------|
| CTR | (clicks / impressions) × 100 | Click-Through Rate (%) |
| CPC | ad_spend / clicks | Cost Per Click ($) |
| conversion_rate | (conversions / clicks) × 100 | Конверсия из клика (%) |
| CPA | ad_spend / conversions | Cost Per Acquisition ($) |
| ROAS | revenue / ad_spend | Return on Ad Spend (мультипликатор) |
| profit | revenue − ad_spend | **Основная зависимая переменная** ($) |

## Предобработка

Пайплайн предобработки в `src/data/preprocessor.py`:

1. Удаление утечек: `revenue` (ROAS = revenue / ad_spend), `profit` (альтернативная цель)
2. Извлечение временных признаков: `year`, `month`, `week_of_year`, `is_weekend` из `start_date`
3. Кодирование категорий: LabelEncoder → `*_encoded`
4. Масштабирование: StandardScaler для числовых признаков
5. Заполнение пропусков медианами (если есть)

## Использование

```bash
import pandas as pd
from src.data.preprocessor import CampaignDataProcessor

df = pd.read_csv('data/tech_advertising_campaigns_dataset.csv')
processor = CampaignDataProcessor()
df_processed = processor.prepare_features(df, fit=True)
```
