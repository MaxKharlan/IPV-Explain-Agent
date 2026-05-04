### Всей команде: ориентируйтесь на эти JSON-контракты, когда будете обрабатывать входные и отдавать выходные данные.

# JSON-контракты проекта IPV Explain Agent

Ниже список основных JSON-контрактов, которые связывают весь проект.
Идея простая: каждый модуль получает понятный вход и возвращает понятный
выход.

---

## 1. `PositionInput`

Это входная позиция, которую система будет оценивать и объяснять.

### Что хранит

- `position_id` — ID позиции
- `instrument_type` — тип инструмента (`bond`, `option`, `swap`)
- `book` — книга / портфель
- `currency` — валюта
- `quantity` — объём
- `counterparty` — контрагент
- `as_of_dates.t0` — первая дата
- `as_of_dates.t1` — вторая дата
- `instrument` — параметры инструмента

### Пример

```json
{
  "position_id": "POS-001",
  "instrument_type": "option",
  "book": "EQ_VOL",
  "currency": "RUB",
  "quantity": 1000.0,
  "counterparty": "internal",
  "as_of_dates": {
    "t0": "2026-05-01",
    "t1": "2026-05-02"
  },
  "instrument": {
    "underlier": "SBER",
    "option_type": "call",
    "strike": 280.0,
    "maturity_date": "2026-09-20"
  }
}
```

---

## 2. `MarketSnapshot`

Это снимок рыночных данных на конкретную дату.

### Что хранит

- `snapshot_id` — ID снапшота
- `snapshot_date` — дата
- `source` — источник данных
- `spot_prices` — спотовые цены
- `yield_curve` — кривая ставок
- `vol_surface` — поверхность волатильности
- `quality_flags` — флаги качества данных

### Пример

```json
{
  "snapshot_id": "SNAP-2026-05-01",
  "snapshot_date": "2026-05-01",
  "source": "moex",
  "spot_prices": {
    "SBER": 301.55
  },
  "yield_curve": {
    "currency": "RUB",
    "points": [
      {"tenor": "1M", "rate": 0.165},
      {"tenor": "3M", "rate": 0.168}
    ]
  },
  "vol_surface": {
    "underlier": "SBER",
    "points": [
      {"tenor": "1M", "strike": 280.0, "implied_vol": 0.24}
    ]
  },
  "quality_flags": {
    "used_mock_data": false,
    "missing_curve_points": false,
    "surface_interpolated": true
  }
}
```

---

## 3. `PricingResult`

Это результат работы quant-модели.

### Что хранит

- `position_id` — ID позиции
- `instrument_type` — тип инструмента
- `price_t0` — цена на первой дате
- `price_t1` — цена на второй дате
- `currency` — валюта
- `sensitivities` — чувствительности (`delta`, `gamma`, `vega`, `theta`, `rho`)
- `model_name` — название модели
- `model_inputs_summary` — краткая сводка входов модели

### Пример

```json
{
  "position_id": "POS-001",
  "instrument_type": "option",
  "price_t0": 12.41,
  "price_t1": 14.02,
  "currency": "RUB",
  "sensitivities": {
    "delta": 0.53,
    "gamma": 0.012,
    "vega": 4.8,
    "theta": -0.15,
    "rho": 0.9
  },
  "model_name": "black_scholes",
  "model_inputs_summary": {
    "spot_t0": 301.55,
    "spot_t1": 307.10,
    "vol_t0": 0.24,
    "vol_t1": 0.27
  }
}
```

---

## 4. `AttributionOutput`

Это результат PnL attribution: объяснение, из-за чего изменилась стоимость.

### Что хранит

- `position_id` — ID позиции
- `currency` — валюта
- `price_t0` — цена на первой дате
- `price_t1` — цена на второй дате
- `total_pnl` — общее изменение стоимости
- `components.delta_effect` — вклад движения базового актива
- `components.gamma_effect` — нелинейный вклад
- `components.vega_effect` — вклад волатильности
- `components.theta_effect` — вклад времени
- `components.residual` — необъяснённый остаток
- `stress_results` — стресс-сценарии
- `waterfall_components` — компоненты для waterfall chart
- `validation` — проверка качества результата

### Пример

```json
{
  "position_id": "POS-001",
  "currency": "RUB",
  "price_t0": 12.41,
  "price_t1": 14.02,
  "total_pnl": 1.61,
  "components": {
    "delta_effect": 0.92,
    "gamma_effect": 0.13,
    "vega_effect": 0.71,
    "theta_effect": -0.09,
    "residual": -0.06
  },
  "stress_results": [
    {"scenario_name": "parallel_shift_1bp", "pnl": -0.01}
  ],
  "waterfall_components": [
    {"label": "Delta", "value": 0.92},
    {"label": "Gamma", "value": 0.13},
    {"label": "Vega", "value": 0.71},
    {"label": "Theta", "value": -0.09},
    {"label": "Residual", "value": -0.06}
  ],
  "validation": {
    "residual_threshold_passed": true,
    "notes": []
  }
}
```

---

## 5. `NarrativeOutput`

Это результат LLM-слоя: человеческое текстовое объяснение.

### Что хранит

- `position_id` — ID позиции
- `summary` — краткое объяснение
- `detailed_explanation` — подробное объяснение
- `top_drivers` — главные драйверы изменения
- `residual_comment` — комментарий по residual
- `validation_status` — статус проверки текста
- `fallback_used` — использовался ли шаблон вместо LLM

### Пример

```json
{
  "position_id": "POS-001",
  "summary": "Рост справедливой стоимости в основном связан с движением базового актива и ростом implied volatility.",
  "detailed_explanation": "Основной вклад в изменение стоимости дали delta и vega компоненты.",
  "top_drivers": [
    {"name": "delta_effect", "value": 0.92},
    {"name": "vega_effect", "value": 0.71}
  ],
  "residual_comment": "Residual находится в допустимом диапазоне.",
  "validation_status": "passed",
  "fallback_used": false
}
```

---

## 6. `IPVState`

Это общее состояние пайплайна в LangGraph.

### Что хранит

- `position`
- `market_snapshot_t0`
- `market_snapshot_t1`
- `pricing_result`
- `attribution_result`
- `narrative_result`
- `report_result`
- `errors`
- `fallback_flags`

### Пример

```json
{
  "position": "PositionInput",
  "market_snapshot_t0": "MarketSnapshot | null",
  "market_snapshot_t1": "MarketSnapshot | null",
  "pricing_result": "PricingResult | null",
  "attribution_result": "AttributionOutput | null",
  "narrative_result": "NarrativeOutput | null",
  "report_result": "dict | null",
  "errors": [],
  "fallback_flags": {
    "used_mock_market_data": false,
    "used_template_narrative": false
  }
}
```

---

# Коротко весь pipeline

```text
PositionInput
-> MarketSnapshot
-> PricingResult
-> AttributionOutput
-> NarrativeOutput
-> Report
```

---

# Зачем это всё нужно

Эти контракты нужны, чтобы:

- кванты писали математику отдельно
- market data слой поставлял данные в одном формате
- LLM не выдумывал числа, а работал по готовому attribution
- API, UI и отчёты использовали одинаковые структуры
