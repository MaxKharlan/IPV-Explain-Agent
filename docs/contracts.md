# Актуальные контракты проекта IPV Explain Agent

Ниже описаны **текущие рабочие контракты**, которые реально используются в коде проекта.
Это не целевая архитектура “на будущее”, а именно то, что сейчас проходит через pipeline.

---

## 1. `PositionInput`

Входной payload позиции, который получает orchestration-layer.

### Поля

- `position_id` — идентификатор позиции
- `instrument_type` — тип инструмента
- `book` — книга / портфель
- `currency` — валюта результата
- `quantity` — размер позиции
- `counterparty` — контрагент, если передаётся
- `as_of_dates.t0` — начальная дата
- `as_of_dates.t1` — конечная дата
- `instrument` — параметры инструмента

### Важное

Сейчас end-to-end pipeline полноценно поддерживает **option flow**.
`bond` и `swap` частично поддержаны на уровне quant-функций, но не доведены как полный orchestration path.

### Пример

```json
{
  "position_id": "POS-SBER-CALL-001",
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
    "maturity_date": "2026-09-20",
    "vol_t0": 0.24,
    "vol_t1": 0.27
  }
}
```

---

## 2. `MarketSnapshot`

Нормализованный снимок рынка на одну дату, который отдаёт `Market Data Layer`.

### Поля

- `snapshot_id`
- `snapshot_date`
- `source`
- `spot_prices`
- `yield_curve`
- `option_quotes`
- `quality_flags`

### Важное

`MarketSnapshot` сейчас содержит только:

- `spot_prices`
- `yield_curve`
- `option_quotes`

`implied vol` и `vol_surface` в этот контракт **не входят**. Они должны строиться выше, в quant-слое.

### Пример

```json
{
  "snapshot_id": "SNAP-2025-05-05",
  "snapshot_date": "2025-05-05",
  "source": "moex",
  "spot_prices": {
    "SBER": 307.10
  },
  "yield_curve": {
    "snapshot_date": "2025-05-05",
    "currency": "RUB",
    "points": [
      {"tenor": "1Y", "rate": 16.5},
      {"tenor": "2Y", "rate": 16.8},
      {"tenor": "5Y", "rate": 17.1}
    ],
    "source": "moex"
  },
  "option_quotes": {
    "snapshot_date": "2025-05-05",
    "underlier": "SBER",
    "points": [
      {
        "option_type": "call",
        "strike": 280.0,
        "expiry": "2026-09-20",
        "settlement_price": 12.4,
        "instrument_id": "SBERC280"
      }
    ],
    "source": "moex"
  },
  "quality_flags": {
    "used_mock_data": false,
    "missing_curve_points": false,
    "used_mock_option_quotes": false
  }
}
```

---

## 3. `PricingResultMin`

Это **реальный контракт между Pricing Agent и Attribution Engine**, который используется в проекте сейчас.

### Поля

- `position_id`
- `price_t0`
- `price_t1`
- `greeks_t0`

### Структура `greeks_t0`

- `delta`
- `gamma`
- `vega`
- `theta`
- `rho`

### Важное

Это минимальный quant-контракт для attribution.

Он **не совпадает** с более богатым презентационным `PricingResult`, который мог бы включать:

- `instrument_type`
- `currency`
- `model_name`
- `model_inputs_summary`

Сейчас эти поля не участвуют в реальном orchestration flow между pricing и attribution.

### Пример

```json
{
  "position_id": "POS-SBER-CALL-001",
  "price_t0": 12.41,
  "price_t1": 13.86,
  "greeks_t0": {
    "delta": 0.53,
    "gamma": 0.012,
    "vega": 4.80,
    "theta": -0.15,
    "rho": 0.90
  }
}
```

---

## 4. `RiskFactorSnapshot`

Это минимальный снапшот риск-факторов, который получает `Attribution Engine`.

### Поля

- `snapshot_date`
- `spot`
- `vol`
- `rate`

### Важное

Этот контракт не хранится как отдельный API payload, но реально используется внутри `attribution_agent.py`.
Он строится из:

- `MarketSnapshot`
- position-level vol inputs (`vol_t0`, `vol_t1`)

### Пример

```json
{
  "snapshot_date": "2026-05-01",
  "spot": 301.55,
  "vol": 0.24,
  "rate": 0.165
}
```

---

## 5. `AttributionResult`

Это **реальный контракт**, который сейчас возвращает `Attribution Engine` и получает `Narrative Agent`.

### Поля

- `position_id`
- `currency`
- `price_t0`
- `price_t1`
- `total_pnl`
- `components`
- `explained_pnl`
- `explained_ratio`
- `residual_threshold_passed`
- `residual_threshold`
- `notes`

### Структура `components`

- `delta_effect`
- `gamma_effect`
- `vega_effect`
- `theta_effect`
- `rho_effect`
- `residual`

### Важное

Это **не тот же самый формат**, что старый презентационный `AttributionOutput` с:

- `stress_results`
- `waterfall_components`
- `validation`

Сейчас narrative-слой работает именно с `AttributionResult` в этой форме.

### Пример

```json
{
  "position_id": "POS-SBER-CALL-001",
  "currency": "RUB",
  "price_t0": 12.41,
  "price_t1": 13.86,
  "total_pnl": 1.45,
  "components": {
    "delta_effect": 0.17,
    "gamma_effect": -0.01,
    "vega_effect": 1.54,
    "theta_effect": -0.13,
    "rho_effect": -0.07,
    "residual": 0.05
  },
  "explained_pnl": 1.40,
  "explained_ratio": 0.9655,
  "residual_threshold_passed": true,
  "residual_threshold": 0.05,
  "notes": []
}
```

---

## 6. `NarrativeOutput`

Это контракт результата narrative-слоя.

### Поля

- `position_id`
- `summary`
- `detailed_explanation`
- `top_drivers`
- `residual_comment`
- `validation_status`
- `fallback_used`

### Важное

Сейчас narrative может идти по двум путям:

- `GigaChat` path
- template fallback path

Но в обоих случаях наружу возвращается один и тот же `NarrativeOutput`.

### Пример

```json
{
  "position_id": "POS-SBER-CALL-001",
  "summary": "Изменение стоимости составило 1.45 RUB. Главный фактор — вега-фактор размером 1.54 RUB. Второй по значимости фактор — тета-фактор размером -0.13 RUB.",
  "detailed_explanation": "Общее изменение стоимости позиции за период составило 1.45 RUB. Наибольший вклад внес фактор вега, отражающий чувствительность стоимости к изменению волатильности. Следующим по вкладу стал тета-фактор, связанный с временным распадом опциона. Остальные заметные факторы внесли ограниченный вклад. Остаток находится в допустимом диапазоне и подтверждает корректность разложения.",
  "top_drivers": [
    {"name": "vega_effect", "value": 1.54},
    {"name": "theta_effect", "value": -0.13}
  ],
  "residual_comment": "Остаток составляет 0.05 RUB и находится в допустимых пределах.",
  "validation_status": "passed",
  "fallback_used": false
}
```

---

## 7. `IPVState`

Это текущее состояние pipeline между агентами.

### Поля

- `position`
- `market_snapshot_t0`
- `market_snapshot_t1`
- `pricing_result`
- `attribution_result`
- `narrative_result`
- `report_result`
- `errors`
- `fallback_flags`

### Структура `fallback_flags`

- `used_mock_market_data`
- `used_template_narrative`

### Пример

```json
{
  "position": "PositionInput | null",
  "market_snapshot_t0": "MarketSnapshot | null",
  "market_snapshot_t1": "MarketSnapshot | null",
  "pricing_result": "PricingResultMin | null",
  "attribution_result": "AttributionResult | null",
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

## 8. Текущий рабочий pipeline

```text
PositionInput
-> MarketSnapshot(t0, t1)
-> PricingResultMin
-> AttributionResult
-> NarrativeOutput
-> Report payload
```

---

## 9. Текущее положение дел

Сейчас проект стабильно проходит путь:

- `position`
- `market data`
- `pricing`
- `attribution`
- `narrative`

при следующих условиях:

- используется `option`-позиция
- доступен PostgreSQL storage или snapshot пересобирается live/mock путём
- для live narrative настроен `GigaChat`

Если `GigaChat` недоступен, narrative слой уходит в template fallback, но общий pipeline остаётся рабочим.
