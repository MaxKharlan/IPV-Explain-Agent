"""Pydantic-контракты Attribution Engine.

Этот модуль описывает интерфейс между Pricing Agent и Attribution Engine,
а также форму выходных данных, которые потом получает Narrative Agent.

Конвенция греков: ВСЕ греки — raw (математические частные производные),
без масштабирования на 1bp/1%. Это критично для воспроизводимости
attribution. Любые pre-scale конверсии должны делаться выше по стеку.
"""

from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class RiskFactorSnapshot(BaseModel):
    """Снапшот рыночных риск-факторов на конкретную дату.

    Используется как t0/t1 вход для Taylor-декомпозиции. Поля минимальны —
    ровно то, что нужно для вычисления ΔS, Δσ, Δt, Δr.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    snapshot_date: date = Field(..., description="Дата снапшота (для расчёта Δt)")
    spot: float = Field(..., gt=0, description="Цена базового актива S")
    vol: float = Field(..., ge=0, description="Implied volatility σ в десятичной форме (0.24 = 24%)")
    rate: Optional[float] = Field(default=None, description="Безрисковая ставка r (десятичная). None если не используется")


class Greeks(BaseModel):
    """Сырые греки от Pricing Agent на t0.

    Конвенция (важно!):
        delta = ∂V/∂S        — на единицу спота
        gamma = ∂²V/∂S²      — на единицу^2 спота
        vega  = ∂V/∂σ        — на единицу σ (НЕ на 1%; для 1% умножить на 0.01)
        theta = ∂V/∂t        — на год календарного времени (для долгого call < 0)
        rho   = ∂V/∂r        — на единицу ставки (НЕ на 1bp)

    Если Pricing Agent отдаёт scaled-греки — нормализуй ДО вызова Attribution Engine.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    delta: float = Field(..., description="∂V/∂S")
    gamma: float = Field(..., description="∂²V/∂S²")
    vega: float = Field(..., description="∂V/∂σ (per 1.0 σ unit)")
    theta: float = Field(..., description="∂V/∂t (per year, calendar time)")
    rho: Optional[float] = Field(default=None, description="∂V/∂r (per 1.0 rate unit)")


class Position(BaseModel):
    """Минимальные метаданные позиции, нужные для attribution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    position_id: str
    instrument_type: Literal["bond", "option", "swap"]
    quantity: float = Field(..., description="Notional или количество единиц")
    currency: str = Field(default="RUB")
    underlier: Optional[str] = Field(default=None, description="Тикер базового актива")
    book: Optional[str] = Field(default=None, description="Книга/портфель — для группировки")


class PricingResultMin(BaseModel):
    """Минимальный контракт Pricing Agent для Attribution Engine.

    Полная схема — в src/api/schemas.py (PricingResult). Здесь только то,
    что нужно для Taylor-разложения: цены на t0/t1 и греки на t0.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    position_id: str
    price_t0: float
    price_t1: float
    greeks_t0: Greeks


class AttributionComponents(BaseModel):
    """Покомпонентная декомпозиция PnL в per-unit price terms."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    delta_effect: float
    gamma_effect: float
    vega_effect: float
    theta_effect: float
    rho_effect: float = 0.0
    residual: float = Field(..., description="Unexplained ε = ΔPnL - сумма объяснённых компонент")


class AttributionResult(BaseModel):
    """Итог Taylor-разложения PnL для одной позиции.

    Все суммы — в per-unit price (умножь на position.quantity для notional).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    position_id: str
    currency: str
    price_t0: float
    price_t1: float
    total_pnl: float
    components: AttributionComponents
    explained_pnl: float = Field(..., description="Сумма всех объяснённых компонент (без residual)")
    explained_ratio: float = Field(..., description="explained_pnl / total_pnl. 1.0 если total_pnl ≈ 0")
    residual_threshold_passed: bool
    residual_threshold: float = Field(..., description="Использованный порог |ε|/|ΔPnL|")
    notes: list[str] = Field(default_factory=list, description="Диагностические сообщения")

# ---------------------------------------------------------------------------
# Stress-testing schemas
# ---------------------------------------------------------------------------


class CurveShiftScenario(BaseModel):
    """Именованный сценарий сдвига yield curve.

    shifts_by_tenor — словарь {tenor: shift_in_rate_units}.
    Например, {"1M": -0.0001, "10Y": 0.0010} = -1bp на 1M, +10bp на 10Y.
    Тенры, не указанные в словаре, считаются НЕ сдвинутыми (shift = 0).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., min_length=1)
    shifts_by_tenor: dict[str, float] = Field(
        ..., description="Сдвиги в абсолютных rate units (0.0001 = 1bp)"
    )

    def is_parallel(self) -> bool:
        """True если все ненулевые сдвиги одинаковы."""
        non_zero = [v for v in self.shifts_by_tenor.values() if v != 0.0]
        if not non_zero:
            return True  # нулевой сценарий тривиально параллелен
        return all(v == non_zero[0] for v in non_zero) and all(
            v == non_zero[0] or v == 0.0 for v in self.shifts_by_tenor.values()
        )

    def parallel_size(self) -> Optional[float]:
        """Размер параллельного сдвига или None, если сценарий непараллелен."""
        if not self.is_parallel():
            return None
        non_zero = [v for v in self.shifts_by_tenor.values() if v != 0.0]
        return non_zero[0] if non_zero else 0.0


class CurveSensitivities(BaseModel):
    """Чувствительности по точкам yield curve (key-rate sensitivities).

    Convention: pv01_by_tenor[tenor] = ∂V/∂r_tenor — изменение цены на
    единицу ставки в данном тенре (per 1.0 rate unit, НЕ per 1bp).

    Примеры:
        - 10Y bond, modified duration ≈ 7, price ≈ 100:
            pv01_by_tenor["10Y"] ≈ -700.0 (цена падает при росте ставки)
        - 5Y receiver swap (получаем фикс, платим плавающий):
            pv01_by_tenor["5Y"] ≈ +400.0 (ценность для рисивера растёт при росте ставок? — no, depends)

    Семантически это аналог KRD (Key Rate Duration), переведённый в
    абсолютные деньги. Для облигаций и свопов — основа стресс-тестов.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pv01_by_tenor: dict[str, float] = Field(
        ..., description="∂V/∂r_tenor (per 1.0 rate unit)"
    )


class StressScenarioResult(BaseModel):
    """Результат одного стресс-сценария."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_name: str
    pnl: Optional[float] = Field(default=None, description="None если данных не хватило")
    method: Literal["scalar_rho", "curve_pv01", "skipped"] = "skipped"
    notes: list[str] = Field(default_factory=list)


class StressReport(BaseModel):
    """Агрегированный отчёт по набору стресс-сценариев."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    position_id: str
    currency: str
    base_price: float = Field(..., description="Цена позиции на дату стресса (для контекста)")
    results: list[StressScenarioResult]

    def pnl_dict(self) -> dict[str, float]:
        """Совместимость с простым dict[str, float] интерфейсом.

        Возвращает только успешно посчитанные сценарии. Skipped — отсутствуют.
        """
        return {r.scenario_name: r.pnl for r in self.results if r.pnl is not None}

# ---------------------------------------------------------------------------
# Backtesting schemas
# ---------------------------------------------------------------------------


class BacktestCase(BaseModel):
    """Один кейс для бэктеста attribution.

    Связывает воедино позицию, pricing-результат и два рыночных снапшота,
    гарантируя типобезопасность (без рассинхронизации параллельных списков).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    position: Position
    pricing: PricingResultMin
    rf_t0: RiskFactorSnapshot
    rf_t1: RiskFactorSnapshot


class BacktestError(BaseModel):
    """Запись об ошибке для одного кейса бэктеста."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    position_id: str
    error_type: str
    message: str


class BacktestPositionResult(BaseModel):
    """Результат attribution для одной позиции в рамках бэктеста."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    position_id: str
    instrument_type: str
    currency: str
    book: Optional[str] = None
    total_pnl: float
    residual: float = Field(..., description="Знаковый ε")
    residual_ratio: Optional[float] = Field(
        default=None, description="|ε|/|ΔPnL|. None если |ΔPnL| ≈ 0"
    )
    threshold_passed: bool


class BacktestStatistics(BaseModel):
    """Агрегированная статистика по группе позиций."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_total: int
    n_passed: int
    n_breached: int
    n_zero_pnl: int = Field(..., description="Позиции с |ΔPnL| ≈ 0 (исключены из ratio-stat)")
    pass_rate: float
    # Статистики по |ε|/|ΔPnL|. None если все позиции zero-PnL.
    mean_abs_residual_ratio: Optional[float] = None
    median_abs_residual_ratio: Optional[float] = None
    p95_abs_residual_ratio: Optional[float] = None
    p99_abs_residual_ratio: Optional[float] = None
    # Знаковая статистика — для детекции системного смещения
    mean_signed_residual: float
    std_signed_residual: float


class BacktestReport(BaseModel):
    """Полный отчёт по бэктесту attribution на наборе исторических позиций."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    overall: BacktestStatistics
    by_instrument_type: dict[str, BacktestStatistics]
    by_currency: dict[str, BacktestStatistics]
    by_book: dict[str, BacktestStatistics] = Field(default_factory=dict)
    breached_positions: list[BacktestPositionResult]
    all_results: list[BacktestPositionResult]
    errors: list[BacktestError] = Field(default_factory=list)
    residual_threshold: float