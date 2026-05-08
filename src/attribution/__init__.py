"""Attribution Engine — публичный API.

Три модуля, объединённых в единый пакет:
    - taylor_decomp: PnL Taylor-разложение по грекам
    - stress_tests:  стресс-сценарии (parallel/twist/butterfly)
    - backtesting:   model validation на исторических позициях

Пример использования:
    >>> from src.attribution import (
    ...     run_attribution, run_stress_scenarios, backtest_attribution,
    ...     standard_scenario_set,
    ... )
"""

from src.attribution.backtesting import backtest_attribution
from src.attribution.schemas import (
    AttributionComponents,
    AttributionResult,
    BacktestCase,
    BacktestError,
    BacktestPositionResult,
    BacktestReport,
    BacktestStatistics,
    CurveSensitivities,
    CurveShiftScenario,
    Greeks,
    Position,
    PricingResultMin,
    RiskFactorSnapshot,
    StressReport,
    StressScenarioResult,
)
from src.attribution.stress_tests import (
    BP,
    DEFAULT_TENORS,
    butterfly,
    parallel_shift,
    run_stress_scenarios,
    standard_scenario_set,
    twist,
)
from src.attribution.taylor_decomp import (
    DEFAULT_RESIDUAL_THRESHOLD,
    ZERO_PNL_TOLERANCE,
    run_attribution,
)

__all__ = [
    # --- Core attribution ---
    "run_attribution",
    "DEFAULT_RESIDUAL_THRESHOLD",
    "ZERO_PNL_TOLERANCE",
    # --- Stress testing ---
    "run_stress_scenarios",
    "parallel_shift",
    "twist",
    "butterfly",
    "standard_scenario_set",
    "DEFAULT_TENORS",
    "BP",
    # --- Backtesting ---
    "backtest_attribution",
    # --- Schemas: inputs ---
    "Position",
    "Greeks",
    "RiskFactorSnapshot",
    "PricingResultMin",
    # --- Schemas: attribution output ---
    "AttributionComponents",
    "AttributionResult",
    # --- Schemas: stress ---
    "CurveShiftScenario",
    "CurveSensitivities",
    "StressScenarioResult",
    "StressReport",
    # --- Schemas: backtest ---
    "BacktestCase",
    "BacktestError",
    "BacktestPositionResult",
    "BacktestStatistics",
    "BacktestReport",
]

__version__ = "0.1.0"