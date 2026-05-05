"""Стресс-тестирование позиций через sensitivity-based approximation.

Поддерживает:
    - параллельные сдвиги кривой (через скалярный rho или PV01-вектор)
    - twist (steepener / flattener) — линейный наклон
    - butterfly — изменение кривизны
    - произвольные кастомные сценарии через CurveShiftScenario

Математика (first-order):

    Для скалярного rho (option-style):
        ΔPnL ≈ ρ × shift           (только для параллельных сценариев)

    Для curve sensitivities (bond/swap-style):
        ΔPnL ≈ Σ_tenor PV01_tenor × shift_tenor

Границы применимости:
    - First-order: convexity не учитывается. Для shift > 50bp ошибка может
      быть существенной — лучше через repricing.
    - Sensitivities берутся at base state. При больших шоках вокруг ATM-points
      нелинейность в опционах становится важной (gamma, vanna).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from loguru import logger

from src.attribution.schemas import (
    CurveSensitivities,
    CurveShiftScenario,
    Greeks,
    Position,
    StressReport,
    StressScenarioResult,
)

# Конвенция тенров по умолчанию — стандартный grid для рублёвой кривой ОФЗ
DEFAULT_TENORS: tuple[str, ...] = ("1M", "3M", "6M", "1Y", "2Y", "5Y", "10Y", "20Y")

# 1bp в абсолютных rate units
BP: float = 1e-4


# ---------------------------------------------------------------------------
# Конструкторы стандартных сценариев
# ---------------------------------------------------------------------------


def parallel_shift(bps: float, tenors: Sequence[str] = DEFAULT_TENORS) -> CurveShiftScenario:
    """Создаёт параллельный сдвиг кривой на заданное число bp.

    Args:
        bps: Размер сдвига в bp (целочисленность не требуется).
        tenors: Список тенров, на которые применяется сдвиг.

    Returns:
        Сценарий с одинаковым сдвигом по всем тенрам.
    """
    if not tenors:
        raise ValueError("tenors must not be empty")
    shift = bps * BP
    sign = "+" if bps >= 0 else "-"
    return CurveShiftScenario(
        name=f"parallel_shift_{sign}{abs(int(bps))}bp",
        shifts_by_tenor={t: shift for t in tenors},
    )


def twist(slope_bps: float, tenors: Sequence[str] = DEFAULT_TENORS) -> CurveShiftScenario:
    """Twist-сценарий: short rate -slope_bps, long rate +slope_bps, линейная интерполяция.

    Положительный slope_bps = steepener (длинные ставки растут сильнее коротких).
    Отрицательный = flattener.

    Math: для тенров [t_0, ..., t_{n-1}] вес i-го = -1 + 2·i/(n-1) ∈ [-1, +1].
    Сдвиг тенра i = slope_bps × BP × вес.

    Args:
        slope_bps: Амплитуда наклона в bp (от -slope до +slope по концам).
        tenors: Тенры curve. Должно быть >= 2.

    Returns:
        Сценарий twist.
    """
    n = len(tenors)
    if n < 2:
        raise ValueError(f"twist needs >= 2 tenors, got {n}")

    shifts: dict[str, float] = {}
    for i, t in enumerate(tenors):
        weight = -1.0 + 2.0 * i / (n - 1)
        shifts[t] = slope_bps * BP * weight

    direction = "steepener" if slope_bps > 0 else "flattener"
    return CurveShiftScenario(
        name=f"twist_{direction}_{abs(int(slope_bps))}bp",
        shifts_by_tenor=shifts,
    )


def butterfly(wing_bps: float, tenors: Sequence[str] = DEFAULT_TENORS) -> CurveShiftScenario:
    """Butterfly-сценарий: концы кривой +wing_bps, середина -2·wing_bps.

    Положительный wing_bps = «банановая» дуга (концы вверх, середина вниз) →
    кривизна становится более выпуклой вниз.

    Args:
        wing_bps: Сдвиг крыльев в bp.
        tenors: Тенры curve. Должно быть >= 3.

    Returns:
        Сценарий butterfly.
    """
    n = len(tenors)
    if n < 3:
        raise ValueError(f"butterfly needs >= 3 tenors, got {n}")

    mid_idx = n // 2
    shifts: dict[str, float] = {}
    for i, t in enumerate(tenors):
        if i == 0 or i == n - 1:
            shifts[t] = wing_bps * BP
        elif i == mid_idx:
            shifts[t] = -2.0 * wing_bps * BP
        else:
            shifts[t] = 0.0

    direction = "wings_up" if wing_bps > 0 else "wings_down"
    return CurveShiftScenario(
        name=f"butterfly_{direction}_{abs(int(wing_bps))}bp",
        shifts_by_tenor=shifts,
    )


def standard_scenario_set(tenors: Sequence[str] = DEFAULT_TENORS) -> list[CurveShiftScenario]:
    """Стандартный набор сценариев для daily IPV-стресса.

    Включает:
        - параллельные ±1bp, ±10bp, ±100bp (6 сценариев)
        - twist steepener / flattener ±25bp (2 сценария)
        - butterfly wings_up / wings_down ±25bp (2 сценария)

    Returns:
        Список из 10 стандартных CurveShiftScenario.
    """
    scenarios: list[CurveShiftScenario] = []
    for bps in (-100, -10, -1, 1, 10, 100):
        scenarios.append(parallel_shift(bps, tenors))
    scenarios.append(twist(+25, tenors))
    scenarios.append(twist(-25, tenors))
    scenarios.append(butterfly(+25, tenors))
    scenarios.append(butterfly(-25, tenors))
    return scenarios


# ---------------------------------------------------------------------------
# Численное ядро
# ---------------------------------------------------------------------------


def _estimate_scenario_pnl(
    scenario: CurveShiftScenario,
    greeks: Greeks,
    curve_sens: CurveSensitivities | None,
) -> StressScenarioResult:
    """Считает PnL impact одного сценария, выбирая лучший доступный метод.

    Приоритет методов:
        1. curve_pv01 — если задан CurveSensitivities (любой сценарий)
        2. scalar_rho — если параллельный сценарий и есть rho
        3. skipped   — non-parallel без curve_sens, или нет ни rho ни PV01

    Args:
        scenario: Сценарий стресс-теста.
        greeks: Греки позиции (содержат rho).
        curve_sens: Опциональные key-rate sensitivities.

    Returns:
        StressScenarioResult с pnl, методом и диагностикой.
    """
    notes: list[str] = []

    # ---------- Метод 1: PV01 по тенрам ----------
    if curve_sens is not None:
        pnl = np.float64(0.0)
        missing_tenors: list[str] = []
        unused_pv01: list[str] = []

        for tenor, shift in scenario.shifts_by_tenor.items():
            if shift == 0.0:
                continue
            pv01 = curve_sens.pv01_by_tenor.get(tenor)
            if pv01 is None:
                missing_tenors.append(tenor)
                continue
            pnl += np.float64(pv01) * np.float64(shift)

        # Тенры в PV01, на которые мы не наложили сдвиг — это нормально (=0),
        # но если в сценарии они присутствуют как 0 — отметим явно
        for tenor in curve_sens.pv01_by_tenor:
            if tenor not in scenario.shifts_by_tenor:
                unused_pv01.append(tenor)

        if missing_tenors:
            notes.append(
                f"PV01 not provided for tenors {missing_tenors}; their shift contribution skipped."
            )
        if unused_pv01:
            notes.append(
                f"PV01 present but no shift for tenors {unused_pv01} (treated as zero shift)."
            )

        return StressScenarioResult(
            scenario_name=scenario.name,
            pnl=float(pnl),
            method="curve_pv01",
            notes=notes,
        )

    # ---------- Метод 2: скалярный rho (только parallel) ----------
    if scenario.is_parallel() and greeks.rho is not None:
        size = scenario.parallel_size()
        # size не может быть None если is_parallel(), но контракт явно
        size_val = 0.0 if size is None else size
        pnl = float(np.float64(greeks.rho) * np.float64(size_val))
        return StressScenarioResult(
            scenario_name=scenario.name,
            pnl=pnl,
            method="scalar_rho",
            notes=notes,
        )

    # ---------- Метод 3: skip ----------
    if not scenario.is_parallel():
        notes.append(
            f"Non-parallel scenario without CurveSensitivities — cannot approximate via scalar rho."
        )
    else:
        notes.append("Parallel scenario but rho is None — cannot approximate.")

    return StressScenarioResult(
        scenario_name=scenario.name,
        pnl=None,
        method="skipped",
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Публичный API
# ---------------------------------------------------------------------------


def run_stress_scenarios(
    position: Position,
    greeks: Greeks,
    base_price: float,
    scenarios: Sequence[CurveShiftScenario],
    *,
    curve_sensitivities: CurveSensitivities | None = None,
) -> StressReport:
    """Запускает набор стресс-сценариев и возвращает агрегированный отчёт.

    Args:
        position: Метаданные позиции.
        greeks: Греки позиции (rho используется для параллельных сценариев).
        base_price: Базовая цена позиции (для контекста в отчёте).
        scenarios: Список сценариев. Можно использовать standard_scenario_set().
        curve_sensitivities: Опциональные KRD/PV01 по тенрам — если есть,
            применяются для всех сценариев (включая twist/butterfly).

    Returns:
        StressReport. Используй .pnl_dict() для совместимости с dict[str, float].

    Raises:
        ValueError: если scenarios пуст.
    """
    if not scenarios:
        raise ValueError("scenarios must not be empty")

    logger.debug(
        "Running {} stress scenarios for {} ({})",
        len(scenarios),
        position.position_id,
        "with curve sensitivities" if curve_sensitivities else "rho-only fallback",
    )

    results = [
        _estimate_scenario_pnl(s, greeks, curve_sensitivities) for s in scenarios
    ]

    skipped = [r for r in results if r.method == "skipped"]
    if skipped:
        logger.warning(
            "{}/{} scenarios skipped for {}: {}",
            len(skipped),
            len(results),
            position.position_id,
            [r.scenario_name for r in skipped],
        )

    return StressReport(
        position_id=position.position_id,
        currency=position.currency,
        base_price=float(base_price),
        results=results,
    )