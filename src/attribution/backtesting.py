"""Бэктест PnL attribution на наборе исторических позиций.

Цель модуля — отвечать на главный вопрос model validation:
"Насколько хорошо наша Taylor-декомпозиция объясняет реальные PnL движения?"

Метрики:
    - pass_rate: доля позиций с |ε|/|ΔPnL| < threshold
    - mean / median / p95 / p99 of |ε|/|ΔPnL| — распределение точности
    - mean signed residual — индикатор системного bias
    - срезы по instrument_type / currency / book

Семантика fail-soft:
    Кейсы с битыми данными (mismatched ids, инверсия дат, etc.) не падают
    весь бэктест — складываются в errors с message и продолжаем дальше.
    Это критично для validation на тысячах исторических записей.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from loguru import logger

from src.attribution.schemas import (
    BacktestCase,
    BacktestError,
    BacktestPositionResult,
    BacktestReport,
    BacktestStatistics,
)
from src.attribution.taylor_decomp import (
    DEFAULT_RESIDUAL_THRESHOLD,
    ZERO_PNL_TOLERANCE,
    run_attribution,
)


def _run_one_case(
    case: BacktestCase,
    threshold: float,
) -> tuple[BacktestPositionResult | None, BacktestError | None]:
    """Прогоняет attribution для одного кейса с soft error handling.

    Returns:
        (result, None) при успехе, (None, error) при ошибке валидации.
    """
    pid = case.position.position_id
    try:
        attribution = run_attribution(
            position=case.position,
            pricing=case.pricing,
            rf_t0=case.rf_t0,
            rf_t1=case.rf_t1,
            residual_threshold=threshold,
        )
    except ValueError as e:
        return None, BacktestError(
            position_id=pid, error_type="ValueError", message=str(e)
        )
    except Exception as e:  # pragma: no cover — safety net на неожиданное
        logger.exception("Unexpected error in backtest case {}", pid)
        return None, BacktestError(
            position_id=pid, error_type=type(e).__name__, message=str(e)
        )

    abs_total = abs(attribution.total_pnl)
    if abs_total < ZERO_PNL_TOLERANCE:
        ratio: float | None = None
    else:
        ratio = abs(attribution.components.residual) / abs_total

    return (
        BacktestPositionResult(
            position_id=pid,
            instrument_type=case.position.instrument_type,
            currency=case.position.currency,
            book=case.position.book,
            total_pnl=attribution.total_pnl,
            residual=attribution.components.residual,
            residual_ratio=ratio,
            threshold_passed=attribution.residual_threshold_passed,
        ),
        None,
    )


def _aggregate_statistics(results: Sequence[BacktestPositionResult]) -> BacktestStatistics:
    """Считает агрегированную статистику по списку результатов.

    NaN/None в residual_ratio (zero-PnL cases) исключаются из percentiles,
    но учитываются в signed residual статистиках.
    """
    n_total = len(results)
    if n_total == 0:
        return BacktestStatistics(
            n_total=0, n_passed=0, n_breached=0, n_zero_pnl=0,
            pass_rate=0.0,
            mean_signed_residual=0.0, std_signed_residual=0.0,
        )

    n_passed = sum(1 for r in results if r.threshold_passed)
    n_breached = n_total - n_passed

    valid_ratios = [r.residual_ratio for r in results if r.residual_ratio is not None]
    n_zero_pnl = n_total - len(valid_ratios)

    if valid_ratios:
        ratios_arr = np.asarray(valid_ratios, dtype=np.float64)
        mean_ratio: float | None = float(np.mean(ratios_arr))
        median_ratio: float | None = float(np.median(ratios_arr))
        p95: float | None = float(np.percentile(ratios_arr, 95))
        p99: float | None = float(np.percentile(ratios_arr, 99))
    else:
        mean_ratio = median_ratio = p95 = p99 = None

    signed = np.asarray([r.residual for r in results], dtype=np.float64)
    mean_signed = float(np.mean(signed))
    # Sample std (ddof=1) для n>=2; иначе 0
    std_signed = float(np.std(signed, ddof=1)) if n_total > 1 else 0.0

    return BacktestStatistics(
        n_total=n_total,
        n_passed=n_passed,
        n_breached=n_breached,
        n_zero_pnl=n_zero_pnl,
        pass_rate=n_passed / n_total,
        mean_abs_residual_ratio=mean_ratio,
        median_abs_residual_ratio=median_ratio,
        p95_abs_residual_ratio=p95,
        p99_abs_residual_ratio=p99,
        mean_signed_residual=mean_signed,
        std_signed_residual=std_signed,
    )


def _group_by(
    results: Sequence[BacktestPositionResult],
    key_fn: Callable[[BacktestPositionResult], str | None],
) -> dict[str, BacktestStatistics]:
    """Группирует результаты по ключу и считает статистику в каждой группе.

    Позиции с key_fn(r) == None в группировку не попадают.
    """
    groups: dict[str, list[BacktestPositionResult]] = {}
    for r in results:
        key = key_fn(r)
        if key is None:
            continue
        groups.setdefault(key, []).append(r)
    return {k: _aggregate_statistics(v) for k, v in groups.items()}


def backtest_attribution(
    cases: Sequence[BacktestCase],
    *,
    residual_threshold: float = DEFAULT_RESIDUAL_THRESHOLD,
) -> BacktestReport:
    """Запускает attribution на наборе исторических кейсов и агрегирует метрики.

    Args:
        cases: Последовательность BacktestCase. Не должна быть пустой.
        residual_threshold: Порог |ε|/|ΔPnL| для pass/fail. По умолчанию 5%.

    Returns:
        BacktestReport с overall и срезовыми статистиками + список breached.

    Raises:
        ValueError: cases пуст или некорректный threshold.
    """
    if not cases:
        raise ValueError("cases must not be empty")
    if not (0.0 < residual_threshold < 1.0):
        raise ValueError(
            f"residual_threshold must be in (0, 1), got {residual_threshold}"
        )

    logger.debug(
        "Backtesting attribution on {} cases (threshold={:.2%})",
        len(cases), residual_threshold,
    )

    all_results: list[BacktestPositionResult] = []
    errors: list[BacktestError] = []

    for case in cases:
        result, error = _run_one_case(case, residual_threshold)
        if error is not None:
            errors.append(error)
        if result is not None:
            all_results.append(result)

    if errors:
        logger.warning(
            "{}/{} cases failed validation: {}",
            len(errors), len(cases),
            [e.position_id for e in errors[:5]] + (["..."] if len(errors) > 5 else []),
        )

    overall = _aggregate_statistics(all_results)
    by_instrument = _group_by(all_results, lambda r: r.instrument_type)
    by_currency = _group_by(all_results, lambda r: r.currency)
    by_book = _group_by(all_results, lambda r: r.book)
    breached = [r for r in all_results if not r.threshold_passed]

    if overall.n_total > 0:
        logger.info(
            "Backtest done: pass_rate={:.2%}, breached={}, zero_pnl={}, errors={}",
            overall.pass_rate, overall.n_breached, overall.n_zero_pnl, len(errors),
        )

    return BacktestReport(
        overall=overall,
        by_instrument_type=by_instrument,
        by_currency=by_currency,
        by_book=by_book,
        breached_positions=breached,
        all_results=all_results,
        errors=errors,
        residual_threshold=residual_threshold,
    )