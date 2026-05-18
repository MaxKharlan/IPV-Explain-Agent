"""High-level orchestration scaffold."""

from __future__ import annotations

from typing import Any

from src.agents.attribution_agent import run_attribution_agent
from src.agents.market_data_agent import run_market_data_agent
from src.agents.narrative_agent import run_narrative_agent
from src.agents.pricing_agent import run_pricing_agent
from src.agents.report_agent import run_report_agent
from src.agents.state import IPVState, create_initial_state


def get_position(position_id: str) -> dict[str, Any]:
    """Возвращает нормализованные данные о позиции."""
    raise NotImplementedError


def load_market_snapshot(snapshot_date: str, instrument_context: dict[str, Any]):
    """Возвращает нормализованную картину рынка за определённую дату."""
    raise NotImplementedError


def compute_fair_value(
    position: dict[str, Any],
    market_t0: dict[str, Any],
    market_t1: dict[str, Any],
):
    """Результаты модели, необходимые для атрибуции."""
    raise NotImplementedError


def run_attribution(pricing_result: dict[str, Any]):
    """Возвращает готовую к объяснению разбивку прибыли и убытка."""
    raise NotImplementedError


def generate_narrative(attribution: dict[str, Any]):
    """Возвращает проверенный текстовый LLM вывод."""
    raise NotImplementedError


def build_report(
    position: dict[str, Any],
    attribution: dict[str, Any],
    narrative: dict[str, Any],
):
    """Возвращает payload отчета для экспорта в формат PDF или HTML."""
    raise NotImplementedError


def initialize_pipeline(position: dict[str, Any] | None = None) -> IPVState:
    """Создаёт стартовое состояние pipeline."""
    return create_initial_state(position=position)


def run_pipeline_until_attribution(position: dict[str, Any]) -> IPVState:
    """Прогоняет pipeline от позиции до attribution_result."""
    state = initialize_pipeline(position=position)
    state = run_market_data_agent(state)
    state = run_pricing_agent(state)
    state = run_attribution_agent(state)
    return state


def run_pipeline(position: dict[str, Any]) -> IPVState:
    """Прогоняет pipeline до narrative и report результата."""
    state = run_pipeline_until_attribution(position)
    state = run_narrative_agent(state)
    state = run_report_agent(state)
    return state
