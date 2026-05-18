"""Narrative Agent scaffold for pipeline orchestration."""

from __future__ import annotations

from src.agents.state import IPVState


def build_template_narrative_payload(state: IPVState) -> dict[str, object]:
    """Готовит payload для narrative layer из attribution-результата."""
    return {
        "position": state.get("position"),
        "attribution_result": state.get("attribution_result"),
    }


def run_narrative_agent(state: IPVState) -> IPVState:
    """Заполняет state полем narrative_result.

    Здесь сначала появится template-based narrative,
    а живой LLM можно подключить позже.
    """
    _ = state
    raise NotImplementedError("Narrative generation will be implemented in the next step.")
