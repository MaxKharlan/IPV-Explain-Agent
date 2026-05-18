"""Narrative Agent scaffold for pipeline orchestration."""

from __future__ import annotations

from src.agents.state import IPVState


def build_template_narrative_payload(state: IPVState) -> dict[str, object]:
    """Готовит payload для narrative layer из attribution-результата."""
    return {
        "position": state.get("position"),
        "attribution_result": state.get("attribution_result"),
    }


def _format_driver(name: str) -> str:
    """Преобразует техническое имя компоненты в человекочитаемое."""
    return name.replace("_effect", "").replace("_", " ")


def _pick_top_drivers(components: dict[str, float], limit: int = 2) -> list[dict[str, float | str]]:
    """Выбирает главные драйверы по абсолютной величине эффекта."""
    ranked = sorted(
        (
            {"name": key, "value": value}
            for key, value in components.items()
            if key != "residual"
        ),
        key=lambda item: abs(float(item["value"])),
        reverse=True,
    )
    return ranked[:limit]


def generate_template_narrative(state: IPVState) -> dict[str, object]:
    """Строит шаблонное объяснение по attribution_result."""
    position = state.get("position")
    attribution = state.get("attribution_result")
    if position is None or attribution is None:
        raise ValueError("Narrative agent requires position and attribution_result.")

    components = attribution.get("components", {})
    if not isinstance(components, dict):
        raise ValueError("Attribution components must be a dictionary.")

    top_drivers = _pick_top_drivers(components)
    if not top_drivers:
        summary = "Изменение стоимости пока не имеет значимых объяснённых драйверов."
        detailed = summary
    else:
        driver_names = ", ".join(_format_driver(str(item["name"])) for item in top_drivers)
        total_pnl = float(attribution.get("total_pnl", 0.0))
        summary = (
            f"Позиция {position['position_id']} изменилась на {total_pnl:.4f} "
            f"в основном за счёт факторов: {driver_names}."
        )
        detailed = (
            "Основные вкладчики в explain: "
            + "; ".join(
                f"{_format_driver(str(item['name']))} = {float(item['value']):.4f}"
                for item in top_drivers
            )
            + "."
        )

    residual = float(components.get("residual", 0.0))
    residual_comment = (
        "Residual находится в допустимом диапазоне."
        if attribution.get("residual_threshold_passed", False)
        else "Residual превышает допустимый порог и требует дополнительной проверки."
    )

    return {
        "position_id": position["position_id"],
        "summary": summary,
        "detailed_explanation": detailed,
        "top_drivers": top_drivers,
        "residual_comment": f"{residual_comment} residual={residual:.4f}",
        "validation_status": "passed" if attribution.get("residual_threshold_passed", False) else "warning",
        "fallback_used": True,
    }


def run_narrative_agent(state: IPVState) -> IPVState:
    """Заполняет state полем narrative_result.

    Здесь сначала появится template-based narrative,
    а живой LLM можно подключить позже.
    """
    narrative_result = generate_template_narrative(state)
    state["narrative_result"] = narrative_result
    state["fallback_flags"]["used_template_narrative"] = True
    return state
