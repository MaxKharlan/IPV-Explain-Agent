"""Template-based narrative generation with validation rules."""

from __future__ import annotations

from typing import Any

from src.agents.state import IPVState


REQUIRED_COMPONENT_KEYS = (
    "delta_effect",
    "gamma_effect",
    "vega_effect",
    "theta_effect",
    "residual",
)


def build_template_narrative_payload(state: IPVState) -> dict[str, object]:
    """Готовит нормализованный payload для explain-слоя."""
    position = state.get("position")
    attribution = state.get("attribution_result")
    if position is None or attribution is None:
        raise ValueError("Narrative agent requires position and attribution_result.")

    components = attribution.get("components", {})
    if not isinstance(components, dict):
        raise ValueError("Attribution components must be a dictionary.")

    validate_attribution_payload(position, attribution)

    return {
        "position_id": position["position_id"],
        "instrument_type": position.get("instrument_type"),
        "total_pnl": float(attribution.get("total_pnl", 0.0)),
        "currency": position.get("currency", attribution.get("currency", "RUB")),
        "components": {
            key: float(components[key])
            for key in REQUIRED_COMPONENT_KEYS
        },
        "residual_threshold_passed": bool(attribution.get("residual_threshold_passed", False)),
    }


def validate_attribution_payload(
    position: dict[str, Any],
    attribution: dict[str, Any],
) -> None:
    """Проверяет минимальную консистентность narrative input."""
    if attribution.get("position_id") != position.get("position_id"):
        raise ValueError("Narrative payload position_id mismatch.")

    components = attribution.get("components", {})
    if not isinstance(components, dict):
        raise ValueError("Attribution components must be a dictionary.")

    missing = [key for key in REQUIRED_COMPONENT_KEYS if key not in components]
    if missing:
        raise ValueError(f"Attribution components missing keys: {', '.join(missing)}")


def _format_driver(name: str) -> str:
    """Преобразует техническое имя компоненты в человекочитаемое."""
    return name.replace("_effect", "").replace("_", " ")


def _signed_direction(value: float) -> str:
    """Возвращает словесное направление вклада."""
    if value > 0:
        return "положительный"
    if value < 0:
        return "отрицательный"
    return "нейтральный"


def _pick_top_drivers(
    components: dict[str, float],
    limit: int = 2,
) -> list[dict[str, float | str]]:
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


def _build_summary(payload: dict[str, object], top_drivers: list[dict[str, float | str]]) -> str:
    """Строит короткое summary по валидированному payload."""
    total_pnl = float(payload["total_pnl"])
    position_id = str(payload["position_id"])
    if not top_drivers:
        return f"Позиция {position_id} изменилась на {total_pnl:.4f}, но значимые explain-драйверы не выделены."

    driver_names = ", ".join(_format_driver(str(item["name"])) for item in top_drivers)
    return (
        f"Позиция {position_id} изменилась на {total_pnl:.4f} "
        f"в основном за счёт факторов: {driver_names}."
    )


def _build_detailed_explanation(top_drivers: list[dict[str, float | str]]) -> str:
    """Строит детальное описание top drivers."""
    if not top_drivers:
        return "Объясняющие компоненты отсутствуют или равны нулю."

    return "Основные вкладчики в explain: " + "; ".join(
        (
            f"{_format_driver(str(item['name']))} = {float(item['value']):.4f} "
            f"({_signed_direction(float(item['value']))} вклад)"
        )
        for item in top_drivers
    ) + "."


def _build_residual_comment(payload: dict[str, object]) -> str:
    """Строит комментарий по residual."""
    components = payload["components"]
    assert isinstance(components, dict)
    residual = float(components["residual"])
    passed = bool(payload["residual_threshold_passed"])
    if passed:
        return f"Residual находится в допустимом диапазоне: {residual:.4f}."
    return f"Residual превышает допустимый порог и требует проверки: {residual:.4f}."


def generate_template_narrative(state: IPVState) -> dict[str, object]:
    """Строит шаблонное объяснение по attribution_result."""
    payload = build_template_narrative_payload(state)
    components = payload["components"]
    assert isinstance(components, dict)

    top_drivers = _pick_top_drivers(components)
    summary = _build_summary(payload, top_drivers)
    detailed = _build_detailed_explanation(top_drivers)
    residual_comment = _build_residual_comment(payload)
    validation_status = "passed" if payload["residual_threshold_passed"] else "warning"

    return {
        "position_id": payload["position_id"],
        "summary": summary,
        "detailed_explanation": detailed,
        "top_drivers": top_drivers,
        "residual_comment": residual_comment,
        "validation_status": validation_status,
        "fallback_used": True,
    }


def run_narrative_agent(state: IPVState) -> IPVState:
    """Заполняет state полем narrative_result."""
    narrative_result = generate_template_narrative(state)
    state["narrative_result"] = narrative_result
    state["fallback_flags"]["used_template_narrative"] = True
    return state
