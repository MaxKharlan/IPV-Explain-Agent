"""Narrative generation with template rules and GigaChat integration."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.agents.state import IPVState


REQUIRED_COMPONENT_KEYS = (
    "delta_effect",
    "gamma_effect",
    "vega_effect",
    "theta_effect",
    "residual",
)

REQUIRED_NARRATIVE_KEYS = (
    "position_id",
    "summary",
    "detailed_explanation",
    "top_drivers",
    "residual_comment",
    "validation_status",
    "fallback_used",
)


class GigaChatClient:
    """Минимальный HTTP-клиент для narrative generation через GigaChat."""

    def __init__(
        self,
        *,
        api_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: int = 30,
    ) -> None:
        self.api_url = api_url or os.getenv("GIGACHAT_API_URL")
        self.api_key = api_key or os.getenv("GIGACHAT_API_KEY")
        self.model = model or os.getenv("GIGACHAT_MODEL", "GigaChat")
        self.timeout_seconds = timeout_seconds

    def is_configured(self) -> bool:
        """Проверяет, что для GigaChat заданы обязательные параметры."""
        return bool(self.api_url and self.api_key)

    def build_headers(self) -> dict[str, str]:
        """Строит HTTP headers для вызова GigaChat."""
        if not self.api_key:
            raise ValueError("GigaChat API key is not configured.")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def build_request_body(self, prompt: str) -> dict[str, object]:
        """Строит request body для chat-like GigaChat API."""
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a financial explainability assistant. "
                        "Return only valid JSON without markdown fences."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": 0.2,
        }

    def _extract_json_payload(self, raw_response: dict[str, Any]) -> dict[str, Any]:
        """Извлекает narrative JSON из ответа API."""
        if "summary" in raw_response:
            return raw_response

        choices = raw_response.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            content = message.get("content")
            if isinstance(content, str):
                return json.loads(content)

        raise ValueError("Unable to extract narrative JSON from GigaChat response.")

    def generate_narrative(self, prompt: str) -> dict[str, Any]:
        """Выполняет HTTP-вызов к GigaChat и возвращает narrative JSON."""
        if not self.api_url:
            raise ValueError("GigaChat API URL is not configured.")

        request = Request(
            self.api_url,
            data=json.dumps(self.build_request_body(prompt)).encode("utf-8"),
            headers=self.build_headers(),
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw_response = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(f"GigaChat HTTP error: {exc.code}") from exc
        except URLError as exc:
            raise RuntimeError(f"GigaChat connection error: {exc.reason}") from exc

        return self._extract_json_payload(raw_response)


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


def build_gigachat_prompt(payload: dict[str, object]) -> str:
    """Строит structured prompt для GigaChat на основе attribution payload."""
    components = payload["components"]
    assert isinstance(components, dict)
    top_drivers = _pick_top_drivers(components)
    return (
        "Сформируй JSON-объект с полями "
        "`position_id`, `summary`, `detailed_explanation`, `top_drivers`, "
        "`residual_comment`, `validation_status`, `fallback_used`.\n"
        "Не выдумывай новые числа и используй только переданные данные.\n"
        f"position_id={payload['position_id']}\n"
        f"instrument_type={payload['instrument_type']}\n"
        f"currency={payload['currency']}\n"
        f"total_pnl={float(payload['total_pnl']):.6f}\n"
        f"components={json.dumps(components, ensure_ascii=False)}\n"
        f"top_drivers={json.dumps(top_drivers, ensure_ascii=False)}\n"
        f"residual_threshold_passed={payload['residual_threshold_passed']}\n"
        "Поле `fallback_used` должно быть false."
    )


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


def validate_narrative_output(
    narrative: dict[str, Any],
    payload: dict[str, object],
) -> dict[str, Any]:
    """Проверяет минимальную консистентность narrative output."""
    missing = [key for key in REQUIRED_NARRATIVE_KEYS if key not in narrative]
    if missing:
        raise ValueError(f"Narrative output missing keys: {', '.join(missing)}")

    if narrative["position_id"] != payload["position_id"]:
        raise ValueError("Narrative output position_id mismatch.")

    top_drivers = narrative.get("top_drivers", [])
    if not isinstance(top_drivers, list):
        raise ValueError("Narrative top_drivers must be a list.")

    allowed_names = {
        item["name"]
        for item in _pick_top_drivers(payload["components"])  # type: ignore[arg-type]
    }
    for item in top_drivers:
        if not isinstance(item, dict) or item.get("name") not in allowed_names:
            raise ValueError("Narrative top_drivers contain unsupported driver names.")

    expected_status = "passed" if payload["residual_threshold_passed"] else "warning"
    if narrative["validation_status"] != expected_status:
        raise ValueError("Narrative validation_status is inconsistent with residual check.")

    return narrative


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


def generate_gigachat_narrative(
    state: IPVState,
    *,
    client: GigaChatClient | None = None,
) -> dict[str, Any]:
    """Строит narrative через GigaChat и валидирует ответ."""
    payload = build_template_narrative_payload(state)
    prompt = build_gigachat_prompt(payload)
    gigachat_client = client or GigaChatClient()
    if not gigachat_client.is_configured():
        raise RuntimeError("GigaChat client is not configured.")

    narrative = gigachat_client.generate_narrative(prompt)
    narrative["fallback_used"] = False
    return validate_narrative_output(narrative, payload)


def run_narrative_agent(state: IPVState) -> IPVState:
    """Заполняет state полем narrative_result."""
    client = GigaChatClient()
    if client.is_configured():
        narrative_result = generate_gigachat_narrative(state, client=client)
        state["narrative_result"] = narrative_result
        state["fallback_flags"]["used_template_narrative"] = False
    else:
        narrative_result = generate_template_narrative(state)
        state["narrative_result"] = narrative_result
        state["fallback_flags"]["used_template_narrative"] = True
    return state
