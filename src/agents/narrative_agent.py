"""Narrative generation with template rules and GigaChat integration."""

from __future__ import annotations

import json
import os
import ssl
from typing import Any
from uuid import uuid4
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import certifi

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
        access_token: str | None = None,
        auth_url: str | None = None,
        model: str | None = None,
        scope: str | None = None,
        timeout_seconds: int = 30,
    ) -> None:
        self.api_url = api_url or os.getenv("GIGACHAT_API_URL")
        self.api_key = api_key or os.getenv("GIGACHAT_API_KEY") or os.getenv("GIGACHAT_AUTH_KEY")
        self.access_token = access_token or os.getenv("GIGACHAT_ACCESS_TOKEN")
        self.auth_url = auth_url or os.getenv(
            "GIGACHAT_AUTH_URL",
            "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
        )
        self.model = model or os.getenv("GIGACHAT_MODEL", "GigaChat")
        self.scope = scope or os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
        self.timeout_seconds = timeout_seconds

    def is_configured(self) -> bool:
        """Проверяет, что для GigaChat заданы обязательные параметры."""
        return bool(self.api_url and (self.access_token or self.api_key))

    def build_headers(self) -> dict[str, str]:
        """Строит HTTP headers для вызова GigaChat."""
        token = self.get_access_token()
        return {
            "Authorization": f"Bearer {token}",
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
                        "Ты — модуль финансовой объяснимости в системе IPV Explain Agent. "
                        "Возвращай только валидный JSON без markdown и без текста вне JSON. "
                        "Пиши все текстовые поля строго на русском языке. "
                        "Не выдумывай числа, не меняй входные данные и не добавляй новые факторы."
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

    def build_ssl_context(self) -> ssl.SSLContext:
        """Строит SSL context для HTTPS-вызова.

        По умолчанию используется certifi bundle.
        Для локальной отладки можно отключить проверку через
        `GIGACHAT_VERIFY_SSL=false`.
        """
        verify_ssl = os.getenv("GIGACHAT_VERIFY_SSL", "true").lower() != "false"
        if not verify_ssl:
            return ssl._create_unverified_context()
        return ssl.create_default_context(cafile=certifi.where())

    def get_access_token(self) -> str:
        """Возвращает access token для chat/completions.

        Если он не задан напрямую, получает его через OAuth endpoint.
        """
        if self.access_token:
            return self.access_token
        return self.request_access_token()

    def request_access_token(self) -> str:
        """Обменивает authorization key на access token."""
        if not self.auth_url:
            raise ValueError("GigaChat auth URL is not configured.")
        if not self.api_key:
            raise ValueError("GigaChat authorization key is not configured.")

        request = Request(
            self.auth_url,
            data=f"scope={self.scope}".encode("utf-8"),
            headers={
                "Authorization": f"Basic {self.api_key}",
                "RqUID": str(uuid4()),
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(
                request,
                timeout=self.timeout_seconds,
                context=self.build_ssl_context(),
            ) as response:
                raw_response = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(f"GigaChat auth HTTP error: {exc.code}") from exc
        except URLError as exc:
            raise RuntimeError(f"GigaChat auth connection error: {exc.reason}") from exc

        access_token = raw_response.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise RuntimeError("GigaChat auth response did not contain access_token.")
        self.access_token = access_token
        return access_token

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
            with urlopen(
                request,
                timeout=self.timeout_seconds,
                context=self.build_ssl_context(),
            ) as response:
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
    expected_status = "passed" if payload["residual_threshold_passed"] else "warning"
    return (
        "Сформируй JSON по правилам ниже.\n"
        "Правила:\n"
        "1. Используй только входные данные.\n"
        "2. Не добавляй новые факторы.\n"
        "3. Не оставляй пустые строки.\n"
        "4. Не меняй знак чисел и не округляй их до потери смысла.\n"
        "5. Обязательно используй числовые значения из входных данных в summary, detailed_explanation и residual_comment.\n"
        "   Упоминай total_pnl, значения главных факторов и остаток.\n"
        "6. Пиши все текстовые поля строго на русском языке.\n"
        "7. Не используй англоязычные термины, если есть нормальный русский эквивалент.\n"
        "8. Используй формулировки:\n"
        "   - «изменение стоимости» вместо «PnL»\n"
        "   - «остаток» вместо «Residual»\n"
        "   - «основные факторы» вместо «top drivers»\n"
        "   - «вклад» вместо «effect», когда речь о тексте\n"
        f"9. Если residual_threshold_passed=true, то validation_status={expected_status}.\n"
        f"10. Если residual_threshold_passed=false, то validation_status={expected_status}.\n"
        "11. fallback_used=false.\n"
        "12. top_drivers верни в том же виде, как они переданы во входе.\n"
        "13. summary — ровно 3 полных предложения:\n"
        "   - предложение 1: итоговое изменение стоимости с числом total_pnl и валютой\n"
        "   - предложение 2: главный фактор с его числом\n"
        "   - предложение 3: второй по значимости фактор с его числом\n"
        "14. detailed_explanation — ровно 5 полных предложений:\n"
        "   - предложение 1: общее изменение стоимости с числом total_pnl\n"
        "   - предложение 2: вклад самого сильного фактора с числом\n"
        "   - предложение 3: вклад второго фактора с числом\n"
        "   - предложение 4: вклад остальных заметных факторов с числами\n"
        "   - предложение 5: отдельный вывод про остаток с числом\n"
        "15. residual_comment — ровно 1 полное предложение про остаток с числом.\n"
        "Формат ответа:\n"
        "{\n"
        '  "position_id": "string",\n'
        '  "summary": "non-empty string",\n'
        '  "detailed_explanation": "non-empty string",\n'
        '  "top_drivers": [{"name": "delta_effect", "value": 0.0}],\n'
        '  "residual_comment": "non-empty string",\n'
        f'  "validation_status": "{expected_status}",\n'
        '  "fallback_used": false\n'
        "}\n"
        "Данные для ответа:\n"
        f"position_id={payload['position_id']}\n"
        f"instrument_type={payload['instrument_type']}\n"
        f"currency={payload['currency']}\n"
        f"total_pnl={float(payload['total_pnl']):.6f}\n"
        f"components={json.dumps(components, ensure_ascii=False)}\n"
        f"top_drivers={json.dumps(top_drivers, ensure_ascii=False)}\n"
        f"residual_threshold_passed={payload['residual_threshold_passed']}\n"
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


def _fallback_to_template(state: IPVState, error: Exception | None = None) -> IPVState:
    """Переключает narrative layer на шаблонный fallback."""
    if error is not None:
        state["errors"].append(f"narrative_fallback: {error}")
    narrative_result = generate_template_narrative(state)
    state["narrative_result"] = narrative_result
    state["fallback_flags"]["used_template_narrative"] = True
    return state


def run_narrative_agent(state: IPVState) -> IPVState:
    """Заполняет state полем narrative_result."""
    client = GigaChatClient()
    if client.is_configured():
        try:
            narrative_result = generate_gigachat_narrative(state, client=client)
            state["narrative_result"] = narrative_result
            state["fallback_flags"]["used_template_narrative"] = False
            return state
        except Exception as exc:
            return _fallback_to_template(state, error=exc)
    return _fallback_to_template(state)
