"""Простой клиент для работы с MOEX ISS API."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen


MOEX_ISS_BASE_URL = "https://iss.moex.com/iss"


@dataclass(slots=True)
class MoexClient:
    """Минимальный клиент для запросов к MOEX ISS API."""

    base_url: str = MOEX_ISS_BASE_URL
    timeout_seconds: float = 10.0

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Выполняет GET-запрос к MOEX и возвращает JSON-ответ."""
        query = urlencode(params or {})
        url = f"{self.base_url}/{path}.json"
        if query:
            url = f"{url}?{query}"

        with urlopen(url, timeout=self.timeout_seconds) as response:
            return json.load(response)


def fetch_spot_prices(
    as_of_date: str,
    securities: list[str] | None = None,
    board: str = "TQBR",
    engine: str = "stock",
    market: str = "shares",
    client: MoexClient | None = None,
) -> dict[str, Any]:
    """Получает сырые спотовые цены акций на выбранную дату."""
    moex = client or MoexClient()
    security_filter = ",".join(securities or ["SBER"])

    return moex.get_json(
        f"history/engines/{engine}/markets/{market}/boards/{board}/securities",
        params={
            "date": as_of_date,
            "securities": security_filter,
        },
    )


def fetch_price_history(
    security: str,
    date_from: str,
    date_to: str,
    board: str = "TQBR",
    engine: str = "stock",
    market: str = "shares",
    client: MoexClient | None = None,
) -> dict[str, Any]:
    """Получает сырую историю котировок по инструменту за диапазон дат."""
    moex = client or MoexClient()
    return moex.get_json(
        f"history/engines/{engine}/markets/{market}/boards/{board}/securities/{security}",
        params={
            "from": date_from,
            "till": date_to,
        },
    )


def fetch_yield_curve(
    as_of_date: str,
    currency: str = "RUB",
    client: MoexClient | None = None,
) -> dict[str, Any]:
    """Загружает сырые точки кривой ставок для выбранной даты."""
    moex = client or MoexClient()

    return moex.get_json(
        "engines/stock/zcyc",
        params={
            "date": as_of_date,
            "currency": currency,
        },
    )


def fetch_option_quotes(
    as_of_date: str,
    underlier: str,
    board: str = "ROPD",
    engine: str = "futures",
    market: str = "options",
    client: MoexClient | None = None,
) -> dict[str, Any]:
    """Загружает сырые котировки опционов для построения поверхности волатильности."""
    moex = client or MoexClient()

    return moex.get_json(
        f"history/engines/{engine}/markets/{market}/boards/{board}/securities",
        params={
            "date": as_of_date,
            "q": underlier,
        },
    )
