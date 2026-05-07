"""Клиент MOEX ISS API и нормализация рыночных данных."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen


MOEX_ISS_BASE_URL = "https://iss.moex.com/iss"

UNDERLIER_ALIASES: dict[str, set[str]] = {
    "SBER": {"SBER", "SBRF"},
}


@dataclass(slots=True)
class RawMoexResponse:
    """Сырой ответ MOEX ISS API до нормализации."""

    endpoint: str
    params: dict[str, Any]
    payload: dict[str, Any]


@dataclass(slots=True)
class NormalizedQuote:
    """Нормализованная котировка базового актива."""

    symbol: str
    snapshot_date: str
    price: float
    currency: str
    board: str
    source: str = "moex"


@dataclass(slots=True)
class CurvePoint:
    """Одна точка кривой ставок."""

    tenor: str
    rate: float


@dataclass(slots=True)
class YieldCurveSnapshot:
    """Нормализованная кривая ставок на дату."""

    snapshot_date: str
    currency: str
    points: list[CurvePoint] = field(default_factory=list)
    source: str = "moex"


@dataclass(slots=True)
class OptionQuote:
    """Нормализованная котировка опциона."""

    underlier: str
    option_type: str
    strike: float
    expiry: str
    settlement_price: float
    instrument_id: str
    source: str = "moex"


@dataclass(slots=True)
class OptionQuoteSnapshot:
    """Нормализованный набор опционных котировок по базовому активу."""

    snapshot_date: str
    underlier: str
    points: list[OptionQuote] = field(default_factory=list)
    source: str = "moex"


@dataclass(slots=True)
class MarketSnapshot:
    """Итоговый нормализованный снимок рынка для quant и attribution слоёв."""

    snapshot_id: str
    snapshot_date: str
    source: str
    spot_prices: dict[str, float]
    yield_curve: YieldCurveSnapshot
    option_quotes: dict[str, Any] | None = None
    quality_flags: dict[str, bool] = field(default_factory=dict)


MOCK_PRICE_HISTORY: dict[str, list[tuple[str, float]]] = {
    "SBER": [
        ("2025-04-28", 314.23),
        ("2025-04-29", 311.45),
        ("2025-04-30", 308.90),
        ("2025-05-02", 299.80),
        ("2025-05-05", 294.64),
    ],
}

MOCK_YIELD_CURVE_POINTS: list[tuple[str, float]] = [
    ("1Y", 16.5),
    ("2Y", 16.8),
    ("5Y", 17.1),
]

MOCK_OPTION_QUOTES: dict[str, list[dict[str, Any]]] = {
    "SBER": [
        {
            "option_type": "call",
            "strike": 280.0,
            "expiry": "2026-09-20",
            "settlement_price": 12.4,
            "instrument_id": "SBERC280",
        },
        {
            "option_type": "put",
            "strike": 290.0,
            "expiry": "2026-09-20",
            "settlement_price": 8.1,
            "instrument_id": "SBERP290",
        },
    ]
}


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


def _rows_from_block(payload: dict[str, Any], block_name: str) -> list[dict[str, Any]]:
    """Преобразует табличный блок MOEX в список словарей."""
    block = payload.get(block_name, {})
    columns = block.get("columns", [])
    data = block.get("data", [])
    return [dict(zip(columns, row, strict=False)) for row in data]


def _to_float(value: Any, default: float = 0.0) -> float:
    """Приводит числовые значения к float."""
    if value in (None, "", "nan"):
        return default
    return float(value)


def normalize_raw_response(
    endpoint: str,
    params: dict[str, Any],
    payload: dict[str, Any],
) -> RawMoexResponse:
    """Упаковывает сырой ответ MOEX в типизированную структуру."""
    return RawMoexResponse(endpoint=endpoint, params=params, payload=payload)


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
        params={"date": as_of_date, "securities": security_filter},
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
        params={"from": date_from, "till": date_to},
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
        params={"date": as_of_date, "currency": currency},
    )


def fetch_option_quotes(
    as_of_date: str,
    underlier: str,
    board: str = "ROPD",
    engine: str = "futures",
    market: str = "options",
    client: MoexClient | None = None,
) -> dict[str, Any]:
    """Загружает live option chain для выбранного базового актива."""
    moex = client or MoexClient()
    payload = moex.get_json(
        f"engines/{engine}/markets/{market}/boards/{board}/securities",
        params={"iss.only": "securities"},
    )
    return payload


def normalize_spot_quotes(
    payload: dict[str, Any],
    *,
    currency: str = "RUB",
    board: str = "TQBR",
    source: str = "moex",
) -> list[NormalizedQuote]:
    """Нормализует историю/спот котировок акций из ответа MOEX."""
    rows = _rows_from_block(payload, "history")
    quotes: list[NormalizedQuote] = []
    for row in rows:
        symbol = row.get("SECID")
        snapshot_date = row.get("TRADEDATE")
        price = row.get("CLOSE") or row.get("LEGALCLOSEPRICE") or row.get("MARKETPRICE2")
        if not symbol or not snapshot_date or price in (None, ""):
            continue
        quotes.append(
            NormalizedQuote(
                symbol=str(symbol),
                snapshot_date=str(snapshot_date),
                price=_to_float(price),
                currency=currency,
                board=board,
                source=source,
            )
        )
    return quotes


def normalize_yield_curve(
    payload: dict[str, Any],
    *,
    snapshot_date: str,
    currency: str = "RUB",
    source: str = "moex",
) -> YieldCurveSnapshot:
    """Нормализует ответ MOEX по кривой ставок."""
    rows = _rows_from_block(payload, "yearyields")
    points: list[CurvePoint] = []
    for row in rows:
        years = row.get("period") or row.get("YEAR") or row.get("years")
        rate = row.get("value") or row.get("YIELD") or row.get("yield")
        if years in (None, "") or rate in (None, ""):
            continue
        points.append(CurvePoint(tenor=f"{_to_float(years):g}Y", rate=_to_float(rate)))
    return YieldCurveSnapshot(
        snapshot_date=snapshot_date,
        currency=currency,
        points=points,
        source=source,
    )


def normalize_option_quotes(
    payload: dict[str, Any],
    *,
    snapshot_date: str,
    underlier: str,
    source: str = "moex",
) -> OptionQuoteSnapshot:
    """Нормализует live option chain в option_quotes для quant-слоя."""
    rows = _rows_from_block(payload, "securities")
    points: list[OptionQuote] = []
    allowed_underliers = UNDERLIER_ALIASES.get(underlier, {underlier})
    for row in rows:
        asset_code = row.get("ASSETCODE")
        if asset_code not in allowed_underliers:
            continue
        strike = row.get("STRIKE")
        option_type = row.get("OPTIONTYPE")
        expiry = row.get("LASTTRADEDATE")
        settlement_price = row.get("PREVSETTLEPRICE")
        instrument_id = row.get("SECID")
        if (
            strike in (None, "")
            or option_type in (None, "")
            or expiry in (None, "")
            or settlement_price in (None, "")
            or instrument_id in (None, "")
        ):
            continue
        points.append(
            OptionQuote(
                underlier=underlier,
                option_type="call" if str(option_type).upper().startswith("C") else "put",
                strike=_to_float(strike),
                expiry=str(expiry),
                settlement_price=_to_float(settlement_price),
                instrument_id=str(instrument_id),
                source=source,
            )
        )
    return OptionQuoteSnapshot(
        snapshot_date=snapshot_date,
        underlier=underlier,
        points=points,
        source=source,
    )


def option_quotes_to_dict(snapshot: OptionQuoteSnapshot) -> dict[str, Any]:
    """Преобразует option quotes в JSON-friendly словарь."""
    return {
        "snapshot_date": snapshot.snapshot_date,
        "underlier": snapshot.underlier,
        "points": [
            {
                "option_type": point.option_type,
                "strike": point.strike,
                "expiry": point.expiry,
                "settlement_price": point.settlement_price,
                "instrument_id": point.instrument_id,
            }
            for point in snapshot.points
        ],
        "source": snapshot.source,
    }


def load_price_history(
    security: str,
    date_from: str,
    date_to: str,
    *,
    currency: str = "RUB",
    board: str = "TQBR",
    engine: str = "stock",
    market: str = "shares",
    source: str = "moex",
    client: MoexClient | None = None,
) -> list[NormalizedQuote]:
    """Загружает и сразу нормализует историю котировок за диапазон дат."""
    payload = fetch_price_history(
        security=security,
        date_from=date_from,
        date_to=date_to,
        board=board,
        engine=engine,
        market=market,
        client=client,
    )
    return normalize_spot_quotes(payload, currency=currency, board=board, source=source)


def get_quote_for_date(quotes: list[NormalizedQuote], snapshot_date: str) -> NormalizedQuote | None:
    """Возвращает котировку на конкретную дату из нормализованной истории."""
    for quote in quotes:
        if quote.snapshot_date == snapshot_date:
            return quote
    return None


def load_mock_price_history(
    security: str,
    *,
    currency: str = "RUB",
    board: str = "TQBR",
    source: str = "mock",
) -> list[NormalizedQuote]:
    """Возвращает mock-историю котировок для локальной разработки и демо."""
    fallback_security = security if security in MOCK_PRICE_HISTORY else "SBER"
    rows = MOCK_PRICE_HISTORY.get(fallback_security, [])
    return [
        NormalizedQuote(
            symbol=fallback_security,
            snapshot_date=snapshot_date,
            price=price,
            currency=currency,
            board=board,
            source=source,
        )
        for snapshot_date, price in rows
    ]


def load_mock_option_quotes(
    snapshot_date: str,
    *,
    underlier: str = "SBER",
    source: str = "mock",
) -> dict[str, Any]:
    """Возвращает mock option quotes для локальной разработки и демо."""
    fallback_underlier = underlier if underlier in MOCK_OPTION_QUOTES else "SBER"
    snapshot = OptionQuoteSnapshot(
        snapshot_date=snapshot_date,
        underlier=fallback_underlier,
        points=[
            OptionQuote(
                underlier=fallback_underlier,
                option_type=item["option_type"],
                strike=item["strike"],
                expiry=item["expiry"],
                settlement_price=item["settlement_price"],
                instrument_id=item["instrument_id"],
                source=source,
            )
            for item in MOCK_OPTION_QUOTES[fallback_underlier]
        ],
        source=source,
    )
    return option_quotes_to_dict(snapshot)


def safe_load_price_history(
    security: str,
    date_from: str,
    date_to: str,
    *,
    currency: str = "RUB",
    board: str = "TQBR",
    engine: str = "stock",
    market: str = "shares",
    client: MoexClient | None = None,
) -> tuple[list[NormalizedQuote], bool]:
    """Пытается загрузить историю из MOEX, а при ошибке возвращает mock-данные."""
    try:
        quotes = load_price_history(
            security=security,
            date_from=date_from,
            date_to=date_to,
            currency=currency,
            board=board,
            engine=engine,
            market=market,
            source="moex",
            client=client,
        )
        if not quotes:
            raise ValueError("MOEX returned empty price history")
        return quotes, False
    except Exception:
        return load_mock_price_history(security, currency=currency, board=board), True


def load_mock_yield_curve(
    snapshot_date: str,
    *,
    currency: str = "RUB",
    source: str = "mock",
) -> YieldCurveSnapshot:
    """Возвращает mock-кривую ставок для локальной разработки и демо."""
    return YieldCurveSnapshot(
        snapshot_date=snapshot_date,
        currency=currency,
        points=[CurvePoint(tenor=tenor, rate=rate) for tenor, rate in MOCK_YIELD_CURVE_POINTS],
        source=source,
    )


def safe_load_yield_curve(
    snapshot_date: str,
    *,
    currency: str = "RUB",
    client: MoexClient | None = None,
) -> tuple[YieldCurveSnapshot, bool]:
    """Пытается загрузить кривую из MOEX, а при ошибке возвращает mock-данные."""
    try:
        payload = fetch_yield_curve(snapshot_date, currency=currency, client=client)
        curve = normalize_yield_curve(
            payload,
            snapshot_date=snapshot_date,
            currency=currency,
            source="moex",
        )
        if not curve.points:
            raise ValueError("MOEX returned empty yield curve")
        return curve, False
    except Exception:
        return load_mock_yield_curve(snapshot_date, currency=currency), True


def safe_load_option_quotes(
    snapshot_date: str,
    *,
    underlier: str = "SBER",
    client: MoexClient | None = None,
) -> tuple[dict[str, Any], bool]:
    """Пытается загрузить option quotes, а при ошибке возвращает mock-данные."""
    try:
        payload = fetch_option_quotes(snapshot_date, underlier=underlier, client=client)
        option_quotes = normalize_option_quotes(
            payload,
            snapshot_date=snapshot_date,
            underlier=underlier,
            source="moex",
        )
        if not option_quotes.points:
            raise ValueError("MOEX returned no option quotes for underlier")
        return option_quotes_to_dict(option_quotes), False
    except Exception:
        return load_mock_option_quotes(snapshot_date, underlier=underlier), True


def build_market_snapshot(
    *,
    snapshot_date: str,
    spot_quotes: list[NormalizedQuote],
    yield_curve: YieldCurveSnapshot,
    option_quotes: dict[str, Any] | None = None,
    source: str = "moex",
    used_mock_data: bool = False,
    used_mock_option_quotes: bool = False,
) -> MarketSnapshot:
    """Собирает итоговый MarketSnapshot из нормализованных частей."""
    return MarketSnapshot(
        snapshot_id=f"SNAP-{snapshot_date}",
        snapshot_date=snapshot_date,
        source=source,
        spot_prices={quote.symbol: quote.price for quote in spot_quotes},
        yield_curve=yield_curve,
        option_quotes=option_quotes,
        quality_flags={
            "used_mock_data": used_mock_data,
            "missing_curve_points": len(yield_curve.points) == 0,
            "used_mock_option_quotes": used_mock_option_quotes,
        },
    )


def load_mock_market_snapshot(
    snapshot_date: str,
    *,
    security: str = "SBER",
    currency: str = "RUB",
) -> MarketSnapshot:
    """Собирает полностью mock MarketSnapshot."""
    spot_quotes = load_mock_price_history(security, currency=currency)
    quote = get_quote_for_date(spot_quotes, snapshot_date)
    selected_quotes = [quote] if quote is not None else []
    yield_curve = load_mock_yield_curve(snapshot_date, currency=currency)
    option_quotes = load_mock_option_quotes(snapshot_date, underlier=security)
    return build_market_snapshot(
        snapshot_date=snapshot_date,
        spot_quotes=selected_quotes,
        yield_curve=yield_curve,
        option_quotes=option_quotes,
        source="mock",
        used_mock_data=True,
        used_mock_option_quotes=True,
    )


def load_market_snapshot(
    snapshot_date: str,
    *,
    security: str = "SBER",
    currency: str = "RUB",
    board: str = "TQBR",
    include_option_quotes: bool = True,
    client: MoexClient | None = None,
) -> MarketSnapshot:
    """Возвращает готовый MarketSnapshot на одну дату с live/mock fallback."""
    spot_history, used_mock_prices = safe_load_price_history(
        security=security,
        date_from=snapshot_date,
        date_to=snapshot_date,
        currency=currency,
        board=board,
        client=client,
    )
    selected_quote = get_quote_for_date(spot_history, snapshot_date)
    selected_quotes = [selected_quote] if selected_quote is not None else []
    if not selected_quotes and used_mock_prices:
        selected_quote = get_quote_for_date(
            load_mock_price_history(security, currency=currency, board=board),
            snapshot_date,
        )
        selected_quotes = [selected_quote] if selected_quote is not None else []

    yield_curve, used_mock_curve = safe_load_yield_curve(
        snapshot_date,
        currency=currency,
        client=client,
    )

    option_quotes: dict[str, Any] | None = None
    used_mock_option_quotes = False
    if include_option_quotes:
        option_quotes, used_mock_option_quotes = safe_load_option_quotes(
            snapshot_date,
            underlier=security,
            client=client,
        )

    used_any_mock = used_mock_prices or used_mock_curve or used_mock_option_quotes
    return build_market_snapshot(
        snapshot_date=snapshot_date,
        spot_quotes=selected_quotes,
        yield_curve=yield_curve,
        option_quotes=option_quotes,
        source="mock" if used_any_mock else "moex",
        used_mock_data=used_any_mock,
        used_mock_option_quotes=used_mock_option_quotes,
    )


def load_market_snapshots_for_period(
    t0: str,
    t1: str,
    *,
    security: str = "SBER",
    currency: str = "RUB",
    board: str = "TQBR",
    include_option_quotes: bool = True,
    client: MoexClient | None = None,
) -> tuple[MarketSnapshot, MarketSnapshot]:
    """Возвращает два снапшота рынка для сравнения дат t0 и t1."""
    snapshot_t0 = load_market_snapshot(
        t0,
        security=security,
        currency=currency,
        board=board,
        include_option_quotes=include_option_quotes,
        client=client,
    )
    snapshot_t1 = load_market_snapshot(
        t1,
        security=security,
        currency=currency,
        board=board,
        include_option_quotes=include_option_quotes,
        client=client,
    )
    return snapshot_t0, snapshot_t1
