"""Attribution Agent scaffold for pipeline orchestration."""

from __future__ import annotations

from datetime import date

from src.agents.state import IPVState
from src.agents.pricing_agent import resolve_option_sigmas
from src.attribution import Position, PricingResultMin, RiskFactorSnapshot, run_attribution


def build_risk_factor_inputs(state: IPVState) -> dict[str, object]:
    """Собирает заготовку risk-factor входов из state для Attribution Engine."""
    return {
        "position": state.get("position"),
        "pricing_result": state.get("pricing_result"),
        "market_snapshot_t0": state.get("market_snapshot_t0"),
        "market_snapshot_t1": state.get("market_snapshot_t1"),
    }


def _parse_snapshot_date(snapshot: dict[str, object]) -> date:
    """Преобразует snapshot_date в date."""
    return date.fromisoformat(str(snapshot["snapshot_date"]))


def _normalize_rate(rate: float) -> float:
    """Приводит ставку к decimal rate format."""
    if abs(rate) > 1.0:
        return rate / 100.0
    return rate


def _extract_reference_rate(snapshot: dict[str, object]) -> float | None:
    """Берёт опорную ставку из первой точки yield curve."""
    yield_curve = snapshot.get("yield_curve")
    if not isinstance(yield_curve, dict):
        return None
    points = yield_curve.get("points", [])
    if not points:
        return None
    first = points[0]
    if not isinstance(first, dict) or "rate" not in first:
        return None
    return _normalize_rate(float(first["rate"]))


def _build_position_schema(position: dict[str, object]) -> Position:
    """Преобразует orchestration position payload в Attribution Position schema."""
    instrument = position.get("instrument", {})
    underlier = instrument.get("underlier") if isinstance(instrument, dict) else None
    return Position(
        position_id=str(position["position_id"]),
        instrument_type=str(position["instrument_type"]),
        quantity=float(position.get("quantity", 1.0)),
        currency=str(position.get("currency", "RUB")),
        underlier=underlier,
        book=position.get("book"),
    )


def _build_risk_factor_snapshots(
    position: dict[str, object],
    market_t0: dict[str, object],
    market_t1: dict[str, object],
) -> tuple[RiskFactorSnapshot, RiskFactorSnapshot]:
    """Преобразует market snapshot в risk-factor schemas для attribution."""
    instrument_type = str(position.get("instrument_type"))
    instrument = position.get("instrument", {})
    if not isinstance(instrument, dict):
        raise ValueError("Position.instrument must be a dictionary.")

    rate_t0 = _extract_reference_rate(market_t0)
    rate_t1 = _extract_reference_rate(market_t1)

    if instrument_type == "option":
        underlier = instrument.get("underlier")
        if underlier is None:
            raise ValueError("Position.instrument.underlier is required.")

        sigma0, sigma1 = resolve_option_sigmas(position)
        spot_t0 = float(market_t0["spot_prices"][underlier])
        spot_t1 = float(market_t1["spot_prices"][underlier])
        return (
            RiskFactorSnapshot(
                snapshot_date=_parse_snapshot_date(market_t0),
                spot=spot_t0,
                vol=sigma0,
                rate=rate_t0,
            ),
            RiskFactorSnapshot(
                snapshot_date=_parse_snapshot_date(market_t1),
                spot=spot_t1,
                vol=sigma1,
                rate=rate_t1,
            ),
        )

    # Rate-driven instruments currently use placeholder spot/vol while attribution
    # consumes their rho-based pricing sensitivity and the change in reference rate.
    return (
        RiskFactorSnapshot(
            snapshot_date=_parse_snapshot_date(market_t0),
            spot=1.0,
            vol=0.0,
            rate=rate_t0,
        ),
        RiskFactorSnapshot(
            snapshot_date=_parse_snapshot_date(market_t1),
            spot=1.0,
            vol=0.0,
            rate=rate_t1,
        ),
    )


def run_attribution_agent(state: IPVState) -> IPVState:
    """Заполняет state полем attribution_result."""
    position_payload = state.get("position")
    pricing_payload = state.get("pricing_result")
    market_t0 = state.get("market_snapshot_t0")
    market_t1 = state.get("market_snapshot_t1")
    if (
        position_payload is None
        or pricing_payload is None
        or market_t0 is None
        or market_t1 is None
    ):
        raise ValueError(
            "Attribution agent requires position, pricing_result, and both market snapshots."
        )

    position = _build_position_schema(position_payload)
    pricing = PricingResultMin.model_validate(pricing_payload)
    rf_t0, rf_t1 = _build_risk_factor_snapshots(position_payload, market_t0, market_t1)
    attribution_result = run_attribution(position, pricing, rf_t0, rf_t1)
    state["attribution_result"] = attribution_result.model_dump(mode="json")
    return state
