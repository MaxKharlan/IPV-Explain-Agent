"""FastAPI entrypoint for the IPV Explain Agent service."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from src.agents.orchestrator import run_pipeline
from src.api.schemas import NARRATIVE_OUTPUT_EXAMPLE, POSITION_INPUT_EXAMPLE


app = FastAPI(
    title="IPV Explain Agent API",
    version="0.1.0",
    description="Service entrypoint for running the IPV Explain Agent pipeline.",
)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    """Returns a simple liveness payload."""
    return {"status": "ok"}


@app.post("/pipeline/run")
def run_pipeline_endpoint(position: dict[str, Any]) -> dict[str, Any]:
    """Runs the current orchestration pipeline and returns a public API payload."""
    try:
        state = run_pipeline(position)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "message": str(exc),
                "position_id": position.get("position_id"),
            },
        ) from exc

    report_result = state.get("report_result")
    narrative_result = state.get("narrative_result")
    return {
        "status": "completed",
        "position_id": position.get("position_id"),
        "report_result": report_result,
        "narrative_result": narrative_result,
        "fallback_flags": state.get("fallback_flags", {}),
        "errors": state.get("errors", []),
    }


PIPELINE_RUN_EXAMPLE = {
    "request_example": POSITION_INPUT_EXAMPLE,
    "response_example": {
        "status": "completed",
        "position_id": POSITION_INPUT_EXAMPLE["position_id"],
        "report_result": {
            "report_summary": {
                "position_id": POSITION_INPUT_EXAMPLE["position_id"],
                "instrument_type": POSITION_INPUT_EXAMPLE["instrument_type"],
                "total_pnl": 1.45,
                "summary": NARRATIVE_OUTPUT_EXAMPLE["summary"],
                "used_mock_market_data": False,
                "used_template_narrative": False,
            }
        },
        "narrative_result": NARRATIVE_OUTPUT_EXAMPLE,
        "fallback_flags": {
            "used_mock_market_data": False,
            "used_template_narrative": False,
        },
        "errors": [],
    },
}
