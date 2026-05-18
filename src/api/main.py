"""FastAPI entrypoint for the IPV Explain Agent service."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from src.api.schemas import POSITION_INPUT_EXAMPLE


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
def run_pipeline_stub(position: dict[str, Any]) -> dict[str, Any]:
    """Placeholder pipeline endpoint.

    The full orchestrator wiring is added in the next API step. For now the
    endpoint confirms the input shape and returns a stub status.
    """
    return {
        "status": "accepted",
        "message": "Pipeline API entrypoint is ready for orchestrator wiring.",
        "position_id": position.get("position_id"),
    }


PIPELINE_RUN_EXAMPLE = {
    "request_example": POSITION_INPUT_EXAMPLE,
    "response_example": {
        "status": "accepted",
        "message": "Pipeline API entrypoint is ready for orchestrator wiring.",
        "position_id": POSITION_INPUT_EXAMPLE["position_id"],
    },
}
