"""Streamlit dashboard for running the IPV Explain Agent pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.orchestrator import run_pipeline
from src.api.schemas import POSITION_INPUT_EXAMPLE


def get_default_position_payload() -> dict[str, Any]:
    """Returns the default demo payload shown in the dashboard."""
    return POSITION_INPUT_EXAMPLE


def format_components_for_table(components: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Converts attribution components into a simple table-friendly structure."""
    if not components:
        return []
    return [
        {"component": name, "value": value}
        for name, value in components.items()
    ]


def render_pipeline_result(st, state: dict[str, Any]) -> None:
    """Renders the main pipeline artifacts in the dashboard."""
    narrative = state.get("narrative_result") or {}
    attribution = state.get("attribution_result") or {}
    report = state.get("report_result") or {}
    fallback_flags = state.get("fallback_flags") or {}
    errors = state.get("errors") or []

    st.subheader("Narrative")
    st.markdown(f"**Summary:** {narrative.get('summary', 'n/a')}")
    st.markdown(f"**Details:** {narrative.get('detailed_explanation', 'n/a')}")
    st.markdown(f"**Residual:** {narrative.get('residual_comment', 'n/a')}")
    st.markdown(f"**Validation:** `{narrative.get('validation_status', 'n/a')}`")

    st.subheader("Attribution")
    st.metric("Total PnL", attribution.get("total_pnl", "n/a"))
    component_rows = format_components_for_table(attribution.get("components"))
    if component_rows:
        st.table(component_rows)

    st.subheader("Flags")
    st.json(
        {
            "fallback_flags": fallback_flags,
            "errors": errors,
        }
    )

    st.subheader("Report Payload")
    st.json(report)


def main() -> None:
    """Launches the Streamlit dashboard."""
    import streamlit as st

    st.set_page_config(page_title="IPV Explain Agent", layout="wide")
    st.title("IPV Explain Agent")
    st.caption("Demo dashboard for the pipeline: market data -> pricing -> attribution -> narrative.")

    default_payload = json.dumps(get_default_position_payload(), ensure_ascii=False, indent=2)
    payload_text = st.text_area(
        "PositionInput JSON",
        value=default_payload,
        height=320,
    )

    run_clicked = st.button("Run Pipeline", type="primary")

    if run_clicked:
        try:
            position_payload = json.loads(payload_text)
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON: {exc}")
            return

        with st.spinner("Running IPV pipeline..."):
            try:
                state = run_pipeline(position_payload)
            except Exception as exc:
                st.error(f"Pipeline failed: {exc}")
                return

        st.success("Pipeline completed.")
        render_pipeline_result(st, state)


if __name__ == "__main__":
    main()
