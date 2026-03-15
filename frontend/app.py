"""Streamlit frontend for Smart GIS Chatbot."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

try:
    import plotly.express as px
except ImportError:
    px = None


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

BACKEND_URL = "http://localhost:8000"
SUGGESTIONS = [
    "Find hospitals in Chennai",
    "Show police stations in Coimbatore",
    "Find schools near Madurai",
    "Where are accident hotspots in Tamil Nadu?",
    "What districts have the most infrastructure?",
]


def backend_health() -> bool:
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=(1, 2))
        return response.ok
    except requests.RequestException:
        return False


def ask_backend(question: str, session_id: str) -> dict[str, Any]:
    response = requests.post(
        f"{BACKEND_URL}/ask",
        json={"question": question, "session_id": session_id},
        timeout=(5, 120),
    )
    response.raise_for_status()
    return response.json()


def render_map(payload: dict[str, Any]) -> None:
    components.iframe(src=f"{BACKEND_URL}{payload['map_url']}", height=620, scrolling=True)


def render_metrics(metadata: dict[str, Any]) -> None:
    total_value = metadata.get("total_incidents", metadata.get("total_features", 0))
    district_counts = metadata.get("district_counts", metadata.get("district_density", []))
    cols = st.columns(4)
    cols[0].metric("Total Records", total_value)
    cols[1].metric("Clusters", metadata.get("cluster_count", 0))
    cols[2].metric("District Rows", len(district_counts))
    cols[3].metric("Top Locations", len(metadata.get("top_locations", [])))


def render_charts(metadata: dict[str, Any]) -> None:
    district_data = metadata.get("district_density", metadata.get("district_counts", []))
    district_counts = pd.DataFrame(district_data)
    chart_cols = st.columns(2)

    if not district_counts.empty:
        district_counts = district_counts.sort_values("record_count", ascending=False)
        if px is not None:
            bar_chart = px.bar(
                district_counts.head(10),
                x="district",
                y="record_count",
                title="District Distribution",
                color="record_count",
                color_continuous_scale="Tealgrn",
            )
            bar_chart.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=320)
            chart_cols[0].plotly_chart(bar_chart, use_container_width=True)
        else:
            chart_cols[0].bar_chart(district_counts.set_index("district")["record_count"])
    else:
        chart_cols[0].info("No district count data available.")

    top_locations = pd.DataFrame(metadata.get("top_locations", []))
    if not top_locations.empty and {"lat", "lon"}.issubset(top_locations.columns):
        if px is not None:
            scatter = px.scatter_geo(
                top_locations,
                lat="lat",
                lon="lon",
                hover_name="name" if "name" in top_locations.columns else None,
                title="Top Locations",
            )
            scatter.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
            chart_cols[1].plotly_chart(scatter, use_container_width=True)
        else:
            chart_cols[1].dataframe(top_locations, use_container_width=True, hide_index=True)
    else:
        chart_cols[1].info("No location highlights available.")


def render_tables(metadata: dict[str, Any]) -> None:
    top_locations = metadata.get("top_locations", [])
    if top_locations:
        st.markdown("### Top Locations")
        st.dataframe(pd.DataFrame(top_locations), use_container_width=True, hide_index=True)

    district_data = metadata.get("district_density", metadata.get("district_counts", []))
    if district_data:
        st.markdown("### District Analytics")
        st.dataframe(pd.DataFrame(district_data), use_container_width=True, hide_index=True)


def render_result(item: dict[str, Any]) -> None:
    payload = item["payload"]
    metadata = payload.get("metadata", {})

    with st.chat_message("user"):
        st.markdown(item["question"])

    with st.chat_message("assistant"):
        st.markdown("### AI Geospatial Insight")
        st.write(payload["insight"])
        st.caption(
            f"Source: {metadata.get('source', 'unknown')} | LLM: {payload.get('llm_provider', 'none')} | Session: {payload.get('session_id', '')[:8]}"
        )

        render_metrics(metadata)

        st.markdown("### Interactive Map")
        render_map(payload)

        st.markdown("### Analytics")
        render_charts(metadata)
        render_tables(metadata)

        with st.expander("Parsed Query"):
            st.json(payload.get("parsed_query", {}))

        with st.expander("Raw Metadata"):
            st.json(metadata)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(8,145,178,0.16), transparent 28%),
                radial-gradient(circle at top left, rgba(245,158,11,0.12), transparent 25%),
                #07111f;
        }
        .block-container {
            max-width: 1280px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero-card {
            padding: 1.2rem 1.4rem;
            border: 1px solid rgba(148,163,184,0.18);
            background: rgba(15,23,42,0.72);
            border-radius: 20px;
            backdrop-filter: blur(12px);
            margin-bottom: 1rem;
        }
        .status-pill {
            display: inline-block;
            padding: 0.25rem 0.7rem;
            border-radius: 999px;
            font-size: 0.9rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="Smart GIS Chatbot", page_icon=":world_map:", layout="wide")
inject_styles()

if "history" not in st.session_state:
    st.session_state.history = []

if "draft_query" not in st.session_state:
    st.session_state.draft_query = ""

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex

backend_online = backend_health()

with st.sidebar:
    st.markdown("## Smart GIS Copilot")
    status_html = (
        '<span class="status-pill" style="background:#052e16;color:#86efac;">Backend Online</span>'
        if backend_online
        else '<span class="status-pill" style="background:#7f1d1d;color:#fecaca;">Backend Offline</span>'
    )
    st.markdown(status_html, unsafe_allow_html=True)
    st.caption(f"API endpoint: `{BACKEND_URL}`")
    st.caption(f"Session ID: `{st.session_state.session_id[:8]}`")
    if not backend_online:
        st.error("FastAPI is offline. Start `uvicorn backend.main:app --reload` before sending queries.")

    st.markdown("### Suggested Queries")
    for suggestion in SUGGESTIONS:
        if st.button(suggestion, use_container_width=True):
            st.session_state.draft_query = suggestion

    st.markdown("### Supported Modes")
    st.write("- Local GIS: accidents, incidents, infrastructure")
    st.write("- OpenStreetMap: hospitals, schools, police, parks, roads, buildings")
    st.write("- District-level counts and hotspot views")

st.markdown(
    """
    <div class="hero-card">
        <h1>Smart GIS Chatbot</h1>
        <p>Ask any Tamil Nadu geospatial question and get dynamic data retrieval, analytics, maps, and conversational AI explanations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

status_cols = st.columns(3)
status_cols[0].info("Backend Status")
status_cols[0].caption("Online" if backend_online else "Offline")
status_cols[1].info("Processing")
status_cols[1].caption("Idle")
status_cols[2].info("AI Response")
status_cols[2].caption("Waiting")

for item in st.session_state.history:
    render_result(item)

query = st.chat_input("Ask about hospitals, schools, police, hotspots, district counts, or safety gaps...")
if st.session_state.draft_query and query is None:
    query = st.session_state.draft_query
    st.session_state.draft_query = ""

if query:
    if not backend_online:
        st.error("Backend is offline. Start FastAPI on http://localhost:8000 and retry.")
    else:
        status_cols[1].caption("Fetching geospatial data")
        status_cols[2].caption("Generating explanation")
        with st.spinner("Running geospatial assistant..."):
            try:
                payload = ask_backend(query, st.session_state.session_id)
            except requests.HTTPError as exc:
                detail = exc.response.text if exc.response is not None else str(exc)
                st.error(f"Backend request failed: {detail}")
            except requests.RequestException as exc:
                st.error(f"Unable to reach the FastAPI backend at {BACKEND_URL}: {exc}")
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
            else:
                st.session_state.history.append({"question": query, "payload": payload})
                st.rerun()
