"""FastAPI application for Smart GIS Chatbot."""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ai_module.assistant_engine import GeospatialAssistant
from gis_engine.spatial_analysis import GENERATED_MAPS_DIR


logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Natural language geospatial question.")
    session_id: str | None = Field(default=None, description="Conversation/session identifier.")


class AskResponse(BaseModel):
    session_id: str
    parsed_query: dict[str, Any]
    map_path: str
    map_url: str
    insight: str
    metadata: dict[str, Any]
    backend_status: str = "online"
    llm_provider: str = "none"


app = FastAPI(
    title="Smart GIS Chatbot API",
    version="3.0.0",
    description="General-purpose Tamil Nadu geospatial assistant using FastAPI, LangChain agents, GeoPandas, Folium, and OSM.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

assistant = GeospatialAssistant()
GENERATED_MAPS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/maps", StaticFiles(directory=Path(GENERATED_MAPS_DIR)), name="maps")


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


def run_analysis(question: str, session_id: str) -> dict[str, Any]:
    return assistant.ask(question, session_id)


@app.post("/ask", response_model=AskResponse)
async def ask_question(payload: AskRequest) -> AskResponse:
    session_id = payload.session_id or uuid.uuid4().hex
    try:
        result = await asyncio.to_thread(run_analysis, payload.question, session_id)
        return AskResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Geospatial assistant failed for session %s", session_id)
        raise HTTPException(status_code=500, detail=f"Assistant failed: {exc}") from exc
