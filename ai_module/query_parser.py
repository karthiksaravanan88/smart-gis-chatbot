"""Natural language parsing for general-purpose Tamil Nadu geospatial questions."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

try:
    from langchain_core.prompts import ChatPromptTemplate
except ImportError:  # pragma: no cover
    ChatPromptTemplate = None  # type: ignore[assignment]

from ai_module.llm_client import build_chat_model


LOCAL_DATASETS = {"accidents", "incidents", "infrastructure"}
KNOWN_DISTRICTS = {
    "Chennai",
    "Coimbatore",
    "Cuddalore",
    "Dindigul",
    "Kancheepuram",
    "Kanchipuram",
    "Madurai",
    "Namakkal",
    "Salem",
    "Theni",
    "Thiruvallur",
    "Tiruppur",
    "Tiruchirappalli",
    "Tirunelveli",
    "Vellore",
    "Thoothukudi",
    "Erode",
    "Thanjavur",
    "Tamil Nadu",
}
ENTITY_ALIASES = {
    "hospital": ("amenity", "hospital"),
    "hospitals": ("amenity", "hospital"),
    "school": ("amenity", "school"),
    "schools": ("amenity", "school"),
    "police station": ("amenity", "police"),
    "police stations": ("amenity", "police"),
    "police": ("amenity", "police"),
    "park": ("leisure", "park"),
    "parks": ("leisure", "park"),
    "road": ("highway", "*"),
    "roads": ("highway", "*"),
    "building": ("building", "yes"),
    "buildings": ("building", "yes"),
}


class GeoIntent(BaseModel):
    source: str = Field(default="local")
    dataset: str | None = Field(default=None)
    entity: str | None = Field(default=None)
    location: str = Field(default="Tamil Nadu")
    analysis: str = Field(default="map")
    filters: dict[str, Any] = Field(default_factory=dict)
    buffer_distance_m: int = Field(default=1000, ge=100, le=100000)
    explanation_level: str = Field(default="simple")


class GeoIntentParser:
    def __init__(self) -> None:
        self.prompt = None
        if ChatPromptTemplate is not None:
            self.prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        (
                            "You are a Tamil Nadu geospatial query planner. "
                            "Classify the user's question for either the local datasets "
                            "(accidents, incidents, infrastructure) or OpenStreetMap data. "
                            "Use source=local for accidents/incidents/infrastructure analytics. "
                            "Use source=osm for hospitals, schools, police stations, parks, roads, and buildings. "
                            "Use analysis=count for district rankings and counts, analysis=hotspot for hotspot/density questions, "
                            "analysis=nearby for near/within questions, analysis=recommendation for planning questions like "
                            "'Where should new hospitals be built?'."
                        ),
                    ),
                    (
                        "human",
                        (
                            "Conversation history:\n{conversation_history}\n\n"
                            "Question:\n{question}\n\n"
                            "Return source, dataset, entity, location, analysis, filters, buffer_distance_m, explanation_level."
                        ),
                    ),
                ]
            )

    def parse(self, question: str, conversation_history: str = "") -> GeoIntent:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        llm_intent = self._parse_with_llm(question, conversation_history)
        if llm_intent is not None:
            return llm_intent
        return self._fallback_parse(question, conversation_history)

    def _parse_with_llm(self, question: str, conversation_history: str) -> GeoIntent | None:
        if self.prompt is None:
            return None
        llm = build_chat_model(temperature=0.1)
        if llm is None:
            return None
        try:
            structured = llm.with_structured_output(GeoIntent)
            response = structured.invoke(
                self.prompt.format_messages(
                    question=question,
                    conversation_history=conversation_history or "No prior conversation.",
                )
            )
            return self._normalize_intent(response)
        except Exception:
            return None

    def _fallback_parse(self, question: str, conversation_history: str) -> GeoIntent:
        lowered = question.lower()
        dataset = self._detect_local_dataset(lowered)
        entity = self._detect_entity(lowered)
        location = self._extract_location(question) or self._extract_location(conversation_history) or "Tamil Nadu"
        analysis = self._detect_analysis(lowered)
        filters: dict[str, Any] = {}
        district = self._extract_district(question) or self._extract_district(conversation_history)
        if district and district != "Tamil Nadu":
            filters["district"] = district
        if entity:
            filters["entity"] = entity

        source = "local" if dataset else "osm"
        if "recommend" in lowered or "should new" in lowered or "better" in lowered:
            analysis = "recommendation"

        return self._normalize_intent(
            GeoIntent(
                source=source,
                dataset=dataset,
                entity=entity or dataset,
                location=location,
                analysis=analysis,
                filters=filters,
                buffer_distance_m=self._detect_buffer_distance(lowered),
                explanation_level="technical" if "technical" in lowered else "simple",
            )
        )

    def _normalize_intent(self, intent: GeoIntent) -> GeoIntent:
        dataset = intent.dataset if intent.dataset in LOCAL_DATASETS else None
        entity = self._canonicalize_entity(intent.entity or dataset or "feature")
        location = self._canonicalize_location(intent.location)
        source = "local" if dataset else intent.source
        if source not in {"local", "osm"}:
            source = "local" if dataset else "osm"
        analysis = (intent.analysis or "map").lower().replace("-", "_")
        aliases = {"density": "hotspot", "cluster": "hotspot", "district": "count"}
        analysis = aliases.get(analysis, analysis)
        if analysis not in {"map", "count", "hotspot", "nearby", "recommendation"}:
            analysis = "map"
        filters = dict(intent.filters or {})
        if location != "Tamil Nadu" and "district" not in filters and location in KNOWN_DISTRICTS:
            filters["district"] = location
        if entity:
            filters["entity"] = entity
        return GeoIntent(
            source=source,
            dataset=dataset,
            entity=entity,
            location=location,
            analysis=analysis,
            filters=filters,
            buffer_distance_m=max(100, int(intent.buffer_distance_m or 1000)),
            explanation_level=intent.explanation_level or "simple",
        )

    def _detect_local_dataset(self, lowered: str) -> str | None:
        if any(word in lowered for word in ("accident", "crash", "collision")):
            return "accidents"
        if any(word in lowered for word in ("incident", "event")):
            return "incidents"
        if "infrastructure" in lowered:
            return "infrastructure"
        return None

    def _detect_entity(self, lowered: str) -> str | None:
        for alias in ENTITY_ALIASES:
            if alias in lowered:
                return alias.rstrip("s")
        dataset = self._detect_local_dataset(lowered)
        return dataset.rstrip("s") if dataset else None

    def _canonicalize_entity(self, entity: str) -> str:
        normalized = entity.strip().lower()
        if normalized.endswith("s") and normalized[:-1] in {item.rstrip("s") for item in ENTITY_ALIASES}:
            normalized = normalized[:-1]
        if normalized in {"accident", "incident"}:
            return f"{normalized}s"
        return normalized

    def _extract_location(self, text: str) -> str | None:
        for district in KNOWN_DISTRICTS:
            if district.lower() in text.lower():
                return "Kancheepuram" if district == "Kanchipuram" else district
        for pattern in (r"in\s+([A-Za-z\s]+)", r"near\s+([A-Za-z\s]+)"):
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                value = match.group(1).strip(" ?.")
                return self._canonicalize_location(value)
        return None

    def _extract_district(self, text: str) -> str | None:
        location = self._extract_location(text)
        if location in KNOWN_DISTRICTS:
            return location
        return None

    def _canonicalize_location(self, location: str | None) -> str:
        if not location:
            return "Tamil Nadu"
        lowered = location.lower().strip()
        if "tamil nadu" in lowered:
            return "Tamil Nadu"
        for district in KNOWN_DISTRICTS:
            if district.lower() == lowered:
                return "Kancheepuram" if district == "Kanchipuram" else district
        return location.title()

    def _detect_analysis(self, lowered: str) -> str:
        if any(term in lowered for term in ("hotspot", "hotspots", "density", "dangerous", "risk")):
            return "hotspot"
        if any(term in lowered for term in ("which district", "most", "count", "how many")):
            return "count"
        if any(term in lowered for term in ("near", "nearby", "within", "around")):
            return "nearby"
        return "map"

    def _detect_buffer_distance(self, lowered: str) -> int:
        match = re.search(r"(\d+)\s*(km|kilometers|kilometres|m|meters|metres)", lowered)
        if not match:
            return 1000
        value = int(match.group(1))
        unit = match.group(2)
        return value * 1000 if unit.startswith("k") else value


def parse_geo_intent(question: str, conversation_history: str = "") -> GeoIntent:
    return GeoIntentParser().parse(question, conversation_history=conversation_history)
