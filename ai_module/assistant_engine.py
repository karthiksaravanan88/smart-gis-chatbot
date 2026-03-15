"""Agent-backed orchestration for the geospatial assistant."""

from __future__ import annotations

import json
import uuid
from typing import Any

from langchain_classic.agents import AgentType, initialize_agent
from langchain_core.tools import ToolException

try:
    from langchain_classic.agents import Tool
except ImportError:  # pragma: no cover
    Tool = None  # type: ignore[assignment]

from ai_module.conversation import ConversationManager
from ai_module.insight_generator import InsightGenerator
from ai_module.llm_client import build_chat_model, get_llm_provider, llm_enabled
from ai_module.query_parser import GeoIntentParser, parse_geo_intent
from data_sources.openstreetmap import OpenStreetMapQueryService
from gis_engine.spatial_analysis import SpatialAnalysisEngine
from map_service.map_generator import MapGenerator


class LocalGISAnalysisService:
    def __init__(self, spatial_engine: SpatialAnalysisEngine | None = None, map_generator: MapGenerator | None = None) -> None:
        self.spatial_engine = spatial_engine or SpatialAnalysisEngine()
        self.map_generator = map_generator or MapGenerator()
        self.intent_parser = GeoIntentParser()

    def run(self, question: str, conversation_history: str = "", map_filename: str | None = None) -> dict[str, Any]:
        intent = parse_geo_intent(question, conversation_history)
        dataset = intent.dataset or "accidents"
        parsed_query = {
            "dataset": dataset,
            "analysis": self._local_analysis(intent.analysis),
            "filters": {key: value for key, value in intent.filters.items() if key in {"district", "keyword"}},
            "buffer_distance_m": intent.buffer_distance_m,
            "explanation_level": intent.explanation_level,
        }
        result = self.spatial_engine.run(parsed_query)
        filename = map_filename or f"{uuid.uuid4().hex}.html"
        map_path = self.map_generator.create_map(result, filename)
        metadata = dict(result.metadata)
        metadata.update(
            {
                "source": "local",
                "entity": dataset,
                "location": intent.location,
                "district_counts": metadata.get("district_density", []),
                "total_features": metadata.get("total_incidents", 0),
            }
        )
        return {
            "source": "local",
            "parsed_query": parsed_query,
            "map_path": str(map_path),
            "map_url": f"/maps/{map_path.name}",
            "metadata": metadata,
        }

    def _local_analysis(self, analysis: str) -> str:
        if analysis == "count":
            return "spatial_join"
        if analysis == "nearby":
            return "buffer"
        if analysis in {"hotspot", "recommendation"}:
            return "heatmap"
        return "heatmap"


class GeospatialAssistant:
    def __init__(self) -> None:
        self.spatial_engine = SpatialAnalysisEngine()
        self.map_generator = MapGenerator()
        self.local_service = LocalGISAnalysisService(self.spatial_engine, self.map_generator)
        self.osm_service = OpenStreetMapQueryService(self.spatial_engine, self.map_generator)
        self.intent_parser = GeoIntentParser()
        self.conversation_manager = ConversationManager()
        self.insight_generator = InsightGenerator()

    def ask(self, question: str, session_id: str) -> dict[str, Any]:
        conversation_history = self.conversation_manager.get_history_text(session_id)
        if llm_enabled():
            routed = self._run_agent(question, conversation_history)
        else:
            routed = self._route_without_llm(question, conversation_history)

        insight = self._build_insight(question, routed, conversation_history)
        self.conversation_manager.add_turn(session_id, question, insight)
        routed["session_id"] = session_id
        routed["insight"] = insight
        routed["backend_status"] = "online"
        routed["llm_provider"] = get_llm_provider()
        return routed

    def _run_agent(self, question: str, conversation_history: str) -> dict[str, Any]:
        llm = build_chat_model(temperature=0.2)
        if llm is None or Tool is None:
            return self._route_without_llm(question, conversation_history)

        def local_tool(tool_input: str) -> str:
            result = self.local_service.run(tool_input, conversation_history)
            return json.dumps(result, ensure_ascii=True)

        def osm_tool(tool_input: str) -> str:
            intent = parse_geo_intent(tool_input, conversation_history)
            map_filename = f"{uuid.uuid4().hex}.html"
            result = self.osm_service.run(intent, map_filename)
            return json.dumps(result, ensure_ascii=True)

        tools = [
            Tool(
                name="local_gis_analysis_tool",
                func=local_tool,
                description=(
                    "Use for local project datasets only: accidents, incidents, infrastructure, district accident counts, accident hotspots, "
                    "and accident cluster questions in Tamil Nadu."
                ),
            ),
            Tool(
                name="openstreetmap_query_tool",
                func=osm_tool,
                description=(
                    "Use for dynamic OpenStreetMap data such as hospitals, schools, police stations, parks, roads, and buildings in Tamil Nadu."
                ),
            ),
        ]
        executor = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
            max_iterations=3,
        )
        prompt = (
            "You are a Tamil Nadu geospatial assistant. "
            "Pick exactly one tool based on the user's question. "
            "Use the local GIS tool for accidents/incidents/infrastructure. "
            "Use the OpenStreetMap tool for hospitals, schools, police stations, parks, roads, and buildings.\n\n"
            f"Conversation history:\n{conversation_history or 'No prior conversation.'}\n\n"
            f"User question:\n{question}"
        )
        try:
            result = executor.invoke({"input": prompt})
        except Exception:
            return self._route_without_llm(question, conversation_history)
        steps = result.get("intermediate_steps", [])
        if not steps:
            return self._route_without_llm(question, conversation_history)
        observation = steps[-1][1]
        try:
            return json.loads(observation)
        except json.JSONDecodeError as exc:
            raise ToolException(f"Tool output was not valid JSON: {observation}") from exc

    def _route_without_llm(self, question: str, conversation_history: str) -> dict[str, Any]:
        intent = self.intent_parser.parse(question, conversation_history)
        map_filename = f"{uuid.uuid4().hex}.html"
        if intent.source == "local":
            return self.local_service.run(question, conversation_history, map_filename)
        return self.osm_service.run(intent, map_filename)

    def _build_insight(self, question: str, routed: dict[str, Any], conversation_history: str) -> str:
        return self.insight_generator.generate(
            question,
            routed.get("parsed_query", {}),
            routed.get("metadata", {}),
            conversation_history=conversation_history,
        )
