"""Conversational explanation generation for local GIS and OSM results."""

from __future__ import annotations

import json
from typing import Any

try:
    from langchain_core.prompts import ChatPromptTemplate
except ImportError:  # pragma: no cover
    ChatPromptTemplate = None  # type: ignore[assignment]

from ai_module.llm_client import build_chat_model, llm_enabled


class InsightGenerator:
    def __init__(self) -> None:
        self.prompt = None
        if ChatPromptTemplate is not None:
            self.prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        (
                            "You are Smart GIS Copilot, a conversational Tamil Nadu geospatial analyst. "
                            "Explain map results clearly, mention the most relevant districts or locations, and add one practical implication."
                        ),
                    ),
                    (
                        "human",
                        (
                            "Conversation history:\n{conversation_history}\n\n"
                            "User question:\n{question}\n\n"
                            "Structured result summary:\n{summary}\n\n"
                            "Respond in 2-4 short paragraphs."
                        ),
                    ),
                ]
            )

    def generate(
        self,
        question: str,
        parsed_query: dict[str, Any],
        metadata: dict[str, Any],
        conversation_history: str = "",
    ) -> str:
        summary = self._summary_payload(parsed_query, metadata)
        llm_response = self._generate_with_llm(question, summary, conversation_history)
        if llm_response:
            return llm_response
        return self._fallback(summary)

    def _generate_with_llm(self, question: str, summary: dict[str, Any], conversation_history: str) -> str | None:
        if self.prompt is None or not llm_enabled():
            return None
        llm = build_chat_model(temperature=0.7)
        if llm is None:
            return None
        try:
            response = llm.invoke(
                self.prompt.format_messages(
                    question=question,
                    summary=json.dumps(summary, ensure_ascii=True, indent=2),
                    conversation_history=conversation_history or "No prior conversation.",
                )
            )
            content = getattr(response, "content", "")
            if isinstance(content, list):
                content = " ".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            text = str(content).strip()
            return text or None
        except Exception:
            return None

    def _summary_payload(self, parsed_query: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            "source": metadata.get("source", parsed_query.get("source", "unknown")),
            "entity": metadata.get("entity", parsed_query.get("entity", parsed_query.get("dataset"))),
            "location": metadata.get("location", parsed_query.get("location", "Tamil Nadu")),
            "analysis": metadata.get("analysis", parsed_query.get("analysis", "map")),
            "total_incidents": metadata.get("total_incidents"),
            "total_features": metadata.get("total_features"),
            "cluster_count": metadata.get("cluster_count"),
            "top_districts": metadata.get("top_districts", metadata.get("district_counts", []))[:5],
            "top_locations": metadata.get("top_locations", [])[:5],
            "filters": metadata.get("filters", {}),
        }

    def _fallback(self, summary: dict[str, Any]) -> str:
        entity = (summary.get("entity") or "features").replace("_", " ")
        location = summary.get("location", "Tamil Nadu")
        analysis = summary.get("analysis", "map")
        top_districts = summary.get("top_districts", [])
        top_names = ", ".join(item.get("district", "Unknown") for item in top_districts[:3]) or location
        total = summary.get("total_incidents") or summary.get("total_features") or 0

        if analysis == "count":
            return (
                f"The district-level count for {entity} in {location} points most strongly to {top_names}. "
                f"The current result set includes {total} mapped records, so those leading districts appear to be the main concentration areas.\n\n"
                "This view is useful when you need to compare service coverage or incident pressure across districts."
            )
        if analysis == "hotspot":
            return (
                f"The hotspot pattern for {entity} in {location} is concentrated around {top_names}. "
                f"The mapped result set contains {total} records, which suggests the distribution is not uniform across Tamil Nadu.\n\n"
                "Those high-density pockets are the best places to investigate operational causes or intervention priorities."
            )
        if analysis == "nearby":
            return (
                f"The nearby search for {entity} in {location} returned {total} mapped records within the selected area. "
                "This helps identify what services or risk points sit close to the location you asked about.\n\n"
                "Use the clustered markers to inspect individual sites and their local context."
            )
        if analysis == "recommendation":
            return (
                f"Based on the mapped {entity} distribution in {location}, the strongest coverage or pressure patterns appear around {top_names}. "
                "That means districts with heavy demand but weaker existing coverage should be prioritized first.\n\n"
                "A sensible next step is to compare these map results with population and road-access data before placing new facilities."
            )
        return (
            f"The assistant mapped {total} {entity} records for {location}. "
            f"The strongest concentrations are around {top_names}, which gives you a practical view of where activity or coverage is most visible.\n\n"
            "You can use the map to drill into individual sites and then ask follow-up questions about those areas."
        )


def generate_insight(
    question: str,
    parsed_query: dict[str, Any],
    metadata: dict[str, Any],
    conversation_history: str = "",
) -> str:
    return InsightGenerator().generate(question, parsed_query, metadata, conversation_history)
