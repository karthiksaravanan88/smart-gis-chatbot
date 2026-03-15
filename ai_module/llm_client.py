"""Shared LLM client helpers with Ollama-first fallback to OpenAI."""

from __future__ import annotations

import os

try:
    from langchain_community.chat_models import ChatOllama
except ImportError:  # pragma: no cover - installed in this repo env
    ChatOllama = None  # type: ignore[assignment]

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - dependency is declared in requirements
    ChatOpenAI = None  # type: ignore[assignment]


DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")


def get_llm_provider() -> str:
    explicit = os.getenv("SMART_GIS_LLM_PROVIDER", "").strip().lower()
    if explicit in {"ollama", "openai"}:
        return explicit
    if os.getenv("OLLAMA_MODEL"):
        return "ollama"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "none"


def llm_enabled() -> bool:
    provider = get_llm_provider()
    if provider == "ollama":
        return ChatOllama is not None
    if provider == "openai":
        return bool(os.getenv("OPENAI_API_KEY")) and ChatOpenAI is not None
    return False


def build_chat_model(*, temperature: float = 0.4):
    provider = get_llm_provider()
    if provider == "ollama" and ChatOllama is not None:
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
            temperature=temperature,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    if provider == "openai" and ChatOpenAI is not None and os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    return None
