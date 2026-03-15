"""Conversation memory helpers for session-aware GIS chat."""

from __future__ import annotations

from threading import Lock

try:
    from langchain_classic.memory import ConversationBufferMemory
except ImportError:  # pragma: no cover - dependency is installed in this repo env
    ConversationBufferMemory = None  # type: ignore[assignment]


class ConversationManager:
    def __init__(self) -> None:
        self._sessions: dict[str, object] = {}
        self._lock = Lock()

    def _get_memory(self, session_id: str):
        if ConversationBufferMemory is None:
            return None

        with self._lock:
            memory = self._sessions.get(session_id)
            if memory is None:
                memory = ConversationBufferMemory(
                    return_messages=True,
                    memory_key="history",
                    input_key="input",
                    output_key="output",
                )
                self._sessions[session_id] = memory
            return memory

    def get_history_text(self, session_id: str) -> str:
        memory = self._get_memory(session_id)
        if memory is None:
            return ""

        variables = memory.load_memory_variables({})
        history = variables.get("history", [])
        lines: list[str] = []
        for message in history:
            role = getattr(message, "type", "assistant").capitalize()
            lines.append(f"{role}: {message.content}")
        return "\n".join(lines)

    def add_turn(self, session_id: str, user_message: str, assistant_message: str) -> None:
        memory = self._get_memory(session_id)
        if memory is None:
            return
        memory.save_context({"input": user_message}, {"output": assistant_message})

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
