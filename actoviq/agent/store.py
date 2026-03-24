from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class AgentStore(Protocol):
    """Tiny key-value store protocol for runtime artifacts."""

    def get(self, key: str) -> Any:
        ...

    def set(self, key: str, value: Any) -> None:
        ...

    def delete(self, key: str) -> None:
        ...


class InMemoryAgentStore:
    """Default process-local store implementation."""

    def __init__(self):
        self._data: Dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self._data.get(key)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def keys(self):
        return list(self._data.keys())
