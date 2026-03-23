"""
Public runtime exports and runtime resolver.

`agents.py` stays as a thin compatibility layer:
- ReAct runtime implementation lives in `agent/runtimes/react/runtime.py`
"""

from __future__ import annotations

from typing import Type

from .graph import GraphKlynxAgent
from .runtimes.react.runtime import KlynxGeneralAgent, KlynxAgent


def resolve_agent_class(mode: str = "react") -> Type[GraphKlynxAgent]:
    normalized = str(mode or "react").strip().lower()
    if normalized in {"", "react", "default"}:
        return KlynxAgent
    raise ValueError(
        f"Unknown agent mode: {mode}. "
        f"Supported modes: react."
    )


__all__ = [
    "KlynxGeneralAgent",
    "KlynxAgent",
    "resolve_agent_class",
]
