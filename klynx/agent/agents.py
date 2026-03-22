"""
Public runtime exports and runtime resolver.

`agents.py` stays as a thin compatibility layer:
- ReAct runtime implementation lives in `agent/runtimes/react/runtime.py`
- Request runtime implementation lives in `agent/runtimes/request_orchestrator/runtime.py`
"""

from __future__ import annotations

from typing import Type

from .graph import GraphKlynxAgent
from .runtimes.react.runtime import KlynxGeneralAgent, KlynxAgent
from .runtimes.request_orchestrator.runtime import RequestOrchestratorAgent


RequestAgent = RequestOrchestratorAgent
ReqAgent = RequestOrchestratorAgent


def resolve_agent_class(mode: str = "react") -> Type[GraphKlynxAgent]:
    normalized = str(mode or "react").strip().lower()
    if normalized in {"", "react", "default"}:
        return KlynxAgent
    if normalized in {
        "request_orchestrator",
        "request",
        "requestagent",
        "request_agent",
        "req-agent",
        "req_agent",
    }:
        return RequestOrchestratorAgent
    raise ValueError(
        f"Unknown agent mode: {mode}. "
        f"Supported modes: react, request_orchestrator (alias: request)."
    )


__all__ = [
    "KlynxGeneralAgent",
    "KlynxAgent",
    "RequestOrchestratorAgent",
    "RequestAgent",
    "ReqAgent",
    "resolve_agent_class",
]

