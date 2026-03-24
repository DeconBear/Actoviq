"""Routing policy for Klynx agent graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class RoutingDecision:
    """Single routing decision result."""

    route: str
    reason: str
    warning: str = ""


class RoutingPolicy:
    """Pure routing policy to keep graph transitions testable."""

    @staticmethod
    def decide(state: Dict[str, Any], max_iterations: Optional[int]) -> RoutingDecision:
        iteration = int(state.get("iteration_count", 0) or 0)

        if bool(state.get("task_completed", False)):
            return RoutingDecision(route="end_direct", reason="")

        pending_tools = state.get("pending_tool_calls", []) or []
        if pending_tools:
            return RoutingDecision(
                route="tools",
                reason=f"({len(pending_tools)})",
            )

        if bool(state.get("needs_user_confirmation", False)):
            return RoutingDecision(route="end_direct", reason="")

        if max_iterations is not None and iteration >= max_iterations:
            return RoutingDecision(
                route="end_direct",
                reason=f"({max_iterations})",
                warning=", max_iterations .",
            )

        if bool(state.get("ended_without_tools", False)):
            return RoutingDecision(route="end_direct", reason=",")

        if str(state.get("last_action", "") or "").strip().lower() == "clarify":
            return RoutingDecision(route="end_direct", reason="")

        return RoutingDecision(route="agent", reason="")
