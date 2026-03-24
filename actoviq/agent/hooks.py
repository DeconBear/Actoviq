from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class AgentHookContext:
    """Runtime context passed to hook callbacks."""

    state: Dict[str, Any]
    iteration: int
    thread_id: str
    working_dir: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class AgentHook(Protocol):
    """Optional hook interface.

    Each method may return a patch dict.

    Patch contract:
    - before_prompt: {"messages": List[Any], "state": Dict[str, Any]}
    - after_model: {"state": Dict[str, Any], ... model fields ...}
    - after_tools: {"state": Dict[str, Any], ... tool fields ...}
    """

    def before_prompt(self, context: AgentHookContext, messages: List[Any]) -> Optional[Dict[str, Any]]:
        ...

    def after_model(self, context: AgentHookContext, model_output: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        ...

    def after_tools(
        self,
        context: AgentHookContext,
        tool_result: Dict[str, Any],
        executed_tools: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        ...


class HookManager:
    """Small hook runner with deterministic merge rules."""

    def __init__(self, hooks: Optional[List[AgentHook]] = None):
        self._hooks: List[AgentHook] = list(hooks or [])

    @property
    def hooks(self) -> List[AgentHook]:
        return list(self._hooks)

    def set_hooks(self, hooks: Optional[List[AgentHook]]) -> None:
        self._hooks = list(hooks or [])

    def add_hook(self, hook: AgentHook) -> None:
        self._hooks.append(hook)

    def clear(self) -> None:
        self._hooks.clear()

    def run_before_prompt(self, context: AgentHookContext, messages: List[Any]) -> Dict[str, Any]:
        out_messages = messages
        out_state: Dict[str, Any] = {}
        for hook in self._hooks:
            fn = getattr(hook, "before_prompt", None)
            if not callable(fn):
                continue
            patch = fn(context, out_messages)
            if not isinstance(patch, dict):
                continue
            if isinstance(patch.get("messages"), list):
                out_messages = patch["messages"]
            if isinstance(patch.get("state"), dict):
                out_state.update(patch["state"])
        return {"messages": out_messages, "state": out_state}

    def run_after_model(self, context: AgentHookContext, model_output: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(model_output or {})
        state_patch: Dict[str, Any] = {}
        for hook in self._hooks:
            fn = getattr(hook, "after_model", None)
            if not callable(fn):
                continue
            patch = fn(context, out)
            if not isinstance(patch, dict):
                continue
            if isinstance(patch.get("state"), dict):
                state_patch.update(patch["state"])
            for key, value in patch.items():
                if key != "state":
                    out[key] = value
        if state_patch:
            out["state"] = {**out.get("state", {}), **state_patch}
        return out

    def run_after_tools(
        self,
        context: AgentHookContext,
        tool_result: Dict[str, Any],
        executed_tools: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        out = dict(tool_result or {})
        state_patch: Dict[str, Any] = {}
        for hook in self._hooks:
            fn = getattr(hook, "after_tools", None)
            if not callable(fn):
                continue
            patch = fn(context, out, executed_tools)
            if not isinstance(patch, dict):
                continue
            if isinstance(patch.get("state"), dict):
                state_patch.update(patch["state"])
            for key, value in patch.items():
                if key != "state":
                    out[key] = value
        if state_patch:
            out.update(state_patch)
        return out


class RuntimeTruthHook:
    """Inject factual corrections when model narration contradicts runtime state."""

    CLAIM_PATTERNS = (
        re.compile(
            r"\b(already|now|i\s+have|i've|implemented|updated|added|fixed|integrated|rewrote|created)\b",
            re.IGNORECASE,
        ),
        re.compile(r".{0,20}(||||||)"),
    )

    def after_model(self, context: AgentHookContext, model_output: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        state = context.state or {}
        last_mutation = dict(state.get("last_mutation", {}) or {})
        if str(last_mutation.get("status", "") or "").strip().lower() != "error":
            return None

        content = str(model_output.get("content", "") or "")
        if not content.strip() or "<runtime_correction>" in content:
            return None
        if not any(pattern.search(content) for pattern in self.CLAIM_PATTERNS):
            return None

        path = str(last_mutation.get("path", "") or "").strip()
        error_kind = str(last_mutation.get("error_kind", "") or "").strip()
        error_excerpt = str(last_mutation.get("error_excerpt", "") or "").strip()
        next_hint = str(last_mutation.get("next_hint", "") or "").strip()

        correction_parts = ["The last mutation attempt failed"]
        if path:
            correction_parts.append(f"for {path}")
        if error_kind:
            correction_parts.append(f"({error_kind})")
        correction = " ".join(correction_parts).strip() + "."
        if error_excerpt:
            correction += f" Evidence: {error_excerpt}."
        if next_hint:
            correction += f" Next step: {next_hint}."

        state_patch: Dict[str, Any] = {"blocked_reason": correction}
        if path:
            state_patch["current_focus"] = path
        return {
            "content": f"<runtime_correction>{correction}</runtime_correction>\n{content}".strip(),
            "state": state_patch,
        }
