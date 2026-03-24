"""Single-pass ReAct subgraph: Think + Act without loop."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from langchain_core.messages import AIMessage

from .actoviq_loop import build_actoviq_initial_state


def _drain_event_buffer(agent) -> List[dict]:
    events: List[dict] = []
    buffer = getattr(agent, "_event_buffer", None)
    while buffer:
        try:
            events.append(buffer.popleft())
        except Exception:
            events.append(buffer.pop(0))
    return events


def think_once_node(agent, state: Dict[str, Any]) -> Dict[str, Any]:
    patch = agent._init_node(state) or {}
    if patch:
        state.update(patch)

    inject_node = getattr(agent, "_inject_system_prompt_node", None)
    if callable(inject_node):
        inject_patch = inject_node(state) or {}
        if inject_patch:
            state.update(inject_patch)

    ctx_patch = agent._load_context_node(state) or {}
    if ctx_patch:
        state.update(ctx_patch)

    inference_patch = agent._model_inference_node(state) or {}
    if inference_patch:
        state.update(inference_patch)
    return state


def act_once_node(agent, state: Dict[str, Any]) -> Dict[str, Any]:
    patch = agent._act_node(state) or {}
    if patch:
        state.update(patch)
    return state


def emit_react_once_done(state: Dict[str, Any]) -> dict:
    answer = ""
    reasoning = ""
    messages = list(state.get("messages", []) or [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            answer = str(getattr(msg, "content", "") or "")
            reasoning = str(
                (getattr(msg, "additional_kwargs", {}) or {}).get("reasoning_content", "")
                or ""
            )
            break

    return {
        "type": "done",
        "content": "",
        "answer": answer,
        "reasoning": reasoning,
        "iteration_count": int(state.get("iteration_count", 0) or 0),
        "total_tokens": int(state.get("total_tokens", 0) or 0),
        "prompt_tokens": int(state.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(state.get("completion_tokens", 0) or 0),
        "tool_calls": list(state.get("pending_tool_calls", []) or []),
        "command_executions": list(state.get("command_executions", []) or []),
    }


def run_react_once_node(runtime, payload: Dict[str, Any]) -> Iterable[dict]:
    agent = runtime._ensure_loop_agent()
    task = str(payload.get("task", "") or "")
    thread_id = str(payload.get("thread_id", "default") or "default")
    invoke_kwargs = dict(payload.get("invoke_kwargs", {}) or {})

    thinking_context = bool(invoke_kwargs.get("thinking_context", False))
    system_prompt_append = str(invoke_kwargs.get("system_prompt_append", "") or "")
    state = build_actoviq_initial_state(
        agent,
        task,
        thread_id=thread_id,
        thinking_context=thinking_context,
        system_prompt_append=system_prompt_append,
    )

    agent._event_buffer.clear()
    if hasattr(agent, "_event_signal"):
        agent._event_signal.clear()
    try:
        state = think_once_node(agent, state)
        state = act_once_node(agent, state)
    except Exception as exc:
        for event in _drain_event_buffer(agent):
            yield event
        yield {"type": "error", "content": f"react_once failed: {exc}"}
        return

    for event in _drain_event_buffer(agent):
        yield event
    yield emit_react_once_done(state)


def build_react_once_subgraph(builder, *, node_name: str = "react_once"):
    builder.add_node(node_name)
    return builder

