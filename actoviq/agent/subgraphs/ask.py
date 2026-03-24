"""Reusable ask-stream helpers."""

from __future__ import annotations

from typing import Iterable, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ...model.adapter import normalize_usage_payload


def build_ask_messages(
    agent,
    message: str,
    *,
    system_prompt: Optional[str] = None,
    thread_id: str = "default",
    history_limit: int = 20,
) -> List:
    default_system = (getattr(agent, "ASK_SYSTEM_PROMPT", "") or "").strip()
    if not default_system:
        default_system = (
            "You are Klynx Ask Assistant. "
            "Answer the user directly and concisely."
        )
    sys_prompt = (system_prompt or default_system).strip()

    history_msgs = []
    pending_checkpoint_id = ""
    pending_once = True
    if hasattr(agent, "get_pending_rollback") and callable(getattr(agent, "get_pending_rollback", None)):
        pending = agent.get_pending_rollback(thread_id=thread_id)
        pending_checkpoint_id = str(pending.get("checkpoint_id", "") or "").strip()
        pending_once = bool(pending.get("once", True))
    app = getattr(agent, "app", None)
    if thread_id and app is not None and hasattr(app, "get_state"):
        builder = getattr(agent, "_build_run_config", None)
        if callable(builder):
            config = builder(
                thread_id=thread_id,
                recursion_limit=None,
                include_pending_rollback=True,
            )
            config.pop("recursion_limit", None)
        else:
            config = {"configurable": {"thread_id": thread_id}}
        current_state = app.get_state(config)
        if current_state and current_state.values:
            history_msgs = current_state.values.get("messages", [])
            if pending_checkpoint_id and pending_once:
                consumer = getattr(agent, "_consume_pending_rollback", None)
                if callable(consumer):
                    consumer(
                        thread_id=thread_id,
                        expected_checkpoint_id=pending_checkpoint_id,
                    )

    messages: List = [SystemMessage(content=sys_prompt)]
    if history_msgs:
        for msg in history_msgs[-max(1, int(history_limit or 20)) :]:
            if isinstance(msg, (HumanMessage, AIMessage)):
                messages.append(msg)
    messages.append(HumanMessage(content=message))
    return messages


def stream_model_answer(agent, messages: List) -> Iterable[dict]:
    model = getattr(agent, "model", None)
    if model is None:
        yield {"type": "error", "content": "Model is not configured."}
        return

    has_stream = hasattr(model, "stream") and callable(model.stream)
    full_content = ""
    full_reasoning = ""
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    try:
        if has_stream:
            for chunk in model.stream(messages):
                reasoning_part = chunk.get("reasoning_content", "")
                content_part = chunk.get("content", "")
                usage = normalize_usage_payload(chunk.get("usage"))
                if usage["total_tokens"] > 0:
                    prompt_tokens = usage["prompt_tokens"]
                    completion_tokens = usage["completion_tokens"]
                    total_tokens = usage["total_tokens"]
                if reasoning_part:
                    full_reasoning += reasoning_part
                    yield {"type": "reasoning_token", "content": reasoning_part}
                if content_part:
                    full_content += content_part
                    yield {"type": "token", "content": content_part}
        else:
            response = model.invoke(messages)
            full_content = response.content or ""
            extract_reasoning = getattr(agent, "_extract_reasoning_content", None)
            if callable(extract_reasoning):
                full_reasoning = extract_reasoning(response) or ""
            usage = normalize_usage_payload(getattr(response, "usage", None))
            prompt_tokens = usage["prompt_tokens"]
            completion_tokens = usage["completion_tokens"]
            total_tokens = usage["total_tokens"]
            if full_reasoning:
                yield {"type": "reasoning", "content": full_reasoning}
            if full_content:
                yield {"type": "answer", "content": full_content}
    except Exception as exc:
        yield {"type": "error", "content": f"ask call failed: {exc}"}
        return

    yield {
        "type": "done",
        "content": "",
        "answer": full_content,
        "reasoning": full_reasoning,
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def stream_ask(
    agent,
    message: str,
    *,
    system_prompt: Optional[str] = None,
    thread_id: str = "default",
    history_limit: int = 20,
) -> Iterable[dict]:
    messages = build_ask_messages(
        agent,
        message,
        system_prompt=system_prompt,
        thread_id=thread_id,
        history_limit=history_limit,
    )
    return stream_model_answer(agent, messages)
