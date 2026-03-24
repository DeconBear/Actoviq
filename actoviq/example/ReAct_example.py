from __future__ import annotations

import json
import os
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

from dotenv import load_dotenv

from actoviq import create_agent, setup_model


load_dotenv()


SCRIPT_DIR = Path(__file__).resolve().parent
LOG_DIR = SCRIPT_DIR / "log"
ACTIVE_LOG_PATH_BY_THREAD: Dict[str, Path] = {}

# ---------------------------------------------------------------------------
# Model selection & hyper-parameters (edit here, no CLI args needed)
# ---------------------------------------------------------------------------
# Mode A (alias/custom route): set MODEL_PROVIDER_OR_ALIAS only, keep MODEL_NAME=None.
# Example alias: "mimo-v2-pro", "gpt-5.2", "deepseek-chat"
# Example custom route: "xiaomi_mimo/mimo-v2-pro"
# MODEL_PROVIDER_OR_ALIAS = "kimi-k2.5"
MODEL_PROVIDER_OR_ALIAS = "xiaomi_mimo/mimo-v2-pro"
# MODEL_PROVIDER_OR_ALIAS = "minimax-m2.5"

# Mode B (provider + model): set both, e.g. provider="openai", model="gpt-5.2"
MODEL_NAME: Optional[str] = None

# Optional explicit API key; keep None to read from env.
MODEL_API_KEY: Optional[str] = None

# Common model kwargs passed to setup_model(...)
MODEL_KWARGS: Dict[str, Any] = {
    "temperature": 0.3,
    "top_p": 0.65,
    # "max_context_tokens": 128000,
    # "max_tokens": 4096,
}


def _supports_color() -> bool:
    if os.getenv("NO_COLOR"):
        return False
    if os.getenv("FORCE_COLOR"):
        return True
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


_USE_COLOR = _supports_color()
_ANSI: Dict[str, str] = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
}


def _c(text: str, *styles: str) -> str:
    if not _USE_COLOR or not styles:
        return text
    prefix = "".join(_ANSI.get(style, "") for style in styles if style in _ANSI)
    if not prefix:
        return text
    return f"{prefix}{text}{_ANSI['reset']}"


def _context_pct_styles(pct: float) -> tuple[str, ...]:
    if pct >= 90.0:
        return ("red", "bold")
    if pct >= 70.0:
        return ("yellow", "bold")
    return ("green", "bold")


def _extract_tool_result_inner(content: str) -> str:
    text = str(content or "")
    match = re.search(r"<tool_result\b[^>]*>\s*(.*?)\s*</tool_result>", text, re.DOTALL | re.IGNORECASE)
    return (match.group(1).strip() if match else text.strip()) or ""


def _extract_tool_result_status(content: str, default: str = "") -> str:
    text = str(content or "")
    match = re.search(r'<tool_result\b[^>]*\bstatus="([^"]+)"', text, re.IGNORECASE)
    if match:
        return str(match.group(1) or "").strip().lower()
    return str(default or "").strip().lower()


def _colorize_patch_line(line: str) -> str:
    text = str(line or "")
    if text.startswith("+") and not text.startswith("+++"):
        return _c(text, "green")
    if text.startswith("-") and not text.startswith("---"):
        return _c(text, "red")
    if text.startswith("@@"):
        return _c(text, "cyan", "bold")
    if text.startswith("*** "):
        return _c(text, "cyan", "bold")
    if text.startswith(" "):
        return _c(text, "dim")
    return _c(text, "dim")


def _print_apply_patch_feedback(
    executed_tools: List[Dict[str, Any]],
    full_messages: List[Dict[str, Any]],
) -> None:
    patch_tool_messages: List[str] = []
    for item in full_messages:
        content = str(item.get("content", "") or "")
        if "<tool_result tool=\"apply_patch\"" in content:
            patch_tool_messages.append(content)

    message_index = 0
    patch_index = 0
    for tool in list(executed_tools or []):
        if str(tool.get("tool", "") or "").strip() != "apply_patch":
            continue

        patch_index += 1
        success = bool(tool.get("success", False))
        params = dict(tool.get("params", {}) or {})
        patch_text = str(params.get("patch", "") or "").strip()

        result_message = ""
        if message_index < len(patch_tool_messages):
            result_message = patch_tool_messages[message_index]
            message_index += 1

        result_status = _extract_tool_result_status(
            result_message,
            default=("success" if success else "error"),
        )
        result_inner = _extract_tool_result_inner(result_message)
        if result_inner:
            result_inner = re.sub(r"</?success>", "", result_inner, flags=re.IGNORECASE).strip()
            result_inner = re.sub(r"</?error>", "", result_inner, flags=re.IGNORECASE).strip()

        status_text = result_status.upper() if result_status else ("SUCCESS" if success else "ERROR")
        status_style = ("green", "bold") if (success or result_status == "success") else ("red", "bold")
        print(_c(f"[apply_patch #{patch_index}] {status_text}", *status_style))

        if result_inner:
            print(f"  {_c('结果:', 'dim')} { _c(result_inner, *status_style) }")
        elif result_message:
            print(f"  {_c('结果:', 'dim')} {_c('(无可解析结果文本)', 'yellow')}")
        else:
            print(f"  {_c('结果:', 'dim')} {_c('(本轮未捕获 tool_result 消息)', 'yellow')}")

        if not patch_text:
            print(f"  {_c('Patch:', 'dim')} {_c('(空 patch)', 'yellow')}")
            continue

        print(f"  {_c('Patch:', 'dim')}")
        for raw_line in patch_text.splitlines():
            print("    " + _colorize_patch_line(raw_line))


def _ensure_debug_env() -> None:
    """终端输出策略：显示 thinking，隐藏完整 tool_result 内容。"""
    os.environ["KLYNX_STREAM_VERBOSE"] = "0"
    os.environ["KLYNX_STREAM_SHOW_REASONING"] = "1"


def _sanitize_log_id(raw_id: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(raw_id))
    return cleaned or "default"


def _build_log_path(thread_id: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_id = _sanitize_log_id(thread_id)
    return LOG_DIR / f"{safe_id}_{timestamp}.log"


def _append_log(log_path: Path, record_type: str, payload: Dict[str, Any]) -> None:
    record: Dict[str, Any] = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "type": record_type,
        **payload,
    }
    try:
        line = json.dumps(record, ensure_ascii=False, default=str)
        safe_line = line.encode("utf-8", errors="replace").decode("utf-8")
        with log_path.open("a", encoding="utf-8", errors="replace") as f:
            f.write(safe_line)
            f.write("\n")
    except Exception as exc:
        # 调试日志失败不应中断主流程。
        print(f"[Warning] 日志写入失败: {exc}")


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _build_round_usage_summary(
    result: Dict[str, Any],
    round_metrics: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    token_events = list(round_metrics.get("token_usage_events", []) or [])
    context_events = list(round_metrics.get("context_stats_events", []) or [])

    prompt_tokens = sum(_to_int(item.get("prompt_tokens", 0), 0) for item in token_events)
    completion_tokens = sum(_to_int(item.get("completion_tokens", 0), 0) for item in token_events)
    total_tokens = sum(_to_int(item.get("total_tokens", 0), 0) for item in token_events)

    # 兜底：若本轮未收到 token_usage 事件，则回退到 done 结果里的统计。
    if prompt_tokens <= 0:
        prompt_tokens = _to_int(result.get("prompt_tokens", 0), 0)
    if completion_tokens <= 0:
        completion_tokens = _to_int(result.get("completion_tokens", 0), 0)
    if total_tokens <= 0:
        total_tokens = _to_int(result.get("total_tokens", 0), 0)

    context_total_tokens = 0
    context_max_tokens = 0
    context_usage_pct = 0.0
    context_breakdown: Dict[str, Any] = {}
    context_source = ""

    for item in reversed(context_events):
        total = _to_int(item.get("context_total_tokens", 0), 0)
        max_tokens = _to_int(item.get("context_max_tokens", 0), 0)
        pct = _to_float(item.get("context_usage_pct", 0.0), 0.0)
        if total > 0 or max_tokens > 0:
            context_total_tokens = total
            context_max_tokens = max_tokens
            context_usage_pct = pct
            context_breakdown = dict(item.get("context_breakdown", {}) or {})
            context_source = str(item.get("usage_source", "") or "").strip()
            break

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "token_event_count": len(token_events),
        "context_event_count": len(context_events),
        "context_total_tokens": context_total_tokens,
        "context_max_tokens": context_max_tokens,
        "context_usage_pct": context_usage_pct,
        "context_breakdown": context_breakdown,
        "context_usage_source": context_source,
    }


def _print_round_usage_summary(thread_id: str, summary: Dict[str, Any]) -> None:
    prompt_tokens = _to_int(summary.get("prompt_tokens", 0), 0)
    completion_tokens = _to_int(summary.get("completion_tokens", 0), 0)
    total_tokens = _to_int(summary.get("total_tokens", 0), 0)
    token_event_count = _to_int(summary.get("token_event_count", 0), 0)

    context_total = _to_int(summary.get("context_total_tokens", 0), 0)
    context_max = _to_int(summary.get("context_max_tokens", 0), 0)
    context_pct = _to_float(summary.get("context_usage_pct", 0.0), 0.0)
    context_source = str(summary.get("context_usage_source", "") or "").strip()
    context_breakdown = dict(summary.get("context_breakdown", {}) or {})

    print(_c(f"[Token统计][Thread {thread_id}]", "cyan", "bold"))
    print(
        "  "
        + _c("Prompt:", "cyan")
        + f" {_c(f'{prompt_tokens:,}', 'blue')} | "
        + _c("Completion:", "magenta")
        + f" {_c(f'{completion_tokens:,}', 'magenta')} | "
        + _c("Total:", "yellow", "bold")
        + f" {_c(f'{total_tokens:,}', 'yellow', 'bold')} "
        + _c(f"(events={token_event_count})", "dim")
    )
    if context_max > 0:
        pct_styles = _context_pct_styles(context_pct)
        print(
            "  "
            + _c("上下文:", "cyan")
            + f" {_c(f'{context_total:,}', 'blue')} / {_c(f'{context_max:,}', 'dim')} "
            + _c(f"({context_pct:.1f}%)", *pct_styles)
            + (
                f" {_c('|', 'dim')} {_c(f'source={context_source}', 'dim')}"
                if context_source
                else ""
            )
        )
    elif context_total > 0:
        print(
            "  "
            + _c("上下文:", "cyan")
            + f" {_c(f'{context_total:,} tokens', 'blue')}"
            + (
                f" {_c('|', 'dim')} {_c(f'source={context_source}', 'dim')}"
                if context_source
                else ""
            )
        )
    else:
        print("  " + _c("上下文:", "cyan") + " " + _c("(本轮无可用 context_stats)", "dim"))

    if context_breakdown:
        top_items = sorted(
            context_breakdown.items(),
            key=lambda kv: _to_int(kv[1], 0),
            reverse=True,
        )[:6]
        parts = [_c(f"{k}: {_to_int(v, 0):,}", "dim") for k, v in top_items]
        print("  " + _c("上下文分项:", "cyan") + " " + _c(" | ", "dim").join(parts))


def _serialize_messages(messages: Any) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    if not isinstance(messages, list):
        return serialized
    for index, msg in enumerate(messages):
        serialized.append(
            {
                "index": index,
                "message_type": msg.__class__.__name__,
                "content": str(getattr(msg, "content", msg) or ""),
                "additional_kwargs": dict(getattr(msg, "additional_kwargs", {}) or {}),
            }
        )
    return serialized


class _DebugIOHook:
    """记录模型完整输入输出到日志。"""

    def __init__(self, log_resolver: Callable[[str], Optional[Path]]) -> None:
        self._log_resolver = log_resolver

    def _write(self, thread_id: str, record_type: str, payload: Dict[str, Any]) -> None:
        log_path = self._log_resolver(thread_id)
        if not log_path:
            return
        _append_log(log_path, record_type, payload)

    def before_prompt(self, context, messages):  # noqa: ANN001
        self._write(
            context.thread_id,
            "hook_before_prompt",
            {
                "iteration": int(getattr(context, "iteration", 0) or 0),
                "thread_id": context.thread_id,
                "messages": _serialize_messages(messages),
            },
        )
        return None

    def after_model(self, context, model_output):  # noqa: ANN001
        self._write(
            context.thread_id,
            "hook_after_model",
            {
                "iteration": int(getattr(context, "iteration", 0) or 0),
                "thread_id": context.thread_id,
                "model_output": dict(model_output or {}),
            },
        )
        return None

    def after_tools(self, context, tool_result, executed_tools):  # noqa: ANN001
        tool_result_dict = dict(tool_result or {})
        full_messages = _serialize_messages(tool_result_dict.get("messages", []))
        self._write(
            context.thread_id,
            "hook_after_tools",
            {
                "iteration": int(getattr(context, "iteration", 0) or 0),
                "thread_id": context.thread_id,
                "executed_tools": list(executed_tools or []),
                "tool_result": {
                    **tool_result_dict,
                    "messages": full_messages,
                },
            },
        )
        _print_apply_patch_feedback(
            executed_tools=list(executed_tools or []),
            full_messages=full_messages,
        )
        return None


def _resolve_active_log_path(thread_id: str) -> Optional[Path]:
    return ACTIVE_LOG_PATH_BY_THREAD.get(str(thread_id or "").strip())


def _run_stream_with_full_logging(
    agent,
    *,
    task: str,
    thread_id: str,
    system_prompt_append: str = "",
    stage: str,
) -> dict:
    """
    运行终端流并记录完整输入输出事件。

    - 日志文件命名：{thread_id}_{timestamp}.log
    - 每条 invoke 事件都按 JSONL 落盘，便于完整回放。
    """
    log_path = _build_log_path(thread_id)
    _append_log(
        log_path,
        "invoke_start",
        {
            "stage": stage,
            "thread_id": thread_id,
            "task": task,
            "system_prompt_append": system_prompt_append,
        },
    )

    round_metrics: Dict[str, List[Dict[str, Any]]] = {
        "token_usage_events": [],
        "context_stats_events": [],
    }

    original_invoke = agent.invoke
    ACTIVE_LOG_PATH_BY_THREAD[thread_id] = log_path

    def _logged_invoke(
        task: str,
        thread_id: str = "default",
        thinking_context: bool = False,
        system_prompt_append: str = "",
    ) -> Iterator[Dict[str, Any]]:
        _append_log(
            log_path,
            "invoke_input",
            {
                "thread_id": thread_id,
                "task": task,
                "thinking_context": bool(thinking_context),
                "system_prompt_append": system_prompt_append,
            },
        )
        for event in original_invoke(
            task=task,
            thread_id=thread_id,
            thinking_context=thinking_context,
            system_prompt_append=system_prompt_append,
        ):
            if isinstance(event, dict):
                etype = str(event.get("type", "") or "").strip()
                if etype == "token_usage":
                    round_metrics["token_usage_events"].append(dict(event))
                elif etype == "context_stats":
                    round_metrics["context_stats_events"].append(dict(event))
            _append_log(log_path, "event", {"thread_id": thread_id, "event": event})
            yield event

    agent.invoke = _logged_invoke
    try:
        print(f"[调试] 全量日志: {log_path}")
        result = agent.run_terminal_agent_stream(
            task=task,
            thread_id=thread_id,
            system_prompt_append=system_prompt_append,
        )
        result = dict(result or {})
        usage_summary = _build_round_usage_summary(result, round_metrics)
        result["_round_metrics"] = round_metrics
        result["_round_usage"] = usage_summary
        _print_round_usage_summary(thread_id, usage_summary)
        _append_log(
            log_path,
            "invoke_done",
            {
                "stage": stage,
                "thread_id": thread_id,
                "result": result,
            },
        )
        return result
    except Exception as exc:
        _append_log(
            log_path,
            "invoke_error",
            {
                "stage": stage,
                "thread_id": thread_id,
                "error": str(exc),
            },
        )
        raise
    finally:
        agent.invoke = original_invoke
        ACTIVE_LOG_PATH_BY_THREAD.pop(thread_id, None)


def _extract_plan_request(user_input: str) -> tuple[str, bool]:
    """显式规划入口。默认直接执行，只有 /plan|plan:|spec: 才进入规划。"""
    stripped = str(user_input or "").strip()
    lowered = stripped.lower()
    if lowered in {"/plan", "plan", "spec"}:
        return "", True

    for prefix in ("/plan ", "plan:", "spec:"):
        if lowered.startswith(prefix):
            return stripped[len(prefix) :].strip(), True

    return stripped, False


def _build_selected_model():
    provider_or_alias = str(MODEL_PROVIDER_OR_ALIAS or "").strip()
    model_name = str(MODEL_NAME or "").strip() or None
    api_key = str(MODEL_API_KEY or "").strip() or None
    model_kwargs = dict(MODEL_KWARGS or {})

    if model_name:
        return setup_model(provider_or_alias, model_name, api_key, **model_kwargs)
    if api_key:
        return setup_model(provider_or_alias, api_key=api_key, **model_kwargs)
    return setup_model(provider_or_alias, **model_kwargs)


def _build_agent(working_dir: str):
    model = _build_selected_model()
    agent = create_agent(
        working_dir=working_dir,
        model=model,
        memory_dir=working_dir,
        load_project_docs=False,
    )
    if hasattr(agent, "add_tools"):
        pass
        # agent.add_tools("group:terminal")
        # agent.add_tools("group:tui")
    mcp_config = Path(working_dir) / ".klynx" / "mcp_servers.json"
    if mcp_config.exists():
        agent.add_mcp(str(mcp_config))

    if hasattr(agent, "add_hook"):
        agent.add_hook(_DebugIOHook(_resolve_active_log_path))
    return agent


def _copy_planning_context(agent, source_thread_id: str, target_thread_id: str) -> None:
    current_state = agent.app.get_state(
        {"configurable": {"thread_id": source_thread_id}}
    )
    if not current_state or not current_state.values:
        return

    state_to_copy = {
        "messages": current_state.values.get("messages", []),
        "overall_goal": current_state.values.get("overall_goal", ""),
        "context_summary": current_state.values.get("context_summary", ""),
    }
    agent.app.update_state(
        {"configurable": {"thread_id": target_thread_id}}, state_to_copy
    )


def do_planning(agent, user_input: str, working_dir: str, thread_id: str) -> str:
    """生成 Spec/spec.md，用户确认后返回 spec 内容。"""
    spec_path = Path(working_dir) / "Spec" / "spec.md"
    planner_task = f"""
用户有一个新的项目需求: "{user_input}"

请你作为架构师和需求规划师，分析这个需求，并在当前目录下执行：
1. 检查是否存在 `Spec` 文件夹，不存在则创建。
2. 在 `Spec` 文件夹下创建 `spec.md`。
3. 在 `spec.md` 中写下详细的实现规划、功能列表和技术细节。

完成后告诉我你已经写好了。
""".strip()

    print("\n>>> [规划阶段] 生成设计文档...")
    _run_stream_with_full_logging(
        agent,
        task=planner_task,
        thread_id=thread_id,
        stage="planning",
    )

    if not spec_path.exists():
        print(f"\n[Error] 未生成 {spec_path}")
        return ""

    print("\n" + "#" * 60)
    print(f"[人工确认] 已生成: {spec_path}")
    print("输入 'y' 继续执行，输入 'n' 取消。")

    while True:
        user_confirm = input("是否继续执行？(y/n) > ").strip().lower()
        if user_confirm in {"", "y", "yes"}:
            return spec_path.read_text(encoding="utf-8")
        if user_confirm in {"n", "no"}:
            print("[系统] 规划已取消，等待下一条输入。\n")
            return ""


def do_execution(agent, task: str, thread_id: str) -> dict:
    """执行任务。"""
    print("\n>>> [执行阶段] Agent 开始工作...")
    return _run_stream_with_full_logging(
        agent,
        task=task,
        thread_id=thread_id,
        stage="execution",
    )


def _print_context(agent, thread_id: str) -> None:
    state_values = agent.get_context(thread_id)
    if not state_values:
        print(f"\n[上下文] Thread: {thread_id} (当前为空)\n")
        return

    msgs = state_values.get("messages", [])
    last_prompt_tokens = int(state_values.get("prompt_tokens", 0) or 0)
    total_tokens = int(state_values.get("total_tokens", 0) or 0)

    goal = state_values.get("overall_goal", "")
    print(f"\n[上下文] Thread: {thread_id}")
    print(f"  - 历史消息: {len(msgs)} 条")
    print(f"  - 最近一次 Prompt 用量: {last_prompt_tokens} Tokens")
    print(f"  - 累计模型用量: {total_tokens} Tokens")
    print(f"  - 当前目标: {goal if goal else '(无)'}\n")


def main():
    _ensure_debug_env()
    working_dir = os.getcwd()

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Actoviq Agent — 精简版计划/执行示例")
    print(f"工作目录: {working_dir}")
    print("=" * 60)

    try:
        agent = _build_agent(working_dir)
    except Exception as exc:
        print(f"[Error] 模型或 Agent 初始化失败: {exc}")
        return

    print("\n" + "=" * 60)
    print("      输入 '/plan 需求描述' 显式进入 Spec 规划流程")
    print("=" * 60 + "\n")

    thread_id = uuid.uuid4().hex[:8]
    round_count = 0
    total_tokens = 0

    while True:
        try:
            user_input = input(f"[轮次 {round_count + 1}] > ").strip()

            task_input, use_planning = _extract_plan_request(user_input)
            if not task_input.strip():
                if use_planning:
                    print("\n[系统] /plan 后需要补充需求描述\n")
                continue

            result = {}
            if use_planning:
                print("\n[规划] 使用显式 /plan 入口生成 Spec...")
                temp_plan_thread_id = f"plan_{uuid.uuid4().hex[:8]}"
                _copy_planning_context(agent, thread_id, temp_plan_thread_id)
                spec_content = do_planning(
                    agent, task_input, working_dir, temp_plan_thread_id
                )
                if spec_content:
                    executor_task = f"""
你是一个执行工程师。以下是我们在 `Spec/spec.md` 中确认的执行计划。

<spec_content>
{spec_content}
</spec_content>

请严格按照这份计划直接开始实现。
""".strip()
                    result = do_execution(
                        agent, task=executor_task, thread_id=thread_id
                    )
            else:
                result = do_execution(agent, task=task_input, thread_id=thread_id)

            round_count += 1
            if result:
                round_usage = dict(result.get("_round_usage", {}) or {})
                round_tokens = int(
                    round_usage.get(
                        "total_tokens",
                        result.get("total_tokens", 0),
                    )
                    or 0
                )
                round_prompt = int(
                    round_usage.get(
                        "prompt_tokens",
                        result.get("prompt_tokens", 0),
                    )
                    or 0
                )
                round_completion = int(
                    round_usage.get(
                        "completion_tokens",
                        result.get("completion_tokens", 0),
                    )
                    or 0
                )
                total_tokens += round_tokens
                print(f"\n[第 {round_count} 轮完成]")
                print(
                    f"本轮 Token: {round_tokens} (Prompt: {round_prompt}, Completion: {round_completion})"
                    f" | 累计 Token: {total_tokens}\n"
                )

        except KeyboardInterrupt:
            print(f"\n\n[系统] 用户中断，退出程序")
            print(f"[统计] 共 {round_count} 轮对话，累计 Token: {total_tokens}")
            break
        except EOFError:
            break


if __name__ == "__main__":
    main()


