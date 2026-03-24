"""Terminal stream renderer for Klynx agents."""

from __future__ import annotations

import os
import re
import sys
import time
from typing import Any, Dict


def _safe_print(*args, **kwargs) -> None:
    """Print safely in Windows consoles that may not support full UTF-8."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = " ".join(str(item) for item in args)
        encoding = sys.stdout.encoding or "utf-8"
        safe_text = text.encode(encoding, errors="replace").decode(encoding)
        print(
            safe_text,
            **{k: v for k, v in kwargs.items() if k not in {"end", "flush"}},
            end=kwargs.get("end", "\n"),
            flush=kwargs.get("flush", False),
        )


def _env_bool(env_key: str, default: bool = False) -> bool:
    text = str(os.getenv(env_key, "1" if default else "0") or "").strip().lower()
    if text in {"1", "true", "yes", "on", "y"}:
        return True
    if text in {"0", "false", "no", "off", "n"}:
        return False
    return default


def _env_float(env_key: str, default: float) -> float:
    raw = str(os.getenv(env_key, "") or "").strip()
    if not raw:
        return default
    try:
        return max(0.0, float(raw))
    except Exception:
        return default


def _env_int(env_key: str, default: int) -> int:
    raw = str(os.getenv(env_key, "") or "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except Exception:
        return default


def _request_agent_cancel(agent: Any) -> None:
    """Best-effort cancellation signal for invoke() graph thread."""
    cancel_event = getattr(agent, "_cancel_event", None)
    if cancel_event is None:
        return
    try:
        cancel_event.set()
    except Exception:
        return


def _find_first_tag(text: str, tags: tuple[str, ...]) -> tuple[int, str]:
    lowered = text.lower()
    first_pos = -1
    first_tag = ""
    for tag in tags:
        pos = lowered.find(tag)
        if pos >= 0 and (first_pos < 0 or pos < first_pos):
            first_pos = pos
            first_tag = tag
    return first_pos, first_tag


def _strip_think_blocks(text: str) -> str:
    raw = str(text or "")
    cleaned = re.sub(
        r"(?is)<think>\s*.*?\s*</think>\s*|<thinking>\s*.*?\s*</thinking>\s*",
        "",
        raw,
    )
    return cleaned.strip()


class _ThinkingBlockStripper:
    """Stateful stripper for <think>/<thinking> blocks in streamed answer text."""

    _OPEN_TAGS = ("<think>", "<thinking>")
    _CLOSE_TAGS = ("</think>", "</thinking>")
    _CARRY_MAX = max(len(tag) for tag in (_OPEN_TAGS + _CLOSE_TAGS)) - 1

    def __init__(self) -> None:
        self._buffer = ""
        self._inside_block = False

    def feed(self, text: str) -> str:
        if not text:
            return ""
        self._buffer += str(text)
        output: list[str] = []

        while self._buffer:
            if self._inside_block:
                close_pos, close_tag = _find_first_tag(self._buffer, self._CLOSE_TAGS)
                if close_pos < 0:
                    if len(self._buffer) > self._CARRY_MAX:
                        self._buffer = self._buffer[-self._CARRY_MAX :]
                    return "".join(output)
                self._buffer = self._buffer[close_pos + len(close_tag) :]
                self._inside_block = False
                continue

            open_pos, open_tag = _find_first_tag(self._buffer, self._OPEN_TAGS)
            if open_pos < 0:
                if len(self._buffer) <= self._CARRY_MAX:
                    return "".join(output)
                output.append(self._buffer[:-self._CARRY_MAX])
                self._buffer = self._buffer[-self._CARRY_MAX :]
                return "".join(output)

            if open_pos > 0:
                output.append(self._buffer[:open_pos])
            self._buffer = self._buffer[open_pos + len(open_tag) :]
            self._inside_block = True

        return "".join(output)

    def flush(self) -> str:
        if self._inside_block:
            self._buffer = ""
            return ""
        tail = self._buffer
        self._buffer = ""
        return tail


class _ThrottledWriter:
    """Buffer token output briefly to reduce per-token flush overhead."""

    def __init__(self, flush_interval_s: float, flush_chars: int):
        self.flush_interval_s = max(0.0, flush_interval_s)
        self.flush_chars = max(1, flush_chars)
        self._parts: list[str] = []
        self._char_count = 0
        self._last_flush = time.monotonic()

    def write(self, text: str) -> None:
        if not text:
            return
        chunk = str(text)
        self._parts.append(chunk)
        self._char_count += len(chunk)
        now = time.monotonic()
        should_flush = (
            "\n" in chunk
            or self._char_count >= self.flush_chars
            or (now - self._last_flush) >= self.flush_interval_s
        )
        if should_flush:
            self.flush()

    def flush(self, force: bool = False) -> None:
        if not self._parts:
            if force:
                try:
                    sys.stdout.flush()
                except Exception:
                    pass
            return

        payload = "".join(self._parts)
        self._parts.clear()
        self._char_count = 0
        self._last_flush = time.monotonic()

        encoding = sys.stdout.encoding or "utf-8"
        try:
            sys.stdout.write(payload)
        except UnicodeEncodeError:
            safe_payload = payload.encode(encoding, errors="replace").decode(encoding)
            sys.stdout.write(safe_payload)
        sys.stdout.flush()


def _extract_xml_attr(text: str, attr: str) -> str:
    match = re.search(rf'{attr}="([^"]*)"', text)
    return match.group(1).strip() if match else ""


def _parse_terminal_payload(content: str) -> Dict[str, Any]:
    text = str(content or "")
    tag_match = re.search(r"<terminal_(output|command|wait)\s+[^>]*>", text)
    if not tag_match:
        return {}

    parsed: Dict[str, Any] = {
        "tag": tag_match.group(1),
        "name": _extract_xml_attr(text, "name"),
        "op_id": _extract_xml_attr(text, "op_id"),
        "status": _extract_xml_attr(text, "status").lower(),
        "matched_pattern": _extract_xml_attr(text, "matched_pattern"),
        "timed_out": _extract_xml_attr(text, "timed_out").lower() in {"1", "true", "yes", "on"},
        "has_new_output": _extract_xml_attr(text, "has_new_output").lower() in {"1", "true", "yes", "on"},
    }
    body_match = re.search(
        r"<terminal_(?:output|command|wait)[^>]*>\s*(.*?)\s*</terminal_(?:output|command|wait)>",
        text,
        re.DOTALL,
    )
    parsed["body"] = body_match.group(1).strip() if body_match else ""
    return parsed


def _format_terminal_wait_message(parsed: Dict[str, Any]) -> str:
    if not parsed:
        return ""

    status = str(parsed.get("status", "") or "").strip().lower()
    tag = str(parsed.get("tag", "") or "").strip().lower()
    name = str(parsed.get("name", "") or "").strip() or "default"
    op_id = str(parsed.get("op_id", "") or "").strip()
    has_new_output = bool(parsed.get("has_new_output", False))
    matched_pattern = str(parsed.get("matched_pattern", "") or "").strip()
    body = str(parsed.get("body", "") or "").strip()
    op_suffix = f" (op_id={op_id})" if op_id else ""

    if tag == "command" and status in {"pending", "running"}:
        return f"[终端] {name}{op_suffix} 命令已发出，正在后台运行，等待后续输出。"
    if status in {"pending", "running"}:
        if matched_pattern:
            return f"[终端] {name}{op_suffix} 正在等待匹配输出: {matched_pattern}"
        if has_new_output and body and body != "(无新输出)":
            return f"[终端] {name}{op_suffix} 收到新输出，但命令仍在运行，继续等待。"
        return f"[终端] {name}{op_suffix} 仍在运行，等待终端输出或命令完成。"
    if status == "timeout" or bool(parsed.get("timed_out", False)):
        return f"[终端] {name}{op_suffix} 等待超时。"
    return ""


def run_terminal_agent_stream(
    agent,
    task: str,
    thread_id: str,
    system_prompt_append: str = "",
) -> dict:
    """Run agent.invoke() and render compact terminal output."""
    verbose = _env_bool("KLYNX_STREAM_VERBOSE", default=False)
    show_reasoning = _env_bool("KLYNX_STREAM_SHOW_REASONING", default=False)
    flush_interval_s = _env_float("KLYNX_STREAM_FLUSH_INTERVAL_S", default=0.05)
    flush_chars = _env_int("KLYNX_STREAM_FLUSH_CHARS", default=64)

    writer = _ThrottledWriter(flush_interval_s=flush_interval_s, flush_chars=flush_chars)
    result: Dict[str, Any] = {}
    current_tool_name = ""
    stream_open = False
    stream_kind = ""
    printed_answer_header = False
    printed_reasoning_header = False
    has_streamed_answer = False
    has_streamed_reasoning = False
    answer_stripper = _ThinkingBlockStripper()

    def _close_stream_line() -> None:
        nonlocal stream_open, stream_kind
        if not stream_open:
            return
        if stream_kind == "answer":
            remaining = answer_stripper.flush()
            if remaining:
                writer.write(remaining)
        writer.flush(force=True)
        _safe_print()
        stream_open = False
        stream_kind = ""

    def _open_stream(kind: str) -> bool:
        nonlocal stream_open, stream_kind, printed_answer_header, printed_reasoning_header
        if kind == "reasoning" and not show_reasoning:
            return False
        if stream_open and stream_kind == kind:
            return True
        _close_stream_line()
        if kind == "answer" and not printed_answer_header:
            _safe_print("[回答] ", end="", flush=False)
            printed_answer_header = True
        elif kind == "reasoning" and not printed_reasoning_header:
            _safe_print("[思考] ", end="", flush=False)
            printed_reasoning_header = True
        else:
            _safe_print("", end="", flush=False)
        stream_open = True
        stream_kind = kind
        return True

    _safe_print("\n" + "=" * 60)
    _safe_print(f"[Agent] (Thread: {thread_id})")
    _safe_print("-" * 60)

    try:
        for event in agent.invoke(
            task=task,
            thread_id=thread_id,
            system_prompt_append=system_prompt_append,
        ):
            etype = str(event.get("type", "") or "")
            content = event.get("content", "")

            if etype == "done":
                _close_stream_line()
                result = dict(event)
                continue

            if etype == "token":
                if _open_stream("answer"):
                    cleaned_chunk = answer_stripper.feed(str(content or ""))
                    if cleaned_chunk:
                        writer.write(cleaned_chunk)
                    has_streamed_answer = True
                continue

            if etype == "reasoning_token":
                if _open_stream("reasoning"):
                    writer.write(str(content or ""))
                    has_streamed_reasoning = True
                continue

            # 非流式事件：先结束当前流式行，避免输出交错。
            _close_stream_line()

            if etype == "reasoning":
                if show_reasoning and not has_streamed_reasoning:
                    _safe_print(f"[思考] {content}")
                continue
            if etype == "answer":
                if not has_streamed_answer:
                    rendered = _strip_think_blocks(str(content or ""))
                    if rendered:
                        _safe_print(f"[回答] {rendered}")
                continue
            if etype == "summary":
                _safe_print(f"[总结] {content}")
                continue
            if etype == "tool_exec":
                _safe_print(content)
                tool_match = re.match(r"\[(?:工具|Tool)\s+\d+/\d+\]\s+(.+)$", str(content or "").strip())
                if tool_match:
                    current_tool_name = tool_match.group(1).strip()
                    if current_tool_name in {"wait_terminal_until", "run_and_wait", "write_stdin"}:
                        _safe_print("  [终端] 正在等待终端输出或命令完成，这一步可能需要几秒钟。")
                    elif current_tool_name in {"run_in_terminal", "exec_command", "launch_interactive_session"}:
                        _safe_print("  [终端] 正在启动后台终端命令，后续可继续读取或等待输出。")
                continue
            if etype == "tool_result":
                is_terminal_tool = current_tool_name in {
                    "run_in_terminal",
                    "exec_command",
                    "write_stdin",
                    "launch_interactive_session",
                    "close_exec_session",
                    "read_terminal",
                    "read_terminal_since_last",
                    "wait_terminal_until",
                    "run_and_wait",
                }
                parsed = _parse_terminal_payload(str(content or "")) if is_terminal_tool else {}
                terminal_wait_message = _format_terminal_wait_message(parsed)
                if terminal_wait_message:
                    _safe_print(f"  {terminal_wait_message}")
                if verbose:
                    _safe_print(f"  结果:\n{content}")
                continue
            if etype == "tool_calls":
                _safe_print(content)
                continue
            if etype == "iteration":
                if verbose:
                    _safe_print(content)
                continue
            if etype == "token_usage":
                if verbose:
                    _safe_print(content)
                continue
            if etype == "context_stats":
                if verbose:
                    _safe_print(content)
                continue
            if etype == "routing":
                if verbose:
                    _safe_print(content)
                continue
            if etype == "complete":
                _safe_print(content)
                continue
            if etype == "warning":
                _safe_print(f"[Warning] {content}")
                continue
            if etype == "error":
                _safe_print(f"[Error] {content}")
                continue
            if etype == "info":
                if verbose:
                    text = str(content or "").strip()
                    if text and text not in {"[]", "[] ..."}:
                        _safe_print(text)
                continue

    except KeyboardInterrupt:
        _request_agent_cancel(agent)
        _close_stream_line()
        _safe_print("\n[SYS] Agent 已中断。")
    except Exception as exc:
        _close_stream_line()
        _safe_print(f"\n[Error] 渲染失败: {exc}")

    _close_stream_line()
    _safe_print("-" * 60)
    round_tokens = int(result.get("total_tokens", 0) or 0)
    task_completed = bool(result.get("task_completed", False))
    _safe_print(f"[Agent] 完成: {task_completed} | Token: {round_tokens}")
    _safe_print("=" * 60 + "\n")
    return result


def run_terminal_ask_stream(
    agent,
    message: str,
    system_prompt: str = None,
    thread_id: str = "default",
) -> str:
    """Run agent.ask() and render compact streaming output."""
    show_reasoning = _env_bool("KLYNX_STREAM_SHOW_REASONING", default=False)
    flush_interval_s = _env_float("KLYNX_STREAM_FLUSH_INTERVAL_S", default=0.05)
    flush_chars = _env_int("KLYNX_STREAM_FLUSH_CHARS", default=64)
    writer = _ThrottledWriter(flush_interval_s=flush_interval_s, flush_chars=flush_chars)

    full_answer = ""
    stream_open = False
    stream_kind = ""
    printed_answer_header = False
    printed_reasoning_header = False
    answer_stripper = _ThinkingBlockStripper()

    def _close_stream_line() -> None:
        nonlocal stream_open, stream_kind
        if not stream_open:
            return
        if stream_kind == "answer":
            remaining = answer_stripper.flush()
            if remaining:
                writer.write(remaining)
        writer.flush(force=True)
        _safe_print()
        stream_open = False
        stream_kind = ""

    def _open_stream(kind: str) -> bool:
        nonlocal stream_open, stream_kind, printed_answer_header, printed_reasoning_header
        if kind == "reasoning" and not show_reasoning:
            return False
        if stream_open and stream_kind == kind:
            return True
        _close_stream_line()
        if kind == "answer" and not printed_answer_header:
            _safe_print("[回答] ", end="", flush=False)
            printed_answer_header = True
        elif kind == "reasoning" and not printed_reasoning_header:
            _safe_print("[思考] ", end="", flush=False)
            printed_reasoning_header = True
        stream_open = True
        stream_kind = kind
        return True

    try:
        for event in agent.ask(message=message, system_prompt=system_prompt, thread_id=thread_id):
            etype = str(event.get("type", "") or "")
            content = event.get("content", "")

            if etype == "token":
                if _open_stream("answer"):
                    cleaned_chunk = answer_stripper.feed(str(content or ""))
                    if cleaned_chunk:
                        writer.write(cleaned_chunk)
                continue

            if etype == "reasoning_token":
                if _open_stream("reasoning"):
                    writer.write(str(content or ""))
                continue

            _close_stream_line()

            if etype == "reasoning":
                if show_reasoning:
                    _safe_print(f"[思考] {content}")
                continue
            if etype == "answer":
                rendered = _strip_think_blocks(str(content or ""))
                if rendered:
                    _safe_print(f"[回答] {rendered}")
                continue
            if etype == "error":
                _safe_print(f"[Error] {content}")
                continue
            if etype == "done":
                full_answer = str(event.get("answer", "") or "")
    except KeyboardInterrupt:
        _request_agent_cancel(agent)
        _close_stream_line()
        _safe_print("\n[SYS] Ask 已中断。")

    _close_stream_line()
    return full_answer
