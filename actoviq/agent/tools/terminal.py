"""Compatibility terminal manager built on top of InteractiveExecManager."""

from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional

from .interactive_exec import InteractiveExecManager


class TerminalManager:
    """
    Compatibility wrapper for legacy terminal tools.

    The legacy API keeps the named terminal/session abstraction, while the
    actual process lifecycle is delegated to InteractiveExecManager.
    """

    MARKER_PREFIX = "__KLYNX_CMD_DONE__"
    ERROR_KEYWORDS = ("traceback", "error", "exception", "failed", "fatal")

    def __init__(
        self,
        default_cwd: str = ".",
        interactive_exec_manager: Optional[InteractiveExecManager] = None,
    ):
        self.default_cwd = os.path.abspath(default_cwd)
        self.exec_manager = interactive_exec_manager or InteractiveExecManager(self.default_cwd)
        self.sessions: Dict[str, str] = {}
        self.output_buffers: Dict[str, List[str]] = {}
        self.command_states: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def _resolve_cwd(self, cwd: Optional[str]) -> str:
        base = self.default_cwd
        raw = str(cwd or "").strip()
        if not raw:
            return base
        target = raw if os.path.isabs(raw) else os.path.join(base, raw)
        resolved = os.path.abspath(target)
        if not os.path.isdir(resolved):
            raise FileNotFoundError(f"working directory does not exist: {resolved}")
        return resolved

    @staticmethod
    def _shell_command() -> str:
        return "cmd.exe" if os.name == "nt" else "/bin/bash"

    def _session_id(self, name: str) -> str:
        return str(self.sessions.get(str(name or "").strip(), "") or "").strip()

    def _append_output_chunk(self, name: str, body: str) -> None:
        payload = str(body or "")
        if not payload or payload == "()":
            return
        chunks = payload.splitlines(keepends=True)
        if not chunks:
            chunks = [payload]
        self.output_buffers.setdefault(name, []).extend(chunks)

    def _read_exec_payload(
        self,
        name: str,
        *,
        yield_time_ms: int = 0,
        max_output_chars: int = 64000,
    ) -> Dict[str, Any]:
        session_id = self._session_id(name)
        if not session_id:
            return {"error": f"terminal session '{name}' does not exist"}
        state = self.exec_manager.read_session_state(
            session_id,
            yield_time_ms=yield_time_ms,
            max_output_chars=max_output_chars,
        )
        if state.get("error"):
            self._cleanup_session(name)
            return state
        self._append_output_chunk(name, str(state.get("body", "") or ""))
        return state

    def create_terminal(self, name: str, cwd: Optional[str] = None) -> str:
        terminal_name = str(name or "").strip()
        if not terminal_name:
            return "<error>terminal name must not be empty</error>"
        if terminal_name in self.sessions:
            return f"<error>terminal session '{terminal_name}' already exists</error>"
        try:
            run_cwd = self._resolve_cwd(cwd)
            session = self.exec_manager.create_session(
                cmd=self._shell_command(),
                workdir=run_cwd,
                tty=True,
                timeout_ms=0,
                shell=False,
                login=False,
            )
            self.sessions[terminal_name] = session.session_id
            self.output_buffers[terminal_name] = []
            self.command_states[terminal_name] = {}
            self._read_exec_payload(terminal_name, yield_time_ms=120, max_output_chars=12000)
            return f"<success>terminal session '{terminal_name}' created (CWD: {run_cwd})</success>"
        except Exception as exc:
            return f"<error>failed to create terminal: {str(exc)}</error>"

    def _build_marker_line(self, op_id: str) -> str:
        if os.name == "nt":
            return f"echo {self.MARKER_PREFIX}{op_id}__=%ERRORLEVEL%"
        return f"printf '{self.MARKER_PREFIX}{op_id}__=%s\\n' $?"

    def run_command(self, name: str, command: str, op_id: Optional[str] = None) -> str:
        terminal_name = str(name or "").strip()
        command_text = str(command or "").strip()
        if not command_text:
            return "<error>command must not be empty</error>"
        session_id = self._session_id(terminal_name)
        if not session_id:
            return f"<error>terminal session '{terminal_name}' does not exist</error>"

        session = self.exec_manager.get_session(session_id)
        if session is None or not session.is_alive():
            self._cleanup_session(terminal_name)
            return f"<error>terminal session '{terminal_name}' has exited</error>"

        self._read_exec_payload(terminal_name, yield_time_ms=40, max_output_chars=24000)
        buffer_len = len(self.output_buffers.get(terminal_name, []))
        actual_op_id = str(op_id or f"cmd_{int(time.time() * 1000)}").strip()
        wrapped = command_text + "\n" + self._build_marker_line(actual_op_id) + "\n"
        result_xml = self.exec_manager.write_stdin(
            session_id=session_id,
            chars=wrapped,
            yield_time_ms=80,
            max_output_tokens=6000,
        )
        parsed = self._parse_terminal_payload(result_xml)
        if parsed.get("body"):
            self._append_output_chunk(terminal_name, str(parsed.get("body", "") or ""))
        self.command_states.setdefault(terminal_name, {})[actual_op_id] = {
            "op_id": actual_op_id,
            "command": command_text,
            "marker": f"{self.MARKER_PREFIX}{actual_op_id}__=",
            "start_index": buffer_len,
            "last_read_index": buffer_len,
            "status": "pending",
            "exit_code": None,
            "started_at": int(time.time()),
            "finished_at": None,
        }
        return (
            f'<terminal_command name="{terminal_name}" op_id="{actual_op_id}" status="pending">'
            "command sent; use read_terminal / wait_terminal_until to inspect output"
            "</terminal_command>"
        )

    def _resolve_target_command(self, name: str, op_id: str) -> Optional[Dict[str, Any]]:
        commands = self.command_states.get(name, {})
        if op_id and op_id in commands:
            return commands[op_id]
        active = [
            item for item in commands.values() if str(item.get("status", "")).lower() in {"pending", "running"}
        ]
        if active:
            active.sort(key=lambda item: int(item.get("started_at", 0) or 0))
            return active[-1]
        if commands:
            all_commands = list(commands.values())
            all_commands.sort(key=lambda item: int(item.get("started_at", 0) or 0))
            return all_commands[-1]
        return None

    def _read_command_status(
        self,
        name: str,
        command_state: Dict[str, Any],
        lines: int,
    ) -> str:
        buffer_lines = self.output_buffers.get(name, [])
        start_index = int(command_state.get("start_index", 0) or 0)
        last_read_index = int(command_state.get("last_read_index", start_index) or start_index)
        op_id = str(command_state.get("op_id", "") or "").strip()
        marker = str(command_state.get("marker", "") or "")
        command_slice = buffer_lines[start_index:]
        joined = "".join(command_slice)
        marker_match = re.search(re.escape(marker) + r"(-?\d+)", joined) if marker else None

        visible_lines: List[str] = []
        marker_index = None
        if marker_match is not None:
            exit_code = int(marker_match.group(1))
            command_state["exit_code"] = exit_code
            command_state["status"] = "completed" if exit_code == 0 else "failed"
            command_state["finished_at"] = int(time.time())
            for idx in range(start_index, len(buffer_lines)):
                if marker in buffer_lines[idx]:
                    marker_index = idx
                    break
        else:
            session = self.exec_manager.get_session(self._session_id(name))
            if session is None or not session.is_alive():
                command_state["status"] = "failed"
                command_state["exit_code"] = session.refresh_exit_code() if session is not None else 1
                command_state["finished_at"] = int(time.time())
            elif len(buffer_lines) > start_index:
                command_state["status"] = "running"
            else:
                command_state["status"] = "pending"

        end_index = marker_index if marker_index is not None else len(buffer_lines)
        if end_index > last_read_index:
            visible_lines = buffer_lines[last_read_index:end_index]
            command_state["last_read_index"] = end_index

        trimmed_lines = [line.rstrip("\r\n") for line in visible_lines if line.strip()]
        if lines > 0 and len(trimmed_lines) > lines:
            trimmed_lines = trimmed_lines[-lines:]
        payload = "\n".join(trimmed_lines).strip() or "()"
        status = str(command_state.get("status", "pending") or "pending")
        exit_code = command_state.get("exit_code")
        exit_attr = "" if exit_code is None else f' exit_code="{int(exit_code)}"'
        has_new_output = "true" if any(line.strip() for line in visible_lines) else "false"
        return (
            f'<terminal_output name="{name}" op_id="{op_id}" status="{status}"'
            f'{exit_attr} has_new_output="{has_new_output}">\n'
            f"{payload}\n"
            "</terminal_output>"
        )

    def read_terminal(self, name: str, lines: int = 50, op_id: Optional[str] = None) -> str:
        terminal_name = str(name or "").strip()
        if terminal_name not in self.sessions:
            return f"<error>terminal session '{terminal_name}' does not exist</error>"
        state = self._read_exec_payload(terminal_name, yield_time_ms=20, max_output_chars=64000)
        if state.get("error"):
            return f"<error>{state['error']}</error>"
        target = self._resolve_target_command(terminal_name, str(op_id or "").strip())
        if target is not None:
            return self._read_command_status(terminal_name, target, int(lines or 50))
        buffer_lines = self.output_buffers.get(terminal_name, [])
        tail = [line.rstrip("\r\n") for line in buffer_lines[-int(lines or 50):] if line.strip()]
        payload = "\n".join(tail).strip() if tail else "()"
        return (
            f'<terminal_output name="{terminal_name}" status="idle" has_new_output="{str(bool(tail)).lower()}">\n'
            f"{payload}\n"
            "</terminal_output>"
        )

    def read_terminal_since_last(self, name: str, lines: int = 50, op_id: Optional[str] = None) -> str:
        return self.read_terminal(name=name, lines=lines, op_id=op_id)

    def wait_terminal_until(
        self,
        name: str,
        op_id: Optional[str] = None,
        pattern: Optional[str] = None,
        timeout_ms: int = 30000,
        poll_interval_ms: int = 300,
    ) -> str:
        terminal_name = str(name or "").strip()
        timeout_ms = max(1, int(timeout_ms or 30000))
        poll_interval_ms = max(20, int(poll_interval_ms or 300))
        started = time.time()
        target_op_id = str(op_id or "").strip()
        matched_pattern = ""
        collected_chunks: List[str] = []
        target_pattern = str(pattern or "").strip()

        while (time.time() - started) * 1000 < timeout_ms:
            payload = self.read_terminal(name=terminal_name, lines=200, op_id=target_op_id)
            parsed = self._parse_terminal_payload(payload)
            if not target_op_id:
                target_op_id = str(parsed.get("op_id", "") or "").strip()

            body = str(parsed.get("body", "") or "").strip()
            if body and body != "()":
                collected_chunks.append(body)

            aggregated = "\n".join(chunk for chunk in collected_chunks if chunk).strip()
            status = str(parsed.get("status", "") or "pending").strip().lower()
            exit_code = str(parsed.get("exit_code", "") or "").strip()
            has_new_output = str(parsed.get("has_new_output", "") or "false").strip().lower()

            if target_pattern:
                try:
                    matched = bool(re.search(target_pattern, aggregated, re.IGNORECASE))
                except re.error:
                    matched = target_pattern.lower() in aggregated.lower()
                if matched:
                    matched_pattern = target_pattern
                    return (
                        f'<terminal_wait name="{terminal_name}" op_id="{target_op_id}" status="{status or "running"}" '
                        f'exit_code="{exit_code}" matched_pattern="{matched_pattern}" '
                        f'has_new_output="{has_new_output}" timed_out="false">\n'
                        f"{aggregated or '()'}\n"
                        "</terminal_wait>"
                    )

            lowered = aggregated.lower()
            if any(keyword in lowered for keyword in self.ERROR_KEYWORDS):
                return (
                    f'<terminal_wait name="{terminal_name}" op_id="{target_op_id}" status="{status or "running"}" '
                    f'exit_code="{exit_code}" matched_pattern="" has_new_output="{has_new_output}" timed_out="false">\n'
                    f"{aggregated or '()'}\n"
                    "</terminal_wait>"
                )

            if status in {"completed", "failed", "idle"}:
                return (
                    f'<terminal_wait name="{terminal_name}" op_id="{target_op_id}" status="{status}" '
                    f'exit_code="{exit_code}" matched_pattern="{matched_pattern}" '
                    f'has_new_output="{has_new_output}" timed_out="false">\n'
                    f"{aggregated or body or '()'}\n"
                    "</terminal_wait>"
                )

            time.sleep(poll_interval_ms / 1000.0)

        final_output = "\n".join(chunk for chunk in collected_chunks if chunk).strip() or "()"
        return (
            f'<terminal_wait name="{terminal_name}" op_id="{target_op_id}" status="timeout" '
            f'matched_pattern="{matched_pattern}" has_new_output="{str(bool(collected_chunks)).lower()}" '
            'timed_out="true">\n'
            f"{final_output}\n"
            "</terminal_wait>"
        )

    def run_and_wait(
        self,
        name: str,
        command: str,
        timeout_ms: int = 30000,
        pattern: Optional[str] = None,
        poll_interval_ms: int = 300,
    ) -> str:
        op_id = f"cmd_{int(time.time() * 1000)}"
        sent = self.run_command(name=name, command=command, op_id=op_id)
        if "<error>" in str(sent or "").lower():
            return sent
        return self.wait_terminal_until(
            name=name,
            op_id=op_id,
            pattern=pattern,
            timeout_ms=timeout_ms,
            poll_interval_ms=poll_interval_ms,
        )

    def _parse_terminal_payload(self, payload: str) -> Dict[str, Any]:
        text = str(payload or "")
        parsed: Dict[str, Any] = {"body": text.strip()}
        match = re.search(r"<terminal_(?:output|command|wait)\s+([^>]*)>", text)
        if not match:
            return parsed
        attrs = match.group(1)
        for key in ("name", "op_id", "status", "exit_code", "has_new_output", "matched_pattern", "timed_out"):
            attr_match = re.search(rf'{key}="([^"]*)"', attrs)
            if attr_match:
                parsed[key] = attr_match.group(1).strip()
        body_match = re.search(
            r"<terminal_(?:output|command|wait)[^>]*>\s*(.*?)\s*</terminal_(?:output|command|wait)>",
            text,
            re.DOTALL,
        )
        if body_match:
            parsed["body"] = body_match.group(1).strip()
        return parsed

    def _cleanup_session(self, name: str) -> None:
        self.sessions.pop(name, None)
        self.output_buffers.pop(name, None)
        self.command_states.pop(name, None)

    def close_terminal(self, name: str) -> None:
        terminal_name = str(name or "").strip()
        session_id = self._session_id(terminal_name)
        if session_id:
            self.exec_manager.close_exec_session(session_id)
        self._cleanup_session(terminal_name)
