"""Interactive command execution with shared PTY/pipe session management."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

if sys.platform == "win32":
    try:
        from winpty import PtyProcess
    except ImportError:
        PtyProcess = None
else:
    try:
        from ptyprocess import PtyProcessUnicode
    except ImportError:
        PtyProcessUnicode = None


_READ_POLL_SECONDS = 0.05
_DEFAULT_ROWS = 24
_DEFAULT_COLS = 80


def _escape_xml_attr(text: Any) -> str:
    return (
        str(text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


@dataclass
class InteractiveSession:
    session_id: str
    command: str
    cwd: str
    tty: bool
    shell: bool
    login: bool
    timeout_ms: int
    transport: str
    process: Any
    started_at: int
    rows: int = _DEFAULT_ROWS
    cols: int = _DEFAULT_COLS
    _buffer_parts: List[str] = field(default_factory=list)
    _buffer_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _read_offset: int = 0
    _alive: bool = True
    exit_code: Optional[int] = None
    timed_out: bool = False
    last_output_at: int = 0

    def append_output(self, text: str) -> None:
        if not text:
            return
        with self._buffer_lock:
            self._buffer_parts.append(str(text))
            self.last_output_at = int(time.time() * 1000)

    def has_output(self) -> bool:
        with self._buffer_lock:
            return bool(self._buffer_parts)

    def snapshot_new(self, max_chars: int) -> Tuple[str, bool]:
        with self._buffer_lock:
            buffer_text = "".join(self._buffer_parts)
            start = max(0, min(self._read_offset, len(buffer_text)))
            chunk = buffer_text[start:]
            self._read_offset = len(buffer_text)
        return self._limit_text(chunk, max_chars), bool(chunk)

    def snapshot_tail(self, max_chars: int) -> str:
        with self._buffer_lock:
            buffer_text = "".join(self._buffer_parts)
        return self._limit_text(buffer_text, max_chars)

    @staticmethod
    def _limit_text(text: str, max_chars: int) -> str:
        raw = str(text or "")
        if max_chars <= 0 or len(raw) <= max_chars:
            return raw
        return raw[-max_chars:]

    def refresh_exit_code(self) -> Optional[int]:
        if self.exit_code is not None:
            return self.exit_code
        if self.tty:
            for attr in ("exitstatus", "returncode", "status"):
                value = getattr(self.process, attr, None)
                if isinstance(value, int):
                    self.exit_code = value
                    return self.exit_code
            return self.exit_code
        try:
            self.exit_code = self.process.poll()
        except Exception:
            self.exit_code = None
        return self.exit_code

    def is_alive(self) -> bool:
        if self.timed_out:
            return False
        if not self._alive:
            return False
        if self.tty:
            try:
                alive = bool(self.process.isalive())
            except Exception:
                alive = False
        else:
            try:
                alive = self.process.poll() is None
            except Exception:
                alive = False
        if not alive:
            self._alive = False
            self.refresh_exit_code()
        return alive

    def mark_reader_done(self) -> None:
        self._alive = False
        self.refresh_exit_code()

    def write(self, chars: str) -> Tuple[bool, str]:
        if not self.is_alive():
            return False, "interactive session is not running"
        try:
            if self.tty:
                self.process.write(chars)
            else:
                if getattr(self.process, "stdin", None) is None:
                    return False, "session does not support stdin"
                self.process.stdin.write(chars)
                self.process.stdin.flush()
            return True, ""
        except Exception as exc:
            return False, str(exc)

    def close(self) -> None:
        self._alive = False
        try:
            if self.tty:
                self.process.terminate(force=True)
            else:
                if self.process.poll() is None:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=0.5)
                    except Exception:
                        self.process.kill()
        except Exception:
            pass
        self.refresh_exit_code()


class InteractiveExecManager:
    """Shared manager for PTY and pipe-backed interactive sessions."""

    def __init__(self, default_cwd: str = "."):
        self.default_cwd = os.path.abspath(default_cwd)
        self.sessions: Dict[str, InteractiveSession] = {}
        self.lock = threading.Lock()

    def _resolve_cwd(self, workdir: Optional[str]) -> str:
        base = Path(self.default_cwd)
        target = Path(workdir).expanduser() if workdir else base
        if not target.is_absolute():
            target = base / target
        resolved = str(target.resolve())
        if not os.path.isdir(resolved):
            raise FileNotFoundError(f"working directory does not exist: {resolved}")
        return resolved

    def _build_spawn_spec(
        self,
        command: str,
        *,
        tty: bool,
        shell: bool,
        login: bool,
    ) -> Union[str, List[str]]:
        raw_command = str(command or "").strip()
        if shell:
            if os.name == "nt":
                wrapped_command = self._wrap_powershell_utf8_command(raw_command)
                args = [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    wrapped_command,
                ]
                return subprocess.list2cmdline(args) if tty else args
            shell_flag = "-lc" if login else "-c"
            return ["/bin/bash", shell_flag, raw_command]

        if tty:
            if os.name == "nt":
                return raw_command
            return shlex.split(raw_command)

        if os.name == "nt":
            return raw_command
        return shlex.split(raw_command)

    @staticmethod
    def _wrap_powershell_utf8_command(command: str) -> str:
        """Ensure Windows PowerShell interactive sessions output UTF-8 text."""
        raw = str(command or "").strip()
        prefix = (
            "$OutputEncoding=[System.Text.UTF8Encoding]::new($false); "
            "[Console]::InputEncoding=[System.Text.UTF8Encoding]::new($false); "
            "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); "
            "$PSDefaultParameterValues['Out-File:Encoding']='utf8'; "
            "$PSDefaultParameterValues['Set-Content:Encoding']='utf8'; "
            "$PSDefaultParameterValues['Add-Content:Encoding']='utf8'; "
        )
        return f"{prefix}{raw}" if raw else prefix

    @staticmethod
    def _max_output_chars(max_output_tokens: int) -> int:
        try:
            tokens = int(max_output_tokens or 0)
        except Exception:
            tokens = 0
        tokens = max(tokens, 128)
        return min(tokens * 4, 24000)

    def _spawn_session(
        self,
        *,
        command: str,
        cwd: str,
        tty: bool,
        shell: bool,
        login: bool,
        timeout_ms: int,
        rows: int,
        cols: int,
    ) -> InteractiveSession:
        session_id = f"exec_{uuid.uuid4().hex[:10]}"
        started_at = int(time.time() * 1000)
        spec = self._build_spawn_spec(command, tty=tty, shell=shell, login=login)

        if tty:
            if sys.platform == "win32":
                if PtyProcess is None:
                    raise RuntimeError("pywinpty is required for PTY sessions on Windows")
                process = PtyProcess.spawn(
                    str(spec),
                    dimensions=(rows, cols),
                    cwd=cwd,
                )
            else:
                if PtyProcessUnicode is None:
                    raise RuntimeError("ptyprocess is required for PTY sessions on Unix-like platforms")
                argv = spec if isinstance(spec, list) else shlex.split(str(spec))
                process = PtyProcessUnicode.spawn(
                    argv,
                    dimensions=(rows, cols),
                    cwd=cwd,
                )
        else:
            process = subprocess.Popen(
                spec,
                shell=False,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=cwd,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

        session = InteractiveSession(
            session_id=session_id,
            command=str(command or "").strip(),
            cwd=cwd,
            tty=tty,
            shell=shell,
            login=login,
            timeout_ms=max(0, int(timeout_ms or 0)),
            transport="pty" if tty else "pipe",
            process=process,
            started_at=started_at,
            rows=rows,
            cols=cols,
        )
        reader = threading.Thread(
            target=self._read_session_output,
            args=(session,),
            daemon=True,
        )
        reader.start()
        return session

    def create_session(
        self,
        *,
        cmd: str,
        workdir: Optional[str] = None,
        tty: bool = False,
        timeout_ms: int = 0,
        shell: bool = True,
        login: bool = True,
        rows: int = _DEFAULT_ROWS,
        cols: int = _DEFAULT_COLS,
    ) -> InteractiveSession:
        command = str(cmd or "").strip()
        if not command:
            raise ValueError("exec command must not be empty")
        cwd = self._resolve_cwd(workdir)
        session = self._spawn_session(
            command=command,
            cwd=cwd,
            tty=bool(tty),
            shell=bool(shell),
            login=bool(login),
            timeout_ms=int(timeout_ms or 0),
            rows=max(2, int(rows or _DEFAULT_ROWS)),
            cols=max(2, int(cols or _DEFAULT_COLS)),
        )
        with self.lock:
            self.sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[InteractiveSession]:
        with self.lock:
            return self.sessions.get(str(session_id or "").strip())

    def _read_session_output(self, session: InteractiveSession) -> None:
        try:
            if session.tty:
                while session._alive:
                    try:
                        data = session.process.read(4096)
                        if data:
                            session.append_output(data)
                            continue
                    except EOFError:
                        break
                    except Exception:
                        if not session._alive:
                            break
                    if not session.is_alive():
                        break
                    time.sleep(_READ_POLL_SECONDS)
            else:
                stdout = getattr(session.process, "stdout", None)
                if stdout is None:
                    return
                while True:
                    line = stdout.readline()
                    if line:
                        session.append_output(line)
                        continue
                    if session.process.poll() is not None:
                        remainder = stdout.read()
                        if remainder:
                            session.append_output(remainder)
                        break
                    time.sleep(_READ_POLL_SECONDS)
        finally:
            session.mark_reader_done()

    def _enforce_timeout(self, session: InteractiveSession) -> None:
        if session.timed_out or session.timeout_ms <= 0:
            return
        now_ms = int(time.time() * 1000)
        if now_ms - int(session.started_at or 0) < session.timeout_ms:
            return
        if not session.is_alive():
            return
        session.timed_out = True
        session.append_output("\n[interactive session timed out]\n")
        session.close()
        if session.exit_code is None:
            session.exit_code = -1

    def _session_status(self, session: InteractiveSession) -> str:
        self._enforce_timeout(session)
        if session.timed_out:
            return "timeout"
        if session.is_alive():
            return "running" if session.has_output() else "pending"
        exit_code = session.refresh_exit_code()
        if exit_code is None:
            return "completed"
        return "completed" if int(exit_code) == 0 else "failed"

    def read_session_state(
        self,
        session_id: str,
        *,
        yield_time_ms: int = 0,
        max_output_chars: int = 0,
    ) -> Dict[str, Any]:
        session = self.get_session(session_id)
        if session is None:
            return {"error": f"interactive session missing: {str(session_id or '').strip()}"}
        wait_ms = max(0, int(yield_time_ms or 0))
        deadline = time.time() + (wait_ms / 1000.0)
        while time.time() < deadline:
            self._enforce_timeout(session)
            if not session.is_alive():
                break
            time.sleep(min(_READ_POLL_SECONDS, max(0.0, deadline - time.time())))

        if not session.is_alive():
            time.sleep(_READ_POLL_SECONDS)

        status = self._session_status(session)
        exit_code = session.refresh_exit_code()
        limit = int(max_output_chars or 0)
        body, has_new_output = session.snapshot_new(limit if limit > 0 else 10 ** 9)
        return {
            "session_id": session.session_id,
            "status": status,
            "exit_code": exit_code,
            "body": body or "",
            "has_new_output": bool(has_new_output),
            "timed_out": bool(session.timed_out),
            "transport": session.transport,
            "rows": session.rows,
            "cols": session.cols,
        }

    def _yield_session(
        self,
        session: InteractiveSession,
        *,
        yield_time_ms: int,
        max_output_tokens: int,
    ) -> str:
        state = self.read_session_state(
            session.session_id,
            yield_time_ms=yield_time_ms,
            max_output_chars=self._max_output_chars(max_output_tokens),
        )
        if state.get("error"):
            return f"<error>{state['error']}</error>"
        status = str(state.get("status", "") or "")
        exit_code = state.get("exit_code")
        body = str(state.get("body", "") or "")
        if not body:
            body = "()"
        tag = "terminal_output" if status in {"completed", "failed", "timeout"} else "terminal_command"
        exit_attr = f' exit_code="{int(exit_code)}"' if isinstance(exit_code, int) else ""
        return (
            f'<{tag} name="{_escape_xml_attr(session.session_id)}" '
            f'session_id="{_escape_xml_attr(session.session_id)}" '
            f'op_id="{_escape_xml_attr(session.session_id)}" '
            f'status="{_escape_xml_attr(status)}"{exit_attr} '
            f'has_new_output="{str(bool(state.get("has_new_output", False))).lower()}" '
            f'timed_out="{str(bool(state.get("timed_out", False))).lower()}" '
            f'transport="{_escape_xml_attr(str(state.get("transport", "") or ""))}">'
            f"\n{body}\n"
            f"</{tag}>"
        )

    def exec_command(
        self,
        *,
        cmd: str,
        workdir: Optional[str] = None,
        tty: bool = False,
        yield_time_ms: int = 1200,
        max_output_tokens: int = 1200,
        timeout_ms: int = 0,
        shell: bool = True,
        login: bool = True,
    ) -> str:
        command = str(cmd or "").strip()
        if not command:
            return "<error>exec_command requires a non-empty command</error>"
        try:
            session = self.create_session(
                cmd=command,
                workdir=workdir,
                tty=bool(tty),
                timeout_ms=int(timeout_ms or 0),
                shell=bool(shell),
                login=bool(login),
            )
            return self._yield_session(
                session,
                yield_time_ms=yield_time_ms,
                max_output_tokens=max_output_tokens,
            )
        except Exception as exc:
            return f"<error>exec_command failed: {str(exc)}</error>"

    def write_stdin(
        self,
        *,
        session_id: str,
        chars: str = "",
        yield_time_ms: int = 800,
        max_output_tokens: int = 1200,
    ) -> str:
        target_id = str(session_id or "").strip()
        if not target_id:
            return "<error>write_stdin requires session_id</error>"
        session = self.get_session(target_id)
        if session is None:
            return f"<error>interactive session missing: {target_id}</error>"

        payload = str(chars or "")
        if payload and not session.tty:
            return "<error>非 PTY 会话不支持写入 stdin；仅支持 chars='' 轮询新输出</error>"
        if payload:
            ok, error_message = session.write(payload)
            if not ok:
                return f"<error>write_stdin failed: {error_message}</error>"

        return self._yield_session(
            session,
            yield_time_ms=yield_time_ms,
            max_output_tokens=max_output_tokens,
        )

    def close_exec_session(self, session_id: str) -> str:
        target_id = str(session_id or "").strip()
        if not target_id:
            return "<error>close_exec_session requires session_id</error>"
        with self.lock:
            session = self.sessions.pop(target_id, None)
        if session is None:
            return f"<error>interactive session missing: {target_id}</error>"
        session.close()
        exit_code = session.refresh_exit_code()
        exit_attr = f' exit_code="{int(exit_code)}"' if isinstance(exit_code, int) else ""
        return (
            f'<terminal_command name="{_escape_xml_attr(target_id)}" '
            f'session_id="{_escape_xml_attr(target_id)}" '
            f'op_id="{_escape_xml_attr(target_id)}" '
            f'status="closed"{exit_attr} has_new_output="false" timed_out="false" '
            f'transport="{_escape_xml_attr(session.transport)}">'
            "session closed"
            "</terminal_command>"
        )
