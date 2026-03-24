"""TUI interaction manager built on top of InteractiveExecManager."""

from __future__ import annotations

import hashlib
import os
import re
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import pyte

from .interactive_exec import InteractiveExecManager


KEY_MAP = {
    "Enter": "\r",
    "Return": "\r",
    "Tab": "\t",
    "Escape": "\x1b",
    "Esc": "\x1b",
    "Backspace": "\x7f",
    "Delete": "\x1b[3~",
    "Space": " ",
    "Up": "\x1b[A",
    "Down": "\x1b[B",
    "Right": "\x1b[C",
    "Left": "\x1b[D",
    "Home": "\x1b[H",
    "End": "\x1b[F",
    "PageUp": "\x1b[5~",
    "PageDown": "\x1b[6~",
    "Insert": "\x1b[2~",
    "F1": "\x1bOP",
    "F2": "\x1bOQ",
    "F3": "\x1bOR",
    "F4": "\x1bOS",
    "F5": "\x1b[15~",
    "F6": "\x1b[17~",
    "F7": "\x1b[18~",
    "F8": "\x1b[19~",
    "F9": "\x1b[20~",
    "F10": "\x1b[21~",
    "F11": "\x1b[23~",
    "F12": "\x1b[24~",
    "Ctrl-A": "\x01",
    "Ctrl-B": "\x02",
    "Ctrl-C": "\x03",
    "Ctrl-D": "\x04",
    "Ctrl-E": "\x05",
    "Ctrl-F": "\x06",
    "Ctrl-G": "\x07",
    "Ctrl-H": "\x08",
    "Ctrl-K": "\x0b",
    "Ctrl-L": "\x0c",
    "Ctrl-N": "\x0e",
    "Ctrl-O": "\x0f",
    "Ctrl-P": "\x10",
    "Ctrl-R": "\x12",
    "Ctrl-S": "\x13",
    "Ctrl-T": "\x14",
    "Ctrl-U": "\x15",
    "Ctrl-V": "\x16",
    "Ctrl-W": "\x17",
    "Ctrl-X": "\x18",
    "Ctrl-Y": "\x19",
    "Ctrl-Z": "\x1a",
}

KEY_ALIASES = {
    "space": "Space",
    "enter": "Enter",
    "return": "Return",
    "tab": "Tab",
    "esc": "Esc",
    "escape": "Escape",
    "backspace": "Backspace",
    "delete": "Delete",
    "del": "Delete",
    "home": "Home",
    "end": "End",
    "pageup": "PageUp",
    "page_up": "PageUp",
    "pagedown": "PageDown",
    "page_down": "PageDown",
    "insert": "Insert",
    "up": "Up",
    "down": "Down",
    "left": "Left",
    "right": "Right",
    "f1": "F1",
    "f2": "F2",
    "f3": "F3",
    "f4": "F4",
    "f5": "F5",
    "f6": "F6",
    "f7": "F7",
    "f8": "F8",
    "f9": "F9",
    "f10": "F10",
    "f11": "F11",
    "f12": "F12",
}

for _char in "abcdefghijklmnopqrstuvwxyz":
    KEY_ALIASES[f"ctrl-{_char}"] = f"Ctrl-{_char.upper()}"


class TUISession:
    """Single TUI session backed by a PTY session from InteractiveExecManager."""

    def __init__(
        self,
        *,
        name: str,
        command: str,
        exec_manager: InteractiveExecManager,
        rows: int = 30,
        cols: int = 100,
        cwd: Optional[str] = None,
    ):
        self.name = name
        self.rows = rows
        self.cols = cols
        self.exec_manager = exec_manager
        self.screen = pyte.Screen(cols, rows)
        self.stream = pyte.Stream(self.screen)
        self.session = exec_manager.create_session(
            cmd=command,
            workdir=cwd,
            tty=True,
            timeout_ms=0,
            shell=False,
            login=False,
            rows=rows,
            cols=cols,
        )
        self.session_id = self.session.session_id
        self._alive = True
        self._lock = threading.Lock()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        while self._alive:
            state = self.exec_manager.read_session_state(
                self.session_id,
                yield_time_ms=40,
                max_output_chars=200000,
            )
            if state.get("error"):
                self._alive = False
                break
            data = str(state.get("body", "") or "")
            if data:
                with self._lock:
                    self.stream.feed(data)
            status = str(state.get("status", "") or "").strip().lower()
            if status in {"completed", "failed", "timeout", "closed"}:
                self._alive = False
                break
            if not data:
                time.sleep(0.01)

    @staticmethod
    def _normalize_color(value: Any) -> str:
        color = str(value or "").strip().lower()
        if color in {"", "default", "foreground", "background"}:
            return ""
        return color

    def _render_plain_line(self, row: int) -> str:
        line = self.screen.buffer[row]
        text = ""
        for col in range(self.cols):
            char = line[col]
            text += char.data if char.data else " "
        return text.rstrip()

    def _render_semantic_line(self, row: int, cursor_row: int, cursor_col: int) -> str:
        line = self.screen.buffer[row]
        plain_text = self._render_plain_line(row)
        non_space_cells = []
        reverse_count = 0
        bold_count = 0
        fg_values = set()
        bg_values = set()

        for col in range(self.cols):
            char = line[col]
            cell_text = char.data if char.data else " "
            if not cell_text.strip():
                continue
            non_space_cells.append(char)
            if getattr(char, "reverse", False):
                reverse_count += 1
            if getattr(char, "bold", False):
                bold_count += 1
            fg_value = self._normalize_color(getattr(char, "fg", ""))
            bg_value = self._normalize_color(getattr(char, "bg", ""))
            if fg_value:
                fg_values.add(fg_value)
            if bg_value:
                bg_values.add(bg_value)

        markers: List[str] = []
        if row == cursor_row:
            markers.append("cursor")
        if reverse_count:
            if reverse_count >= max(1, len(non_space_cells) // 2):
                markers.append("selected")
            else:
                markers.append("reverse")
        if bold_count:
            markers.append("bold")
        if len(fg_values) == 1:
            markers.append(f"fg={next(iter(fg_values))}")
        if len(bg_values) == 1:
            markers.append(f"bg={next(iter(bg_values))}")

        if not markers:
            return plain_text
        marker_prefix = "".join(f"[{marker}]" for marker in markers)
        if plain_text:
            return f"{marker_prefix} {plain_text}"
        return marker_prefix

    def get_screen_text(self, semantic: bool = False) -> List[str]:
        with self._lock:
            cursor_row = self.screen.cursor.y
            cursor_col = self.screen.cursor.x
            lines: List[str] = []
            for row in range(self.rows):
                text = (
                    self._render_semantic_line(row, cursor_row, cursor_col)
                    if semantic
                    else self._render_plain_line(row)
                )
                lines.append(text.rstrip())
            return lines

    def get_cursor(self) -> tuple[int, int]:
        with self._lock:
            return (self.screen.cursor.y, self.screen.cursor.x)

    def get_screen_hash(self) -> str:
        lines = self.get_screen_text(semantic=True)
        content = "\n".join(lines)
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def write(self, data: str) -> None:
        ok, error_message = self.session.write(data)
        if not ok:
            raise RuntimeError(error_message)

    def is_alive(self) -> bool:
        return self._alive and self.session.is_alive()

    def close(self) -> None:
        self._alive = False
        self.exec_manager.close_exec_session(self.session_id)


class TUIManager:
    """Manage TUI sessions and expose screen-oriented operations."""

    def __init__(
        self,
        default_cwd: str = ".",
        screen_update_callback: Optional[Callable] = None,
        interactive_exec_manager: Optional[InteractiveExecManager] = None,
    ):
        self.sessions: Dict[str, TUISession] = {}
        self.default_cwd = os.path.abspath(default_cwd)
        self.observations: Dict[str, Dict[str, Any]] = {}
        self.screen_update_callback = screen_update_callback
        self.exec_manager = interactive_exec_manager or InteractiveExecManager(self.default_cwd)

    def _snapshot_session(self, name: str) -> Dict[str, Any]:
        if name not in self.sessions:
            raise KeyError(name)
        session = self.sessions[name]
        try:
            plain_lines = session.get_screen_text(semantic=False)
            lines = session.get_screen_text(semantic=True)
        except TypeError:
            # Test doubles and older session implementations may only expose get_screen_text().
            fallback_lines = session.get_screen_text()
            plain_lines = list(fallback_lines or [])
            lines = list(fallback_lines or [])
        cursor_row, cursor_col = session.get_cursor()
        screen_hash = session.get_screen_hash()
        return {
            "name": name,
            "lines": lines,
            "plain_lines": plain_lines,
            "cursor_row": cursor_row,
            "cursor_col": cursor_col,
            "screen_hash": screen_hash,
            "timestamp": int(time.time()),
        }

    def _set_observation(self, name: str, snapshot: Dict[str, Any]) -> None:
        previous = self.observations.get(name, {}) or {}
        self.observations[name] = {
            "lines": list(snapshot.get("lines", []) or []),
            "plain_lines": list(snapshot.get("plain_lines", []) or []),
            "cursor_row": int(snapshot.get("cursor_row", 0) or 0),
            "cursor_col": int(snapshot.get("cursor_col", 0) or 0),
            "screen_hash": str(snapshot.get("screen_hash", "") or ""),
            "timestamp": int(snapshot.get("timestamp", 0) or int(time.time())),
            "full_layout_read": bool(snapshot.get("full_layout_read", previous.get("full_layout_read", False))),
        }

    def _get_observation(self, name: str) -> Dict[str, Any]:
        return dict(self.observations.get(name, {}) or {})

    @staticmethod
    def _compute_changed_rows(before_lines: List[str], after_lines: List[str]) -> List[int]:
        max_len = max(len(before_lines), len(after_lines))
        changed: List[int] = []
        for idx in range(max_len):
            before = before_lines[idx] if idx < len(before_lines) else ""
            after = after_lines[idx] if idx < len(after_lines) else ""
            if before != after:
                changed.append(idx)
        return changed

    def _lines_excerpt(self, lines: List[str], rows: List[int], context: int = 1) -> str:
        if not lines:
            return ""
        selected = set()
        for row in rows:
            for idx in range(max(0, row - context), min(len(lines), row + context + 1)):
                selected.add(idx)
        if not selected:
            return ""
        rendered = []
        for idx in sorted(selected):
            rendered.append(f'    <line row="{idx}">{self._escape_xml(lines[idx])}</line>')
        return "\n".join(rendered)

    def _extract_tui_block(self, text: str, tag_name: str) -> str:
        match = re.search(
            rf"<{re.escape(tag_name)}(?:\s+[^>]*)?>\s*(.*?)\s*</{re.escape(tag_name)}>",
            str(text or ""),
            re.IGNORECASE | re.DOTALL,
        )
        return match.group(1) if match else ""

    @staticmethod
    def _find_text_matches(lines: List[str], text: str, case_insensitive: bool = True) -> List[Dict[str, Any]]:
        matches: List[Dict[str, Any]] = []
        if not text:
            return matches
        needle = text.lower() if case_insensitive else text
        for idx, line in enumerate(lines):
            haystack = line.lower() if case_insensitive else line
            if needle in haystack:
                matches.append({"row": idx, "line": line})
        return matches

    def open_tui(self, name: str, command: str, rows: int = 30, cols: int = 100) -> str:
        session_name = str(name or "").strip()
        if not session_name:
            return "<error>TUI session name must not be empty</error>"
        if session_name in self.sessions:
            return f"<error>TUI session '{session_name}' already exists</error>"
        try:
            session = TUISession(
                name=session_name,
                command=str(command or "").strip(),
                exec_manager=self.exec_manager,
                rows=int(rows or 24),
                cols=int(cols or 80),
                cwd=self.default_cwd,
            )
            self.sessions[session_name] = session
            ready = self._wait_for_initial_screen(session, timeout=3.0)
            snapshot = self._snapshot_session(session_name)
            snapshot["full_layout_read"] = False
            self._set_observation(session_name, snapshot)
            self._notify_screen_update(session_name)
            first_hash = snapshot["screen_hash"]
            status = "first screen captured" if ready else "screen not ready yet"
            excerpt = self._lines_excerpt(snapshot.get("lines", []), list(range(min(len(snapshot.get("lines", [])), 8))), context=0)
            return (
                f"<success>TUI session '{session_name}' started "
                f"(command: {command}, {cols}x{rows}); {status}; screen_hash={first_hash}</success>\n"
                f'<tui_open name="{session_name}" rows="{session.rows}" cols="{session.cols}" '
                f'cursor_row="{snapshot["cursor_row"]}" cursor_col="{snapshot["cursor_col"]}" '
                f'screen_hash="{first_hash}" status="{self._escape_xml(status)}">\n'
                "  <summary>, <tui_views> .</summary>\n"
                f"  <first_screen>\n{excerpt}\n  </first_screen>\n"
                "</tui_open>"
            )
        except Exception as exc:
            return f"<error>failed to start TUI: {str(exc)}</error>"

    def read_tui(self, name: str, skip_empty_lines: bool = True) -> str:
        if name not in self.sessions:
            return f"<error>TUI session '{name}' does not exist</error>"
        session = self.sessions[name]
        if not session.is_alive():
            del self.sessions[name]
            self.observations.pop(name, None)
            return f"<error>TUI session '{name}' has exited</error>"

        snapshot = self._snapshot_session(name)
        lines = snapshot["lines"]
        cursor_row = snapshot["cursor_row"]
        cursor_col = snapshot["cursor_col"]
        screen_hash = snapshot["screen_hash"]
        previous = self._get_observation(name)
        first_full_read = not bool(previous.get("full_layout_read", False))
        if first_full_read:
            skip_empty_lines = False
        non_empty_lines = sum(1 for line in lines if line.strip())
        xml_lines = [
            f'<tui_screen name="{name}" rows="{session.rows}" cols="{session.cols}" '
            f'cursor_row="{cursor_row}" cursor_col="{cursor_col}" '
            f'line_count="{len(lines)}" non_empty_lines="{non_empty_lines}" '
            f'screen_hash="{screen_hash}" initial_read="{str(first_full_read).lower()}">'
        ]
        for row_idx, line in enumerate(lines):
            if not skip_empty_lines or line.strip():
                xml_lines.append(f'  <line row="{row_idx}">{self._escape_xml(line)}</line>')
        xml_lines.append("</tui_screen>")
        snapshot["full_layout_read"] = True
        self._set_observation(name, snapshot)
        return "\n".join(xml_lines)

    def send_keys(self, name: str, keys: str) -> str:
        if name not in self.sessions:
            return f"<error>TUI session '{name}' does not exist</error>"
        session = self.sessions[name]
        if not session.is_alive():
            del self.sessions[name]
            self.observations.pop(name, None)
            return f"<error>TUI session '{name}' has exited</error>"

        try:
            data = self._parse_keys(keys)
        except ValueError as exc:
            return f"<error>{str(exc)}</error>"

        before_snapshot = self._snapshot_session(name)
        old_hash = before_snapshot["screen_hash"]
        session.write(data)
        self._wait_for_update(session, old_hash, timeout=2.0)
        after_snapshot = self._snapshot_session(name)
        new_hash = after_snapshot["screen_hash"]
        changed = new_hash != old_hash
        changed_rows = self._compute_changed_rows(before_snapshot.get("lines", []), after_snapshot.get("lines", []))
        changed_rows_text = ",".join(str(idx) for idx in changed_rows)
        region_excerpt = self._lines_excerpt(after_snapshot.get("lines", []), changed_rows[:6], context=1)
        cursor_moved = (
            int(before_snapshot.get("cursor_row", 0) or 0) != int(after_snapshot.get("cursor_row", 0) or 0)
            or int(before_snapshot.get("cursor_col", 0) or 0) != int(after_snapshot.get("cursor_col", 0) or 0)
        )
        self._set_observation(name, after_snapshot)
        self._notify_screen_update(name)
        return (
            f'<tui_send name="{name}" changed="{str(changed).lower()}" '
            f'old_hash="{old_hash}" new_hash="{new_hash}" changed_rows="{changed_rows_text}" '
            f'cursor_moved="{str(cursor_moved).lower()}">\n'
            f'  <keys>{self._escape_xml(keys)}</keys>\n'
            f'  <summary>{self._escape_xml("" if changed else "")}</summary>\n'
            f'  <after_excerpt>\n{region_excerpt}\n  </after_excerpt>\n'
            f'  <region_excerpt>\n{region_excerpt}\n  </region_excerpt>\n'
            "</tui_send>"
        )

    def read_tui_diff(self, name: str) -> str:
        if name not in self.sessions:
            return f"<error>TUI session '{name}' does not exist</error>"
        session = self.sessions[name]
        if not session.is_alive():
            del self.sessions[name]
            self.observations.pop(name, None)
            return f"<error>TUI session '{name}' has exited</error>"

        previous = self._get_observation(name)
        current = self._snapshot_session(name)
        before_lines = list(previous.get("lines", []) or [])
        after_lines = list(current.get("lines", []) or [])
        changed_rows = self._compute_changed_rows(before_lines, after_lines)
        changed_rows_text = ",".join(str(idx) for idx in changed_rows)
        cursor_moved = (
            int(previous.get("cursor_row", 0) or 0) != int(current.get("cursor_row", 0) or 0)
            or int(previous.get("cursor_col", 0) or 0) != int(current.get("cursor_col", 0) or 0)
        )
        before_excerpt = self._lines_excerpt(before_lines, changed_rows[:6], context=1)
        after_excerpt = self._lines_excerpt(after_lines, changed_rows[:6], context=1)
        self._set_observation(name, current)
        summary = (
            f"changed_rows={changed_rows_text or 'none'}"
            if changed_rows
            else ("cursor moved" if cursor_moved else "no visible change")
        )
        return (
            f'<tui_diff name="{name}" previous_hash="{self._escape_xml(str(previous.get("screen_hash", "") or ""))}" '
            f'current_hash="{current["screen_hash"]}" changed_rows="{changed_rows_text}" '
            f'cursor_moved="{str(cursor_moved).lower()}">\n'
            f'  <summary>{self._escape_xml(summary)}</summary>\n'
            f'  <before>\n{before_excerpt}\n  </before>\n'
            f'  <after>\n{after_excerpt}\n  </after>\n'
            "</tui_diff>"
        )

    def read_tui_region(self, name: str, start_row: int, end_row: int) -> str:
        if name not in self.sessions:
            return f"<error>TUI session '{name}' does not exist</error>"
        session = self.sessions[name]
        if not session.is_alive():
            del self.sessions[name]
            self.observations.pop(name, None)
            return f"<error>TUI session '{name}' has exited</error>"

        current = self._snapshot_session(name)
        lines = current.get("lines", [])
        start = max(0, int(start_row or 0))
        end = min(len(lines) - 1, int(end_row or 0))
        if end < start:
            start, end = end, start
        xml_lines = [f'<tui_region name="{name}" start_row="{start}" end_row="{end}" screen_hash="{current["screen_hash"]}">']
        for row_idx in range(start, end + 1):
            xml_lines.append(f'  <line row="{row_idx}">{self._escape_xml(lines[row_idx])}</line>')
        xml_lines.append("</tui_region>")
        self._set_observation(name, current)
        return "\n".join(xml_lines)

    def find_text_in_tui(self, name: str, text: str, case_insensitive: bool = True) -> str:
        if name not in self.sessions:
            return f"<error>TUI session '{name}' does not exist</error>"
        session = self.sessions[name]
        if not session.is_alive():
            del self.sessions[name]
            self.observations.pop(name, None)
            return f"<error>TUI session '{name}' has exited</error>"

        current = self._snapshot_session(name)
        matches = self._find_text_matches(current.get("lines", []), str(text or ""), bool(case_insensitive))
        xml_lines = [
            f'<tui_text_matches name="{name}" text="{self._escape_xml(str(text or ""))}" '
            f'case_insensitive="{str(bool(case_insensitive)).lower()}" match_count="{len(matches)}" '
            f'screen_hash="{current["screen_hash"]}">'
        ]
        for item in matches[:20]:
            row = int(item.get("row", 0) or 0)
            context_rows = list(range(max(0, row - 1), min(len(current.get("lines", [])), row + 2)))
            xml_lines.append(f'  <match row="{row}">')
            for idx in context_rows:
                xml_lines.append(f'    <line row="{idx}">{self._escape_xml(current["lines"][idx])}</line>')
            xml_lines.append("  </match>")
        xml_lines.append("</tui_text_matches>")
        self._set_observation(name, current)
        return "\n".join(xml_lines)

    def send_keys_and_read(self, name: str, keys: str, delay_ms: int = 100) -> str:
        result = self.send_keys(name=name, keys=keys)
        if "<error>" in result.lower():
            return result
        if delay_ms and int(delay_ms) > 0:
            time.sleep(max(0, int(delay_ms)) / 1000.0)
        current = self._snapshot_session(name)
        old_hash_match = re.search(r'old_hash="([^"]*)"', result)
        new_hash_match = re.search(r'new_hash="([^"]*)"', result)
        changed_match = re.search(r'changed="(true|false)"', result)
        match = re.search(r'changed_rows="([^"]*)"', result)
        changed_rows_text = match.group(1).strip() if match else ""
        after_excerpt = self._extract_tui_block(result, "after_excerpt")
        return (
            f'<tui_action name="{name}" keys="{self._escape_xml(keys)}" '
            f'changed="{changed_match.group(1) if changed_match else "false"}" '
            f'old_hash="{self._escape_xml(old_hash_match.group(1) if old_hash_match else "")}" '
            f'new_hash="{self._escape_xml(new_hash_match.group(1) if new_hash_match else current["screen_hash"])}" '
            f'changed_rows="{changed_rows_text}" screen_hash="{current["screen_hash"]}">\n'
            f'  <summary>{self._escape_xml(", after_excerpt .")}</summary>\n'
            f'  <after_excerpt>\n{after_excerpt}\n  </after_excerpt>\n'
            f"{result}\n"
            "</tui_action>"
        )

    def wait_tui_until(
        self,
        name: str,
        text: Optional[str] = None,
        hash_change: bool = False,
        timeout_ms: int = 5000,
        poll_interval_ms: int = 200,
    ) -> str:
        if name not in self.sessions:
            return f"<error>TUI session '{name}' does not exist</error>"
        session = self.sessions[name]
        if not session.is_alive():
            del self.sessions[name]
            self.observations.pop(name, None)
            return f"<error>TUI session '{name}' has exited</error>"

        previous = self._get_observation(name)
        baseline_hash = str(previous.get("screen_hash", "") or "")
        baseline_lines = list(previous.get("lines", []) or [])
        started = time.time()
        timeout_ms = max(1, int(timeout_ms or 5000))
        poll_interval_ms = max(20, int(poll_interval_ms or 200))
        target_text = str(text or "")
        while (time.time() - started) * 1000 < timeout_ms:
            current = self._snapshot_session(name)
            lines = current.get("lines", [])
            text_hits = self._find_text_matches(lines, target_text, True) if target_text else []
            changed = hash_change and current.get("screen_hash", "") != baseline_hash
            if text_hits or changed:
                self._set_observation(name, current)
                hit_rows = ",".join(str(item.get("row", "")) for item in text_hits[:10])
                changed_rows = self._compute_changed_rows(baseline_lines, lines)
                changed_rows_text = ",".join(str(item) for item in changed_rows[:20])
                excerpt_rows = [int(item.get("row", 0) or 0) for item in text_hits[:4]] or changed_rows[:6]
                excerpt = self._lines_excerpt(lines, excerpt_rows, context=1)
                return (
                    f'<tui_wait name="{name}" status="matched" '
                    f'text="{self._escape_xml(target_text)}" hash_change="{str(changed).lower()}" '
                    f'match_rows="{self._escape_xml(hit_rows)}" changed_rows="{self._escape_xml(changed_rows_text)}" '
                    f'screen_hash="{current["screen_hash"]}">\n'
                    f'  <summary>{self._escape_xml(", excerpt .")}</summary>\n'
                    f'  <excerpt>\n{excerpt}\n  </excerpt>\n'
                    "</tui_wait>"
                )
            time.sleep(poll_interval_ms / 1000.0)

        current = self._snapshot_session(name)
        self._set_observation(name, current)
        return (
            f'<tui_wait name="{name}" status="timeout" text="{self._escape_xml(target_text)}" '
            f'hash_change="false" screen_hash="{current["screen_hash"]}">\n'
            "()\n"
            "</tui_wait>"
        )

    def close_tui(self, name: str) -> str:
        if name not in self.sessions:
            return f"<error>TUI session '{name}' does not exist</error>"
        self.sessions[name].close()
        del self.sessions[name]
        self.observations.pop(name, None)
        return f"<success>TUI session '{name}' closed</success>"

    def render_to_console(self, name: str) -> None:
        if name not in self.sessions:
            return
        session = self.sessions[name]
        if not session.is_alive():
            return
        lines = session.get_screen_text()
        cursor_row, cursor_col = session.get_cursor()
        cols = session.cols
        header_tail = "-" * max(cols - len(name) - 7, 0)
        self._safe_console_print(f"\n  +-- TUI: {name} {header_tail}+")
        for row_idx, line in enumerate(lines):
            padded = line.ljust(cols)[:cols]
            if row_idx == cursor_row:
                chars = list(padded)
                if cursor_col < len(chars):
                    chars[cursor_col] = f"\033[7m{chars[cursor_col]}\033[0m"
                padded = "".join(chars)
            self._safe_console_print(f"  |{padded}|")
        self._safe_console_print(f"  +{'-' * cols}+")
        self._safe_console_print(f"  Cursor: ({cursor_row}, {cursor_col})")
        self._safe_console_print("")

    def _notify_screen_update(self, name: str) -> None:
        if not self.screen_update_callback:
            return
        if name not in self.sessions:
            return
        session = self.sessions[name]
        if not session.is_alive():
            return
        try:
            lines = session.get_screen_text()
            cursor_row, cursor_col = session.get_cursor()
            self.screen_update_callback(name, lines, cursor_row, cursor_col)
        except Exception:
            pass

    def _parse_keys(self, keys: str) -> str:
        if not isinstance(keys, str):
            raise ValueError("send_keys keys must be a string")
        if not keys.strip():
            raise ValueError("send_keys keys must not be blank; use Space for a space key")
        tokens = keys.split()
        if not tokens:
            raise ValueError("send_keys could not parse any tokens; use Space for a space key")
        result = []
        for token in tokens:
            canonical = self._normalize_key_token(token)
            if canonical in KEY_MAP:
                result.append(KEY_MAP[canonical])
            else:
                result.append(token)
        return "".join(result)

    def _normalize_key_token(self, token: str) -> str:
        raw = str(token or "").strip()
        if not raw:
            return ""
        if raw in KEY_MAP:
            return raw
        alias_hit = KEY_ALIASES.get(raw.lower())
        return alias_hit or raw

    def _wait_for_initial_screen(self, session: TUISession, timeout: float = 3.0) -> bool:
        started = time.time()
        interval = 0.05
        last_hash = ""
        stable_hash_count = 0
        while time.time() - started < timeout:
            semantic_lines = session.get_screen_text(semantic=True)
            if any(line.strip() for line in semantic_lines):
                return True
            current_hash = session.get_screen_hash()
            if current_hash and current_hash == last_hash:
                stable_hash_count += 1
                if stable_hash_count >= 2:
                    return True
            else:
                last_hash = current_hash
                stable_hash_count = 1 if current_hash else 0
            time.sleep(interval)
            interval = min(interval * 1.5, 0.2)
        return False

    def _wait_for_update(self, session: TUISession, old_hash: str, timeout: float = 2.0) -> None:
        started = time.time()
        interval = 0.05
        while time.time() - started < timeout:
            time.sleep(interval)
            new_hash = session.get_screen_hash()
            if new_hash != old_hash:
                time.sleep(0.1)
                return
            interval = min(interval * 1.5, 0.2)

    @staticmethod
    def _escape_xml(text: str) -> str:
        return (
            str(text or "")
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    @staticmethod
    def _safe_console_print(text: str) -> None:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe_text = text
        try:
            safe_text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        except Exception:
            pass
        print(safe_text)
