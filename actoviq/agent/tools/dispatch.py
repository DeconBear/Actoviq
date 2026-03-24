"""
Klynx Agent - Tool dispatch mixin

Contains tool execution dispatch and formatting logic used by KlynxAgent.
"""

import inspect
import hashlib
import html
import json
import os
import re
import shlex
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage

from ..state import AgentState
from .syntax import SyntaxChecker
from .registry import ToolRegistry


class ToolDispatchMixin:
    """Tool execution and output formatting for KlynxAgent."""
    TOOL_RESULT_INLINE_LIMIT = 4000
    TOOL_RESULT_PREVIEW_LIMIT = 1200
    TUI_RESULT_INLINE_LIMIT = 900
    TUI_PROGRESS_ENTRY_KEEP = 8
    MUTATION_HISTORY_KEEP = 20
    TUI_VERIFICATION_KEEP = 20
    COMMAND_VERIFICATION_KEEP = 20
    APPLY_PATCH_FALLBACK_THRESHOLD = 3
    MUTATION_TOOLS = {"apply_patch"}
    TUI_STATUS_KEYWORDS = (
        "paused",
        "running",
        "resume",
        "game over",
        "暂停",
        "运行",
        "继续",
        "游戏结束",
    )
    TUI_GAMEPLAY_KEYWORDS = (
        "moved",
        "move blocked",
        "blocked",
        "cannot move",
        "win",
        "won",
        "victory",
        "移动",
        "阻挡",
        "无法移动",
        "胜利",
        "通关",
    )
    TUI_MENU_KEYWORDS = (
        "lobby",
        "menu",
        "select game",
        "game hall",
        "游戏大厅",
        "菜单",
        "选择游戏",
        "开始游戏",
    )
    TUI_GAMEPLAY_SCREEN_KEYWORDS = (
        "steps",
        "time",
        "difficulty",
        "player",
        "goal",
        "exit",
        "步数",
        "时间",
        "难度",
        "玩家",
        "坐标",
    )
    TUI_MOVEMENT_KEYS = frozenset({"up", "down", "left", "right", "w", "a", "s", "d", "h", "j", "k", "l"})
    TUI_CONFIRM_KEYS = frozenset({"enter", "return", "space"})
    TUI_NAVIGATION_KEYS = frozenset({"up", "down", "left", "right", "tab", "shift+tab", "j", "k"})
    TUI_VERIFICATION_TEMPLATES = {
        "selection_navigation": {
            "assertion": "A navigation key should move the highlighted selection in a predictable way.",
            "pass_condition": "selected option changes or highlighted row changes consistently.",
        },
        "maze_enter_game": {
            "assertion": "Entering the maze should transition from lobby/menu to the maze gameplay screen.",
            "pass_condition": "screen contains maze/gameplay markers instead of menu-only content.",
        },
        "maze_key_response": {
            "assertion": "After a movement key, at least one gameplay signal should change.",
            "pass_condition": "step count, player position, or move/block/win status changes.",
        },
        "maze_readability": {
            "assertion": "Maze screen should expose distinct markers and readable controls/info.",
            "pass_condition": "player/goal/wall markers are distinguishable and controls/info remain visible.",
        },
        "generic_tui_assertion": {
            "assertion": "A TUI bug is fixed only when a domain-specific observable changes as expected.",
            "pass_condition": "observable state changes beyond screen hash or cursor movement.",
        },
    }
    COMMAND_VERIFICATION_TEMPLATES = {
        "maze_movement_behavior": {
            "assertion": "Command output should prove whether maze movement is single-cell, blocked, or jump-like.",
            "pass_condition": "stdout includes before/after positions or delta values for sampled moves.",
        },
        "maze_input_mapping": {
            "assertion": "Per-key command output should show whether input handling maps to one movement result or a blocked/no-op result.",
            "pass_condition": "stdout includes handled/moved state or per-key before/after positions.",
        },
        "generic_command_assertion": {
            "assertion": "A command-based bug check is complete only when stdout contains domain-specific evidence, not just exit_code=0.",
            "pass_condition": "stdout includes structured semantic facts that prove or disprove a hypothesis.",
        },
    }
    PYTHON_FOREGROUND_SAFE_MODULES = frozenset(
        {
            "black",
            "compileall",
            "coverage",
            "ensurepip",
            "mypy",
            "pip",
            "py_compile",
            "pytest",
            "ruff",
            "unittest",
            "venv",
        }
    )
    PYTHON_ONE_SHOT_FLAGS = frozenset({"-h", "--help", "-V", "--version"})
    TOOL_ARTIFACT_KEEP_MAX = 300
    TOOL_DEDUPE_WINDOW_DEFAULT = 3
    TOOL_DEDUPE_KEEP_MAX = 400
    SOFT_DOOM_LOOP_THRESHOLD = 3
    SOFT_REPEATED_READ_THRESHOLD = 3
    SOFT_READS_PER_PATH_THRESHOLD = 5
    TOOL_RESULT_EXTERNALIZE_EXCLUDED = {"read_file"}
    TOOL_LEDGER_KEEP_MAX = 80
    TOOL_REUSE_CACHE_KEEP_MAX = 120
    TOOL_REUSE_PREVIEW_LIMIT = 800
    FULL_INLINE_PRIMARY_TOOLS = frozenset({"execute_command", "read_file"})
    TOOL_OUTPUT_DELIVERY_DEFAULT = "full_inline"
    TOOL_OUTPUT_HARD_CEILING_DEFAULT = 200000
    PLAN_STATE_TOOLS = {"state_update"}
    PARALLEL_SAFE_TOOLS = frozenset({"search_in_files", "read_file", "list_directory"})
    SERIAL_ONLY_TOOLS = frozenset(
        {
            "parallel_tool_call",
            "apply_patch",
            "state_update",
            "run_subtask",
            "write_stdin",
            "close_exec_session",
            "exec_command",
            "launch_interactive_session",
            "run_in_terminal",
            "run_and_wait",
            "open_tui",
            "send_keys",
            "send_keys_and_read",
            "wait_tui_until",
            "browser_act",
        }
    )
    COMMAND_TRACKED_TOOLS = {
        "execute_command",
        "run_in_terminal",
        "read_terminal",
        "read_terminal_since_last",
        "wait_terminal_until",
        "run_and_wait",
        "exec_command",
        "write_stdin",
        "close_exec_session",
        "launch_interactive_session",
        "check_syntax",
    }
    FILE_VIEW_SNIPPETS_PER_FILE = 4
    FILE_VIEW_FILES_MAX = 12
    LAST_READ_CHUNKS_KEEP_MAX = 24
    FILE_VIEW_CHUNKS_KEEP_MAX = 24
    TUI_VIEW_LINES_MAX = 18
    TUI_VIEW_SESSIONS_MAX = 8
    LAST_TUI_SNAPSHOTS_KEEP_MAX = 24

    def _resolve_runtime_checkpoint_id(self, thread_id: str) -> str:
        normalized_thread = str(thread_id or "").strip()
        if not normalized_thread:
            return ""
        app = getattr(self, "app", None)
        if app is None or not hasattr(app, "get_state"):
            return ""
        try:
            snapshot = app.get_state({"configurable": {"thread_id": normalized_thread}})
            cfg = dict(getattr(snapshot, "config", {}) or {}) if snapshot else {}
            configurable = dict(cfg.get("configurable", {}) or {})
            return str(configurable.get("checkpoint_id", "") or "").strip()
        except Exception:
            return ""

    def _normalize_task_plan(self, task_plan: Any) -> List[Dict[str, str]]:
        normalized: List[Dict[str, str]] = []
        if not isinstance(task_plan, list):
            return normalized
        for idx, item in enumerate(task_plan):
            if isinstance(item, dict):
                step_id = str(item.get("id", "") or f"step_{idx + 1}").strip()
                title = str(
                    item.get("title", "")
                    or item.get("task", "")
                    or item.get("name", "")
                    or step_id
                ).strip()
            else:
                step_id = f"step_{idx + 1}"
                title = str(item).strip()
            if not step_id:
                step_id = f"step_{idx + 1}"
            if not title:
                title = step_id
            normalized.append({"id": step_id, "title": title})
        return normalized

    def _first_pending_step_id(self, task_plan: List[Dict[str, str]], completed_steps: List[str]) -> str:
        completed = {str(step).strip() for step in (completed_steps or []) if str(step).strip()}
        for step in task_plan:
            step_id = str(step.get("id", "") or "").strip()
            if step_id and step_id not in completed:
                return step_id
        return ""

    def _ensure_step_stats(self, step_execution_stats: Dict[str, Any], step_id: str) -> Dict[str, Any]:
        if not step_id:
            return {}
        snapshot = dict(step_execution_stats.get(step_id, {}) or {})
        snapshot.setdefault("tools", 0)
        snapshot.setdefault("reads_per_file", {})
        snapshot.setdefault("tool_retries", {})
        step_execution_stats[step_id] = snapshot
        return snapshot

    def _append_summary_event(
        self,
        summary_events: List[Dict[str, Any]],
        event: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        summary_events.append(event)
        if len(summary_events) > 200:
            summary_events = summary_events[-200:]
        return summary_events

    def _build_step_checkpoint_event(
        self,
        step_id: str,
        step_title: str,
        executed_tools: List[Dict[str, Any]],
        tool_artifacts: List[Dict[str, Any]],
        blocked_reason: str,
    ) -> Dict[str, Any]:
        recent_tools = [str(item.get("tool", "") or "").strip() for item in (executed_tools or [])[-8:]]
        recent_tools = [tool for tool in recent_tools if tool]
        artifact_ids = []
        for item in (tool_artifacts or [])[-8:]:
            if not isinstance(item, dict):
                continue
            artifact_id = str(item.get("id", "") or "").strip()
            if artifact_id:
                artifact_ids.append(artifact_id)
        done_text = step_title or step_id
        if recent_tools:
            done_text = f"{done_text}(: {', '.join(recent_tools)})"
        open_risks = []
        if blocked_reason:
            open_risks.append(blocked_reason)
        return {
            "id": f"chk_{uuid.uuid4().hex[:12]}",
            "type": "step_checkpoint",
            "timestamp": int(time.time()),
            "step_id": step_id,
            "done": done_text,
            "artifacts": artifact_ids,
            "open_risks": open_risks,
            "summary": f"{step_id}: {done_text}",
        }

    def _resolve_budget_limit(
        self,
        state: AgentState,
        key: str,
        default_value: int,
    ) -> int:
        try:
            value = int(state.get(key, default_value) or default_value)
        except Exception:
            value = default_value
        return max(1, value)

    def _record_step_tool_usage(
        self,
        step_execution_stats: Dict[str, Any],
        step_id: str,
        tool_name: str,
        params: Dict[str, Any],
        failed: bool,
    ) -> Dict[str, Any]:
        snapshot = self._ensure_step_stats(step_execution_stats, step_id)
        if not snapshot:
            return step_execution_stats
        snapshot["tools"] = int(snapshot.get("tools", 0) or 0) + 1

        if tool_name == "read_file":
            path = str(params.get("path", "") or "").strip()
            if path:
                reads = dict(snapshot.get("reads_per_file", {}) or {})
                reads[path] = int(reads.get(path, 0) or 0) + 1
                snapshot["reads_per_file"] = reads

        if failed:
            retries = dict(snapshot.get("tool_retries", {}) or {})
            retries[tool_name] = int(retries.get(tool_name, 0) or 0) + 1
            snapshot["tool_retries"] = retries

        step_execution_stats[step_id] = snapshot
        return step_execution_stats

    def _is_tui_tool(self, tool_name: str) -> bool:
        return str(tool_name or "").strip() in {
            "open_tui",
            "read_tui",
            "read_tui_diff",
            "read_tui_region",
            "find_text_in_tui",
            "send_keys",
            "send_keys_and_read",
            "wait_tui_until",
        }

    def _coerce_bool(self, value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        return default

    def _extract_tui_screen_hash(self, output: str) -> str:
        text = str(output or "")
        match = re.search(r'screen_hash="([0-9a-fA-F]{32})"', text)
        if match:
            return match.group(1).lower()
        match = re.search(r'new_hash="([0-9a-fA-F]{32})"', text)
        if match:
            return match.group(1).lower()
        return ""

    def _extract_tui_status_tokens(self, output: str) -> List[str]:
        text = str(output or "")
        lowered = text.lower()
        compact = re.sub(r"\s+", "", lowered)
        hits = []
        for keyword in self.TUI_STATUS_KEYWORDS:
            token = str(keyword or "").strip()
            if not token:
                continue
            normalized = re.sub(r"\s+", "", token.lower())
            if normalized and normalized in compact:
                hits.append(normalized)
        return sorted(set(hits))

    def _is_mutation_tool(self, tool_name: str) -> bool:
        return str(tool_name or "").strip() in self.MUTATION_TOOLS

    def _extract_xml_text(self, text: str, tag_name: str) -> str:
        match = re.search(
            rf"<{re.escape(tag_name)}[^>]*>(.*?)</{re.escape(tag_name)}>",
            str(text or ""),
            re.DOTALL | re.IGNORECASE,
        )
        if not match:
            return ""
        raw = re.sub(r"<[^>]+>", " ", match.group(1))
        raw = html.unescape(raw)
        return re.sub(r"\s+", " ", raw).strip()

    def _extract_mutation_paths(self, tool_name: str, params: Dict[str, Any]) -> List[str]:
        files: List[str] = []
        if tool_name == "apply_patch":
            files = self._extract_patch_paths(str((params or {}).get("patch", "") or ""))
        return [path for path in files if path]

    def _summarize_mutation_intent(self, tool_name: str, params: Dict[str, Any], files: List[str]) -> str:
        if tool_name == "apply_patch":
            target_text = ", ".join(files[:3]) or "patch target"
            return f"apply_patch on {target_text}"
        return f"mutation via {str(tool_name or '').strip()}"

    def _classify_mutation_error_kind(self, tool_name: str, output: str) -> str:
        lowered = str(output or "").lower()
        if not lowered or "<error>" not in lowered:
            return ""
        if "unable to locate" in lowered or "hunk" in lowered or "" in lowered:
            return "hunk_mismatch"
        if "search/replace" in lowered or "search block" in lowered:
            return "anchor_mismatch"
        if "parse" in lowered or "" in lowered:
            return "parse_error"
        if "" in lowered or "does not exist" in lowered or "" in lowered or "already exists" in lowered:
            return "path_error"
        return "tool_error"

    def _mutation_retry_hint(self, error_kind: str) -> str:
        if error_kind == "hunk_mismatch":
            return (
                "The patch syntax parsed, but the target context did not match. "
                "Read the exact target slice, preserve exact whitespace/context lines, "
                "and reduce the patch scope before retrying."
            )
        if error_kind == "anchor_mismatch":
            return (
                "The anchor text did not match the file content. Read the exact target "
                "slice and keep SEARCH/context lines exact before retrying."
            )
        if error_kind == "parse_error":
            return (
                "Fix the patch syntax before retrying. Use *** Begin Patch / *** End Patch "
                "and prefix every change line with a space, +, or -."
            )
        if error_kind == "path_error":
            return "Verify the target path before retrying the mutation."
        return "Collect direct evidence before retrying the mutation."

    @staticmethod
    def _coerce_non_negative_int(value: Any, default: int = 0) -> int:
        try:
            return max(int(value), 0)
        except Exception:
            return max(int(default), 0)

    def _build_directory_probe_hint(self) -> str:
        shell_allowed = bool(getattr(self, "allow_shell_commands", True))
        has_list_tool = bool("list_directory" in (getattr(self, "tools", {}) or {}))
        if shell_allowed and has_list_tool:
            return "Use execute_command (`dir`/`ls`) or list_directory to confirm the path."
        if shell_allowed:
            return "Use execute_command (`dir`/`ls`) to confirm the path."
        if has_list_tool:
            return "Use list_directory to confirm the path."
        return "Use read_file with a known path to confirm the target location."

    def _resolve_read_file_preview_limit(self, params: Dict[str, Any]) -> int:
        start_line = self._coerce_non_negative_int((params or {}).get("start_line"), 0)
        end_line = self._coerce_non_negative_int((params or {}).get("end_line"), 0)
        limit = self._coerce_non_negative_int((params or {}).get("limit"), 0)
        if limit > 0:
            return min(limit, 400)
        if start_line > 0 and end_line >= start_line:
            return min((end_line - start_line + 1), 400)
        if end_line > 0:
            return min(end_line, 400)
        return 80

    def _build_read_file_terminal_hint(self, params: Dict[str, Any]) -> str:
        path = str((params or {}).get("path", "") or "").strip()
        if not bool(getattr(self, "allow_shell_commands", True)):
            return self._build_directory_probe_hint()
        if not path:
            return 'execute_command: Get-Content "<path>" -TotalCount 80'
        preview_limit = self._resolve_read_file_preview_limit(params)
        escaped = path.replace('"', '`"')
        return (
            "execute_command: "
            f'Get-Content "{escaped}" -TotalCount {max(preview_limit, 1)}'
        )

    def _is_read_only_execute_command(self, params: Dict[str, Any]) -> bool:
        command = str((params or {}).get("command", "") or "").strip().lower()
        if not command:
            return False
        if any(token in command for token in ("&&", "||", ";", "|", ">", ">>")):
            return False
        if any(
            token in command
            for token in (
                " rm ",
                " del ",
                " move ",
                " mv ",
                " copy ",
                " cp ",
                " chmod ",
                " chown ",
                " git commit",
                " git push",
                " git reset",
                " apply_patch",
            )
        ):
            return False
        read_markers = (
            "rg ",
            "grep ",
            "findstr ",
            "find ",
            "ls",
            "dir",
            "pwd",
            "get-childitem",
            "get-location",
            "cat ",
            "get-content",
            "git status",
            "git diff",
            "git show",
            "git log",
        )
        return any(marker in command for marker in read_markers)

    def _tool_parallel_safety(self, tool_name: str, params: Dict[str, Any]) -> str:
        normalized = str(tool_name or "").strip()
        if normalized in self.SERIAL_ONLY_TOOLS:
            return "serial"
        if normalized in self.PARALLEL_SAFE_TOOLS:
            return "parallel"
        if normalized == "execute_command" and self._is_read_only_execute_command(params):
            return "parallel"
        return "serial"

    def _is_tool_active(self, tool_name: str) -> bool:
        normalized = str(tool_name or "").strip()
        if not normalized:
            return False
        active_tools = getattr(self, "tools", {}) or {}
        if not isinstance(active_tools, dict) or not active_tools:
            return True
        return normalized in active_tools
    def _expand_parallel_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        expanded: List[Dict[str, Any]] = []
        expanded_count = 0

        for item in tool_calls or []:
            if not isinstance(item, dict):
                continue
            tool_name = str(item.get("tool", "") or "").strip()
            params = item.get("params", {})
            if not isinstance(params, dict):
                params = {}

            if tool_name != "parallel_tool_call":
                call_id = str(item.get("call_id", "") or item.get("id", "") or item.get("tool_call_id", "")).strip()
                payload = {"tool": tool_name, "params": params}
                if call_id:
                    payload["call_id"] = call_id
                expanded.append(payload)
                continue

            calls = params.get("calls", [])
            if not isinstance(calls, list) or not calls:
                self._emit("warning", "[Tool Parser] parallel_tool_call ignored: calls is empty or invalid.")
                continue

            for nested in calls:
                if not isinstance(nested, dict):
                    continue
                nested_tool = str(nested.get("tool", "") or "").strip()
                if not nested_tool or nested_tool == "parallel_tool_call":
                    continue
                nested_params = nested.get("params", {})
                if not isinstance(nested_params, dict):
                    nested_params = {}
                nested_call_id = str(nested.get("call_id", "") or nested.get("id", "") or nested.get("tool_call_id", "")).strip()
                nested_payload = {"tool": nested_tool, "params": nested_params}
                if nested_call_id:
                    nested_payload["call_id"] = nested_call_id
                expanded.append(nested_payload)
                expanded_count += 1

        if expanded_count > 0:
            self._emit("info", f"[Tool Parser] parallel_tool_call expanded to {expanded_count} call(s).")
        return expanded

    def _prefetch_parallel_tool_outputs(
        self,
        *,
        tool_calls: List[Dict[str, Any]],
        loaded_skill_names: List[str],
        skill_context: str,
    ) -> Tuple[Dict[int, Dict[str, Any]], int]:
        eligible: List[Tuple[int, str, Dict[str, Any]]] = []
        for idx, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                continue
            tool_name = str(tool_call.get("tool", "") or "").strip()
            params = dict(tool_call.get("params", {}) or {})
            if not self._is_tool_active(tool_name):
                continue
            if self._tool_parallel_safety(tool_name, params) != "parallel":
                continue
            eligible.append((idx, tool_name, params))

        if len(eligible) < 2:
            return {}, 0

        prefetched: Dict[int, Dict[str, Any]] = {}
        max_workers = min(4, len(eligible))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {}
            for idx, tool_name, params in eligible:
                future = pool.submit(
                    self._execute_single_tool_action,
                    tool_name=tool_name,
                    params=params,
                    loaded_skill_names=list(loaded_skill_names),
                    skill_context=str(skill_context or ""),
                )
                future_map[future] = (idx, tool_name, params)

            for future in as_completed(future_map):
                idx, tool_name, params = future_map[future]
                try:
                    output, _, _ = future.result()
                    prefetched[idx] = {
                        "tool": tool_name,
                        "params": params,
                        "output": output,
                        "error": "",
                    }
                except Exception as exc:
                    prefetched[idx] = {
                        "tool": tool_name,
                        "params": params,
                        "output": "",
                        "error": str(exc),
                    }
        return prefetched, len(eligible)

    def _update_apply_patch_failure_streak(
        self,
        *,
        output_text: str,
        failure_streak: int,
    ) -> int:
        failed = "<error>" in str(output_text or "").lower()
        if failed:
            return self._coerce_non_negative_int(failure_streak, 0) + 1
        return 0

    def _build_mutation_failure_hint(self, tool_name: str, error_kind: str) -> str:
        tool = str(tool_name or "").strip()
        kind = str(error_kind or "").strip()
        if tool == "apply_patch":
            if kind == "hunk_mismatch":
                return (
                    "<hint>apply_patch parsed successfully, but the target context did not match "
                    "(not a patch syntax error). Read the exact file slice first, preserve exact "
                    "whitespace/context lines, keep the patch minimal, and remember that @@ line "
                    "numbers are separators only and are not used for positioning.</hint>"
                )
            if kind == "parse_error":
                return (
                    "<hint>apply_patch syntax is invalid. Use either *** Begin Patch / *** End Patch "
                    "or unified diff (--- / +++ with @@ hunks), and prefix every hunk line with "
                    "a space, +, or -.</hint>"
                )
            if kind == "path_error":
                return (
                    "<hint>Verify the target path before retrying apply_patch. "
                    f"{self._build_directory_probe_hint()}</hint>"
                )
            return (
                "<hint>Read the exact target file slice before the next patch and keep stable "
                "context lines.</hint>"
            )
        return ""

    def _build_mutation_record(self, tool_name: str, params: Dict[str, Any], output: str) -> Dict[str, Any]:
        if not self._is_mutation_tool(tool_name):
            return {}
        files = self._extract_mutation_paths(tool_name, params if isinstance(params, dict) else {})
        status = "error" if "<error>" in str(output or "").lower() else "success"
        error_kind = self._classify_mutation_error_kind(tool_name, output) if status == "error" else ""
        error_excerpt = self._extract_xml_text(output, "error")
        if not error_excerpt and status == "error":
            error_excerpt = str(output or "").splitlines()[0][:240]
        return {
            "tool": tool_name,
            "path": files[0] if files else "",
            "files": files,
            "status": status,
            "file_changed": bool(status == "success" and files),
            "error_kind": error_kind,
            "error_excerpt": error_excerpt,
            "claimed_intent": self._summarize_mutation_intent(tool_name, params if isinstance(params, dict) else {}, files),
            "next_hint": self._mutation_retry_hint(error_kind) if status == "error" else "Verify the changed file against the task goal.",
            "timestamp": int(time.time()),
        }

    def _update_mutation_truth_state(
        self,
        *,
        recent_mutations: List[Dict[str, Any]],
        pending_verification_targets: List[str],
        mutation_record: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[str], str]:
        if not mutation_record:
            last_mutation = dict(recent_mutations[-1]) if recent_mutations else {}
            return (
                last_mutation,
                recent_mutations[-self.MUTATION_HISTORY_KEEP :],
                pending_verification_targets[-20:],
                self._build_mutation_truth_digest(recent_mutations, pending_verification_targets),
            )

        recent = list(recent_mutations or [])
        recent.append(dict(mutation_record))
        if len(recent) > self.MUTATION_HISTORY_KEEP:
            recent = recent[-self.MUTATION_HISTORY_KEEP :]

        targets = [str(item).strip() for item in (pending_verification_targets or []) if str(item).strip()]
        record_paths = [
            str(item).strip()
            for item in (mutation_record.get("files", []) or [])
            if str(item).strip()
        ]
        if mutation_record.get("file_changed"):
            for path in record_paths:
                if path not in targets:
                    targets.append(path)
        else:
            targets = [path for path in targets if path not in set(record_paths)]
        if len(targets) > 20:
            targets = targets[-20:]

        return (
            dict(mutation_record),
            recent,
            targets,
            self._build_mutation_truth_digest(recent, targets),
        )

    def _build_mutation_truth_digest(
        self,
        recent_mutations: List[Dict[str, Any]],
        pending_verification_targets: List[str],
    ) -> str:
        payload = {
            "recent": [
                {
                    "tool": str(item.get("tool", "") or ""),
                    "path": str(item.get("path", "") or ""),
                    "status": str(item.get("status", "") or ""),
                    "file_changed": bool(item.get("file_changed", False)),
                    "error_kind": str(item.get("error_kind", "") or ""),
                }
                for item in (recent_mutations or [])[-8:]
                if isinstance(item, dict)
            ],
            "pending_verification_targets": [
                str(item).strip()
                for item in (pending_verification_targets or [])[-8:]
                if str(item).strip()
            ],
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:24]

    def _extract_tui_line_texts(self, output: str) -> List[str]:
        matches = re.findall(r"<line[^>]*>(.*?)</line>", str(output or ""), re.DOTALL | re.IGNORECASE)
        lines = [re.sub(r"\s+", " ", html.unescape(match)).strip() for match in matches]
        return [line for line in lines if line]

    def _extract_tui_block_lines(self, output: str, block_name: str) -> List[str]:
        match = re.search(
            rf"<{re.escape(block_name)}[^>]*>(.*?)</{re.escape(block_name)}>",
            str(output or ""),
            re.DOTALL | re.IGNORECASE,
        )
        if not match:
            return []
        block = match.group(1)
        lines = re.findall(r"<line[^>]*>(.*?)</line>", block, re.DOTALL | re.IGNORECASE)
        rendered = [re.sub(r"\s+", " ", html.unescape(item)).strip() for item in lines]
        if rendered:
            return [line for line in rendered if line]
        plain = re.sub(r"<[^>]+>", "\n", block)
        return [re.sub(r"\s+", " ", html.unescape(line)).strip() for line in plain.splitlines() if line.strip()]

    def _extract_tui_selected_labels(self, lines: List[str]) -> List[str]:
        selected = []
        for line in lines:
            if "[selected]" not in line.lower():
                continue
            normalized = re.sub(r"\[selected\]\s*", "", line, flags=re.IGNORECASE).strip()
            if normalized:
                selected.append(normalized)
        return selected

    def _extract_tui_step_values(self, output: str) -> List[int]:
        values: List[int] = []
        patterns = (
            r"(?:steps?|step_count|move_count|moves|步数|步)\s*[:=：]\s*(\d+)",
            r"(?:steps?|step_count|moves|步数)\s+(\d+)",
        )
        for pattern in patterns:
            for match in re.findall(pattern, str(output or ""), re.IGNORECASE):
                try:
                    values.append(int(match))
                except Exception:
                    continue
        return values

    def _extract_tui_position_values(self, output: str) -> List[str]:
        positions: List[str] = []
        patterns = (
            r"(?:player|position|pos|coords?|玩家|位置|坐标)\s*[:=：]\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?",
            r"x\s*=\s*(\d+)\s*[, ]+\s*y\s*=\s*(\d+)",
        )
        for pattern in patterns:
            for x_val, y_val in re.findall(pattern, str(output or ""), re.IGNORECASE):
                positions.append(f"{x_val},{y_val}")
        return positions

    def _extract_tui_gameplay_tokens(self, output: str) -> List[str]:
        lowered = str(output or "").lower()
        hits = []
        for token in self.TUI_GAMEPLAY_KEYWORDS:
            normalized = str(token or "").strip().lower()
            if not normalized:
                continue
            if re.search(r"[a-z]", normalized):
                if re.search(rf"\b{re.escape(normalized)}\b", lowered):
                    hits.append(normalized)
            elif normalized in lowered:
                hits.append(normalized)
        return sorted(set(hits))

    def _text_contains_any_token(self, text: str, tokens: Tuple[str, ...]) -> bool:
        lowered = str(text or "").lower()
        for token in tokens:
            normalized = str(token or "").strip().lower()
            if not normalized:
                continue
            if re.search(r"[a-z]", normalized):
                if re.search(rf"(?<![a-z0-9_]){re.escape(normalized)}(?![a-z0-9_])", lowered):
                    return True
            elif normalized in lowered:
                return True
        return False

    def _extract_tui_input_tokens(self, output: str, params: Dict[str, Any]) -> List[str]:
        raw_keys = str((params or {}).get("keys", "") or "").strip()
        if not raw_keys:
            raw_keys = self._extract_xml_attr(output, "keys")
        if not raw_keys:
            key_match = re.search(r"<keys>\s*(.*?)\s*</keys>", str(output or ""), re.IGNORECASE | re.DOTALL)
            raw_keys = key_match.group(1).strip() if key_match else ""
        if not raw_keys:
            return []
        normalized = (
            raw_keys.replace(",", " ")
            .replace(";", " ")
            .replace("/", " ")
            .replace("|", " ")
            .replace("+", " + ")
        )
        tokens = []
        pending_plus = False
        for chunk in normalized.split():
            piece = str(chunk or "").strip().lower()
            if not piece:
                continue
            if piece == "+":
                pending_plus = True
                continue
            if pending_plus and tokens:
                tokens[-1] = f"{tokens[-1]}+{piece}"
                pending_plus = False
                continue
            pending_plus = False
            tokens.append(piece)
        return tokens

    def _classify_tui_scene(self, lines: List[str]) -> str:
        text_blob = "\n".join(str(line or "") for line in (lines or [])).strip()
        if not text_blob:
            return "unknown"
        if (
            self._text_contains_any_token(text_blob, self.TUI_GAMEPLAY_SCREEN_KEYWORDS)
            or bool(self._extract_tui_step_values(text_blob))
            or bool(self._extract_tui_position_values(text_blob))
        ):
            return "maze_gameplay"
        if self._extract_tui_selected_labels(lines) or self._text_contains_any_token(text_blob, self.TUI_MENU_KEYWORDS):
            return "menu"
        return "unknown"

    def _default_tui_verification_targets(self) -> List[Dict[str, str]]:
        ordered_goals = [
            "selection_navigation",
            "maze_enter_game",
            "maze_key_response",
            "maze_readability",
        ]
        targets: List[Dict[str, str]] = []
        for goal in ordered_goals:
            template = dict(self.TUI_VERIFICATION_TEMPLATES.get(goal, {}) or {})
            if not template:
                continue
            targets.append(
                {
                    "goal": goal,
                    "assertion": str(template.get("assertion", "") or ""),
                    "pass_condition": str(template.get("pass_condition", "") or ""),
                }
            )
        return targets

    def _ensure_tui_verification_targets(
        self,
        tui_verification_targets: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        existing = [item for item in (tui_verification_targets or []) if isinstance(item, dict)]
        if existing:
            return existing
        return self._default_tui_verification_targets()

    def _infer_tui_verification_goal(
        self,
        *,
        state: AgentState,
        tool_name: str,
        params: Dict[str, Any],
        output: str,
    ) -> str:
        task_text = " ".join(
            [
                str(state.get("overall_goal", "") or ""),
                str(state.get("current_task", "") or ""),
                str(state.get("current_focus", "") or ""),
            ]
        ).lower()
        all_lines = self._extract_tui_line_texts(output)
        before_lines = self._extract_tui_block_lines(output, "before")
        after_lines = (
            self._extract_tui_block_lines(output, "after")
            or self._extract_tui_block_lines(output, "after_excerpt")
            or self._extract_tui_block_lines(output, "excerpt")
            or self._extract_tui_block_lines(output, "region_excerpt")
            or all_lines
        )
        before_scene = self._classify_tui_scene(before_lines)
        after_scene = self._classify_tui_scene(after_lines)
        key_tokens = self._extract_tui_input_tokens(output, params)
        movement_intent = bool(set(key_tokens) & self.TUI_MOVEMENT_KEYS)
        confirm_intent = bool(set(key_tokens) & self.TUI_CONFIRM_KEYS)
        navigation_intent = bool(set(key_tokens) & self.TUI_NAVIGATION_KEYS) or confirm_intent
        maze_focus = self._text_contains_any_token(task_text, ("maze", "迷宫"))
        movement_focus = self._text_contains_any_token(
            task_text,
            ("key response", "movement", "move", "controls not work", "按键响应", "移动", "控制失效"),
        )
        readability_focus = self._text_contains_any_token(
            task_text,
            ("readability", "readable", "可读", "marker", "legend", "说明", "layout", "render"),
        )
        navigation_focus = self._text_contains_any_token(
            task_text,
            ("selected", "highlight", "navigation", "navigate", "选项", "菜单", "selection"),
        )

        if after_scene == "menu":
            if maze_focus and confirm_intent:
                return "maze_enter_game"
            if navigation_intent or navigation_focus:
                return "selection_navigation"
            if maze_focus:
                return "maze_enter_game"
        if before_scene == "menu" and after_scene == "maze_gameplay":
            return "maze_enter_game"
        if after_scene == "maze_gameplay":
            if readability_focus:
                return "maze_readability"
            if confirm_intent and before_scene == "menu":
                return "maze_enter_game"
            if movement_intent or movement_focus:
                return "maze_key_response"
            if maze_focus:
                return "maze_enter_game"
        if maze_focus:
            if readability_focus:
                return "maze_readability"
            if confirm_intent:
                return "maze_enter_game"
            if movement_intent:
                return "maze_key_response"
            if tool_name in {"open_tui", "read_tui", "read_tui_region", "find_text_in_tui", "wait_tui_until"}:
                return "maze_enter_game"
            if movement_focus:
                return "maze_key_response"
        if navigation_focus or self._extract_tui_selected_labels(after_lines):
            return "selection_navigation"
        return "generic_tui_assertion"

    def _build_tui_verification_record(
        self,
        *,
        state: AgentState,
        tool_name: str,
        params: Dict[str, Any],
        output: str,
        tui_progressed: bool,
    ) -> Dict[str, Any]:
        if not self._is_tui_tool(tool_name) or "<error>" in str(output or "").lower():
            return {}

        goal = self._infer_tui_verification_goal(
            state=state,
            tool_name=tool_name,
            params=params,
            output=output,
        )
        template = dict(self.TUI_VERIFICATION_TEMPLATES.get(goal, self.TUI_VERIFICATION_TEMPLATES["generic_tui_assertion"]))
        before_lines = self._extract_tui_block_lines(output, "before")
        after_lines = (
            self._extract_tui_block_lines(output, "after")
            or self._extract_tui_block_lines(output, "after_excerpt")
            or self._extract_tui_block_lines(output, "excerpt")
            or self._extract_tui_block_lines(output, "region_excerpt")
        )
        all_lines = self._extract_tui_line_texts(output)
        step_values = self._extract_tui_step_values(output)
        position_values = self._extract_tui_position_values(output)
        gameplay_tokens = self._extract_tui_gameplay_tokens(output)
        key_tokens = self._extract_tui_input_tokens(output, params)
        movement_intent = bool(set(key_tokens) & self.TUI_MOVEMENT_KEYS)
        confirm_intent = bool(set(key_tokens) & self.TUI_CONFIRM_KEYS)
        navigation_intent = bool(set(key_tokens) & self.TUI_NAVIGATION_KEYS) or confirm_intent
        before_scene = self._classify_tui_scene(before_lines)
        after_scene = self._classify_tui_scene(after_lines or all_lines)
        selected_before = self._extract_tui_selected_labels(before_lines)
        selected_after = self._extract_tui_selected_labels(after_lines or all_lines)
        before_evidence = "; ".join(selected_before or before_lines[:2])[:240]
        after_evidence = "; ".join(selected_after or after_lines[:2] or all_lines[:2])[:240]
        status = "observed"
        reason = "screen observed; no semantic assertion evaluated yet"

        if goal == "selection_navigation":
            if selected_before and selected_after and selected_before[-1] != selected_after[-1]:
                status = "passed"
                reason = "selected option changed"
            elif navigation_intent and tui_progressed and selected_after and not selected_before:
                status = "passed"
                reason = "selected option is explicitly observable after the interaction"
            elif navigation_intent and not tui_progressed:
                status = "failed"
                reason = "screen did not change after navigation input"
            elif after_scene == "menu":
                reason = "menu state observed; waiting for a navigation interaction"
        elif goal == "maze_key_response":
            if len(step_values) >= 2 and step_values[0] != step_values[-1]:
                status = "passed"
                reason = f"step count changed from {step_values[0]} to {step_values[-1]}"
                before_evidence = before_evidence or f"steps={step_values[0]}"
                after_evidence = after_evidence or f"steps={step_values[-1]}"
            elif len(step_values) >= 2:
                status = "failed"
                reason = f"step count stayed at {step_values[-1]}"
            elif len(position_values) >= 2 and position_values[0] != position_values[-1]:
                status = "passed"
                reason = f"player position changed from {position_values[0]} to {position_values[-1]}"
                before_evidence = before_evidence or f"player={position_values[0]}"
                after_evidence = after_evidence or f"player={position_values[-1]}"
            elif len(position_values) >= 2:
                status = "failed"
                reason = f"player position stayed at {position_values[-1]}"
            elif gameplay_tokens:
                status = "passed"
                reason = f"gameplay status changed: {', '.join(gameplay_tokens[:3])}"
                after_evidence = after_evidence or ", ".join(gameplay_tokens[:3])
            elif movement_intent and not tui_progressed:
                status = "failed"
                reason = "screen did not change after movement input"
            elif movement_intent and after_scene == "maze_gameplay":
                status = "failed"
                reason = "screen changed but gameplay signal did not change"
            elif movement_intent:
                reason = "movement input sent; collect gameplay evidence from a region or full-screen read"
            elif after_scene != "maze_gameplay":
                reason = "maze gameplay screen not visible; key-response assertion not evaluated"
            else:
                reason = "not a movement interaction; key-response assertion not evaluated"
        elif goal == "maze_readability":
            if after_scene != "maze_gameplay":
                reason = "maze gameplay screen not visible yet"
            else:
                text_blob = "\n".join(after_lines or all_lines)
                player_visible = any(token in text_blob for token in ("@", "player", "", "P "))
                goal_visible = any(token in text_blob for token in ("goal", "", "exit", " G", "E "))
                wall_visible = any(token in text_blob for token in ("#", "wall", "", "█"))
                info_visible = any(token.lower() in text_blob.lower() for token in ("controls", "keys", "", "", "help"))
                if player_visible and goal_visible and wall_visible and info_visible:
                    status = "passed"
                    reason = "markers and controls are distinguishable"
                else:
                    status = "failed"
                    missing = []
                    if not player_visible:
                        missing.append("player marker")
                    if not goal_visible:
                        missing.append("goal marker")
                    if not wall_visible:
                        missing.append("wall marker")
                    if not info_visible:
                        missing.append("controls/info")
                    reason = "missing readability evidence: " + ", ".join(missing)
        elif goal == "maze_enter_game":
            if after_scene == "maze_gameplay":
                status = "passed"
                reason = "screen shows gameplay markers"
            elif confirm_intent and not tui_progressed:
                status = "failed"
                reason = "screen did not change after enter/start interaction"
            elif confirm_intent and before_scene == "menu" and after_scene == "menu":
                status = "failed"
                reason = "enter/start interaction did not leave the menu"
            elif after_scene == "menu":
                reason = "lobby/menu observed; waiting for an enter/start interaction"
            else:
                reason = "screen changed but gameplay markers are still missing"
        else:
            if gameplay_tokens or (selected_before and selected_after and selected_before[-1] != selected_after[-1]):
                status = "passed"
                reason = "domain-specific observable changed"
            elif key_tokens and not tui_progressed:
                status = "failed"
                reason = "screen did not change after the TUI interaction"

        return {
            "goal": goal,
            "status": status,
            "assertion": str(template.get("assertion", "") or ""),
            "pass_condition": str(template.get("pass_condition", "") or ""),
            "before_evidence": before_evidence,
            "after_evidence": after_evidence,
            "reason": reason,
            "tui_progressed": bool(tui_progressed),
            "tool": tool_name,
            "timestamp": int(time.time()),
        }

    def _update_tui_verification_state(
        self,
        *,
        recent_tui_verifications: List[Dict[str, Any]],
        verification_record: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str]:
        if not verification_record:
            last_verification = dict(recent_tui_verifications[-1]) if recent_tui_verifications else {}
            return (
                last_verification,
                recent_tui_verifications[-self.TUI_VERIFICATION_KEEP :],
                self._build_tui_verification_digest(recent_tui_verifications),
            )

        recent = list(recent_tui_verifications or [])
        recent.append(dict(verification_record))
        if len(recent) > self.TUI_VERIFICATION_KEEP:
            recent = recent[-self.TUI_VERIFICATION_KEEP :]
        return (
            dict(verification_record),
            recent,
            self._build_tui_verification_digest(recent),
        )

    def _build_tui_verification_digest(self, recent_tui_verifications: List[Dict[str, Any]]) -> str:
        payload = [
            {
                "goal": str(item.get("goal", "") or ""),
                "status": str(item.get("status", "") or ""),
                "reason": str(item.get("reason", "") or "")[:120],
                "tui_progressed": bool(item.get("tui_progressed", False)),
            }
            for item in (recent_tui_verifications or [])[-8:]
            if isinstance(item, dict)
        ]
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:24]

    def _extract_command_text(self, tool_name: str, params: Dict[str, Any]) -> str:
        if tool_name in {"exec_command", "launch_interactive_session"}:
            return str((params or {}).get("cmd", "") or (params or {}).get("command", "") or "").strip()
        return str((params or {}).get("command", "") or (params or {}).get("path", "") or "").strip()

    def _extract_command_semantic_text(self, tool_name: str, output: str) -> str:
        if tool_name in {
            "run_in_terminal",
            "read_terminal",
            "read_terminal_since_last",
            "wait_terminal_until",
            "run_and_wait",
            "exec_command",
            "write_stdin",
            "close_exec_session",
            "launch_interactive_session",
        }:
            terminal_state = self._parse_terminal_output(output)
            body = str(terminal_state.get("body", "") or "").strip()
            return "" if body == "()" else body
        return self._extract_command_stdout(output)

    def _dedupe_text_items(self, items: List[str], keep: int = 6) -> List[str]:
        seen = set()
        deduped: List[str] = []
        for item in items:
            text = str(item or "").strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(text)
            if len(deduped) >= keep:
                break
        return deduped

    def _extract_command_move_transitions(self, output: str) -> List[Dict[str, Any]]:
        text = str(output or "")
        pattern = re.compile(
            r"(?mi)^\s*(?:\d+\.\s*)?(?P<label>[^:\n]{0,32}?)?\s*:?\s*"
            r"(?:success=(?P<success>true|false)\s*,\s*)?"
            r"\((?P<x1>-?\d+)\s*,\s*(?P<y1>-?\d+)\)\s*->\s*"
            r"\((?P<x2>-?\d+)\s*,\s*(?P<y2>-?\d+)\)\s*,\s*"
            r"delta=\((?P<dx>-?\d+)\s*,\s*(?P<dy>-?\d+)\)\s*$"
        )
        transitions: List[Dict[str, Any]] = []
        for match in pattern.finditer(text):
            label = str(match.group("label") or "").strip().strip(".")
            success_text = str(match.group("success") or "").strip().lower()
            transitions.append(
                {
                    "label": label,
                    "success": None if not success_text else success_text == "true",
                    "x1": int(match.group("x1")),
                    "y1": int(match.group("y1")),
                    "x2": int(match.group("x2")),
                    "y2": int(match.group("y2")),
                    "dx": int(match.group("dx")),
                    "dy": int(match.group("dy")),
                }
            )
        return transitions

    def _extract_command_key_mapping_results(self, output: str) -> List[Dict[str, Any]]:
        text = str(output or "")
        pattern = re.compile(
            r"(?mi)^\s*(?P<key>[a-z0-9_+-]+)\s*->\s*"
            r"handled=(?P<handled>true|false)\s*,\s*"
            r"moved=(?P<moved>true|false)\s*,\s*"
            r"pos=\((?P<x1>-?\d+)\s*,\s*(?P<y1>-?\d+)\)\s*->\s*"
            r"\((?P<x2>-?\d+)\s*,\s*(?P<y2>-?\d+)\)\s*$"
        )
        mappings: List[Dict[str, Any]] = []
        for match in pattern.finditer(text):
            x1 = int(match.group("x1"))
            y1 = int(match.group("y1"))
            x2 = int(match.group("x2"))
            y2 = int(match.group("y2"))
            mappings.append(
                {
                    "key": str(match.group("key") or "").strip().lower(),
                    "handled": str(match.group("handled") or "").strip().lower() == "true",
                    "moved": str(match.group("moved") or "").strip().lower() == "true",
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "dx": x2 - x1,
                    "dy": y2 - y1,
                }
            )
        return mappings

    def _infer_command_verification_goal(
        self,
        *,
        state: AgentState,
        tool_name: str,
        params: Dict[str, Any],
        output: str,
    ) -> str:
        semantic_text = self._extract_command_semantic_text(tool_name, output)
        if not semantic_text:
            return ""

        task_text = " ".join(
            [
                str(state.get("overall_goal", "") or ""),
                str(state.get("current_task", "") or ""),
                str(state.get("current_focus", "") or ""),
            ]
        ).lower()
        command_text = self._extract_command_text(tool_name, params).lower()
        semantic_lower = semantic_text.lower()
        transitions = self._extract_command_move_transitions(semantic_text)
        mappings = self._extract_command_key_mapping_results(semantic_text)
        maze_focus = self._text_contains_any_token(
            " ".join((task_text, command_text, semantic_lower)),
            ("maze", "", "move_player", "handle_input", "direction"),
        )
        movement_focus = self._text_contains_any_token(
            " ".join((task_text, command_text)),
            ("movement", "move", "jump", "delta", "", "", ""),
        )
        input_focus = self._text_contains_any_token(
            " ".join((task_text, command_text)),
            ("input", "key", "keyboard", "", "", "handle_input"),
        )

        if mappings and (maze_focus or input_focus):
            return "maze_input_mapping"
        if transitions and (maze_focus or movement_focus or input_focus):
            return "maze_movement_behavior"
        if mappings or transitions:
            return "generic_command_assertion"
        return ""

    def _build_command_verification_record(
        self,
        *,
        state: AgentState,
        tool_name: str,
        params: Dict[str, Any],
        output: str,
    ) -> Dict[str, Any]:
        if tool_name not in self.COMMAND_TRACKED_TOOLS or "<error>" in str(output or "").lower():
            return {}

        semantic_text = self._extract_command_semantic_text(tool_name, output)
        if not semantic_text:
            return {}

        goal = self._infer_command_verification_goal(
            state=state,
            tool_name=tool_name,
            params=params,
            output=output,
        )
        if not goal:
            return {}

        template = dict(
            self.COMMAND_VERIFICATION_TEMPLATES.get(
                goal,
                self.COMMAND_VERIFICATION_TEMPLATES["generic_command_assertion"],
            )
        )
        command_text = self._extract_command_text(tool_name, params)
        command_preview = command_text if len(command_text) <= 160 else command_text[:160] + "..."
        transitions = self._extract_command_move_transitions(semantic_text)
        mappings = self._extract_command_key_mapping_results(semantic_text)
        status = "observed"
        reason = "command ran but did not yet produce enough semantic evidence"
        facts: List[str] = []
        supported_hypotheses: List[str] = []
        disproven_hypotheses: List[str] = []

        if goal == "maze_movement_behavior":
            moved_transitions = [item for item in transitions if int(item.get("dx", 0) or 0) or int(item.get("dy", 0) or 0)]
            blocked_transitions = [
                item
                for item in transitions
                if not (int(item.get("dx", 0) or 0) or int(item.get("dy", 0) or 0))
            ]
            jump_transitions = [
                item
                for item in moved_transitions
                if abs(int(item.get("dx", 0) or 0)) > 1
                or abs(int(item.get("dy", 0) or 0)) > 1
                or abs(int(item.get("dx", 0) or 0)) + abs(int(item.get("dy", 0) or 0)) > 1
            ]
            if jump_transitions:
                sample = jump_transitions[0]
                status = "failed"
                reason = (
                    "jump-like movement observed: "
                    f"{sample.get('label') or 'move'} delta=({sample.get('dx', 0)},{sample.get('dy', 0)})"
                )
                supported_hypotheses.append("movement jumps more than one cell per action")
            elif moved_transitions:
                status = "passed"
                moved_labels = [
                    (
                        f"{item.get('label') or 'move'} "
                        f"({item.get('x1')},{item.get('y1')})->({item.get('x2')},{item.get('y2')}) "
                        f"delta=({item.get('dx')},{item.get('dy')})"
                    ).strip()
                    for item in moved_transitions[:4]
                ]
                reason = "single-cell movement observed in command output"
                facts.extend(moved_labels)
                disproven_hypotheses.append("movement jumps more than one cell per action")
                if blocked_transitions:
                    supported_hypotheses.append("some directions are blocked by maze state or walls")
                    disproven_hypotheses.append("all movement fails")
            elif transitions:
                status = "observed"
                reason = "only blocked or zero-delta moves were observed; step size is still unproven"
                facts.extend(
                    [
                        (
                            f"{item.get('label') or 'move'} "
                            f"({item.get('x1')},{item.get('y1')})->({item.get('x2')},{item.get('y2')}) "
                            f"delta=({item.get('dx')},{item.get('dy')})"
                        ).strip()
                        for item in transitions[:4]
                    ]
                )
        elif goal == "maze_input_mapping":
            moved_mappings = [
                item for item in mappings if bool(item.get("moved", False)) or int(item.get("dx", 0) or 0) or int(item.get("dy", 0) or 0)
            ]
            handled_mappings = [item for item in mappings if bool(item.get("handled", False))]
            jump_mappings = [
                item
                for item in moved_mappings
                if abs(int(item.get("dx", 0) or 0)) > 1
                or abs(int(item.get("dy", 0) or 0)) > 1
                or abs(int(item.get("dx", 0) or 0)) + abs(int(item.get("dy", 0) or 0)) > 1
            ]
            if jump_mappings:
                sample = jump_mappings[0]
                status = "failed"
                reason = (
                    "a single key produced multi-cell movement: "
                    f"{sample.get('key')} delta=({sample.get('dx', 0)},{sample.get('dy', 0)})"
                )
                supported_hypotheses.append("a single key press can trigger jump-like movement")
            elif moved_mappings:
                status = "passed"
                moved_keys = [
                    (
                        f"{item.get('key')} "
                        f"({item.get('x1')},{item.get('y1')})->({item.get('x2')},{item.get('y2')}) "
                        f"delta=({item.get('dx')},{item.get('dy')})"
                    ).strip()
                    for item in moved_mappings[:4]
                ]
                reason = "per-key movement results are explicit and stay within one-cell deltas"
                facts.extend(moved_keys)
                disproven_hypotheses.append("a single key press always causes multi-cell jump behavior")
                if handled_mappings and len(handled_mappings) < len(mappings):
                    supported_hypotheses.append("key outcomes depend on current maze state, not every direction is currently available")
            elif mappings:
                status = "observed"
                reason = "per-key outcomes were sampled but no successful movement was observed yet"
                facts.extend(
                    [
                        (
                            f"{item.get('key')} handled={str(bool(item.get('handled', False))).lower()} "
                            f"moved={str(bool(item.get('moved', False))).lower()} "
                            f"delta=({item.get('dx')},{item.get('dy')})"
                        ).strip()
                        for item in mappings[:4]
                    ]
                )
        else:
            structured_lines = [line.strip() for line in semantic_text.splitlines() if line.strip()]
            if transitions or mappings:
                status = "passed"
                reason = "command output contains structured before/after evidence"
                facts.extend(structured_lines[:4])
            elif structured_lines:
                status = "observed"
                reason = "command output exists but the semantic conclusion is still ambiguous"
                facts.extend(structured_lines[:4])

        return {
            "goal": goal,
            "status": status,
            "assertion": str(template.get("assertion", "") or ""),
            "pass_condition": str(template.get("pass_condition", "") or ""),
            "reason": reason,
            "facts": self._dedupe_text_items(facts, keep=4),
            "supported_hypotheses": self._dedupe_text_items(supported_hypotheses, keep=4),
            "disproven_hypotheses": self._dedupe_text_items(disproven_hypotheses, keep=4),
            "tool": tool_name,
            "command": command_preview,
            "timestamp": int(time.time()),
        }

    def _update_command_verification_state(
        self,
        *,
        command_verification_targets: List[Dict[str, Any]],
        recent_command_verifications: List[Dict[str, Any]],
        verification_record: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]], str]:
        targets = list(command_verification_targets or [])
        if not verification_record:
            last_verification = dict(recent_command_verifications[-1]) if recent_command_verifications else {}
            return (
                targets,
                last_verification,
                recent_command_verifications[-self.COMMAND_VERIFICATION_KEEP :],
                self._build_command_verification_digest(recent_command_verifications),
            )

        goal = str(verification_record.get("goal", "") or "").strip()
        if goal and not any(str(item.get("goal", "") or "").strip() == goal for item in targets if isinstance(item, dict)):
            template = dict(
                self.COMMAND_VERIFICATION_TEMPLATES.get(
                    goal,
                    self.COMMAND_VERIFICATION_TEMPLATES["generic_command_assertion"],
                )
            )
            targets.append(
                {
                    "goal": goal,
                    "assertion": str(template.get("assertion", "") or ""),
                    "pass_condition": str(template.get("pass_condition", "") or ""),
                }
            )

        recent = list(recent_command_verifications or [])
        recent.append(dict(verification_record))
        if len(recent) > self.COMMAND_VERIFICATION_KEEP:
            recent = recent[-self.COMMAND_VERIFICATION_KEEP :]
        return (
            targets[-8:],
            dict(verification_record),
            recent,
            self._build_command_verification_digest(recent),
        )

    def _build_command_verification_digest(self, recent_command_verifications: List[Dict[str, Any]]) -> str:
        payload = [
            {
                "goal": str(item.get("goal", "") or ""),
                "status": str(item.get("status", "") or ""),
                "reason": str(item.get("reason", "") or "")[:120],
                "disproven": [
                    str(h or "").strip()
                    for h in (item.get("disproven_hypotheses", []) or [])[:3]
                    if str(h or "").strip()
                ],
            }
            for item in (recent_command_verifications or [])[-8:]
            if isinstance(item, dict)
        ]
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:24]

    def _derive_convergence_state(
        self,
        *,
        recent_mutations: List[Dict[str, Any]],
        recent_tui_verifications: List[Dict[str, Any]],
        recent_command_verifications: List[Dict[str, Any]],
        next_stall_rounds: int,
        next_tui_stall_rounds: int,
        round_dedupe_hits: int,
    ) -> Tuple[str, str, List[str]]:
        recent_failed_mutations = [
            item
            for item in (recent_mutations or [])[-4:]
            if isinstance(item, dict) and str(item.get("status", "") or "").strip().lower() == "error"
        ]
        if len(recent_failed_mutations) >= 2:
            last_item = recent_failed_mutations[-1]
            same_path_failures = [
                item
                for item in recent_failed_mutations
                if str(item.get("path", "") or "").strip()
                and str(item.get("path", "") or "").strip() == str(last_item.get("path", "") or "").strip()
            ]
            same_error_kind = str(last_item.get("error_kind", "") or "").strip()
            if len(same_path_failures) >= 2 or same_error_kind in {"hunk_mismatch", "anchor_mismatch"}:
                path = str(last_item.get("path", "") or "").strip()
                reason = f"Repeated mutation failures require anchor repair for {path or 'the target file'}."
                requirements = [
                    "Read the exact file slice before the next patch.",
                    "Reduce patch scope and include stable anchor lines.",
                    "Treat hunk mismatch as a context mismatch, not a patch syntax failure.",
                    "Do not claim the file changed until a mutation succeeds or read evidence confirms it.",
                ]
                return "repair_anchor", reason, requirements

        recent_tui = [
            item
            for item in (recent_tui_verifications or [])[-3:]
            if isinstance(item, dict)
        ]
        if recent_tui:
            last_tui = recent_tui[-1]
            if (
                bool(last_tui.get("tui_progressed", False))
                and str(last_tui.get("status", "") or "").strip().lower() == "failed"
            ):
                reason = f"TUI changed without semantic proof for {str(last_tui.get('goal', '') or 'the current assertion')}."
                requirements = [
                    "Write an explicit TUI assertion before the next interaction.",
                    "Collect before/after evidence with read_tui_region, find_text_in_tui, or send_keys_and_read.",
                    "Treat screen-hash change as operational progress only, not bug-fix proof.",
                ]
                return "semantic_verify", reason, requirements

        recent_command = [
            item
            for item in (recent_command_verifications or [])[-4:]
            if isinstance(item, dict)
        ]
        if recent_command:
            last_command = recent_command[-1]
            goal = str(last_command.get("goal", "") or "").strip()
            status = str(last_command.get("status", "") or "").strip().lower()
            last_supported = tuple(
                sorted(
                    str(item).strip()
                    for item in (last_command.get("supported_hypotheses", []) or [])
                    if str(item).strip()
                )
            )
            last_disproven = tuple(
                sorted(
                    str(item).strip()
                    for item in (last_command.get("disproven_hypotheses", []) or [])
                    if str(item).strip()
                )
            )
            semantic_repeats = [
                item
                for item in recent_command
                if str(item.get("goal", "") or "").strip() == goal
                and str(item.get("status", "") or "").strip().lower() == status
                and tuple(
                    sorted(
                        str(h).strip()
                        for h in (item.get("supported_hypotheses", []) or [])
                        if str(h).strip()
                    )
                )
                == last_supported
                and tuple(
                    sorted(
                        str(h).strip()
                        for h in (item.get("disproven_hypotheses", []) or [])
                        if str(h).strip()
                    )
                )
                == last_disproven
            ]
            if (
                goal
                and status in {"passed", "failed"}
                and (last_supported or last_disproven or list(last_command.get("facts", []) or []))
                and len(semantic_repeats) >= 2
            ):
                reason = f"Recent command outputs already established semantic evidence for {goal}."
                requirements = [
                    "Cite the latest command_verification facts before proposing another hypothesis.",
                    "List which hypotheses are disproven and do not reopen them without contradictory evidence.",
                    "Choose the next experiment around the remaining unproven cause instead of re-running the same check.",
                ]
                return "semantic_verify", reason, requirements

        if next_stall_rounds >= 2 or next_tui_stall_rounds >= 2 or round_dedupe_hits >= 2:
            return (
                "summarize_blocker",
                "Recent rounds did not add enough trusted evidence.",
                [
                    "Summarize what is proven.",
                    "Summarize what is disproven.",
                    "Choose the smallest next experiment.",
                ],
            )

        return "normal", "", []

    def _build_tool_result_metadata(
        self,
        *,
        tool_name: str,
        mutation_record: Dict[str, Any],
        tui_verification_record: Dict[str, Any],
        command_verification_record: Dict[str, Any],
        output: str,
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "status": "error" if "<error>" in str(output or "").lower() else "success",
        }
        redirected_tool = self._extract_tool_redirect(str(output or ""))
        if redirected_tool:
            metadata["redirected_to"] = redirected_tool
        if mutation_record:
            metadata["file_changed"] = str(bool(mutation_record.get("file_changed", False))).lower()
            error_kind = str(mutation_record.get("error_kind", "") or "").strip()
            if error_kind:
                metadata["error_kind"] = error_kind
            path = str(mutation_record.get("path", "") or "").strip()
            if path:
                metadata["path"] = path
        if tui_verification_record:
            metadata["tui_assertion_goal"] = str(tui_verification_record.get("goal", "") or "").strip()
            metadata["tui_assertion_status"] = str(tui_verification_record.get("status", "") or "").strip()
            metadata["tui_progressed"] = str(bool(tui_verification_record.get("tui_progressed", False))).lower()
        if command_verification_record:
            metadata["command_assertion_goal"] = str(command_verification_record.get("goal", "") or "").strip()
            metadata["command_assertion_status"] = str(command_verification_record.get("status", "") or "").strip()
        return metadata

    def _extract_xml_attr(self, text: str, attr_name: str) -> str:
        pattern = rf'{re.escape(attr_name)}="([^"]*)"'
        match = re.search(pattern, str(text or ""))
        return match.group(1).strip() if match else ""

    def _escape_tool_result_xml(self, value: Any) -> str:
        escape = getattr(self, "_escape_xml", None)
        if callable(escape):
            try:
                return str(escape(value))
            except Exception:
                pass
        return html.escape(str(value or ""), quote=True)

    def _extract_command_stdout(self, output: str) -> str:
        text = str(output or "")
        match = re.search(
            r"\[STDOUT\]\n(.*?)(?=\n\[STDERR\]|\n\[EXIT CODE\]|$)",
            text,
            re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        if "[STDERR]" in text or "[EXIT CODE]" in text:
            return ""
        return text.strip()

    def _normalize_search_path(self, path_text: str) -> Tuple[str, str]:
        normalized = str(path_text or "").strip()
        if not normalized:
            return "", ""
        try:
            if os.path.isabs(normalized):
                abs_path = os.path.abspath(normalized)
            else:
                abs_path = os.path.abspath(os.path.join(self.working_dir, normalized))
        except Exception:
            abs_path = normalized
        rel_path = abs_path
        try:
            rel_path = os.path.relpath(abs_path, self.working_dir)
        except Exception:
            rel_path = abs_path
        return abs_path, rel_path

    def _append_unique_search_item(
        self,
        items: List[Dict[str, Any]],
        item: Dict[str, Any],
        fingerprint: str,
        keep: int = 200,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        if not fingerprint:
            return items, False
        for existing in reversed(items[-keep:]):
            if str(existing.get("fingerprint", "") or "").strip() == fingerprint:
                return items, False
        enriched = dict(item or {})
        enriched["fingerprint"] = fingerprint
        if not enriched.get("id"):
            enriched["id"] = f"search_{uuid.uuid4().hex[:12]}"
        enriched["timestamp"] = int(time.time())
        items.append(enriched)
        if len(items) > keep:
            del items[:-keep]
        return items, True

    def _extract_search_backend_from_output(self, output: str) -> str:
        match = re.search(r'<search_hits[^>]*\sbackend="([^"]+)"', str(output or ""), re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def _extract_search_backend_reason_from_output(self, output: str) -> str:
        match = re.search(
            r'<search_hits[^>]*\sbackend_reason="([^"]+)"',
            str(output or ""),
            re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()
        return ""

    def _update_command_search_state(
        self,
        *,
        search_hits_index: List[Dict[str, Any]],
        file_candidates: List[Dict[str, Any]],
        evidence_index: List[Dict[str, Any]],
        command: str,
        output: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], bool, str]:
        command_text = str(command or "").strip()
        lowered_command = command_text.lower()
        if not re.search(r"(^|[\s|;&(])rg(?:\.exe)?(?=$|[\s|;&)])", lowered_command):
            return search_hits_index, file_candidates, evidence_index, False, ""

        stdout_text = self._extract_command_stdout(output)
        if not stdout_text:
            return search_hits_index, file_candidates, evidence_index, False, "rg"

        changed = False
        backend = "rg"

        if "--files" in lowered_command:
            for raw_line in stdout_text.splitlines():
                candidate_text = str(raw_line or "").strip()
                if not candidate_text or candidate_text == "--":
                    continue
                abs_path, rel_path = self._normalize_search_path(candidate_text)
                if not abs_path:
                    continue
                fingerprint = "file_candidate:" + hashlib.sha1(abs_path.encode("utf-8")).hexdigest()[:20]
                candidate = {
                    "path": abs_path,
                    "rel_path": rel_path,
                    "summary": rel_path,
                    "source_tool": "execute_command",
                    "backend": backend,
                    "command": command_text,
                }
                file_candidates, appended = self._append_unique_search_item(
                    file_candidates,
                    candidate,
                    fingerprint,
                )
                if appended:
                    changed = True
                    evidence_index, _ = self._append_evidence_item(
                        evidence_index,
                        {
                            "type": "file_candidate",
                            "path": abs_path,
                            "summary": rel_path,
                            "reason": "rg --files",
                            "backend": backend,
                            "command": command_text,
                        },
                        fingerprint,
                    )
            return search_hits_index, file_candidates, evidence_index, changed, backend

        seen_line_fingerprints = set()
        for raw_line in stdout_text.splitlines():
            line_text = str(raw_line or "").rstrip()
            if not line_text or line_text == "--":
                continue
            match = re.match(r"^(.*):(\d+)(?::(\d+))?:(.*)$", line_text)
            if not match:
                continue
            path_text, line_no_text, col_text, preview = match.groups()
            abs_path, rel_path = self._normalize_search_path(path_text)
            if not abs_path:
                continue
            preview_text = str(preview or "").strip()
            if len(preview_text) > 220:
                preview_text = preview_text[:220] + "..."
            try:
                line_no = int(line_no_text)
            except Exception:
                continue
            fingerprint_seed = f"{abs_path}:{line_no}:{preview_text[:180]}"
            if fingerprint_seed in seen_line_fingerprints:
                continue
            seen_line_fingerprints.add(fingerprint_seed)
            hit_id = hashlib.sha1(fingerprint_seed.encode("utf-8")).hexdigest()[:12]
            fingerprint = "command_search:" + hashlib.sha1(fingerprint_seed.encode("utf-8")).hexdigest()[:20]
            hit = {
                "id": hit_id,
                "path": abs_path,
                "rel_path": rel_path,
                "line": line_no,
                "column": str(col_text or "").strip(),
                "score": "rg",
                "summary": preview_text,
                "source_tool": "execute_command",
                "backend": backend,
                "command": command_text,
            }
            search_hits_index, appended = self._append_unique_search_item(
                search_hits_index,
                hit,
                fingerprint,
            )
            if appended:
                changed = True
                evidence_index, _ = self._append_evidence_item(
                    evidence_index,
                    {
                        "type": "command_search_hit",
                        "hit_id": hit_id,
                        "path": abs_path,
                        "line": str(line_no),
                        "score": "rg",
                        "summary": preview_text[:220],
                        "reason": "execute_command rg",
                        "backend": backend,
                        "command": command_text,
                    },
                    fingerprint,
                )
        return search_hits_index, file_candidates, evidence_index, changed, backend

    def _parse_terminal_output(self, output: str) -> Dict[str, Any]:
        text = str(output or "")
        if "<terminal_output" not in text and "<terminal_command" not in text and "<terminal_wait" not in text:
            return {}

        parsed: Dict[str, Any] = {}
        parsed["name"] = self._extract_xml_attr(text, "name")
        parsed["op_id"] = self._extract_xml_attr(text, "op_id")
        parsed["status"] = self._extract_xml_attr(text, "status").lower()
        parsed["matched_pattern"] = self._extract_xml_attr(text, "matched_pattern")
        parsed["timed_out"] = self._extract_xml_attr(text, "timed_out").lower() in {"1", "true", "yes", "on"}

        exit_code_text = self._extract_xml_attr(text, "exit_code")
        if exit_code_text:
            try:
                parsed["exit_code"] = int(exit_code_text)
            except Exception:
                parsed["exit_code"] = None
        else:
            parsed["exit_code"] = None

        has_new_output = self._extract_xml_attr(text, "has_new_output").lower()
        parsed["has_new_output"] = has_new_output in {"1", "true", "yes", "on"}

        body_match = re.search(
            r"<terminal_(?:output|command|wait)[^>]*>\s*(.*?)\s*</terminal_(?:output|command|wait)>",
            text,
            re.DOTALL,
        )
        parsed["body"] = body_match.group(1).strip() if body_match else ""
        return parsed

    def _call_terminal_run_command(self, name: str, command: str, op_id: str) -> str:
        runner = getattr(self.terminal_manager, "run_command")
        try:
            return runner(name, command, op_id=op_id)
        except TypeError:
            return runner(name, command)

    def _call_terminal_read(self, name: str, lines: int, op_id: str = "") -> str:
        reader = getattr(self.terminal_manager, "read_terminal")
        try:
            return reader(name, lines, op_id=op_id)
        except TypeError:
            return reader(name, lines)

    def _call_exec_command(self, params: Dict[str, Any]) -> str:
        if not bool(getattr(self, "allow_shell_commands", True)):
            return "<error>exec_command disabled</error>"

        manager = getattr(self, "interactive_exec_manager", None)
        if manager is None:
            return "<error></error>"

        normalized_workdir = str(params.get("workdir", "") or "").strip() or None
        if normalized_workdir:
            try:
                resolved = ToolRegistry._resolve_path(normalized_workdir)
                if not resolved.is_dir():
                    return f"<error>exec_command workdir is not a directory: {normalized_workdir}</error>"
                normalized_workdir = str(resolved)
            except Exception as exc:
                return f"<error>exec_command invalid workdir: {exc}</error>"

        return manager.exec_command(
            cmd=str(params.get("cmd", "") or params.get("command", "") or "").strip(),
            workdir=normalized_workdir,
            tty=self._coerce_bool(params.get("tty", True), default=True),
            yield_time_ms=int(params.get("yield_time_ms", 1200) or 1200),
            max_output_tokens=int(params.get("max_output_tokens", 1200) or 1200),
            timeout_ms=int(params.get("timeout_ms", 0) or 0),
            shell=self._coerce_bool(params.get("shell", True), default=True),
            login=self._coerce_bool(params.get("login", True), default=True),
        )

    def _call_write_stdin(self, params: Dict[str, Any]) -> str:
        manager = getattr(self, "interactive_exec_manager", None)
        if manager is None:
            return "<error></error>"
        return manager.write_stdin(
            session_id=str(params.get("session_id", "") or "").strip(),
            chars=str(params.get("chars", "") or ""),
            yield_time_ms=int(params.get("yield_time_ms", 800) or 800),
            max_output_tokens=int(params.get("max_output_tokens", 1200) or 1200),
        )

    def _call_close_exec_session(self, params: Dict[str, Any]) -> str:
        manager = getattr(self, "interactive_exec_manager", None)
        if manager is None:
            return "<error></error>"
        return manager.close_exec_session(str(params.get("session_id", "") or "").strip())

    def _looks_like_exec_session_id(self, handle: Any) -> bool:
        return bool(re.match(r"^exec_[a-zA-Z0-9]+$", str(handle or "").strip()))

    def _build_exec_session_namespace_error(self, tool_name: str, session_id: str) -> str:
        exec_id = str(session_id or "").strip()
        return (
            f'<error>{tool_name}(name="{exec_id}")  exec .'
            f'{exec_id}  exec_command / launch_interactive_session  session_id, legacy terminal name.</error>\n'
            f'<hint> write_stdin(session_id="{exec_id}", chars="");'
            f' close_exec_session(session_id="{exec_id}").</hint>'
        )

    @staticmethod
    def _normalize_shell_token(token: Any) -> str:
        return str(token or "").strip().strip("\"'")

    def _split_shell_tokens(self, command: str) -> List[str]:
        command_text = str(command or "").strip()
        if not command_text:
            return []
        try:
            return shlex.split(command_text, posix=(os.name != "nt"))
        except Exception:
            return re.findall(r'"[^"]*"|\'[^\']*\'|\S+', command_text)

    def _classify_python_foreground_command(self, command: str) -> str:
        tokens = self._split_shell_tokens(command)
        if not tokens:
            return ""

        launcher = os.path.basename(self._normalize_shell_token(tokens[0])).lower()
        if launcher.endswith(".exe"):
            launcher = launcher[:-4]
        if launcher not in {"python", "python3", "py"}:
            return ""
        if len(tokens) == 1:
            return "starts a Python REPL"

        normalized_tail = [self._normalize_shell_token(token) for token in tokens[1:]]
        lowered_tail = [token.lower() for token in normalized_tail if token]
        if any(flag in lowered_tail for flag in self.PYTHON_ONE_SHOT_FLAGS):
            return ""

        index = 0
        while index < len(normalized_tail):
            token = normalized_tail[index]
            lowered = token.lower()
            if not lowered:
                index += 1
                continue
            if lowered == "-i":
                return "starts python -i"
            if lowered == "-c":
                return ""
            if lowered == "-m":
                module = normalized_tail[index + 1] if index + 1 < len(normalized_tail) else ""
                module_name = module.lower()
                if not module_name:
                    return ""
                if module_name in self.PYTHON_FOREGROUND_SAFE_MODULES:
                    return ""
                return f"launches python -m {module} as a foreground app"
            if token in {"-W", "-X"}:
                index += 2
                continue
            if token in {
                "-b",
                "-bb",
                "-d",
                "-e",
                "-i",
                "-I",
                "-O",
                "-OO",
                "-P",
                "-q",
                "-s",
                "-S",
                "-u",
                "-v",
                "-vv",
                "-vvv",
                "-3",
                "-2",
                "-x",
            }:
                index += 1
                continue
            if token.startswith("-W") or token.startswith("-X"):
                index += 1
                continue
            if lowered == "-x":
                index += 1
                continue
            if lowered.endswith((".py", ".pyw")):
                return f"launches Python script {os.path.basename(token)} as a foreground app"
            if lowered.startswith("-"):
                index += 1
                continue
            return ""
        return ""

    def _classify_interactive_command(self, command: str) -> str:
        command_text = str(command or "").strip()
        if not command_text:
            return ""
        lowered = command_text.lower()
        python_reason = self._classify_python_foreground_command(command_text)
        if python_reason:
            return python_reason

        patterns = [
            (r"(?:^|\s)--mode\s+tui(?:\s|$)", "--mode tui"),
            (r"\btextual\b", "mentions textual"),
            (r"\bcurses?\b", "mentions curses"),
            (r"^\s*(ipython|node|bash|sh|zsh|fish|cmd(?:\.exe)?|powershell(?:\.exe)?|pwsh|vim|nvim|top|htop|watch|less|more)\s*$", "starts an interactive foreground program"),
        ]
        for pattern, reason in patterns:
            if re.search(pattern, lowered, re.IGNORECASE):
                return reason
        return ""

    def _build_execute_command_guidance_error(self, command: str, reason: str, workdir: str = "") -> str:
        command_text = str(command or "").strip()
        hint_parts = [
            f"/({reason}), execute_command.",
            " exec_command(cmd=..., tty=true). write_stdin(session_id=..., chars='').",
        ]
        if str(workdir or "").strip():
            hint_parts.append(f" workdir='{workdir}',.")
        if os.name == "nt":
            hint_parts.append(" shell  PowerShell: cd /d; workdir, Set-Location \"path\"; python ...")
        return (
            f"<error>execute_command : {command_text}</error>\n"
            f"<hint>{' '.join(hint_parts)}</hint>"
        )

    def _extract_tool_redirect(self, output: str) -> str:
        text = str(output or "")
        match = re.search(r'<tool_redirect\b[^>]*to="([^"]+)"', text)
        return match.group(1).strip() if match else ""

    def _effective_tool_name_for_output(self, tool_name: str, output: str) -> str:
        redirected_tool = self._extract_tool_redirect(output)
        if tool_name == "execute_command" and redirected_tool:
            return redirected_tool
        return tool_name

    def _effective_tool_params_for_output(
        self,
        *,
        tool_name: str,
        params: Dict[str, Any],
        output: str,
    ) -> Dict[str, Any]:
        redirect_tool = self._extract_tool_redirect(output)
        if tool_name == "execute_command" and redirect_tool == "exec_command":
            redirected = dict(params or {})
            redirected["cmd"] = str(params.get("command", "") or params.get("cmd", "") or "").strip()
            redirected["workdir"] = str(params.get("workdir", "") or params.get("cwd", "") or "").strip()
            redirected["tty"] = True
            redirected["yield_time_ms"] = int(params.get("yield_time_ms", 1200) or 1200)
            redirected["max_output_tokens"] = int(params.get("max_output_tokens", 1200) or 1200)
            timeout_ms = params.get("timeout_ms")
            if timeout_ms is None and params.get("timeout") is not None:
                try:
                    timeout_ms = max(0, int(params.get("timeout") or 0)) * 1000
                except Exception:
                    timeout_ms = 0
            redirected["timeout_ms"] = int(timeout_ms or 0)
            redirected["shell"] = self._coerce_bool(params.get("shell", True), default=True)
            redirected["login"] = self._coerce_bool(params.get("login", True), default=True)
            return redirected
        return params

    def _redirect_execute_command_to_exec(self, command: str, workdir: str, reason: str, params: Dict[str, Any]) -> str:
        timeout_ms = params.get("timeout_ms")
        if timeout_ms is None and params.get("timeout") is not None:
            try:
                timeout_ms = max(0, int(params.get("timeout") or 0)) * 1000
            except Exception:
                timeout_ms = 0
        exec_output = self._call_exec_command(
            {
                "cmd": command,
                "workdir": workdir or None,
                "tty": True,
                "yield_time_ms": int(params.get("yield_time_ms", 1200) or 1200),
                "max_output_tokens": int(params.get("max_output_tokens", 1200) or 1200),
                "timeout_ms": int(timeout_ms or 0),
                "shell": self._coerce_bool(params.get("shell", True), default=True),
                "login": self._coerce_bool(params.get("login", True), default=True),
            }
        )
        if "<error>" in str(exec_output or "").lower():
            return (
                f"{exec_output}\n"
                f"<system>execute_command /({reason}), exec_command, exec .</system>"
            )
        return (
            f"{exec_output}\n"
            f'<tool_redirect from="execute_command" to="exec_command" reason="{self._escape_tool_result_xml(reason)}" />\n'
            "<system>execute_command 检测到交互式/前台程序，已自动重定向到 exec_command(tty=true)。"
            "可继续使用 write_stdin(session_id=..., chars='') 发送输入；仅在需要屏幕级观察时再切换到 TUI 工具链。</system>"
        )

    def _call_execute_command(self, params: Dict[str, Any]) -> str:
        command = str(params.get("command", "") or params.get("cmd", "") or "").strip()
        if not command:
            return "<error>execute_command  command</error>"

        workdir = str(params.get("cwd", "") or params.get("workdir", "") or "").strip()
        interactive_reason = self._classify_interactive_command(command)
        if interactive_reason:
            return self._redirect_execute_command_to_exec(
                command=command,
                workdir=workdir,
                reason=interactive_reason,
                params=params,
            )

        if os.name == "nt" and re.search(r"(?i)\bcd\s+/d\b", command):
            return (
                "<error> shell  PowerShell, cmd.exe  cd /d.</error>\n"
                "<hint> workdir;, Set-Location \"path\"; your-command.</hint>"
            )

        timeout_seconds = 30
        try:
            if params.get("timeout_ms") is not None:
                timeout_ms = int(params.get("timeout_ms") or 0)
                if timeout_ms > 0:
                    timeout_seconds = max(1, (timeout_ms + 999) // 1000)
            elif params.get("timeout") is not None:
                timeout_seconds = max(1, int(params.get("timeout") or 30))
        except Exception:
            timeout_seconds = 30

        return ToolRegistry.execute_command(
            command=command,
            timeout=timeout_seconds,
            cwd=workdir or None,
        )

    def _extract_exec_session_id(self, output: str, params: Dict[str, Any]) -> str:
        text = str(output or "")
        for attr in ("session_id", "op_id", "name"):
            value = self._extract_xml_attr(text, attr)
            if value:
                return value
        return str((params or {}).get("session_id", "") or "").strip()

    def _find_command_entry(
        self,
        command_executions: List[Dict[str, Any]],
        *,
        op_id: str = "",
        terminal_name: str = "",
        statuses: Tuple[str, ...] = (),
    ) -> Dict[str, Any]:
        wanted_statuses = {str(item).strip().lower() for item in statuses if str(item).strip()}
        for entry in reversed(command_executions or []):
            if not isinstance(entry, dict):
                continue
            if op_id and str(entry.get("op_id", "") or "").strip() != op_id:
                continue
            if terminal_name and str(entry.get("terminal_name", "") or "").strip() != terminal_name:
                continue
            status = str(entry.get("status", "") or "").strip().lower()
            if wanted_statuses and status not in wanted_statuses:
                continue
            return entry
        return {}

    def _record_command_execution(
        self,
        command_executions: List[Dict[str, Any]],
        *,
        tool_name: str,
        params: Dict[str, Any],
        output: str,
        op_id: str = "",
        target_op_id: str = "",
    ) -> List[Dict[str, Any]]:
        text = str(output or "")
        now_ts = int(time.time())
        terminal_state = self._parse_terminal_output(text)

        if tool_name == "run_in_terminal":
            actual_op_id = str(terminal_state.get("op_id", "") or op_id or "").strip()
            status = str(terminal_state.get("status", "") or "").strip().lower()
            if not status:
                status = "failed" if "<error>" in text.lower() else "pending"
            exit_code = terminal_state.get("exit_code")
            if exit_code is None and status == "failed":
                exit_code = 1

            entry = {
                "op_id": actual_op_id or f"cmd_{uuid.uuid4().hex[:10]}",
                "tool": tool_name,
                "terminal_name": str(params.get("name", "") or "").strip(),
                "command": str(params.get("command", "") or "").strip(),
                "mode": "async",
                "status": status,
                "started_at": now_ts,
                "finished_at": now_ts if status in {"completed", "failed"} else None,
                "exit_code": exit_code,
                "stdout_tail": str(terminal_state.get("body", "") or text)[:600],
                "stderr_tail": "" if status != "failed" else str(terminal_state.get("body", "") or text)[:300],
                "expected_checks": ["", ""],
                "poll_spec": {
                    "tool": "wait_terminal_until",
                    "name": str(params.get("name", "") or "").strip(),
                    "op_id": actual_op_id,
                },
            }
            command_executions.append(entry)
            return command_executions[-200:]

        if tool_name in {"read_terminal", "read_terminal_since_last", "wait_terminal_until"}:
            actual_op_id = str(terminal_state.get("op_id", "") or target_op_id or "").strip()
            entry = self._find_command_entry(
                command_executions,
                op_id=actual_op_id,
                terminal_name=str(params.get("name", "") or "").strip(),
            )
            if not entry:
                return command_executions[-200:]

            status = str(terminal_state.get("status", "") or entry.get("status", "") or "").strip().lower()
            if status:
                entry["status"] = status
            if terminal_state.get("exit_code") is not None:
                entry["exit_code"] = terminal_state.get("exit_code")
            body = str(terminal_state.get("body", "") or "").strip()
            if body and body != "()":
                entry["stdout_tail"] = body[:600]
            entry["last_polled_at"] = now_ts
            if status in {"completed", "failed", "timeout"}:
                entry["finished_at"] = now_ts
                if status == "failed" and not entry.get("stderr_tail"):
                    entry["stderr_tail"] = body[:300]
            return command_executions[-200:]

        if tool_name == "run_and_wait":
            actual_op_id = str(terminal_state.get("op_id", "") or op_id or f"cmd_{uuid.uuid4().hex[:10]}").strip()
            status = str(terminal_state.get("status", "") or "").strip().lower() or ("failed" if "<error>" in text.lower() else "completed")
            entry = {
                "op_id": actual_op_id,
                "tool": tool_name,
                "terminal_name": str(params.get("name", "") or "").strip(),
                "command": str(params.get("command", "") or "").strip(),
                "mode": "async_wait",
                "status": status,
                "started_at": now_ts,
                "finished_at": now_ts if status in {"completed", "failed", "timeout"} else None,
                "exit_code": terminal_state.get("exit_code"),
                "stdout_tail": str(terminal_state.get("body", "") or text)[:600],
                "stderr_tail": "" if status == "completed" else str(terminal_state.get("body", "") or text)[:300],
                "expected_checks": ["", ""],
                "poll_spec": {
                    "tool": "wait_terminal_until",
                    "name": str(params.get("name", "") or "").strip(),
                    "op_id": actual_op_id,
                    "pattern": str(params.get("pattern", "") or "").strip(),
                },
            }
            command_executions.append(entry)
            return command_executions[-200:]

        if tool_name in {"exec_command", "launch_interactive_session"}:
            session_id = self._extract_exec_session_id(text, params)
            status = str(terminal_state.get("status", "") or "").strip().lower()
            if not status:
                status = "failed" if "<error>" in text.lower() else "running"
            body = str(terminal_state.get("body", "") or text).strip()
            command_text = str(params.get("cmd", "") or params.get("command", "") or "").strip()
            entry = {
                "op_id": session_id or f"exec_{uuid.uuid4().hex[:10]}",
                "tool": tool_name,
                "terminal_name": session_id,
                "command": command_text,
                "mode": "interactive_pty" if self._coerce_bool(params.get("tty", tool_name == "launch_interactive_session"), default=(tool_name == "launch_interactive_session")) else "interactive_pipe",
                "status": status,
                "started_at": now_ts,
                "finished_at": now_ts if status in {"completed", "failed", "timeout", "closed"} else None,
                "exit_code": terminal_state.get("exit_code"),
                "stdout_tail": body[:600],
                "stderr_tail": "" if status in {"completed", "running", "pending", "closed"} else body[:300],
                "expected_checks": ["", ""],
                "poll_spec": {
                    "tool": "write_stdin",
                    "session_id": session_id,
                    "chars": "",
                    "yield_time_ms": 800,
                },
            }
            command_executions.append(entry)
            return command_executions[-200:]

        if tool_name == "write_stdin":
            session_id = self._extract_exec_session_id(text, params)
            entry = self._find_command_entry(
                command_executions,
                op_id=session_id,
                terminal_name=session_id,
            )
            if not entry:
                return command_executions[-200:]
            status = str(terminal_state.get("status", "") or entry.get("status", "") or "").strip().lower()
            if status:
                entry["status"] = status
            if terminal_state.get("exit_code") is not None:
                entry["exit_code"] = terminal_state.get("exit_code")
            body = str(terminal_state.get("body", "") or "").strip()
            if body and body != "()":
                entry["stdout_tail"] = body[:600]
            entry["last_polled_at"] = now_ts
            if status in {"completed", "failed", "timeout", "closed"}:
                entry["finished_at"] = now_ts
                if status in {"failed", "timeout"}:
                    entry["stderr_tail"] = body[:300]
            return command_executions[-200:]

        if tool_name == "close_exec_session":
            session_id = self._extract_exec_session_id(text, params)
            entry = self._find_command_entry(
                command_executions,
                op_id=session_id,
                terminal_name=session_id,
            )
            if not entry:
                return command_executions[-200:]
            entry["status"] = "closed"
            entry["finished_at"] = now_ts
            if terminal_state.get("exit_code") is not None:
                entry["exit_code"] = terminal_state.get("exit_code")
            return command_executions[-200:]

        if tool_name not in self.COMMAND_TRACKED_TOOLS:
            return command_executions[-200:]

        command_status = "failed" if "<error>" in text.lower() else "completed"
        command_executions.append(
            {
                "op_id": f"cmd_{uuid.uuid4().hex[:10]}",
                "tool": tool_name,
                "terminal_name": str(params.get("name", "") or "").strip(),
                "command": str(params.get("command", "") or params.get("path", "") or "").strip(),
                "mode": "sync",
                "status": command_status,
                "started_at": now_ts,
                "finished_at": now_ts,
                "exit_code": 1 if command_status == "failed" else 0,
                "stdout_tail": text[:600],
                "stderr_tail": "" if command_status == "completed" else text[:300],
                "expected_checks": ["", ""],
                "poll_spec": {},
            }
        )
        return command_executions[-200:]

    def _detect_tui_progress(
        self,
        tool_name: str,
        output: str,
        last_hash: str,
        last_status_tokens: List[str],
    ) -> Tuple[bool, str, List[str], str]:
        lowered = str(output or "").lower()
        if "<error>" in lowered:
            return False, last_hash, list(last_status_tokens or []), "error"

        parsed_hash = self._extract_tui_screen_hash(output)
        parsed_tokens = self._extract_tui_status_tokens(output)
        prev_set = set(last_status_tokens or [])
        next_set = set(parsed_tokens or [])
        status_changed = bool(parsed_tokens) and (next_set != prev_set)
        hash_changed = bool(parsed_hash and parsed_hash != str(last_hash or ""))

        send_changed = False
        if str(tool_name) in {"send_keys", "send_keys_and_read"}:
            match = re.search(r'<tui_send[^>]*\schanged="(true|false)"', str(output or ""), re.IGNORECASE)
            if match:
                send_changed = match.group(1).lower() == "true"
        elif str(tool_name) == "wait_tui_until":
            send_changed = 'status="matched"' in str(output or "").lower()

        progressed = send_changed or hash_changed or status_changed
        reason_parts = []
        if send_changed:
            reason_parts.append("send_changed=true")
        if hash_changed:
            reason_parts.append("screen_hash_changed")
        if status_changed:
            reason_parts.append("status_tokens_changed")
        if not reason_parts:
            reason_parts.append("no_visible_change")
        next_hash = parsed_hash or str(last_hash or "")
        next_tokens = parsed_tokens if parsed_tokens else list(last_status_tokens or [])
        return progressed, next_hash, next_tokens, ",".join(reason_parts)

    def _compact_tui_progress_entries(self, progress: str) -> str:
        lines = [line for line in str(progress or "").splitlines() if line.strip()]
        if not lines:
            return ""
        non_tui_lines = [line for line in lines if not line.startswith("- [TUI]")]
        tui_lines = [line for line in lines if line.startswith("- [TUI]")]
        if len(tui_lines) <= self.TUI_PROGRESS_ENTRY_KEEP:
            return "\n".join(lines) + "\n"
        merged_tail = tui_lines[-self.TUI_PROGRESS_ENTRY_KEEP :]
        merged_prefix = f"- [TUI] : {len(tui_lines) - len(merged_tail)} "
        compacted = non_tui_lines + [merged_prefix] + merged_tail
        return "\n".join(compacted) + "\n"

    def _tool_executor_node(self, state: AgentState) -> Dict[str, Any]:
        """
         - 
        
         LLM ,.
         LLM ,.
        """
        tool_calls = state.get("pending_tool_calls", [])
        if isinstance(tool_calls, list):
            tool_calls = self._expand_parallel_tool_calls(tool_calls)
        
        # ,
        if not tool_calls:
            return {
                "messages": [],
                "pending_tool_calls": [],
                "last_action": "act"
            }
        
        results: List[HumanMessage] = []
        executed_tools: List[Dict[str, Any]] = []
        iteration = int(state.get("iteration_count", 0) or 0)
        thread_id = str(state.get("thread_id", "") or "")
        runtime_checkpoint_id = self._resolve_runtime_checkpoint_id(thread_id)
        ToolRegistry.set_runtime_context(
            thread_id=thread_id,
            checkpoint_id=runtime_checkpoint_id,
        )

        # (state_update / subtask)
        new_overall_goal = None
        new_current_task = None
        new_task_plan = None
        new_current_step_id = None
        new_completed_steps = None
        new_blocked_reason = None
        loaded_skill_names = list(state.get("loaded_skill_names", []) or [])
        skill_context = (state.get("skill_context", "") or "").strip()
        tool_artifacts = list(state.get("tool_artifacts", []) or [])
        subtask_history = list(state.get("subtask_history", []) or [])
        summary_events = list(state.get("summary_events", []) or [])
        command_executions = list(state.get("command_executions", []) or [])
        tool_call_history = list(state.get("tool_call_history", []) or [])
        tool_ledger_recent = list(state.get("tool_ledger_recent", []) or [])
        tool_reuse_cache = dict(state.get("tool_reuse_cache", {}) or {})
        read_coverage = dict(state.get("read_coverage", {}) or {})
        file_views = dict(state.get("file_views", {}) or {})
        active_file_view_paths = list(state.get("active_file_view_paths", []) or [])
        last_read_chunks = list(state.get("last_read_chunks", []) or [])
        evidence_index = list(state.get("evidence_index", []) or [])
        search_hits_index = list(state.get("search_hits_index", []) or [])
        file_candidates = list(state.get("file_candidates", []) or [])
        last_search_backend = str(state.get("last_search_backend", "") or "").strip()
        last_read_fingerprint = str(state.get("last_read_fingerprint", "") or "").strip()
        trusted_modified_files = list(state.get("trusted_modified_files", []) or [])
        last_patch_summaries = list(state.get("last_patch_summaries", []) or [])
        last_mutation = dict(state.get("last_mutation", {}) or {})
        recent_mutations = list(state.get("recent_mutations", []) or [])
        pending_verification_targets = list(state.get("pending_verification_targets", []) or [])
        mutation_truth_digest = str(state.get("mutation_truth_digest", "") or "").strip()
        recent_terminal_events = list(state.get("recent_terminal_events", []) or [])
        recent_tui_events = list(state.get("recent_tui_events", []) or [])
        recent_exec_sessions = list(state.get("recent_exec_sessions", []) or [])
        tui_views = dict(state.get("tui_views", {}) or {})
        active_tui_view_names = list(state.get("active_tui_view_names", []) or [])
        last_tui_snapshots = list(state.get("last_tui_snapshots", []) or [])
        tui_verification_targets = list(state.get("tui_verification_targets", []) or [])
        last_tui_verification = dict(state.get("last_tui_verification", {}) or {})
        recent_tui_verifications = list(state.get("recent_tui_verifications", []) or [])
        tui_verification_digest = str(state.get("tui_verification_digest", "") or "").strip()
        command_verification_targets = list(state.get("command_verification_targets", []) or [])
        last_command_verification = dict(state.get("last_command_verification", {}) or {})
        recent_command_verifications = list(state.get("recent_command_verifications", []) or [])
        command_verification_digest = str(state.get("command_verification_digest", "") or "").strip()
        active_terminal_op = dict(state.get("active_terminal_op", {}) or {})
        active_tui_focus = dict(state.get("active_tui_focus", {}) or {})
        active_exec_session = dict(state.get("active_exec_session", {}) or {})
        last_terminal_delta = dict(state.get("last_terminal_delta", {}) or {})
        last_terminal_failure = dict(state.get("last_terminal_failure", {}) or {})
        last_exec_output = dict(state.get("last_exec_output", {}) or {})
        last_tui_diff = dict(state.get("last_tui_diff", {}) or {})
        last_tui_region = dict(state.get("last_tui_region", {}) or {})
        last_tui_anchor_match = dict(state.get("last_tui_anchor_match", {}) or {})
        dedupe_hits_total = int(state.get("tool_dedupe_hits", 0) or 0)
        repeated_read_hits_total = int(state.get("repeated_read_hits", 0) or 0)
        artifact_saved_count_total = self._coerce_artifact_metric_int(
            state.get("artifact_saved_count", 0)
        )
        artifact_saved_bytes_total = self._coerce_artifact_metric_int(
            state.get("artifact_saved_bytes", 0)
        )
        artifact_cleanup_count_total = self._coerce_artifact_metric_int(
            state.get("artifact_cleanup_count", 0)
        )
        round_artifact_metrics: Dict[str, int] = {
            "saved_count": 0,
            "saved_bytes": 0,
            "cleanup_count": 0,
        }
        read_file_failure_streaks_raw = state.get("read_file_failure_streaks", {})
        read_file_failure_streaks: Dict[str, int] = {}
        if isinstance(read_file_failure_streaks_raw, dict):
            for key, value in read_file_failure_streaks_raw.items():
                normalized_key = str(key or "").strip()
                if not normalized_key:
                    continue
                read_file_failure_streaks[normalized_key] = self._coerce_non_negative_int(value, 0)
        round_dedupe_hits = 0
        round_repeated_read_hits = 0
        round_progressed = False
        prev_stall_rounds = int(state.get("stall_rounds", 0) or 0)
        prev_tui_stall_rounds = int(state.get("tui_stall_rounds", 0) or 0)
        try:
            tui_stall_threshold = int(
                state.get("tui_stall_threshold", int(getattr(self, "tui_stall_threshold", 3) or 3))
                or int(getattr(self, "tui_stall_threshold", 3) or 3)
            )
        except Exception:
            tui_stall_threshold = int(getattr(self, "tui_stall_threshold", 3) or 3)
        tui_stall_threshold = max(0, tui_stall_threshold)
        last_tui_screen_hash = str(state.get("last_tui_screen_hash", "") or "").strip().lower()
        tui_last_status_tokens = [
            str(token).strip().lower()
            for token in (state.get("tui_last_status_tokens", []) or [])
            if str(token).strip()
        ]
        round_tui_progressed = False
        round_tui_attempted = False
        step_execution_stats = dict(state.get("step_execution_stats", {}) or {})
        should_plan = bool(state.get("should_plan", False))
        apply_patch_failure_streak = self._coerce_non_negative_int(
            state.get("apply_patch_failure_streak", 0),
            0,
        )
        completion_by_tool = False
        effective_task_plan = self._normalize_task_plan(
            new_task_plan if isinstance(new_task_plan, list) else state.get("task_plan", [])
        )
        effective_completed_steps = [
            str(item).strip()
            for item in (
                new_completed_steps
                if isinstance(new_completed_steps, list)
                else (state.get("completed_steps", []) or [])
            )
            if str(item).strip()
        ]
        effective_current_step = str(
            new_current_step_id
            if isinstance(new_current_step_id, str)
            else (state.get("current_step_id", "") or "")
        ).strip()
        if should_plan and effective_task_plan and not effective_current_step:
            effective_current_step = self._first_pending_step_id(effective_task_plan, effective_completed_steps)
            if effective_current_step:
                new_current_step_id = effective_current_step
        max_tools_per_step = self._resolve_budget_limit(
            state=state,
            key="max_tools_per_step",
            default_value=int(getattr(self, "max_tools_per_step", 20) or 20),
        )
        max_reads_per_file_per_step = self._resolve_budget_limit(
            state=state,
            key="max_reads_per_file_per_step",
            default_value=int(getattr(self, "max_reads_per_file_per_step", 6) or 6),
        )
        max_retry_per_tool_per_step = self._resolve_budget_limit(
            state=state,
            key="max_retry_per_tool_per_step",
            default_value=int(getattr(self, "max_retry_per_tool_per_step", 2) or 2),
        )
        prefetched_parallel_outputs, parallel_candidate_count = self._prefetch_parallel_tool_outputs(
            tool_calls=tool_calls,
            loaded_skill_names=loaded_skill_names,
            skill_context=skill_context,
        )
        parallel_executed_count = 0
        if parallel_candidate_count > 1:
            self._emit(
                "info",
                f"[Tool Scheduler] parallel candidates={parallel_candidate_count}",
            )
        
        # 
        for i, tool_call in enumerate(tool_calls, 1):
            tool_name = tool_call.get('tool', 'unknown')
            params = tool_call.get('params', {})
            call_id = self._resolve_tool_call_id(tool_call, tool_name, i)
            command_op_id = ""
            command_target_op_id = ""
            reused_result = False
            reused_cache_entry: Dict[str, Any] = {}
            
            # 
            self._emit("tool_exec", f"[ {i}/{len(tool_calls)}] {tool_name}")
            self._print_tool_params(tool_name, params)
            
            try:
                effective_task_plan = self._normalize_task_plan(
                    new_task_plan if isinstance(new_task_plan, list) else state.get("task_plan", [])
                )
                effective_completed_steps = [
                    str(item).strip()
                    for item in (
                        new_completed_steps
                        if isinstance(new_completed_steps, list)
                        else (state.get("completed_steps", []) or [])
                    )
                    if str(item).strip()
                ]
                effective_current_step = str(
                    new_current_step_id
                    if isinstance(new_current_step_id, str)
                    else (state.get("current_step_id", "") or "")
                ).strip()
                if should_plan and effective_task_plan and not effective_current_step:
                    effective_current_step = self._first_pending_step_id(
                        effective_task_plan,
                        effective_completed_steps,
                    )
                    if effective_current_step:
                        new_current_step_id = effective_current_step
                dedupe_blocked, fingerprint, tool_call_history, duplicate_count = self._should_block_duplicate_tool_call(
                    state=state,
                    tool_call_history=tool_call_history,
                    tool_name=tool_name,
                    params=params,
                    iteration=iteration,
                )
                if dedupe_blocked:
                    round_dedupe_hits += 1
                    reused_cache_entry = dict(tool_reuse_cache.get(fingerprint, {}) or {})
                if not self._is_tool_active(tool_name):
                    output = f"<error>Tool not active in current group set: {tool_name}</error>"
                    self._emit("warning", f"[Tool Policy] blocked inactive tool: {tool_name}")
                    inactive_metadata = {"status": "error", "blocked_reason": "inactive_tool", "call_id": call_id}
                    result_content = self._build_tool_result_message(
                        tool_name=tool_name,
                        params=params if isinstance(params, dict) else {},
                        output=output,
                        tool_artifacts=tool_artifacts,
                        thread_id=thread_id,
                        metadata=inactive_metadata,
                        artifact_metrics=round_artifact_metrics,
                        call_id=call_id,
                        reused=False,
                    )
                    results.append(HumanMessage(content=result_content))
                    tool_ledger_recent = self._append_tool_ledger_recent(
                        tool_ledger_recent,
                        {
                            "call_id": call_id,
                            "tool": tool_name,
                            "fingerprint": fingerprint,
                            "status": "error",
                            "reused": False,
                            "artifact_id": str(inactive_metadata.get("artifact_id", "") or "").strip(),
                            "summary": output,
                        },
                    )
                    executed_tools.append({
                        "tool": tool_name,
                        "params": params,
                        "call_id": call_id,
                        "success": False,
                        "status_label": "inactive",
                        "error": "inactive_tool",
                        "parallel_mode": self._tool_parallel_safety(tool_name, params if isinstance(params, dict) else {}),
                        "deduped": bool(dedupe_blocked),
                        "reused": False,
                        "timestamp": int(time.time()),
                    })
                    progress_lines.append(self._format_progress_entry(tool_name, params, "ERROR"))
                    continue

                # ====== Unified state update ======
                if tool_name == "state_update":
                    new_task_val = params.get("current_task")
                    new_goal_val = params.get("overall_goal")
                    new_task_plan_val = params.get("task_plan")
                    new_current_step_id_val = params.get("current_step_id")
                    new_completed_steps_val = params.get("completed_steps")
                    new_blocked_reason_val = params.get("blocked_reason")
                    todos_val = params.get("todos")
                    outputs = []
                    if new_goal_val:
                        new_overall_goal = str(new_goal_val).strip()
                        outputs.append(f": {new_goal_val}")
                        self._emit("info", f"[/] {new_goal_val}")
                    if new_task_val:
                        new_current_task = str(new_task_val).strip()
                        outputs.append(f": {new_task_val}")
                        self._emit("info", f"[/] {new_task_val}")
                    if isinstance(new_task_plan_val, list):
                        new_task_plan = self._normalize_task_plan(new_task_plan_val)
                        outputs.append(f"({len(new_task_plan)} )")
                        self._emit("info", f"[]  {len(new_task_plan)} ")
                    if new_current_step_id_val is not None:
                        new_current_step_id = str(new_current_step_id_val).strip()
                        if new_current_step_id:
                            outputs.append(f": {new_current_step_id}")
                    if isinstance(new_completed_steps_val, list):
                        completed_unique = []
                        completed_seen = set()
                        for item in new_completed_steps_val:
                            step_id = str(item).strip()
                            if not step_id or step_id in completed_seen:
                                continue
                            completed_seen.add(step_id)
                            completed_unique.append(step_id)
                        new_completed_steps = completed_unique
                        outputs.append(f": {len(new_completed_steps)}")
                    if should_plan:
                        active_plan = self._normalize_task_plan(
                            new_task_plan if isinstance(new_task_plan, list) else state.get("task_plan", [])
                        )
                        active_completed = [
                            str(item).strip()
                            for item in (
                                new_completed_steps
                                if isinstance(new_completed_steps, list)
                                else (state.get("completed_steps", []) or [])
                            )
                            if str(item).strip()
                        ]
                        plan_step_ids = {
                            str(step.get("id", "") or "").strip()
                            for step in active_plan
                            if isinstance(step, dict)
                        }
                        plan_step_ids = {sid for sid in plan_step_ids if sid}
                        if (
                            new_current_step_id
                            and plan_step_ids
                            and new_current_step_id not in plan_step_ids
                        ):
                            fallback_step = self._first_pending_step_id(active_plan, active_completed)
                            new_current_step_id = fallback_step
                            if fallback_step:
                                outputs.append(f": {fallback_step}")
                            else:
                                outputs.append(":")
                        if not new_current_step_id and active_plan:
                            fallback_step = self._first_pending_step_id(active_plan, active_completed)
                            new_current_step_id = fallback_step
                            if fallback_step:
                                outputs.append(f": {fallback_step}")
                    if new_blocked_reason_val is not None:
                        new_blocked_reason = str(new_blocked_reason_val).strip()
                        if new_blocked_reason:
                            outputs.append(f": {new_blocked_reason}")
                    if isinstance(todos_val, list):
                        normalized_plan: List[Dict[str, Any]] = []
                        normalized_completed: List[str] = []
                        normalized_current_step = ""
                        normalized_current_task = ""
                        for idx, item in enumerate(todos_val):
                            if not isinstance(item, dict):
                                continue
                            step_id = str(item.get("id", "") or f"todo_{idx + 1}").strip()
                            title = str(item.get("content", "") or item.get("title", "") or step_id).strip()
                            status = str(item.get("status", "") or "pending").strip().lower()
                            if status not in {"pending", "in_progress", "completed"}:
                                status = "pending"
                            if status == "in_progress" and not normalized_current_step:
                                normalized_current_step = step_id
                                normalized_current_task = title
                            if status == "completed":
                                normalized_completed.append(step_id)
                            normalized_plan.append(
                                {
                                    "id": step_id,
                                    "title": title,
                                    "status": status,
                                }
                            )
                        if not normalized_current_step:
                            for step in normalized_plan:
                                if str(step.get("status", "")).strip() in {"pending", "in_progress"}:
                                    normalized_current_step = str(step.get("id", "") or "").strip()
                                    normalized_current_task = str(step.get("title", "") or "").strip()
                                    break
                        new_task_plan = self._normalize_task_plan(normalized_plan)
                        new_completed_steps = normalized_completed
                        new_current_step_id = normalized_current_step
                        if normalized_current_task:
                            new_current_task = normalized_current_task
                        outputs.append(f"todos={len(normalized_plan)}")

                    output = ". " + (" ; ".join(outputs) if outputs else "()")

                # ======  ======
                elif tool_name == "run_subtask":
                    output, subtask_entry, loaded_skill_names, skill_context = self._execute_subtask_actions(
                        params=params,
                        tool_artifacts=tool_artifacts,
                        loaded_skill_names=loaded_skill_names,
                        skill_context=skill_context,
                        thread_id=thread_id,
                        artifact_metrics=round_artifact_metrics,
                    )
                    if subtask_entry:
                        subtask_history.append(subtask_entry)
                # ======  ======
                else:
                    if dedupe_blocked and reused_cache_entry:
                        output = self._build_reused_tool_output(reused_cache_entry, tool_name)
                        reused_result = True
                    else:
                        prefetched = prefetched_parallel_outputs.get(i - 1)
                        if prefetched is not None:
                            if str(prefetched.get("error", "") or "").strip():
                                raise RuntimeError(str(prefetched.get("error", "") or "").strip())
                            output = prefetched.get("output", "")
                            parallel_executed_count += 1
                        else:
                            if tool_name == "run_in_terminal":
                                command_op_id = f"cmd_{uuid.uuid4().hex[:10]}"
                            elif tool_name in {"read_terminal", "read_terminal_since_last", "wait_terminal_until"}:
                                target_entry = self._find_command_entry(
                                    command_executions,
                                    op_id=str(params.get("op_id", "") or "").strip(),
                                    terminal_name=str(params.get("name", "") or "").strip(),
                                    statuses=("pending", "running"),
                                )
                                command_target_op_id = str(target_entry.get("op_id", "") or "").strip()
                            output, loaded_skill_names, skill_context = self._execute_single_tool_action(
                                tool_name=tool_name,
                                params=params,
                                loaded_skill_names=loaded_skill_names,
                                skill_context=skill_context,
                                command_op_id=command_op_id,
                                command_target_op_id=command_target_op_id,
                                tool_artifacts=tool_artifacts,
                                thread_id=thread_id,
                            )
                
                #  Actionable Hint()
                if "<error>" in output:
                    mutation_error_kind = ""
                    if self._is_mutation_tool(tool_name):
                        mutation_error_kind = self._classify_mutation_error_kind(tool_name, output)
                    if tool_name == "apply_patch":
                        output += "\n" + self._build_mutation_failure_hint(tool_name, mutation_error_kind)

                if tool_name == "apply_patch":
                    apply_patch_failure_streak = self._update_apply_patch_failure_streak(
                        output_text=output,
                        failure_streak=apply_patch_failure_streak,
                    )

                if tool_name == "read_file":
                    read_failure_fingerprint = str(fingerprint or "").strip()
                    failed_now = "<error>" in str(output or "").lower()
                    if failed_now and read_failure_fingerprint:
                        next_streak = (
                            self._coerce_non_negative_int(
                                read_file_failure_streaks.get(read_failure_fingerprint, 0),
                                0,
                            )
                            + 1
                        )
                        read_file_failure_streaks[read_failure_fingerprint] = next_streak
                        if next_streak == 2:
                            fallback_hint = self._build_read_file_terminal_hint(
                                params if isinstance(params, dict) else {}
                            )
                            output += (
                                "\n<hint>read_file failed repeatedly with the same parameters. "
                                "Try a terminal read to verify path/encoding and collect evidence: "
                                f"{self._escape_tool_result_xml(fallback_hint)}</hint>"
                            )
                            self._emit(
                                "warning",
                                "[Hint] read_file repeated failure on identical params. "
                                f"Prefer {fallback_hint}",
                            )
                    elif read_failure_fingerprint:
                        read_file_failure_streaks.pop(read_failure_fingerprint, None)
                
                # ()
                self._print_tool_output(output)

                coverage_changed = False
                evidence_changed = False
                file_views_changed = False
                if tool_name == "search_in_files":
                    backend = self._extract_search_backend_from_output(output)
                    if backend:
                        last_search_backend = backend
                    evidence_index, evidence_changed = self._update_search_evidence_index(
                        evidence_index=evidence_index,
                        output=output,
                    )
                if tool_name == "execute_command" and self._extract_tool_redirect(output) != "exec_command":
                    (
                        search_hits_index,
                        file_candidates,
                        evidence_index,
                        command_search_changed,
                        command_search_backend,
                    ) = self._update_command_search_state(
                        search_hits_index=search_hits_index,
                        file_candidates=file_candidates,
                        evidence_index=evidence_index,
                        command=str((params or {}).get("command", "") or ""),
                        output=output,
                    )
                    if command_search_backend:
                        last_search_backend = command_search_backend
                    evidence_changed = evidence_changed or command_search_changed
                if tool_name == "read_file":
                    (
                        read_coverage,
                        coverage_changed,
                        repeated_hit,
                        new_fingerprint,
                    ) = self._update_read_coverage(read_coverage, output)
                    if new_fingerprint:
                        last_read_fingerprint = new_fingerprint
                    if repeated_hit:
                        round_repeated_read_hits += 1
                    evidence_index, read_evidence_changed = self._update_read_evidence_index(
                        evidence_index=evidence_index,
                        output=output,
                    )
                    evidence_changed = evidence_changed or read_evidence_changed
                    (
                        file_views,
                        active_file_view_paths,
                        last_read_chunks,
                        file_views_changed,
                    ) = self._update_file_views_state(
                        file_views=file_views,
                        active_file_view_paths=active_file_view_paths,
                        last_read_chunks=last_read_chunks,
                        output=output,
                        current_focus=str(state.get("current_focus", "") or ""),
                    )

                tui_progressed = False
                tui_progress_reason = ""
                if self._is_tui_tool(tool_name):
                    round_tui_attempted = True
                    (
                        tui_progressed,
                        last_tui_screen_hash,
                        tui_last_status_tokens,
                        tui_progress_reason,
                    ) = self._detect_tui_progress(
                        tool_name=tool_name,
                        output=output,
                        last_hash=last_tui_screen_hash,
                        last_status_tokens=tui_last_status_tokens,
                    )
                    if tui_progressed:
                        round_tui_progressed = True

                progressed = self._tool_execution_advances_progress(
                    tool_name=tool_name,
                    output=output,
                    coverage_changed=coverage_changed,
                    evidence_changed=evidence_changed,
                    file_views_changed=file_views_changed,
                    tui_progressed=tui_progressed,
                )
                if progressed:
                    round_progressed = True

                effective_tool_name = self._effective_tool_name_for_output(tool_name, str(output or ""))
                effective_params = self._effective_tool_params_for_output(
                    tool_name=tool_name,
                    params=params if isinstance(params, dict) else {},
                    output=str(output or ""),
                )

                if should_plan and tool_name not in self.PLAN_STATE_TOOLS:
                    active_step_for_record = str(
                        new_current_step_id
                        if isinstance(new_current_step_id, str)
                        else (state.get("current_step_id", "") or "")
                    ).strip()
                    if active_step_for_record:
                        failed_now = "<error>" in str(output or "").lower()
                        step_execution_stats = self._record_step_tool_usage(
                            step_execution_stats=step_execution_stats,
                            step_id=active_step_for_record,
                            tool_name=tool_name,
                            params=params if isinstance(params, dict) else {},
                            failed=failed_now,
                        )
                
                command_executions = self._record_command_execution(
                    command_executions,
                    tool_name=effective_tool_name,
                    params=effective_params,
                    output=str(output or ""),
                    op_id=command_op_id,
                    target_op_id=command_target_op_id,
                )
                command_verification_record = self._build_command_verification_record(
                    state=state,
                    tool_name=effective_tool_name,
                    params=effective_params,
                    output=str(output or ""),
                )
                (
                    command_verification_targets,
                    last_command_verification,
                    recent_command_verifications,
                    command_verification_digest,
                ) = self._update_command_verification_state(
                    command_verification_targets=command_verification_targets,
                    recent_command_verifications=recent_command_verifications,
                    verification_record=command_verification_record,
                )
                trusted_modified_files, last_patch_summaries = self._record_patch_success(
                    tool_name=tool_name,
                    params=params if isinstance(params, dict) else {},
                    output=str(output or ""),
                    trusted_modified_files=trusted_modified_files,
                    last_patch_summaries=last_patch_summaries,
                )
                mutation_record = self._build_mutation_record(
                    tool_name=tool_name,
                    params=params if isinstance(params, dict) else {},
                    output=str(output or ""),
                )
                (
                    last_mutation,
                    recent_mutations,
                    pending_verification_targets,
                    mutation_truth_digest,
                ) = self._update_mutation_truth_state(
                    recent_mutations=recent_mutations,
                    pending_verification_targets=pending_verification_targets,
                    mutation_record=mutation_record,
                )
                if self._is_tui_tool(tool_name) or tool_name == "activate_tui_mode":
                    tui_verification_targets = self._ensure_tui_verification_targets(tui_verification_targets)
                tui_verification_record = self._build_tui_verification_record(
                    state=state,
                    tool_name=tool_name,
                    params=params if isinstance(params, dict) else {},
                    output=str(output or ""),
                    tui_progressed=tui_progressed,
                )
                (
                    last_tui_verification,
                    recent_tui_verifications,
                    tui_verification_digest,
                ) = self._update_tui_verification_state(
                    recent_tui_verifications=recent_tui_verifications,
                    verification_record=tui_verification_record,
                )
                if tool_name in self.COMMAND_TRACKED_TOOLS:
                    (
                        recent_terminal_events,
                        terminal_focus,
                        terminal_delta,
                        terminal_failure,
                    ) = self._record_terminal_event(
                        events=recent_terminal_events,
                        tool_name=tool_name,
                        params=params if isinstance(params, dict) else {},
                        output=str(output or ""),
                    )
                    if terminal_focus:
                        active_terminal_op = terminal_focus
                    elif tool_name in {"wait_terminal_until", "run_and_wait", "read_terminal", "read_terminal_since_last"}:
                        status = str(self._parse_terminal_output(str(output or "")).get("status", "") or "").strip().lower()
                        if status in {"completed", "failed", "timeout", "idle"}:
                            active_terminal_op = {}
                    if terminal_delta:
                        last_terminal_delta = terminal_delta
                    if terminal_failure:
                        last_terminal_failure = terminal_failure
                if effective_tool_name in {"exec_command", "write_stdin", "close_exec_session", "launch_interactive_session"}:
                    (
                        recent_exec_sessions,
                        exec_focus,
                        exec_output,
                    ) = self._record_exec_session_event(
                        events=recent_exec_sessions,
                        tool_name=effective_tool_name,
                        params=effective_params,
                        output=str(output or ""),
                    )
                    if exec_focus:
                        active_exec_session = exec_focus
                    elif effective_tool_name in {"write_stdin", "close_exec_session"}:
                        exec_status = str(self._parse_terminal_output(str(output or "")).get("status", "") or "").strip().lower()
                        if exec_status in {"completed", "failed", "timeout", "closed"}:
                            active_exec_session = {}
                    if exec_output:
                        last_exec_output = exec_output
                if self._is_tui_tool(tool_name):
                    (
                        recent_tui_events,
                        tui_focus,
                        tui_diff_entry,
                        tui_region_entry,
                    ) = self._record_tui_event(
                        events=recent_tui_events,
                        tool_name=tool_name,
                        params=params if isinstance(params, dict) else {},
                        output=str(output or ""),
                    )
                    if tui_focus:
                        active_tui_focus = tui_focus
                    if tui_diff_entry:
                        last_tui_diff = tui_diff_entry
                    if tui_region_entry:
                        last_tui_region = tui_region_entry
                    if tool_name in {"find_text_in_tui", "wait_tui_until"} and tui_focus:
                        last_tui_anchor_match = tui_focus
                    (
                        tui_views,
                        active_tui_view_names,
                        last_tui_snapshots,
                        _,
                    ) = self._update_tui_views_state(
                        tui_views=tui_views,
                        active_tui_view_names=active_tui_view_names,
                        last_tui_snapshots=last_tui_snapshots,
                        output=str(output or ""),
                        tool_name=tool_name,
                        params=params if isinstance(params, dict) else {},
                        current_focus=str(state.get("current_focus", "") or ""),
                    )
                elif tool_name == "close_tui" and "<error>" not in str(output or "").lower():
                    closed_name = str(params.get("name", "") or "").strip()
                    if closed_name:
                        tui_views.pop(closed_name, None)
                        active_tui_view_names = [
                            item for item in active_tui_view_names if str(item or "").strip() != closed_name
                        ][: self.TUI_VIEW_SESSIONS_MAX]
                        last_tui_snapshots = [
                            item
                            for item in last_tui_snapshots
                            if str((item or {}).get("session_name", "") or "").strip() != closed_name
                        ][-self.LAST_TUI_SNAPSHOTS_KEEP_MAX :]
                        if str(active_tui_focus.get("session_name", "") or "").strip() == closed_name:
                            active_tui_focus = {}
                        if str(last_tui_diff.get("session_name", "") or "").strip() == closed_name:
                            last_tui_diff = {}
                        if str(last_tui_region.get("session_name", "") or "").strip() == closed_name:
                            last_tui_region = {}
                        if str(last_tui_anchor_match.get("session_name", "") or "").strip() == closed_name:
                            last_tui_anchor_match = {}

                tool_result_metadata = self._build_tool_result_metadata(
                    tool_name=tool_name,
                    mutation_record=mutation_record,
                    tui_verification_record=tui_verification_record,
                    command_verification_record=command_verification_record,
                    output=str(output or ""),
                )
                tool_result_metadata["call_id"] = call_id
                if reused_result:
                    tool_result_metadata["reused"] = "true"
                result_content = self._build_tool_result_message(
                    tool_name=tool_name,
                    params=params,
                    output=output,
                    tool_artifacts=tool_artifacts,
                    thread_id=thread_id,
                    metadata=tool_result_metadata,
                    artifact_metrics=round_artifact_metrics,
                    call_id=call_id,
                    reused=reused_result,
                )
                results.append(HumanMessage(content=result_content))

                artifact_id = str(tool_result_metadata.get("artifact_id", "") or "").strip()
                output_text = str(output or "")
                tool_ledger_recent = self._append_tool_ledger_recent(
                    tool_ledger_recent,
                    {
                        "call_id": call_id,
                        "tool": tool_name,
                        "fingerprint": fingerprint,
                        "status": str(tool_result_metadata.get("status", "") or ""),
                        "reused": reused_result,
                        "artifact_id": artifact_id,
                        "summary": output_text,
                    },
                )
                if str(tool_result_metadata.get("status", "") or "").strip().lower() == "success":
                    tool_reuse_cache[str(fingerprint or "")] = {
                        "timestamp": int(time.time()),
                        "tool": tool_name,
                        "status": str(tool_result_metadata.get("status", "") or ""),
                        "preview": self._shorten_tool_result_preview(output_text, self.TOOL_REUSE_PREVIEW_LIMIT),
                        "artifact_id": artifact_id,
                    }
                    tool_reuse_cache = self._prune_tool_reuse_cache(tool_reuse_cache)

                # 
                executed_tools.append({
                    "tool": tool_name,
                    "params": params,
                    "call_id": call_id,
                    "success": True,
                    "status_label": "",
                    "parallel_mode": self._tool_parallel_safety(tool_name, params if isinstance(params, dict) else {}),
                    "deduped": bool(dedupe_blocked),
                    "reused": bool(reused_result),
                    "tui_progress_reason": tui_progress_reason,
                    "mutation_status": str(mutation_record.get("status", "") or ""),
                    "mutation_file_changed": bool(mutation_record.get("file_changed", False)),
                    "tui_assertion_goal": str(tui_verification_record.get("goal", "") or ""),
                    "tui_assertion_status": str(tui_verification_record.get("status", "") or ""),
                    "command_assertion_goal": str(command_verification_record.get("goal", "") or ""),
                    "command_assertion_status": str(command_verification_record.get("status", "") or ""),
                })
                
            except Exception as e:
                # ,
                error_msg = f": {str(e)}"
                self._emit("error", f"  : {error_msg}")
                error_output = f"<error>{error_msg}</error>"
                if tool_name == "apply_patch":
                    apply_patch_failure_streak = self._update_apply_patch_failure_streak(
                        output_text=error_output,
                        failure_streak=apply_patch_failure_streak,
                    )

                if should_plan and tool_name not in self.PLAN_STATE_TOOLS:
                    active_step_for_record = str(
                        new_current_step_id
                        if isinstance(new_current_step_id, str)
                        else (state.get("current_step_id", "") or "")
                    ).strip()
                    if active_step_for_record:
                        step_execution_stats = self._record_step_tool_usage(
                            step_execution_stats=step_execution_stats,
                            step_id=active_step_for_record,
                            tool_name=tool_name,
                            params=params if isinstance(params, dict) else {},
                            failed=True,
                        )

                mutation_record = self._build_mutation_record(
                    tool_name=tool_name,
                    params=params if isinstance(params, dict) else {},
                    output=error_output,
                )
                (
                    last_mutation,
                    recent_mutations,
                    pending_verification_targets,
                    mutation_truth_digest,
                ) = self._update_mutation_truth_state(
                    recent_mutations=recent_mutations,
                    pending_verification_targets=pending_verification_targets,
                    mutation_record=mutation_record,
                )

                error_metadata = self._build_tool_result_metadata(
                    tool_name=tool_name,
                    mutation_record=mutation_record,
                    tui_verification_record={},
                    command_verification_record={},
                    output=error_output,
                )
                error_metadata["call_id"] = call_id
                result_content = self._build_tool_result_message(
                    tool_name=tool_name,
                    params=params if isinstance(params, dict) else {},
                    output=error_output,
                    tool_artifacts=tool_artifacts,
                    thread_id=thread_id,
                    metadata=error_metadata,
                    artifact_metrics=round_artifact_metrics,
                    call_id=call_id,
                    reused=False,
                )
                results.append(HumanMessage(content=result_content))

                tool_ledger_recent = self._append_tool_ledger_recent(
                    tool_ledger_recent,
                    {
                        "call_id": call_id,
                        "tool": tool_name,
                        "fingerprint": fingerprint,
                        "status": "error",
                        "reused": False,
                        "artifact_id": "",
                        "summary": error_output,
                    },
                )

                executed_tools.append({
                    "tool": tool_name,
                    "params": params,
                    "call_id": call_id,
                    "success": False,
                    "status_label": "",
                    "error": str(e),
                    "parallel_mode": self._tool_parallel_safety(tool_name, params if isinstance(params, dict) else {}),
                    "deduped": False,
                    "reused": False,
                    "mutation_status": str(mutation_record.get("status", "") or ""),
                    "mutation_file_changed": bool(mutation_record.get("file_changed", False)),
                })
        
        self._emit("info", f"[]  {len(tool_calls)} ")
        if parallel_executed_count > 0:
            self._emit(
                "info",
                f"[Tool Scheduler] parallel executed={parallel_executed_count}",
            )
        if self._coerce_artifact_metric_int(round_artifact_metrics.get("saved_count", 0)) > 0:
            self._emit(
                "info",
                "[Tool Artifact] "
                f"saved={self._coerce_artifact_metric_int(round_artifact_metrics.get('saved_count', 0))} "
                f"bytes={self._coerce_artifact_metric_int(round_artifact_metrics.get('saved_bytes', 0))} "
                f"cleanup={self._coerce_artifact_metric_int(round_artifact_metrics.get('cleanup_count', 0))}",
            )
        
        # 
        progress = state.get("progress_summary", "")
        for tool_info in executed_tools:
            tool_name = tool_info["tool"]
            params = tool_info["params"]
            status = str(
                tool_info.get("status_label")
                or ("" if tool_info.get("success", False) else "")
            )
            
            entry = self._format_progress_entry(tool_name, params, status)
            if entry:
                progress += entry + "\n"

        progress = self._compact_tui_progress_entries(progress)
        next_stall_rounds = 0 if round_progressed else (prev_stall_rounds + 1)
        if round_tui_attempted:
            next_tui_stall_rounds = 0 if round_tui_progressed else (prev_tui_stall_rounds + 1)
        else:
            next_tui_stall_rounds = prev_tui_stall_rounds
        evidence_digest = self._build_evidence_digest(
            progress=progress,
            read_coverage=read_coverage,
            file_views=file_views,
            evidence_index=evidence_index,
            search_hits_index=search_hits_index,
            file_candidates=file_candidates,
            executed_tools=executed_tools,
        )

        dedupe_hits_total += round_dedupe_hits
        repeated_read_hits_total += round_repeated_read_hits
        if round_tui_attempted:
            self._emit(
                "info",
                f"[TUI ] TUI, progress={round_tui_progressed}, "
                f"tui_stall_rounds={next_tui_stall_rounds}/{tui_stall_threshold}",
            )

        convergence_mode, convergence_reason, next_step_requirements = self._derive_convergence_state(
            recent_mutations=recent_mutations,
            recent_tui_verifications=recent_tui_verifications,
            recent_command_verifications=recent_command_verifications,
            next_stall_rounds=next_stall_rounds,
            next_tui_stall_rounds=next_tui_stall_rounds,
            round_dedupe_hits=round_dedupe_hits,
        )

        previous_task_plan = self._normalize_task_plan(state.get("task_plan", []) or [])
        final_task_plan = self._normalize_task_plan(
            new_task_plan if isinstance(new_task_plan, list) else previous_task_plan
        )
        previous_completed_steps = [
            str(item).strip()
            for item in (state.get("completed_steps", []) or [])
            if str(item).strip()
        ]
        final_completed_steps = (
            [
                str(item).strip()
                for item in new_completed_steps
                if str(item).strip()
            ]
            if isinstance(new_completed_steps, list)
            else previous_completed_steps
        )
        final_completed_set = set(final_completed_steps)
        previous_completed_set = set(previous_completed_steps)
        final_current_step = str(
            new_current_step_id
            if isinstance(new_current_step_id, str)
            else (state.get("current_step_id", "") or "")
        ).strip()

        if should_plan and final_task_plan:
            if not final_current_step or final_current_step in final_completed_set:
                next_step = self._first_pending_step_id(final_task_plan, final_completed_steps)
                if next_step != final_current_step:
                    new_current_step_id = next_step
                    final_current_step = next_step
                    if next_step:
                        self._emit("info", f"[] : {next_step}")

        checkpoint_count = 0
        plan_title_map = {
            str(item.get("id", "") or "").strip(): str(item.get("title", "") or "").strip()
            for item in final_task_plan
            if isinstance(item, dict)
        }
        for step_id in sorted(final_completed_set - previous_completed_set):
            step_title = plan_title_map.get(step_id, step_id)
            checkpoint_event = self._build_step_checkpoint_event(
                step_id=step_id,
                step_title=step_title,
                executed_tools=executed_tools,
                tool_artifacts=tool_artifacts,
                blocked_reason=str(new_blocked_reason if new_blocked_reason is not None else state.get("blocked_reason", "") or "").strip(),
            )
            summary_events = self._append_summary_event(summary_events, checkpoint_event)
            checkpoint_count += 1
        if checkpoint_count > 0:
            self._emit("info", f"[Checkpoint]  {checkpoint_count} ")

        # Do not interrupt execution for repeated tool/read patterns.
        # Keep convergence stats for observability, but avoid clarification gating.
        needs_user_confirmation = False
        clarification_question = ""
        active_skill_paths: List[str] = []
        skill_path_getter = getattr(self, "get_skill_paths_for_names", None)
        if callable(skill_path_getter):
            active_skill_paths = list(skill_path_getter(loaded_skill_names) or [])
        else:
            for skill_name in loaded_skill_names:
                meta = getattr(self, "skill_registry", {}).get(skill_name, {}) if hasattr(self, "skill_registry") else {}
                skill_path = str(meta.get("skill_md_path", "") or meta.get("skill_dir", "") or "").strip()
                if skill_path:
                    active_skill_paths.append(skill_path)
        result = {
            "messages": results,
            "pending_tool_calls": [],
            "last_action": "clarify" if needs_user_confirmation else "act",
            "empty_response_count": 0,
            "task_completed": bool(completion_by_tool),
            "progress_summary": progress,
            "loaded_skill_names": loaded_skill_names,
            "skill_context": skill_context,
            "active_tool_groups": list(getattr(self, "active_tool_groups", []) or []),
            "active_tool_names": sorted(list(getattr(self, "tools", {}).keys())),
            "active_skill_names": list(loaded_skill_names),
            "active_skill_paths": active_skill_paths,
            "skill_registry_digest": str(getattr(self, "skill_registry_digest", "") or ""),
            "skill_injection_mode": str(getattr(self, "skill_injection_mode", "hybrid") or "hybrid"),
            "permission_mode": str(getattr(self, "permission_mode", "workspace") or "workspace"),
            "allow_shell_commands": bool(getattr(self, "allow_shell_commands", True)),
            "command_executions": command_executions[-200:],
            "tool_artifacts": tool_artifacts,
            "subtask_history": subtask_history[-100:],
            "tool_call_history": tool_call_history[-self.TOOL_DEDUPE_KEEP_MAX :],
            "tool_ledger_recent": tool_ledger_recent[-self.TOOL_LEDGER_KEEP_MAX :],
            "tool_reuse_cache": self._prune_tool_reuse_cache(tool_reuse_cache),
            "parallel_executed_count": int(parallel_executed_count),
            "artifact_saved_count_round": self._coerce_artifact_metric_int(
                round_artifact_metrics.get("saved_count", 0)
            ),
            "artifact_saved_bytes_round": self._coerce_artifact_metric_int(
                round_artifact_metrics.get("saved_bytes", 0)
            ),
            "artifact_cleanup_count_round": self._coerce_artifact_metric_int(
                round_artifact_metrics.get("cleanup_count", 0)
            ),
            "artifact_saved_count": artifact_saved_count_total
            + self._coerce_artifact_metric_int(round_artifact_metrics.get("saved_count", 0)),
            "artifact_saved_bytes": artifact_saved_bytes_total
            + self._coerce_artifact_metric_int(round_artifact_metrics.get("saved_bytes", 0)),
            "artifact_cleanup_count": artifact_cleanup_count_total
            + self._coerce_artifact_metric_int(round_artifact_metrics.get("cleanup_count", 0)),
            "tool_dedupe_hits": dedupe_hits_total,
            "repeated_read_hits": repeated_read_hits_total,
            "read_file_failure_streaks": read_file_failure_streaks,
            "read_coverage": read_coverage,
            "file_views": file_views,
            "active_file_view_paths": active_file_view_paths[: self.FILE_VIEW_FILES_MAX],
            "last_read_chunks": last_read_chunks[-self.LAST_READ_CHUNKS_KEEP_MAX :],
            "last_read_fingerprint": last_read_fingerprint,
            "evidence_index": evidence_index[-300:],
            "evidence_digest": evidence_digest,
            "search_hits_index": search_hits_index[-200:],
            "last_search_backend": last_search_backend,
            "file_candidates": file_candidates[-200:],
            "trusted_modified_files": trusted_modified_files[-50:],
            "last_patch_summaries": last_patch_summaries[-20:],
            "last_mutation": last_mutation,
            "recent_mutations": recent_mutations[-self.MUTATION_HISTORY_KEEP :],
            "pending_verification_targets": pending_verification_targets[-20:],
            "mutation_truth_digest": mutation_truth_digest,
            "recent_terminal_events": recent_terminal_events[-40:],
            "recent_tui_events": recent_tui_events[-40:],
            "recent_exec_sessions": recent_exec_sessions[-40:],
            "tui_views": tui_views,
            "active_tui_view_names": active_tui_view_names[: self.TUI_VIEW_SESSIONS_MAX],
            "last_tui_snapshots": last_tui_snapshots[-self.LAST_TUI_SNAPSHOTS_KEEP_MAX :],
            "tui_verification_targets": tui_verification_targets,
            "last_tui_verification": last_tui_verification,
            "recent_tui_verifications": recent_tui_verifications[-self.TUI_VERIFICATION_KEEP :],
            "tui_verification_digest": tui_verification_digest,
            "command_verification_targets": command_verification_targets[-8:],
            "last_command_verification": last_command_verification,
            "recent_command_verifications": recent_command_verifications[-self.COMMAND_VERIFICATION_KEEP :],
            "command_verification_digest": command_verification_digest,
            "active_terminal_op": active_terminal_op,
            "active_tui_focus": active_tui_focus,
            "active_exec_session": active_exec_session,
            "last_terminal_delta": last_terminal_delta,
            "last_terminal_failure": last_terminal_failure,
            "last_exec_output": last_exec_output,
            "last_tui_diff": last_tui_diff,
            "last_tui_region": last_tui_region,
            "last_tui_anchor_match": last_tui_anchor_match,
            "stall_rounds": next_stall_rounds,
            "tui_stall_rounds": next_tui_stall_rounds,
            "tui_stall_threshold": tui_stall_threshold,
            "last_tui_screen_hash": last_tui_screen_hash,
            "tui_last_status_tokens": tui_last_status_tokens,
            "summary_events": summary_events,
            "blocked_reason": (
                str(new_blocked_reason).strip()
                if new_blocked_reason is not None
                else (
                    convergence_reason
                    if convergence_mode != "normal"
                    else str(state.get("blocked_reason", "") or "").strip()
                )
            ),
            "convergence_mode": convergence_mode,
            "convergence_reason": convergence_reason,
            "next_step_requirements": next_step_requirements,
            "step_execution_stats": step_execution_stats,
            "max_tools_per_step": max_tools_per_step,
            "max_reads_per_file_per_step": max_reads_per_file_per_step,
            "max_retry_per_tool_per_step": max_retry_per_tool_per_step,
            "apply_patch_failure_streak": apply_patch_failure_streak,
            "should_plan": bool(final_task_plan),
            "needs_user_confirmation": needs_user_confirmation,
            "clarification_question": clarification_question,
            "ended_without_tools": False,
            "has_new_user_input": False,
        }
        
        # state_update patch
        if new_overall_goal is not None:
            result["overall_goal"] = new_overall_goal
        if new_current_task is not None:
            result["current_task"] = new_current_task
            result["current_focus"] = new_current_task
        if new_task_plan is not None:
            result["task_plan"] = new_task_plan
        if new_current_step_id is not None:
            result["current_step_id"] = new_current_step_id
        if new_completed_steps is not None:
            result["completed_steps"] = new_completed_steps
        if new_blocked_reason is not None:
            result["blocked_reason"] = new_blocked_reason
        if new_overall_goal is not None and new_current_task is None:
            result["current_focus"] = new_overall_goal
        
        #  TUI ,
        if getattr(self, '_tui_guide_loaded', False):
            result["tui_guide_loaded"] = True

        return self._apply_after_tools_hooks(
            state=state,
            iteration=iteration,
            tool_result=result,
            executed_tools=executed_tools,
        )


    def _apply_after_tools_hooks(
        self,
        state: AgentState,
        iteration: int,
        tool_result: Dict[str, Any],
        executed_tools: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        hook_runner = getattr(self, "_run_after_tools_hooks", None)
        if callable(hook_runner):
            return hook_runner(
                state=state,
                iteration=iteration,
                tool_result=tool_result,
                executed_tools=executed_tools,
            )
        return tool_result

    def _resolve_tool_output_delivery_mode(self) -> str:
        raw_mode = str(
            getattr(
                self,
                "tool_output_delivery_mode",
                self.TOOL_OUTPUT_DELIVERY_DEFAULT,
            )
            or self.TOOL_OUTPUT_DELIVERY_DEFAULT
        ).strip().lower()
        if raw_mode in {"full_inline", "hybrid", "artifact_first"}:
            return raw_mode
        return self.TOOL_OUTPUT_DELIVERY_DEFAULT

    def _resolve_tool_output_hard_ceiling(self) -> int:
        try:
            raw = int(
                getattr(
                    self,
                    "tool_output_hard_ceiling_chars",
                    self.TOOL_OUTPUT_HARD_CEILING_DEFAULT,
                )
                or self.TOOL_OUTPUT_HARD_CEILING_DEFAULT
            )
        except Exception:
            raw = self.TOOL_OUTPUT_HARD_CEILING_DEFAULT
        return max(4000, raw)
    @staticmethod
    def _shorten_tool_result_preview(text: str, limit: int) -> str:
        payload = str(text or "").strip()
        if not payload:
            return ""
        capped = max(int(limit or 0), 80)
        if len(payload) <= capped:
            return payload
        omitted = len(payload) - capped
        return f"{payload[:capped]}\n...[truncated {omitted} chars]"

    def _build_tool_result_message(
        self,
        tool_name: str,
        params: Dict[str, Any],
        output: Any,
        tool_artifacts: List[Dict[str, Any]],
        thread_id: str,
        metadata: Any = None,
        artifact_metrics: Any = None,
        call_id: str = "",
        reused: bool = False,
    ) -> str:
        output_text = str(output or "")
        metadata_ref: Dict[str, Any] = metadata if isinstance(metadata, dict) else {}
        metadata = dict(metadata_ref)
        if "status" not in metadata:
            metadata["status"] = "error" if "<error>" in output_text.lower() else "success"
        if call_id:
            metadata["call_id"] = call_id
        if reused:
            metadata["reused"] = "true"

        mode = self._resolve_tool_output_delivery_mode()
        should_externalize = self._should_externalize_tool_output(tool_name, output_text)
        artifact: Dict[str, Any] = {}
        inline_output = output_text

        if should_externalize:
            artifact = self._store_tool_artifact(
                tool_artifacts=tool_artifacts,
                tool_name=tool_name,
                params=params if isinstance(params, dict) else {},
                output_text=output_text,
                thread_id=thread_id,
            )
            self._accumulate_artifact_metrics(artifact_metrics, artifact)
            artifact_id = str(artifact.get("id", "") or "").strip()
            store_key = str(artifact.get("content_store_key", "") or "").strip()
            if artifact_id:
                metadata["artifact_id"] = artifact_id
            if store_key:
                metadata["content_store_key"] = store_key
            metadata["delivery_mode"] = mode
            metadata["content_truncated"] = "true"
            metadata["total_chars"] = str(len(output_text))
            metadata["total_lines"] = str(max(len(output_text.splitlines()), 1))
            preview_limit = self.TOOL_RESULT_PREVIEW_LIMIT
            if mode == "artifact_first":
                preview_limit = max(400, min(self.TOOL_RESULT_PREVIEW_LIMIT, 800))
            preview_head = self._shorten_tool_result_preview(output_text, preview_limit)
            preview_tail = self._shorten_tool_result_preview(output_text[-preview_limit:], preview_limit)
            receipt_lines = [
                "[tool_output_truncated]",
                f"total_lines={metadata['total_lines']} total_chars={metadata['total_chars']}",
            ]
            if preview_head:
                receipt_lines.extend(["<head>", preview_head, "</head>"])
            if preview_tail and preview_tail != preview_head:
                receipt_lines.extend(["<tail>", preview_tail, "</tail>"])
            if artifact_id:
                receipt_lines.append(f"[artifact_ref] id={artifact_id}")
            receipt_lines.append(
                "[system] Full output is stored internally. Relevant slices will be auto-hydrated into context when needed."
            )
            receipt_lines.append(
                "[system] If evidence is still insufficient, rerun the tool with narrower scope to fetch targeted output."
            )
            inline_output = "\n".join(receipt_lines)

        attr_parts = []
        for key, value in metadata.items():
            key_text = str(key or "").strip()
            value_text = str(value or "").strip()
            if key_text and value_text:
                attr_parts.append(f'{key_text}="{self._escape_tool_result_xml(value_text)}"')
        attrs = " ".join(attr_parts)
        if attrs:
            attrs = " " + attrs
        if isinstance(metadata_ref, dict):
            metadata_ref.clear()
            metadata_ref.update(metadata)
        return (
            f"<tool_result tool=\"{self._escape_tool_result_xml(tool_name)}\"{attrs}>\n"
            f"{inline_output}\n"
            f"</tool_result>"
        )

    def _store_tool_artifact(
        self,
        tool_artifacts: List[Dict[str, Any]],
        tool_name: str,
        params: Dict[str, Any],
        output_text: str,
        thread_id: str,
    ) -> Dict[str, Any]:
        artifact_id = f"tool_{uuid.uuid4().hex[:12]}"
        artifact: Dict[str, Any] = {
            "id": artifact_id,
            "tool": tool_name,
            "params": params,
            "size": len(output_text),
        }
        store_key = f"tool_artifact:{thread_id or 'default'}:{artifact_id}"
        try:
            store = getattr(self, "store", None)
            if store is not None and hasattr(store, "set"):
                store.set(store_key, output_text)
                artifact["content_store_key"] = store_key
            else:
                artifact["content"] = output_text
        except Exception:
            artifact["content"] = output_text
        tool_artifacts.append(artifact)
        cleanup_count = 0
        if len(tool_artifacts) > self.TOOL_ARTIFACT_KEEP_MAX:
            stale_items = list(tool_artifacts[:-self.TOOL_ARTIFACT_KEEP_MAX])
            cleanup_count = len(stale_items)
            for stale in stale_items:
                if not isinstance(stale, dict):
                    continue
                stale_store_key = str(stale.get("content_store_key", "") or "").strip()
                if not stale_store_key:
                    continue
                try:
                    store = getattr(self, "store", None)
                    if store is not None and hasattr(store, "delete"):
                        store.delete(stale_store_key)
                except Exception:
                    continue
            del tool_artifacts[:-self.TOOL_ARTIFACT_KEEP_MAX]
        artifact["cleanup_count"] = cleanup_count
        return artifact

    @staticmethod
    def _coerce_artifact_metric_int(value: Any) -> int:
        try:
            return max(int(value), 0)
        except Exception:
            return 0

    def _accumulate_artifact_metrics(self, artifact_metrics: Any, artifact: Any) -> None:
        if not isinstance(artifact_metrics, dict) or not isinstance(artifact, dict):
            return
        artifact_metrics["saved_count"] = self._coerce_artifact_metric_int(
            artifact_metrics.get("saved_count", 0)
        ) + 1
        artifact_metrics["saved_bytes"] = self._coerce_artifact_metric_int(
            artifact_metrics.get("saved_bytes", 0)
        ) + self._coerce_artifact_metric_int(artifact.get("size", 0))
        artifact_metrics["cleanup_count"] = self._coerce_artifact_metric_int(
            artifact_metrics.get("cleanup_count", 0)
        ) + self._coerce_artifact_metric_int(artifact.get("cleanup_count", 0))

    def _should_externalize_tool_output(self, tool_name: str, output_text: str) -> bool:
        text = str(output_text or "")
        if not text:
            return False
        normalized_tool = str(tool_name or "").strip()
        hard_ceiling = self._resolve_tool_output_hard_ceiling()
        if len(text) > hard_ceiling:
            return True
        if normalized_tool in self.TOOL_RESULT_EXTERNALIZE_EXCLUDED:
            return False

        mode = self._resolve_tool_output_delivery_mode()
        if "<error>" in text.lower() and len(text) <= self.TOOL_RESULT_INLINE_LIMIT:
            return False
        if mode == "artifact_first":
            return True
        if mode == "hybrid":
            inline_limit = self.TOOL_RESULT_INLINE_LIMIT
            if normalized_tool not in self.FULL_INLINE_PRIMARY_TOOLS:
                inline_limit = min(inline_limit, self.TOOL_RESULT_PREVIEW_LIMIT)
            return len(text) > inline_limit
        return False

    def _normalize_tool_params(self, params: Dict[str, Any]) -> str:
        try:
            return json.dumps(params or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except Exception:
            return str(params or {})


    def _tool_call_fingerprint(self, tool_name: str, params: Dict[str, Any]) -> str:
        params_for_fingerprint: Dict[str, Any] = dict(params or {})
        if tool_name == "read_file":
            params_for_fingerprint.pop("reason", None)
            params_for_fingerprint.pop("hit_id", None)
        normalized = self._normalize_tool_params(params_for_fingerprint)
        digest = hashlib.sha256(f"{tool_name}:{normalized}".encode("utf-8")).hexdigest()[:16]
        return f"{tool_name}:{digest}"


    def _resolve_tool_call_id(self, tool_call: Dict[str, Any], tool_name: str, index: int) -> str:
        payload = dict(tool_call or {}) if isinstance(tool_call, dict) else {}
        for key in ("call_id", "id", "tool_call_id"):
            value = str(payload.get(key, "") or "").strip()
            if value:
                return value
            fn = payload.get("function", {}) if isinstance(payload.get("function"), dict) else {}
            nested_value = str(fn.get(key, "") or "").strip()
            if nested_value:
                return nested_value
        return f"call_{int(index or 0)}_{str(tool_name or '').strip() or 'tool'}"

    def _append_tool_ledger_recent(
        self,
        tool_ledger_recent: List[Dict[str, Any]],
        entry: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if not isinstance(entry, dict):
            return tool_ledger_recent
        record = {
            "timestamp": int(time.time()),
            "call_id": str(entry.get("call_id", "") or "").strip(),
            "tool": str(entry.get("tool", "") or "").strip(),
            "fingerprint": str(entry.get("fingerprint", "") or "").strip(),
            "status": str(entry.get("status", "") or "").strip(),
            "reused": bool(entry.get("reused", False)),
            "artifact_id": str(entry.get("artifact_id", "") or "").strip(),
            "summary": self._shorten_tool_result_preview(str(entry.get("summary", "") or ""), 220),
        }
        tool_ledger_recent.append(record)
        if len(tool_ledger_recent) > self.TOOL_LEDGER_KEEP_MAX:
            del tool_ledger_recent[:-self.TOOL_LEDGER_KEEP_MAX]
        return tool_ledger_recent

    def _prune_tool_reuse_cache(self, tool_reuse_cache: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        if not isinstance(tool_reuse_cache, dict):
            return {}
        if len(tool_reuse_cache) <= self.TOOL_REUSE_CACHE_KEEP_MAX:
            return tool_reuse_cache
        ordered = sorted(
            tool_reuse_cache.items(),
            key=lambda item: int((item[1] or {}).get("timestamp", 0) or 0),
            reverse=True,
        )[: self.TOOL_REUSE_CACHE_KEEP_MAX]
        return {str(key): dict(value or {}) for key, value in ordered}

    def _build_reused_tool_output(self, cache_entry: Dict[str, Any], tool_name: str) -> str:
        artifact_id = str(cache_entry.get("artifact_id", "") or "").strip()
        preview = self._shorten_tool_result_preview(
            str(cache_entry.get("preview", "") or ""),
            self.TOOL_REUSE_PREVIEW_LIMIT,
        )
        lines = [f"<info>reused_recent_result tool={tool_name}</info>"]
        if artifact_id:
            lines.append(f"<info>artifact_id={artifact_id}</info>")
        if preview:
            lines.append(preview)
        return "\n".join(lines)

    def _append_tool_call_history(
        self,
        tool_call_history: List[Dict[str, Any]],
        tool_name: str,
        params: Dict[str, Any],
        iteration: int,
        fingerprint: str,
    ) -> List[Dict[str, Any]]:
        tool_call_history.append(
            {
                "fingerprint": fingerprint,
                "tool": str(tool_name or "").strip(),
                "params": self._normalize_tool_params(params),
                "iteration": int(iteration or 0),
                "timestamp": int(time.time()),
            }
        )
        if len(tool_call_history) > self.TOOL_DEDUPE_KEEP_MAX:
            del tool_call_history[:-self.TOOL_DEDUPE_KEEP_MAX]
        return tool_call_history


    def _should_block_duplicate_tool_call(
        self,
        state: AgentState,
        tool_call_history: List[Dict[str, Any]],
        tool_name: str,
        params: Dict[str, Any],
        iteration: int,
    ) -> Tuple[bool, str, List[Dict[str, Any]], int]:
        fingerprint = self._tool_call_fingerprint(tool_name, params)
        dedupe_enabled = bool(state.get("dedupe_tools", True))
        if not dedupe_enabled:
            return (
                False,
                fingerprint,
                self._append_tool_call_history(tool_call_history, tool_name, params, iteration, fingerprint),
                1,
            )

        if tool_name in {"state_update"}:
            return (
                False,
                fingerprint,
                self._append_tool_call_history(tool_call_history, tool_name, params, iteration, fingerprint),
                1,
            )

        try:
            dedupe_window = int(
                state.get("tool_dedupe_window", self.TOOL_DEDUPE_WINDOW_DEFAULT)
                or self.TOOL_DEDUPE_WINDOW_DEFAULT
            )
        except Exception:
            dedupe_window = self.TOOL_DEDUPE_WINDOW_DEFAULT
        dedupe_window = max(1, dedupe_window)

        recent_duplicate_count = 0
        current_iteration = int(iteration or 0)
        for item in reversed(tool_call_history):
            if str(item.get("fingerprint", "") or "") != fingerprint:
                continue
            try:
                prev_iteration = int(item.get("iteration", 0) or 0)
            except Exception:
                prev_iteration = 0
            if (current_iteration - prev_iteration) <= dedupe_window:
                recent_duplicate_count += 1
                continue
            break

        updated_history = self._append_tool_call_history(
            tool_call_history,
            tool_name,
            params,
            iteration,
            fingerprint,
        )
        duplicate_count = recent_duplicate_count + 1
        dedupe_blocked = recent_duplicate_count > 0
        return (
            dedupe_blocked,
            fingerprint,
            updated_history,
            duplicate_count,
        )


    def _parse_file_chunk_meta(self, output: str) -> Dict[str, Any]:
        text = str(output or "")
        match = re.search(r"<file_chunk\s+([^>]*)>", text, re.IGNORECASE)
        if not match:
            return {}
        attrs = {}
        for key, value in re.findall(r'([a-zA-Z0-9_]+)="([^"]*)"', match.group(1)):
            attrs[str(key)] = html.unescape(str(value))
        return attrs

    def _parse_file_chunk_body(self, output: str) -> str:
        text = str(output or "")
        match = re.search(
            r"<file_chunk\b[^>]*>\s*(.*?)\s*</file_chunk>",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return ""
        return match.group(1).strip("\n\r")

    def _parse_file_chunk_lines(self, output: str) -> Dict[int, str]:
        line_map: Dict[int, str] = {}
        body = self._parse_file_chunk_body(output)
        if not body:
            return line_map

        for raw_line in body.splitlines():
            match = re.match(r"^\s*(\d+)\s\|\s?(.*)$", str(raw_line or ""))
            if not match:
                continue
            try:
                line_no = int(match.group(1))
            except Exception:
                continue
            line_map[line_no] = match.group(2)
        return line_map

    def _normalize_file_view_ranges(self, ranges: Any) -> List[List[int]]:
        normalized: List[List[int]] = []
        for item in ranges or []:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            try:
                start_line = int(item[0])
                end_line = int(item[1])
            except Exception:
                continue
            if start_line <= 0 or end_line <= 0:
                continue
            if end_line < start_line:
                start_line, end_line = end_line, start_line
            normalized.append([start_line, end_line])
        normalized.sort(key=lambda pair: (pair[0], pair[1]))
        return normalized

    def _normalize_chunk_records(self, chunks: Any) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for item in chunks or []:
            if not isinstance(item, dict):
                continue
            try:
                start_line = int(item.get("start_line", 0) or 0)
                end_line = int(item.get("end_line", 0) or 0)
            except Exception:
                continue
            if start_line <= 0 or end_line <= 0:
                continue
            if end_line < start_line:
                start_line, end_line = end_line, start_line
            normalized.append(
                {
                    "start_line": start_line,
                    "end_line": end_line,
                    "chunk_id": str(item.get("chunk_id", "") or "").strip(),
                    "content_hash": str(item.get("content_hash", "") or "").strip(),
                    "updated_at": int(item.get("updated_at", 0) or 0),
                }
            )
        normalized.sort(
            key=lambda item: (
                int(item.get("updated_at", 0) or 0),
                int(item.get("start_line", 0) or 0),
                int(item.get("end_line", 0) or 0),
            )
        )
        return normalized[-self.FILE_VIEW_CHUNKS_KEEP_MAX :]

    def _normalize_line_map(self, line_map: Any) -> Dict[int, str]:
        normalized: Dict[int, str] = {}
        if not isinstance(line_map, dict):
            return normalized
        for key, value in line_map.items():
            try:
                line_no = int(key)
            except Exception:
                continue
            if line_no <= 0:
                continue
            normalized[line_no] = str(value or "")
        return normalized

    def _snippet_content_from_line_map(
        self,
        line_map: Dict[int, str],
        start_line: int,
        end_line: int,
    ) -> str:
        lines: List[str] = []
        for line_no in range(start_line, end_line + 1):
            if line_no not in line_map:
                continue
            lines.append(f"{line_no:4d} | {line_map[line_no]}")
        return "\n".join(lines)

    def _rebuild_file_view_snippets(
        self,
        snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        ranges = self._normalize_file_view_ranges(snapshot.get("ranges", []))
        line_map = self._normalize_line_map(snapshot.get("line_map", {}))
        chunks = self._normalize_chunk_records(snapshot.get("chunks", []))
        snippets: List[Dict[str, Any]] = []

        for start_line, end_line in ranges:
            content = self._snippet_content_from_line_map(line_map, start_line, end_line)
            if not content:
                continue
            overlapping_chunks = [
                item
                for item in chunks
                if int(item.get("start_line", 0) or 0) <= end_line
                and int(item.get("end_line", 0) or 0) >= start_line
            ]
            overlapping_chunks.sort(key=lambda item: int(item.get("updated_at", 0) or 0), reverse=True)

            chunk_ids: List[str] = []
            content_hashes: List[str] = []
            latest_updated = int(snapshot.get("last_updated", 0) or 0)
            for item in overlapping_chunks:
                chunk_id = str(item.get("chunk_id", "") or "").strip()
                content_hash = str(item.get("content_hash", "") or "").strip()
                updated_at = int(item.get("updated_at", 0) or 0)
                latest_updated = max(latest_updated, updated_at)
                if chunk_id and chunk_id not in chunk_ids:
                    chunk_ids.append(chunk_id)
                if content_hash and content_hash not in content_hashes:
                    content_hashes.append(content_hash)

            snippets.append(
                {
                    "start_line": start_line,
                    "end_line": end_line,
                    "content": content,
                    "chunk_ids": chunk_ids[:6],
                    "content_hashes": content_hashes[:6],
                    "updated_at": latest_updated,
                }
            )

        snippets.sort(
            key=lambda item: (
                int(item.get("updated_at", 0) or 0),
                int(item.get("end_line", 0) or 0),
            ),
            reverse=True,
        )
        snapshot["ranges"] = ranges
        snapshot["line_map"] = line_map
        snapshot["chunks"] = chunks
        snapshot["snippets"] = snippets[: self.FILE_VIEW_SNIPPETS_PER_FILE]
        return snapshot

    def _append_last_read_chunk(
        self,
        last_read_chunks: List[Dict[str, Any]],
        chunk_meta: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if not isinstance(chunk_meta, dict) or not chunk_meta.get("path"):
            return last_read_chunks
        last_read_chunks.append(dict(chunk_meta))
        if len(last_read_chunks) > self.LAST_READ_CHUNKS_KEEP_MAX:
            del last_read_chunks[:-self.LAST_READ_CHUNKS_KEEP_MAX]
        return last_read_chunks

    def _score_file_view_path(self, path: str, current_focus: str, last_read_chunks: List[Dict[str, Any]]) -> Tuple[int, int]:
        focus_text = str(current_focus or "").strip().lower()
        normalized_path = str(path or "").strip()
        lowered_path = normalized_path.lower()
        score = 0
        if focus_text:
            basename = os.path.basename(lowered_path)
            if basename and basename in focus_text:
                score += 4
            rel_path = lowered_path.replace("\\", "/")
            if rel_path and rel_path in focus_text:
                score += 4
            for segment in [seg for seg in re.split(r"[\\/]+", rel_path) if seg]:
                if len(segment) >= 3 and segment in focus_text:
                    score += 1

        recency = 0
        for item in reversed(last_read_chunks or []):
            if str(item.get("path", "") or "").strip() != normalized_path:
                continue
            recency = int(item.get("updated_at", 0) or 0)
            break
        return score, recency

    def _refresh_active_file_view_paths(
        self,
        file_views: Dict[str, Any],
        active_file_view_paths: List[str],
        last_read_chunks: List[Dict[str, Any]],
        current_focus: str,
    ) -> List[str]:
        candidate_paths = set()
        for path in file_views.keys():
            normalized_path = str(path or "").strip()
            if normalized_path:
                candidate_paths.add(normalized_path)
        for path in active_file_view_paths or []:
            normalized_path = str(path or "").strip()
            if normalized_path and normalized_path in file_views:
                candidate_paths.add(normalized_path)
        for item in last_read_chunks or []:
            normalized_path = str(item.get("path", "") or "").strip()
            if normalized_path and normalized_path in file_views:
                candidate_paths.add(normalized_path)

        ranked = sorted(
            candidate_paths,
            key=lambda path: (
                self._score_file_view_path(path, current_focus, last_read_chunks)[0],
                self._score_file_view_path(path, current_focus, last_read_chunks)[1],
                int((file_views.get(path, {}) or {}).get("last_updated", 0) or 0),
                path,
            ),
            reverse=True,
        )
        return ranked[: self.FILE_VIEW_FILES_MAX]

    def _update_file_views_state(
        self,
        file_views: Dict[str, Any],
        active_file_view_paths: List[str],
        last_read_chunks: List[Dict[str, Any]],
        output: str,
        current_focus: str,
    ) -> Tuple[Dict[str, Any], List[str], List[Dict[str, Any]], bool]:
        meta = self._parse_file_chunk_meta(output)
        if not meta:
            return file_views, active_file_view_paths, last_read_chunks, False

        path = str(meta.get("path", "") or "").strip()
        if not path:
            return file_views, active_file_view_paths, last_read_chunks, False

        try:
            start_line = int(meta.get("start_line", 0) or 0)
            end_line = int(meta.get("end_line", 0) or 0)
            total_lines = int(meta.get("total_lines", 0) or 0)
        except Exception:
            return file_views, active_file_view_paths, last_read_chunks, False

        if start_line <= 0 or end_line <= 0:
            return file_views, active_file_view_paths, last_read_chunks, False
        if end_line < start_line:
            start_line, end_line = end_line, start_line

        line_map = self._parse_file_chunk_lines(output)
        if not line_map:
            return file_views, active_file_view_paths, last_read_chunks, False

        now_ts = int(time.time())
        snapshot = dict(file_views.get(path, {}) or {})
        snapshot["path"] = path
        snapshot["total_lines"] = max(total_lines, 0)
        snapshot["reads"] = int(snapshot.get("reads", 0) or 0) + 1
        snapshot["last_chunk_id"] = str(meta.get("chunk_id", "") or "").strip()
        snapshot["last_content_hash"] = str(meta.get("content_hash", "") or "").strip()
        snapshot["last_updated"] = now_ts

        stored_line_map = self._normalize_line_map(snapshot.get("line_map", {}))
        old_line_map = dict(stored_line_map)
        stored_line_map.update(line_map)
        snapshot["line_map"] = stored_line_map

        old_ranges = self._normalize_file_view_ranges(snapshot.get("ranges", []))
        snapshot["ranges"] = self._merge_ranges(old_ranges, start_line, end_line)

        chunks = self._normalize_chunk_records(snapshot.get("chunks", []))
        chunks.append(
            {
                "start_line": start_line,
                "end_line": end_line,
                "chunk_id": str(meta.get("chunk_id", "") or "").strip(),
                "content_hash": str(meta.get("content_hash", "") or "").strip(),
                "updated_at": now_ts,
            }
        )
        snapshot["chunks"] = chunks[-self.FILE_VIEW_CHUNKS_KEEP_MAX :]

        old_snippets = list(snapshot.get("snippets", []) or [])
        snapshot = self._rebuild_file_view_snippets(snapshot)
        file_views[path] = snapshot

        chunk_meta = {
            "path": path,
            "start_line": start_line,
            "end_line": end_line,
            "chunk_id": str(meta.get("chunk_id", "") or "").strip(),
            "content_hash": str(meta.get("content_hash", "") or "").strip(),
            "updated_at": now_ts,
        }
        last_read_chunks = self._append_last_read_chunk(last_read_chunks, chunk_meta)
        active_file_view_paths = self._refresh_active_file_view_paths(
            file_views=file_views,
            active_file_view_paths=active_file_view_paths,
            last_read_chunks=last_read_chunks,
            current_focus=current_focus,
        )

        changed = (
            snapshot.get("ranges", []) != old_ranges
            or old_line_map != stored_line_map
            or old_snippets != snapshot.get("snippets", [])
        )
        return file_views, active_file_view_paths, last_read_chunks, changed

    def _parse_int_list_attr(self, raw_value: Any) -> List[int]:
        values: List[int] = []
        for chunk in re.split(r"[,\s]+", str(raw_value or "").strip()):
            if not chunk:
                continue
            try:
                values.append(int(chunk))
            except Exception:
                continue
        return values

    def _parse_tui_line_nodes(self, block: str) -> List[Dict[str, Any]]:
        lines: List[Dict[str, Any]] = []
        for row_text, content in re.findall(r'<line\s+row="(-?\d+)">(.*?)</line>', str(block or ""), re.IGNORECASE | re.DOTALL):
            try:
                row = int(row_text)
            except Exception:
                continue
            lines.append(
                {
                    "row": row,
                    "text": html.unescape(str(content or "").strip()),
                }
            )
        return lines

    def _extract_tui_block(self, text: str, tag_name: str) -> str:
        match = re.search(
            rf"<{re.escape(tag_name)}(?:\s+[^>]*)?>\s*(.*?)\s*</{re.escape(tag_name)}>",
            str(text or ""),
            re.IGNORECASE | re.DOTALL,
        )
        return match.group(1) if match else ""

    def _extract_tui_preferred_lines(self, text: str, tool_name: str) -> List[Dict[str, Any]]:
        preferred_blocks = {
            "open_tui": ["first_screen", "viewport", "excerpt"],
            "read_tui": ["viewport", "first_screen", "tui_screen"],
            "read_tui_diff": ["after_excerpt", "after", "excerpt"],
            "read_tui_region": ["region_excerpt", "tui_region", "excerpt"],
            "find_text_in_tui": ["anchors", "excerpt"],
            "send_keys": ["after_excerpt", "region_excerpt", "excerpt"],
            "send_keys_and_read": ["after_excerpt", "region_excerpt", "excerpt"],
            "wait_tui_until": ["excerpt", "after_excerpt", "region_excerpt"],
        }
        for tag_name in preferred_blocks.get(str(tool_name or "").strip(), []):
            block = self._extract_tui_block(text, tag_name)
            parsed = self._parse_tui_line_nodes(block)
            if parsed:
                return parsed
        parsed = self._parse_tui_line_nodes(text)
        deduped: List[Dict[str, Any]] = []
        seen_rows = set()
        for item in parsed:
            row = int(item.get("row", -1) or -1)
            if row in seen_rows:
                continue
            seen_rows.add(row)
            deduped.append(item)
        return deduped

    def _extract_tui_anchor_matches(self, text: str) -> List[Dict[str, Any]]:
        matches: List[Dict[str, Any]] = []
        for attrs_blob, body in re.findall(r"<match\b([^>]*)>(.*?)</match>", str(text or ""), re.IGNORECASE | re.DOTALL):
            attrs = self._parse_tag_attrs(attrs_blob)
            row_text = str(attrs.get("row", "") or "").strip()
            try:
                row = int(row_text)
            except Exception:
                row = -1
            line_nodes = self._parse_tui_line_nodes(body)
            excerpt = ""
            for item in line_nodes:
                snippet = str(item.get("text", "") or "").strip()
                if snippet:
                    excerpt = snippet
                    break
            text_attr = html.unescape(str(attrs.get("text", "") or "").strip())
            matches.append(
                {
                    "row": row,
                    "text": text_attr or excerpt,
                    "excerpt": excerpt,
                }
            )
        return matches

    def _extract_tui_summary(self, text: str, tool_name: str, session_name: str) -> str:
        explicit = html.unescape(self._extract_tui_block(text, "summary")).strip()
        if explicit:
            return explicit[:240]
        lines = self._extract_tui_preferred_lines(text, tool_name)
        for item in lines:
            snippet = str(item.get("text", "") or "").strip()
            if snippet:
                return snippet[:240]
        normalized = re.sub(r"<[^>]+>", " ", str(text or ""))
        normalized = re.sub(r"\s+", " ", html.unescape(normalized)).strip()
        if normalized:
            return normalized[:240]
        return session_name or tool_name

    def _render_tui_excerpt_text(self, lines: List[Dict[str, Any]], limit: int = 4) -> str:
        rendered: List[str] = []
        for item in (lines or [])[: max(1, limit)]:
            try:
                row = int(item.get("row", 0) or 0)
            except Exception:
                row = 0
            rendered.append(f"{row}: {str(item.get('text', '') or '').strip()}")
        return " | ".join(part for part in rendered if part).strip()

    def _score_tui_view_name(
        self,
        session_name: str,
        current_focus: str,
        last_tui_snapshots: List[Dict[str, Any]],
    ) -> Tuple[int, int]:
        focus_text = str(current_focus or "").strip().lower()
        lowered_name = str(session_name or "").strip().lower()
        score = 0
        if focus_text and lowered_name and lowered_name in focus_text:
            score += 4
        recency = 0
        for item in reversed(last_tui_snapshots or []):
            if str((item or {}).get("session_name", "") or "").strip() != session_name:
                continue
            recency = int((item or {}).get("updated_at", 0) or 0)
            break
        return score, recency

    def _refresh_active_tui_view_names(
        self,
        tui_views: Dict[str, Any],
        active_tui_view_names: List[str],
        last_tui_snapshots: List[Dict[str, Any]],
        current_focus: str,
    ) -> List[str]:
        candidates = set()
        for name in tui_views.keys():
            normalized_name = str(name or "").strip()
            if normalized_name:
                candidates.add(normalized_name)
        for name in active_tui_view_names or []:
            normalized_name = str(name or "").strip()
            if normalized_name and normalized_name in tui_views:
                candidates.add(normalized_name)
        for item in last_tui_snapshots or []:
            normalized_name = str((item or {}).get("session_name", "") or "").strip()
            if normalized_name and normalized_name in tui_views:
                candidates.add(normalized_name)

        ranked = sorted(
            candidates,
            key=lambda name: (
                self._score_tui_view_name(name, current_focus, last_tui_snapshots)[0],
                self._score_tui_view_name(name, current_focus, last_tui_snapshots)[1],
                int((tui_views.get(name, {}) or {}).get("updated_at", 0) or 0),
                name,
            ),
            reverse=True,
        )
        return ranked[: self.TUI_VIEW_SESSIONS_MAX]

    def _append_last_tui_snapshot(
        self,
        last_tui_snapshots: List[Dict[str, Any]],
        snapshot_meta: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if not isinstance(snapshot_meta, dict) or not snapshot_meta.get("session_name"):
            return last_tui_snapshots
        last_tui_snapshots.append(dict(snapshot_meta))
        if len(last_tui_snapshots) > self.LAST_TUI_SNAPSHOTS_KEEP_MAX:
            del last_tui_snapshots[:-self.LAST_TUI_SNAPSHOTS_KEEP_MAX]
        return last_tui_snapshots

    def _update_tui_views_state(
        self,
        tui_views: Dict[str, Any],
        active_tui_view_names: List[str],
        last_tui_snapshots: List[Dict[str, Any]],
        output: str,
        tool_name: str,
        params: Dict[str, Any],
        current_focus: str,
    ) -> Tuple[Dict[str, Any], List[str], List[Dict[str, Any]], bool]:
        text = str(output or "")
        if "<error>" in text.lower():
            return tui_views, active_tui_view_names, last_tui_snapshots, False

        session_name = str(self._extract_xml_attr(text, "name") or params.get("name", "") or "").strip()
        if not session_name:
            return tui_views, active_tui_view_names, last_tui_snapshots, False

        snapshot = dict(tui_views.get(session_name, {}) or {})
        previous_snapshot = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, default=str)
        now_ts = int(time.time())

        def _safe_attr_int(attr_name: str) -> int:
            try:
                return int(self._extract_xml_attr(text, attr_name) or 0)
            except Exception:
                return 0

        rows = _safe_attr_int("rows")
        cols = _safe_attr_int("cols")
        cursor_row = _safe_attr_int("cursor_row")
        cursor_col = _safe_attr_int("cursor_col")
        screen_hash = ""
        for attr_name in ("screen_hash", "current_hash", "new_hash"):
            screen_hash = self._extract_xml_attr(text, attr_name)
            if screen_hash:
                break

        visible_lines = self._extract_tui_preferred_lines(text, tool_name)
        region_excerpt = self._parse_tui_line_nodes(self._extract_tui_block(text, "region_excerpt"))
        if not region_excerpt:
            region_excerpt = self._parse_tui_line_nodes(self._extract_tui_block(text, "after_excerpt"))
        if not region_excerpt:
            region_excerpt = self._parse_tui_line_nodes(self._extract_tui_block(text, "excerpt"))
        anchor_matches = self._extract_tui_anchor_matches(text)
        changed_rows = self._parse_int_list_attr(self._extract_xml_attr(text, "changed_rows"))
        match_rows = self._parse_int_list_attr(self._extract_xml_attr(text, "match_rows"))
        if not changed_rows and match_rows:
            changed_rows = match_rows
        summary = self._extract_tui_summary(text, tool_name, session_name)

        snapshot["session_name"] = session_name
        snapshot["rows"] = rows or int(snapshot.get("rows", 0) or 0)
        snapshot["cols"] = cols or int(snapshot.get("cols", 0) or 0)
        snapshot["cursor_row"] = cursor_row if rows or cursor_row else int(snapshot.get("cursor_row", 0) or 0)
        snapshot["cursor_col"] = cursor_col if cols or cursor_col else int(snapshot.get("cursor_col", 0) or 0)
        snapshot["screen_hash"] = screen_hash or str(snapshot.get("screen_hash", "") or "")
        snapshot["last_action"] = tool_name
        snapshot["updated_at"] = now_ts
        snapshot["summary"] = summary
        if visible_lines:
            snapshot["visible_lines"] = visible_lines[: self.TUI_VIEW_LINES_MAX]
        if changed_rows:
            snapshot["last_changed_rows"] = changed_rows[:20]
        if region_excerpt:
            snapshot["last_region_excerpt"] = region_excerpt[: self.TUI_VIEW_LINES_MAX]
        if anchor_matches:
            snapshot["last_anchor_matches"] = anchor_matches[:8]

        tui_views[session_name] = snapshot
        last_tui_snapshots = self._append_last_tui_snapshot(
            last_tui_snapshots,
            {
                "session_name": session_name,
                "screen_hash": snapshot.get("screen_hash", ""),
                "last_action": tool_name,
                "summary": summary,
                "updated_at": now_ts,
                "changed_rows": changed_rows[:20],
            },
        )
        active_tui_view_names = self._refresh_active_tui_view_names(
            tui_views=tui_views,
            active_tui_view_names=active_tui_view_names,
            last_tui_snapshots=last_tui_snapshots,
            current_focus=current_focus,
        )

        changed = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, default=str) != previous_snapshot
        return tui_views, active_tui_view_names, last_tui_snapshots, changed

    def _parse_tag_attrs(self, attr_text: str) -> Dict[str, str]:
        attrs: Dict[str, str] = {}
        for key, value in re.findall(r'([a-zA-Z0-9_]+)="([^"]*)"', str(attr_text or "")):
            attrs[str(key)] = str(value)
        return attrs

    def _append_evidence_item(
        self,
        evidence_index: List[Dict[str, Any]],
        item: Dict[str, Any],
        fingerprint: str,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        if not fingerprint:
            return evidence_index, False
        for existing in reversed(evidence_index[-300:]):
            if str(existing.get("fingerprint", "") or "") == fingerprint:
                return evidence_index, False

        enriched = dict(item or {})
        enriched["fingerprint"] = fingerprint
        if not enriched.get("id"):
            enriched["id"] = f"ev_{uuid.uuid4().hex[:12]}"
        enriched["timestamp"] = int(time.time())
        evidence_index.append(enriched)
        if len(evidence_index) > 500:
            del evidence_index[:-500]
        return evidence_index, True

    def _update_search_evidence_index(
        self,
        evidence_index: List[Dict[str, Any]],
        output: str,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        text = str(output or "")
        backend = self._extract_search_backend_from_output(text)
        backend_reason = self._extract_search_backend_reason_from_output(text)
        changed = False
        for attrs_blob, preview in re.findall(r"<hit\s+([^>]*)>(.*?)</hit>", text, re.IGNORECASE | re.DOTALL):
            attrs = self._parse_tag_attrs(attrs_blob)
            hit_id = str(attrs.get("id", "") or "").strip()
            file_path = str(attrs.get("file_path", "") or "").strip()
            line = str(attrs.get("line", "") or "").strip()
            score = str(attrs.get("score", "") or "").strip()
            preview_text = str(preview or "").strip()
            fingerprint_seed = f"{hit_id}:{file_path}:{line}:{preview_text[:180]}"
            fingerprint = "search:" + hashlib.sha1(fingerprint_seed.encode("utf-8")).hexdigest()[:20]
            item = {
                "type": "search_hit",
                "hit_id": hit_id,
                "path": file_path,
                "line": line,
                "score": score,
                "backend": backend,
                "backend_reason": backend_reason,
                "summary": preview_text[:220],
            }
            evidence_index, appended = self._append_evidence_item(evidence_index, item, fingerprint)
            changed = changed or appended
        return evidence_index, changed

    def _update_read_evidence_index(
        self,
        evidence_index: List[Dict[str, Any]],
        output: str,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        meta = self._parse_file_chunk_meta(output)
        if not meta:
            return evidence_index, False

        path = str(meta.get("path", "") or "").strip()
        start_line = str(meta.get("start_line", "") or "").strip()
        end_line = str(meta.get("end_line", "") or "").strip()
        chunk_id = str(meta.get("chunk_id", "") or "").strip()
        content_hash = str(meta.get("content_hash", "") or "").strip()
        reason = str(meta.get("reason", "") or "").strip()
        hit_id = str(meta.get("hit_id", "") or "").strip()
        mode = str(meta.get("mode", "") or "").strip()
        backend = str(meta.get("backend", "") or "").strip()
        backend_reason = str(meta.get("backend_reason", "") or "").strip()
        range_text = f"{start_line}-{end_line}" if start_line and end_line else ""
        fingerprint_base = chunk_id or f"{path}:{range_text}:{content_hash}"
        if not fingerprint_base:
            return evidence_index, False
        fingerprint = "read:" + hashlib.sha1(fingerprint_base.encode("utf-8")).hexdigest()[:20]
        item = {
            "type": "read_chunk",
            "chunk_id": chunk_id,
            "path": path,
            "range": range_text,
            "content_hash": content_hash,
            "reason": reason,
            "hit_id": hit_id,
            "mode": mode,
            "backend": backend,
            "backend_reason": backend_reason,
            "summary": f"{path}:{range_text}" if path else range_text,
        }
        return self._append_evidence_item(evidence_index, item, fingerprint)


    def _merge_ranges(self, ranges: List[List[int]], start: int, end: int) -> List[List[int]]:
        normalized: List[List[int]] = []
        for item in ranges or []:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            try:
                s = int(item[0])
                e = int(item[1])
            except Exception:
                continue
            if s <= 0 or e <= 0:
                continue
            if e < s:
                s, e = e, s
            normalized.append([s, e])
        normalized.append([start, end])
        normalized.sort(key=lambda pair: (pair[0], pair[1]))

        merged: List[List[int]] = []
        for current in normalized:
            if not merged:
                merged.append(current)
                continue
            last = merged[-1]
            if current[0] <= (last[1] + 1):
                last[1] = max(last[1], current[1])
            else:
                merged.append(current)
        return merged


    def _update_read_coverage(
        self,
        read_coverage: Dict[str, Any],
        output: str,
    ) -> Tuple[Dict[str, Any], bool, bool, str]:
        meta = self._parse_file_chunk_meta(output)
        if not meta:
            return read_coverage, False, False, ""

        path = str(meta.get("path", "") or "").strip()
        if not path:
            return read_coverage, False, False, ""

        try:
            start_line = int(meta.get("start_line", 0) or 0)
            end_line = int(meta.get("end_line", 0) or 0)
            total_lines = int(meta.get("total_lines", 0) or 0)
        except Exception:
            return read_coverage, False, False, ""

        fingerprint = ""
        if start_line > 0 and end_line > 0:
            fingerprint = f"{path}:{start_line}-{end_line}"

        snapshot = dict(read_coverage.get(path, {}) or {})
        old_ranges = list(snapshot.get("ranges", []) or [])
        changed = False
        repeated_hit = False
        if start_line > 0 and end_line > 0 and end_line >= start_line:
            merged_ranges = self._merge_ranges(old_ranges, start_line, end_line)
            changed = merged_ranges != old_ranges
            repeated_hit = not changed
            snapshot["ranges"] = merged_ranges
        else:
            snapshot["ranges"] = old_ranges

        if int(snapshot.get("total_lines", 0) or 0) != total_lines and total_lines >= 0:
            snapshot["total_lines"] = total_lines
            changed = True

        content_hash = str(meta.get("content_hash", "") or "").strip()
        chunk_id = str(meta.get("chunk_id", "") or "").strip()
        mode = str(meta.get("mode", "") or "").strip()
        reason = str(meta.get("reason", "") or "").strip()
        hit_id = str(meta.get("hit_id", "") or "").strip()

        if content_hash and str(snapshot.get("last_content_hash", "") or "") != content_hash:
            snapshot["last_content_hash"] = content_hash
            changed = True
            repeated_hit = False
        if chunk_id:
            snapshot["last_chunk_id"] = chunk_id
        if mode:
            snapshot["last_mode"] = mode
        if reason:
            snapshot["last_reason"] = reason
        if hit_id:
            recent_hit_ids = list(snapshot.get("recent_hit_ids", []) or [])
            if hit_id not in recent_hit_ids:
                recent_hit_ids.append(hit_id)
            snapshot["recent_hit_ids"] = recent_hit_ids[-10:]

        snapshot["reads"] = int(snapshot.get("reads", 0) or 0) + 1
        snapshot["updated_at"] = int(time.time())
        read_coverage[path] = snapshot
        return read_coverage, changed, repeated_hit, fingerprint


    def _tool_execution_advances_progress(
        self,
        tool_name: str,
        output: str,
        coverage_changed: bool,
        evidence_changed: bool = False,
        file_views_changed: bool = False,
        tui_progressed: bool = False,
    ) -> bool:
        lowered = str(output or "").lower()
        if "<error>" in lowered:
            return False
        if tool_name in {"read_tui", "read_tui_diff", "read_tui_region", "find_text_in_tui", "send_keys", "send_keys_and_read", "wait_tui_until"}:
            return bool(tui_progressed)
        if tool_name == "read_file":
            return bool(coverage_changed or evidence_changed or file_views_changed)
        if tool_name == "search_in_files":
            return bool(evidence_changed)
        if tool_name in {"read_terminal", "read_terminal_since_last", "wait_terminal_until", "run_and_wait"}:
            return 'has_new_output="true"' in lowered or 'status="completed"' in lowered or 'status="failed"' in lowered or 'status="timeout"' in lowered
        if tool_name in {"exec_command", "write_stdin", "launch_interactive_session"}:
            return 'has_new_output="true"' in lowered or 'status="completed"' in lowered or 'status="failed"' in lowered or 'status="timeout"' in lowered or 'status="running"' in lowered
        if tool_name == "close_exec_session":
            return 'status="closed"' in lowered
        return True


    def _build_evidence_digest(
        self,
        progress: str,
        read_coverage: Dict[str, Any],
        file_views: Dict[str, Any],
        evidence_index: List[Dict[str, Any]],
        search_hits_index: List[Dict[str, Any]],
        file_candidates: List[Dict[str, Any]],
        executed_tools: List[Dict[str, Any]],
    ) -> str:
        try:
            coverage_brief = {}
            for path, meta in sorted((read_coverage or {}).items(), key=lambda item: str(item[0])):
                if not isinstance(meta, dict):
                    continue
                coverage_brief[str(path)] = {
                    "total_lines": int(meta.get("total_lines", 0) or 0),
                    "ranges": meta.get("ranges", []) or [],
                    "reads": int(meta.get("reads", 0) or 0),
                }
            file_view_brief = {}
            for path, meta in sorted((file_views or {}).items(), key=lambda item: str(item[0])):
                if not isinstance(meta, dict):
                    continue
                file_view_brief[str(path)] = {
                    "ranges": meta.get("ranges", []) or [],
                    "reads": int(meta.get("reads", 0) or 0),
                    "last_updated": int(meta.get("last_updated", 0) or 0),
                }
            payload = {
                "progress_tail": str(progress or "")[-1600:],
                "coverage": coverage_brief,
                "file_views": file_view_brief,
                "evidence_tail": [
                    {
                        "id": str(item.get("id", "") or ""),
                        "type": str(item.get("type", "") or ""),
                        "path": str(item.get("path", "") or ""),
                        "range": str(item.get("range", "") or ""),
                        "hash": str(item.get("content_hash", "") or ""),
                    }
                    for item in (evidence_index or [])[-30:]
                    if isinstance(item, dict)
                ],
                "search_hits_tail": [
                    {
                        "id": str(item.get("id", "") or ""),
                        "path": str(item.get("path", "") or ""),
                        "line": str(item.get("line", "") or ""),
                        "summary": str(item.get("summary", "") or "")[:120],
                    }
                    for item in (search_hits_index or [])[-20:]
                    if isinstance(item, dict)
                ],
                "file_candidates_tail": [
                    str(item.get("path", "") or "")
                    for item in (file_candidates or [])[-20:]
                    if isinstance(item, dict)
                ],
                "recent_tools": [
                    {
                        "tool": str(item.get("tool", "") or ""),
                        "status": str(item.get("status_label", "") or ""),
                        "success": bool(item.get("success", False)),
                    }
                    for item in (executed_tools or [])[-20:]
                ],
            }
            serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:24]
        except Exception:
            return ""

    def _append_recent_event(
        self,
        events: List[Dict[str, Any]],
        event: Dict[str, Any],
        keep: int = 40,
    ) -> List[Dict[str, Any]]:
        if not isinstance(event, dict) or not event:
            return events
        enriched = dict(event)
        enriched.setdefault("timestamp", int(time.time()))
        events.append(enriched)
        if len(events) > keep:
            del events[:-keep]
        return events

    def _extract_patch_paths(self, patch_text: str) -> List[str]:
        paths: List[str] = []
        for line in str(patch_text or "").splitlines():
            for prefix in ("*** Update File: ", "*** File: ", "*** Add File: ", "*** Delete File: ", "*** Move to: "):
                if line.startswith(prefix):
                    path = str(line[len(prefix):] or "").strip()
                    if path and path not in paths:
                        paths.append(path)
        return paths

    def _record_patch_success(
        self,
        *,
        tool_name: str,
        params: Dict[str, Any],
        output: str,
        trusted_modified_files: List[str],
        last_patch_summaries: List[Dict[str, Any]],
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        lowered = str(output or "").lower()
        if "<error>" in lowered:
            return trusted_modified_files, last_patch_summaries

        files: List[str] = []
        if tool_name == "apply_patch":
            files = self._extract_patch_paths(str((params or {}).get("patch", "") or ""))

        if not files:
            return trusted_modified_files, last_patch_summaries

        for path in files:
            if path not in trusted_modified_files:
                trusted_modified_files.append(path)
        summary = {
            "tool": tool_name,
            "files": files,
            "summary": str(output or "").splitlines()[0][:240],
            "timestamp": int(time.time()),
        }
        last_patch_summaries.append(summary)
        if len(last_patch_summaries) > 20:
            last_patch_summaries = last_patch_summaries[-20:]
        if len(trusted_modified_files) > 50:
            trusted_modified_files = trusted_modified_files[-50:]
        return trusted_modified_files, last_patch_summaries

    def _record_terminal_event(
        self,
        *,
        events: List[Dict[str, Any]],
        tool_name: str,
        params: Dict[str, Any],
        output: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        parsed = self._parse_terminal_output(output)
        if not parsed:
            return events, {}, {}, {}
        status = str(parsed.get("status", "") or "").strip().lower()
        body = str(parsed.get("body", "") or "").strip()
        session_name = str(parsed.get("name", "") or params.get("name", "") or "").strip()
        op_id = str(parsed.get("op_id", "") or "").strip()
        exit_code = parsed.get("exit_code")
        has_new_output = bool(parsed.get("has_new_output", False))
        event_type = "command_progress"
        if tool_name in {"run_in_terminal", "exec_command", "launch_interactive_session"}:
            event_type = "command_started"
        elif status == "completed":
            event_type = "command_completed"
        elif status in {"failed", "timeout"}:
            event_type = "command_failed" if status == "failed" else "command_timeout"
        elif tool_name in {"wait_terminal_until", "run_and_wait"} and parsed.get("matched_pattern"):
            event_type = "command_progress"
        summary = body if body and body != "()" else f"{tool_name}:{status or 'pending'}"
        if len(summary) > 240:
            summary = summary[:240] + "..."
        event = {
            "event_type": event_type,
            "session_name": session_name,
            "op_id": op_id,
            "status": status,
            "exit_code": exit_code,
            "summary": summary,
            "has_new_output": has_new_output,
        }
        events = self._append_recent_event(events, event)
        active_op = {}
        if status in {"pending", "running"}:
            active_op = dict(event)
        last_delta = {}
        if has_new_output and body and body != "()":
            last_delta = dict(event)
        last_failure = {}
        if status in {"failed", "timeout"}:
            last_failure = dict(event)
        return events, active_op, last_delta, last_failure

    def _record_exec_session_event(
        self,
        *,
        events: List[Dict[str, Any]],
        tool_name: str,
        params: Dict[str, Any],
        output: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        if tool_name not in {"exec_command", "write_stdin", "close_exec_session", "launch_interactive_session"}:
            return events, {}, {}
        parsed = self._parse_terminal_output(output)
        session_id = self._extract_exec_session_id(output, params)
        if not parsed and tool_name != "close_exec_session":
            return events, {}, {}
        status = str(parsed.get("status", "") or "").strip().lower()
        body = str(parsed.get("body", "") or "").strip()
        if tool_name == "close_exec_session":
            status = status or "closed"
            body = body or ""
        event_type = "session_progress"
        if tool_name in {"exec_command", "launch_interactive_session"}:
            event_type = "session_started"
        elif tool_name == "close_exec_session" or status == "closed":
            event_type = "session_closed"
        elif status == "completed":
            event_type = "session_completed"
        elif status in {"failed", "timeout"}:
            event_type = "session_failed" if status == "failed" else "session_timeout"
        summary = body if body and body != "()" else f"{tool_name}:{status or 'pending'}"
        if len(summary) > 240:
            summary = summary[:240] + "..."
        event = {
            "event_type": event_type,
            "session_id": session_id,
            "status": status,
            "summary": summary,
            "command": str(params.get("cmd", "") or params.get("command", "") or "").strip(),
            "transport": self._extract_xml_attr(str(output or ""), "transport"),
        }
        events = self._append_recent_event(events, event)
        active_session = {}
        if status in {"pending", "running"}:
            active_session = dict(event)
        last_output = {}
        if body and body != "()":
            last_output = dict(event)
        return events, active_session, last_output

    def _record_tui_event(
        self,
        *,
        events: List[Dict[str, Any]],
        tool_name: str,
        params: Dict[str, Any],
        output: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        text = str(output or "")
        if "<error>" in text.lower():
            return events, {}, {}, {}
        session_name = str((params or {}).get("name", "") or "").strip()
        if not session_name:
            session_name = self._extract_xml_attr(text, "name")
        changed_rows = ""
        changed_match = re.search(r'changed_rows="([^"]*)"', text)
        if changed_match:
            changed_rows = changed_match.group(1).strip()
        screen_hash = ""
        for attr in ("screen_hash", "current_hash", "new_hash"):
            screen_hash = self._extract_xml_attr(text, attr)
            if screen_hash:
                break
        event_type = {
            "open_tui": "tui_opened",
            "read_tui": "screen_snapshot",
            "read_tui_diff": "screen_changed",
            "read_tui_region": "region_read",
            "find_text_in_tui": "text_found",
            "send_keys": "keys_sent",
            "send_keys_and_read": "keys_sent",
            "wait_tui_until": "wait_satisfied" if 'status="matched"' in text else "wait_timeout",
        }.get(tool_name, "screen_snapshot")
        excerpt_lines = self._extract_tui_preferred_lines(text, tool_name)
        summary = self._extract_tui_summary(text, tool_name, session_name or tool_name)
        excerpt = self._render_tui_excerpt_text(excerpt_lines, limit=3)
        match_rows = self._extract_xml_attr(text, "match_rows")
        target_text = self._extract_xml_attr(text, "text") or str((params or {}).get("text", "") or "").strip()
        event = {
            "event_type": event_type,
            "session_name": session_name,
            "screen_hash": screen_hash,
            "changed_rows": changed_rows,
            "summary": summary,
            "excerpt": excerpt,
            "match_rows": match_rows,
            "text": target_text,
        }
        events = self._append_recent_event(events, event)
        focus = dict(event)
        if excerpt_lines:
            focus["visible_lines"] = excerpt_lines[:4]
        last_diff = dict(event) if tool_name in {"read_tui_diff", "send_keys", "send_keys_and_read"} else {}
        last_region = dict(event) if tool_name == "read_tui_region" else {}
        return events, focus, last_diff, last_region
    def _execute_single_tool_action(
        self,
        tool_name: str,
        params: Dict[str, Any],
        loaded_skill_names: List[str],
        skill_context: str,
        command_op_id: str = "",
        command_target_op_id: str = "",
        tool_artifacts: Any = None,
        thread_id: str = "",
    ) -> Tuple[str, List[str], str]:
        normalized_tool_name = str(tool_name or "").strip()
        if not self._is_tool_active(normalized_tool_name):
            return (
                f"<error>Tool not active in current group set: {normalized_tool_name}</error>",
                loaded_skill_names,
                skill_context,
            )
        tool_name = normalized_tool_name
        if tool_name == "load_skill":
            skill_name = (
                str(params.get("name", "") or params.get("skill_name", "") or params.get("input", ""))
                .strip()
            )
            full_requested = self._coerce_bool(params.get("full", False), default=False)
            skill_result = self.get_skill_markdown(skill_name)
            if not skill_result.get("ok", False):
                return (
                    f"<error>{skill_result.get('error', '')}</error>",
                    loaded_skill_names,
                    skill_context,
                )

            canonical_name = str(skill_result.get("name", skill_name)).strip()
            if canonical_name and canonical_name not in loaded_skill_names:
                loaded_skill_names.append(canonical_name)

            content_text = str(skill_result.get("content", "") or "")
            content_digest = str(skill_result.get("content_digest", "") or "").strip()
            include_full = bool(full_requested)
            artifact_hint = ""
            context_limit = max(int(getattr(self, "SKILL_CONTEXT_INLINE_CHAR_LIMIT", 6000) or 6000), 1200)
            if include_full and len(content_text) > context_limit:
                include_full = False
                if isinstance(tool_artifacts, list):
                    artifact = self._store_tool_artifact(
                        tool_artifacts=tool_artifacts,
                        tool_name="load_skill",
                        params={"name": canonical_name, "full": True},
                        output_text=content_text,
                        thread_id=thread_id,
                    )
                    artifact_id = str(artifact.get("id", "") or "").strip()
                    if artifact_id:
                        artifact_hint = f"artifact_id={artifact_id}"

            block_builder = getattr(self, "_build_skill_context_block", None)
            if callable(block_builder):
                try:
                    block = str(block_builder(skill_result, source="tool", include_full=include_full))
                except TypeError:
                    block = str(block_builder(skill_result, source="tool"))
            else:
                block = (
                    f"[SKILL] {canonical_name}\n"
                    f"skill_dir: {skill_result.get('skill_dir', '')}\n"
                    f"skill_md: {skill_result.get('skill_md_path', '')}\n"
                    f"description: {skill_result.get('description', '')}"
                )
                if include_full:
                    block = (
                        f"{block}\n"
                        "----- SKILL.md BEGIN -----\n"
                        f"{content_text}\n"
                        "----- SKILL.md END -----"
                    )

            append_block = getattr(self, "_append_skill_context_block", None)
            if callable(append_block):
                skill_context = append_block(skill_context, block, content_digest)
            else:
                if content_digest and f"digest: {content_digest}" in str(skill_context or ""):
                    pass
                else:
                    skill_context = f"{skill_context}\n\n{block}".strip() if skill_context else block

            status = "loaded"
            if full_requested and include_full:
                status = "loaded_full"
            elif full_requested and not include_full:
                status = "loaded_summary_with_artifact"
            message_lines = [f"<success>{status}: {canonical_name}</success>"]
            if full_requested and not include_full:
                message_lines.append(
                    "<info>Full SKILL.md moved out of inline context due budget; relevant slices will be auto-hydrated into context when needed.</info>"
                )
            else:
                message_lines.append("<info>Skill summary is now available in <skill_context>.</info>")
            if artifact_hint:
                message_lines.append(f"<info>{artifact_hint}</info>")
            return (
                "\n".join(message_lines),
                loaded_skill_names,
                skill_context,
            )

        if tool_name in self.external_tool_funcs:
            return (
                str(self._invoke_external_tool(self.external_tool_funcs[tool_name], params)),
                loaded_skill_names,
                skill_context,
            )

        # Dispatch-map first for straightforward tool handlers; special cases stay below.
        simple_dispatch_map = self._build_simple_tool_dispatch_map(
            params=params if isinstance(params, dict) else {},
            command_op_id=command_op_id,
            command_target_op_id=command_target_op_id,
        )
        direct_handler = simple_dispatch_map.get(tool_name)
        if callable(direct_handler):
            return (
                str(direct_handler()),
                loaded_skill_names,
                skill_context,
            )

        if tool_name == "read_terminal":
            session_name = str(params.get("name", "") or "").strip()
            if self._looks_like_exec_session_id(session_name):
                return (
                    self._build_exec_session_namespace_error(tool_name, session_name),
                    loaded_skill_names,
                    skill_context,
                )
            lines = params.get("lines", 50)
            return (
                self._call_terminal_read(
                    session_name,
                    int(lines) if lines else 50,
                    command_target_op_id,
                ),
                loaded_skill_names,
                skill_context,
            )
        if tool_name == "read_terminal_since_last":
            session_name = str(params.get("name", "") or "").strip()
            if self._looks_like_exec_session_id(session_name):
                return (
                    self._build_exec_session_namespace_error(tool_name, session_name),
                    loaded_skill_names,
                    skill_context,
                )
            lines = int(params.get("lines", 100) or 100)
            return (
                self.terminal_manager.read_terminal_since_last(
                    session_name,
                    lines=lines,
                    op_id=str(params.get("op_id", "") or command_target_op_id or "").strip() or None,
                ),
                loaded_skill_names,
                skill_context,
            )
        if tool_name == "wait_terminal_until":
            session_name = str(params.get("name", "") or "").strip()
            if self._looks_like_exec_session_id(session_name):
                return (
                    self._build_exec_session_namespace_error(tool_name, session_name),
                    loaded_skill_names,
                    skill_context,
                )
            return (
                self.terminal_manager.wait_terminal_until(
                    session_name,
                    op_id=str(params.get("op_id", "") or command_target_op_id or "").strip() or None,
                    pattern=str(params.get("pattern", "") or "").strip() or None,
                    timeout_ms=int(params.get("timeout_ms", 30000) or 30000),
                    poll_interval_ms=int(params.get("poll_interval_ms", 300) or 300),
                ),
                loaded_skill_names,
                skill_context,
            )
        if tool_name == "run_and_wait":
            session_name = str(params.get("name", "") or "").strip()
            if self._looks_like_exec_session_id(session_name):
                return (
                    self._build_exec_session_namespace_error(tool_name, session_name),
                    loaded_skill_names,
                    skill_context,
                )
            return (
                self.terminal_manager.run_and_wait(
                    session_name,
                    str(params.get("command", "") or ""),
                    timeout_ms=int(params.get("timeout_ms", 30000) or 30000),
                    pattern=str(params.get("pattern", "") or "").strip() or None,
                    poll_interval_ms=int(params.get("poll_interval_ms", 300) or 300),
                ),
                loaded_skill_names,
                skill_context,
            )
        if tool_name == "open_tui":
            rows = int(params.get("rows", 30) or 30)
            cols = int(params.get("cols", 100) or 100)
            self._tui_guide_loaded = True
            output = self.tui_manager.open_tui(params.get("name"), params.get("command"), rows, cols)
            self.tui_manager.render_to_console(params.get("name"))
            return output, loaded_skill_names, skill_context
        if tool_name == "read_tui":
            skip_empty = params.get("skip_empty_lines", None)
            if skip_empty is None:
                skip_empty = not bool(getattr(self, "full_tui_echo", False))
            skip_empty = self._coerce_bool(
                skip_empty,
                default=(not bool(getattr(self, "full_tui_echo", False))),
            )
            output = self.tui_manager.read_tui(params.get("name"), skip_empty)
            self.tui_manager.render_to_console(params.get("name"))
            return output, loaded_skill_names, skill_context
        if tool_name == "read_tui_diff":
            output = self.tui_manager.read_tui_diff(params.get("name"))
            self.tui_manager.render_to_console(params.get("name"))
            return output, loaded_skill_names, skill_context
        if tool_name == "read_tui_region":
            output = self.tui_manager.read_tui_region(
                params.get("name"),
                int(params.get("start_row", 0) or 0),
                int(params.get("end_row", 0) or 0),
            )
            self.tui_manager.render_to_console(params.get("name"))
            return output, loaded_skill_names, skill_context
        if tool_name == "find_text_in_tui":
            output = self.tui_manager.find_text_in_tui(
                params.get("name"),
                str(params.get("text", "") or ""),
                self._coerce_bool(params.get("case_insensitive", True), default=True),
            )
            self.tui_manager.render_to_console(params.get("name"))
            return output, loaded_skill_names, skill_context
        if tool_name == "send_keys":
            output = self.tui_manager.send_keys(params.get("name"), params.get("keys"))
            self.tui_manager.render_to_console(params.get("name"))
            return output, loaded_skill_names, skill_context
        if tool_name == "send_keys_and_read":
            output = self.tui_manager.send_keys_and_read(
                params.get("name"),
                params.get("keys"),
                delay_ms=int(params.get("delay_ms", 100) or 100),
            )
            self.tui_manager.render_to_console(params.get("name"))
            return output, loaded_skill_names, skill_context
        if tool_name == "wait_tui_until":
            output = self.tui_manager.wait_tui_until(
                params.get("name"),
                text=str(params.get("text", "") or "").strip() or None,
                hash_change=self._coerce_bool(params.get("hash_change", False), default=False),
                timeout_ms=int(params.get("timeout_ms", 5000) or 5000),
                poll_interval_ms=int(params.get("poll_interval_ms", 200) or 200),
            )
            self.tui_manager.render_to_console(params.get("name"))
            return output, loaded_skill_names, skill_context
        if tool_name == "close_tui":
            return self.tui_manager.close_tui(params.get("name")), loaded_skill_names, skill_context
        if tool_name == "activate_tui_mode":
            self._tui_guide_loaded = True
            return (
                "<success>TUI .TUI .\n send_keys_and_read / wait_tui_until / read_tui_diff / read_tui_region, read_tui; <tui_views> ,.</success>",
                loaded_skill_names,
                skill_context,
            )
        if tool_name == "launch_interactive_session":
            command = str(params.get("command", "") or "").strip()
            output = self._call_exec_command(
                {
                    "cmd": command,
                    "tty": True,
                    "yield_time_ms": int(params.get("yield_time_ms", 1200) or 1200),
                    "max_output_tokens": int(params.get("max_output_tokens", 1200) or 1200),
                    "timeout_ms": int(params.get("timeout_ms", 0) or 0),
                    "shell": self._coerce_bool(params.get("shell", True), default=True),
                    "login": self._coerce_bool(params.get("login", True), default=True),
                    "workdir": str(params.get("workdir", "") or "").strip() or None,
                }
            )
            output += "\n<system>已优先走交互式终端层。只有需要屏幕级观察、diff、region 或锚点定位时，才切换到 TUI 工具链。</system>"
            return output, loaded_skill_names, skill_context
        if tool_name == "web_search":
            web_search_available = True
            availability_checker = getattr(self, "_is_web_search_available", None)
            if callable(availability_checker):
                web_search_available = bool(availability_checker())
            if not web_search_available:
                return (
                    "<error>web_search requires Tavily API key. "
                    "Use set_tavily_api('tvly-...') or set TAVILY_API_KEY.</error>",
                    loaded_skill_names,
                    skill_context,
                )
            return (
                self.web_search_tool.search(
                    query=params.get("query", ""),
                    max_results=int(params.get("max_results", 5)),
                    search_depth=params.get("search_depth", "basic"),
                ),
                loaded_skill_names,
                skill_context,
            )
        return ToolRegistry.execute({"tool": tool_name, "params": params}), loaded_skill_names, skill_context

    def _build_simple_tool_dispatch_map(
        self,
        *,
        params: Dict[str, Any],
        command_op_id: str,
        command_target_op_id: str,
    ) -> Dict[str, Any]:
        safe_params = params if isinstance(params, dict) else {}
        return {
            "execute_command": lambda: self._call_execute_command(safe_params),
            "exec_command": lambda: self._call_exec_command(safe_params),
            "write_stdin": lambda: self._call_write_stdin(safe_params),
            "close_exec_session": lambda: self._call_close_exec_session(safe_params),
            "create_terminal": lambda: self.terminal_manager.create_terminal(
                safe_params.get("name"),
                safe_params.get("cwd"),
            ),
            "run_in_terminal": lambda: self._call_terminal_run_command(
                str(safe_params.get("name", "") or ""),
                str(safe_params.get("command", "") or ""),
                command_op_id,
            ),
            "check_syntax": lambda: SyntaxChecker.check_file(safe_params.get("path")),
            "browser_open": lambda: self.browser_manager.goto(safe_params.get("url")),
            "browser_view": lambda: self.browser_manager.get_content(safe_params.get("selector")),
            "browser_act": lambda: self.browser_manager.act(
                safe_params.get("action"),
                safe_params.get("selector"),
                safe_params.get("value"),
            ),
            "browser_scroll": lambda: self.browser_manager.scroll(
                direction=safe_params.get("direction", "down"),
                amount=safe_params.get("amount", None),
            ),
            "browser_screenshot": lambda: self.browser_manager.screenshot(),
            "browser_console_logs": lambda: self.browser_manager.get_console_logs(),
        }


    def _execute_subtask_actions(
        self,
        params: Dict[str, Any],
        tool_artifacts: List[Dict[str, Any]],
        loaded_skill_names: List[str],
        skill_context: str,
        thread_id: str,
        artifact_metrics: Any = None,
    ) -> Tuple[str, Dict[str, Any], List[str], str]:
        title = str(params.get("title", "") or "subtask").strip()
        raw_actions = params.get("actions", [])
        if not isinstance(raw_actions, list) or not raw_actions:
            return "<error>run_subtask  actions </error>", {}, loaded_skill_names, skill_context

        started_at = int(time.time())
        subtask_id = f"subtask_{uuid.uuid4().hex[:10]}"
        action_records: List[Dict[str, Any]] = []
        summary_lines = [f"<subtask id=\"{subtask_id}\" title=\"{title}\">"]

        for idx, action in enumerate(raw_actions, 1):
            if not isinstance(action, dict):
                action_records.append(
                    {"index": idx, "tool": "", "success": False, "error": "action "}
                )
                summary_lines.append(f"- [{idx}] FAIL: action ")
                continue

            action_tool = str(action.get("tool", "") or "").strip()
            action_params = action.get("params", {})
            if not isinstance(action_params, dict):
                action_params = {}
            if not action_tool:
                action_records.append(
                    {"index": idx, "tool": "", "success": False, "error": "tool "}
                )
                summary_lines.append(f"- [{idx}] FAIL: tool ")
                continue
            if action_tool in {"run_subtask", "state_update", "parallel_tool_call"}:
                action_records.append(
                    {
                        "index": idx,
                        "tool": action_tool,
                        "success": False,
                        "error": f"run_subtask  {action_tool}",
                    }
                )
                summary_lines.append(f"- [{idx}] FAIL {action_tool}: run_subtask ")
                continue
            if not self._is_tool_active(action_tool):
                action_records.append(
                    {
                        "index": idx,
                        "tool": action_tool,
                        "success": False,
                        "error": f"tool not active: {action_tool}",
                    }
                )
                summary_lines.append(f"- [{idx}] FAIL {action_tool}: tool not active")
                continue

            try:
                action_output, loaded_skill_names, skill_context = self._execute_single_tool_action(
                    tool_name=action_tool,
                    params=action_params,
                    loaded_skill_names=loaded_skill_names,
                    skill_context=skill_context,
                    tool_artifacts=tool_artifacts,
                    thread_id=thread_id,
                )
                action_output_text = str(action_output)
                success = "<error>" not in action_output_text.lower()
                output_preview = action_output_text
                action_records.append(
                    {
                        "index": idx,
                        "tool": action_tool,
                        "params": action_params,
                        "success": success,
                        "artifact_id": "",
                        "output_preview": output_preview[:600],
                    }
                )
                status = "OK" if success else "FAIL"
                summary_lines.append(f"- [{idx}] {status} {action_tool}: {output_preview[:200]}")
            except Exception as exc:
                action_records.append(
                    {
                        "index": idx,
                        "tool": action_tool,
                        "params": action_params,
                        "success": False,
                        "error": str(exc),
                    }
                )
                summary_lines.append(f"- [{idx}] FAIL {action_tool}: {exc}")

        success_count = sum(1 for item in action_records if item.get("success", False))
        failed_count = len(action_records) - success_count
        summary_lines.append(
            f"<summary total=\"{len(action_records)}\" success=\"{success_count}\" failed=\"{failed_count}\" />"
        )
        summary_lines.append("</subtask>")
        subtask_entry = {
            "id": subtask_id,
            "title": title,
            "started_at": started_at,
            "finished_at": int(time.time()),
            "total_actions": len(action_records),
            "success_actions": success_count,
            "failed_actions": failed_count,
            "actions": action_records,
        }
        return "\n".join(summary_lines), subtask_entry, loaded_skill_names, skill_context


    def _invoke_external_tool(self, func, params: Dict[str, Any]) -> Any:
        """
        .

        :
        1)  kwargs ;
        2)  schema  {"input": "..."},;
        3) .
        """
        try:
            return func(**params)
        except TypeError as first_error:
            #  schema: input ,
            if isinstance(params, dict) and set(params.keys()) == {"input"}:
                try:
                    return func(params["input"])
                except TypeError:
                    pass

            # 
            try:
                sig = inspect.signature(func)
                if len(sig.parameters) == 0:
                    return func()
            except (TypeError, ValueError):
                pass

            raise first_error


    def _print_tool_params(self, tool_name: str, params: Dict[str, Any]) -> None:
        """Emit concise tool parameters in stable `key: value` form for TUI rendering."""
        if not isinstance(params, dict) or not params:
            return

        def _short_text(value: Any, *, limit: int = 180) -> str:
            if isinstance(value, (dict, list, tuple)):
                try:
                    text = json.dumps(value, ensure_ascii=False)
                except Exception:
                    text = str(value)
            else:
                text = str(value)
            text = " ".join(text.split())
            if len(text) > limit:
                return text[: limit - 3] + "..."
            return text

        def _emit_param(key: str, value: Any, *, limit: int = 180) -> None:
            key_text = str(key or "").strip()
            value_text = _short_text(value, limit=limit).strip()
            if not key_text or not value_text:
                return
            self._emit("tool_exec", f"  {key_text}: {value_text}")

        if tool_name == "apply_patch":
            patch_text = str(params.get("patch", "") or "")
            _emit_param("patch_len", len(patch_text))
            if patch_text:
                _emit_param("patch_preview", patch_text, limit=220)
            for key in ("path", "file_path", "target_path"):
                if key in params:
                    _emit_param(key, params.get(key))
            return

        if tool_name == "run_subtask":
            _emit_param("title", params.get("title", ""))
            actions = params.get("actions", [])
            action_count = len(actions) if isinstance(actions, list) else 0
            _emit_param("action_count", action_count)
            return

        # Generic fallback: print every provided param in deterministic key order.
        for key in sorted(params.keys()):
            value = params.get(key)
            if value is None:
                continue
            _emit_param(str(key), value)


    def _print_tool_output(self, output: str, max_lines: int = 20) -> None:
        """()"""
        lines = output.split('\n')
        result_text = "\n".join(lines[:max_lines])
        if len(lines) > max_lines:
            result_text += f"\n    ... ( {len(lines) - max_lines} )"
        self._emit("tool_result", result_text)


    def _format_progress_entry(self, tool_name: str, params: Dict[str, Any], status: str) -> str:
        """"""
        if tool_name == 'list_directory':
            path = params.get('path', '.')
            return f"- : {path} [{status}]"
        elif tool_name == 'read_file':
            path = params.get('path', '')
            return f"- : {path} [{status}]"
        elif tool_name == 'apply_patch':
            return f"-  [{status}]"
        elif tool_name == 'execute_command':
            cmd = params.get('command', '')[:50]
            return f"- : {cmd} [{status}]"
        elif tool_name == 'search_in_files':
            pattern = params.get('pattern', '')[:50]
            path = params.get('path', '.')
            return f"- : '{pattern}' in {path} [{status}]"
        elif tool_name == 'wait_terminal_until':
            name = params.get('name', '')
            return f"- : {name} [{status}]"
        elif tool_name == 'read_terminal_since_last':
            name = params.get('name', '')
            return f"- : {name} [{status}]"
        elif tool_name == 'run_and_wait':
            name = params.get('name', '')
            return f"- : {name} [{status}]"
        elif tool_name == 'load_skill':
            name = params.get('name', params.get('skill_name', params.get('input', '')))
            return f"- : {name} [{status}]"
        elif tool_name == 'run_subtask':
            title = params.get('title', 'subtask')
            return f"- : {title} [{status}]"
        elif tool_name == 'read_tui':
            name = params.get('name', '')
            skip_empty = params.get('skip_empty_lines', 'default')
            return f"- [TUI] : {name} (skip_empty_lines={skip_empty}) [{status}]"
        elif tool_name == 'read_tui_diff':
            name = params.get('name', '')
            return f"- [TUI] : {name} [{status}]"
        elif tool_name == 'read_tui_region':
            name = params.get('name', '')
            return f"- [TUI] : {name} [{status}]"
        elif tool_name == 'find_text_in_tui':
            name = params.get('name', '')
            text = str(params.get('text', '') or '')[:40]
            return f"- [TUI] : {name} '{text}' [{status}]"
        elif tool_name == 'send_keys':
            name = params.get('name', '')
            keys = str(params.get('keys', '') or '')
            if len(keys) > 60:
                keys = keys[:60] + "..."
            return f"- [TUI] : {name} keys='{keys}' [{status}]"
        elif tool_name == 'send_keys_and_read':
            name = params.get('name', '')
            keys = str(params.get('keys', '') or '')
            if len(keys) > 60:
                keys = keys[:60] + "..."
            return f"- [TUI] : {name} keys='{keys}' [{status}]"
        elif tool_name == 'wait_tui_until':
            name = params.get('name', '')
            return f"- [TUI] : {name} [{status}]"
        else:
            return f"-  {tool_name} [{status}]"


























