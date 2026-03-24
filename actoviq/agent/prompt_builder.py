"""
Klynx Agent - 
PromptBuilderMixin , XML ,
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from actoviq.agent.context_manager import TokenCounter


class PromptBuilderMixin:
    """ Mixin -  KlynxAgent """

    # ()
    # Base system prompt is defined in agents.py

    # (90%)
    CONTEXT_SUMMARIZE_THRESHOLD = 0.9
    RECENT_HISTORY_LIMIT = 8
    LIGHT_ASSISTANT_PREVIEW_CHARS = 320
    HEAVY_ASSISTANT_PREVIEW_CHARS = 120
    PROGRESS_SUMMARY_MAX_CHARS = 3200
    CONTEXT_MAX_OPTIONAL_SEGMENTS = 24
    CONTEXT_MAX_OPTIONAL_SEGMENT_TOKENS = 4000
    CONTEXT_MAX_OPTIONAL_TOTAL_TOKENS = 32000
    MJP_TOOL_LEDGER_MAX = 12
    HYDRATED_TOOL_OUTPUT_MAX_ITEMS = 3
    HYDRATED_TOOL_OUTPUT_SNIPPET_CHARS = 1200
    DEFAULT_LOW_INFO_TERMS = (
        "",
        "",
        "",
        "",
        "ok",
        "yes",
        "",
        "",
        "continue",
        "go",
        "y",
    )

    _PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

    def _build_context_slice_budget(self, state: Dict[str, Any]) -> Dict[str, Any]:
        def _coerce(name: str, fallback: int, floor: int, ceil: int) -> int:
            raw = state.get(name, fallback)
            try:
                value = int(raw)
            except Exception:
                value = fallback
            return max(floor, min(value, ceil))

        max_segments = _coerce(
            "max_context_segments_per_round",
            int(getattr(self, "CONTEXT_MAX_OPTIONAL_SEGMENTS", 24) or 24),
            1,
            200,
        )
        max_segment_tokens = _coerce(
            "max_context_segment_tokens",
            int(getattr(self, "CONTEXT_MAX_OPTIONAL_SEGMENT_TOKENS", 4000) or 4000),
            80,
            100000,
        )
        max_total_tokens = _coerce(
            "max_context_total_optional_tokens",
            int(getattr(self, "CONTEXT_MAX_OPTIONAL_TOTAL_TOKENS", 32000) or 32000),
            400,
            500000,
        )
        return {
            "max_segments": max_segments,
            "max_segment_tokens": max_segment_tokens,
            "max_total_tokens": max_total_tokens,
            "used_segments": 0,
            "used_tokens": 0,
            "skipped": [],
        }

    def _append_context_section(
        self,
        *,
        xml_parts: List[str],
        token_stats: Dict[str, int],
        section_name: str,
        section_xml: str,
        budget: Optional[Dict[str, Any]] = None,
        required: bool = False,
    ) -> bool:
        payload = str(section_xml or "").strip()
        if not payload:
            return False

        section_tokens = TokenCounter.estimate_tokens(payload)
        include = True
        skip_reason = ""
        if not required and isinstance(budget, dict):
            max_segments = int(budget.get("max_segments", 0) or 0)
            max_segment_tokens = int(budget.get("max_segment_tokens", 0) or 0)
            max_total_tokens = int(budget.get("max_total_tokens", 0) or 0)
            used_segments = int(budget.get("used_segments", 0) or 0)
            used_tokens = int(budget.get("used_tokens", 0) or 0)

            if max_segments and used_segments >= max_segments:
                include = False
                skip_reason = "max_segments"
            elif max_segment_tokens and section_tokens > max_segment_tokens:
                include = False
                skip_reason = "max_segment_tokens"
            elif max_total_tokens and used_tokens + section_tokens > max_total_tokens:
                include = False
                skip_reason = "max_total_tokens"

            if include:
                budget["used_segments"] = used_segments + 1
                budget["used_tokens"] = used_tokens + section_tokens
            elif skip_reason:
                budget.setdefault("skipped", []).append(
                    {
                        "section": section_name,
                        "reason": skip_reason,
                        "tokens": section_tokens,
                    }
                )

        if not include:
            return False

        xml_parts.append(payload)
        token_stats[section_name] = section_tokens
        return True

    def _extract_latest_user_intent(self, state: Dict[str, Any]) -> str:
        direct_input = str(state.get("user_input", "") or "").strip()
        if direct_input:
            return direct_input

        history = state.get("messages", []) or []
        if not isinstance(history, list):
            return ""
        for msg in reversed(history):
            if not isinstance(msg, HumanMessage):
                continue
            content = str(getattr(msg, "content", "") or "").strip()
            if not content:
                continue
            if "<tool_result" in content.lower():
                continue
            return content
        return ""

    def _build_minimum_judgement_package_xml(
        self,
        *,
        state: Dict[str, Any],
        task_plan: List[Dict[str, Any]],
        current_step_id: str,
        completed_steps: set,
    ) -> str:
        latest_intent = self._truncate_text(self._extract_latest_user_intent(state), 280)
        active_step_id = str(current_step_id or "").strip()
        if not active_step_id:
            active_step_id = self._first_pending_step_from_plan(task_plan, completed_steps)

        active_step_title = ""
        if active_step_id:
            for idx, step in enumerate(task_plan or []):
                if isinstance(step, dict):
                    step_id = str(step.get("id", "") or f"step_{idx + 1}").strip()
                    if step_id != active_step_id:
                        continue
                    active_step_title = str(
                        step.get("title", "")
                        or step.get("task", "")
                        or step.get("name", "")
                        or step_id
                    ).strip()
                    break

        tool_ledger_recent = state.get("tool_ledger_recent", []) or []
        ledger_lines: List[str] = []
        if isinstance(tool_ledger_recent, list):
            for item in tool_ledger_recent[-self.MJP_TOOL_LEDGER_MAX :]:
                if not isinstance(item, dict):
                    continue
                call_id = str(item.get("call_id", "") or "").strip()
                tool = str(item.get("tool", "") or "").strip()
                status = str(item.get("status", "") or "").strip() or "unknown"
                reused = str(bool(item.get("reused", False))).lower()
                artifact_id = str(item.get("artifact_id", "") or "").strip()
                summary = self._truncate_text(str(item.get("summary", "") or "").strip(), 160)
                attr_call = f' call_id="{self._escape_xml(call_id)}"' if call_id else ""
                attr_tool = f' tool="{self._escape_xml(tool)}"' if tool else ""
                attr_artifact = f' artifact_id="{self._escape_xml(artifact_id)}"' if artifact_id else ""
                ledger_lines.append(
                    f'      <tool_call{attr_call}{attr_tool} status="{self._escape_xml(status)}" reused="{reused}"{attr_artifact}>'
                    f"{self._escape_xml(summary)}</tool_call>"
                )

        mjp_lines = ["  <minimum_judgement_package>"]
        if latest_intent:
            mjp_lines.append(f"    <latest_user_intent>{self._escape_xml(latest_intent)}</latest_user_intent>")
        if active_step_id or active_step_title:
            mjp_lines.append(
                f'    <active_step id="{self._escape_xml(active_step_id)}">{self._escape_xml(active_step_title)}</active_step>'
            )
        if ledger_lines:
            mjp_lines.append("    <tool_ledger_recent>")
            mjp_lines.extend(ledger_lines)
            mjp_lines.append("    </tool_ledger_recent>")
        if len(mjp_lines) == 1:
            return ""
        mjp_lines.append("  </minimum_judgement_package>")
        return "\n".join(mjp_lines)

    @staticmethod
    def _extract_query_terms(text: str, max_terms: int = 20) -> List[str]:
        tokens: List[str] = []
        seen = set()
        for raw in re.findall(r"[A-Za-z0-9_./:-]+", str(text or "").lower()):
            token = str(raw or "").strip()
            if len(token) < 3 or token in seen:
                continue
            seen.add(token)
            tokens.append(token)
            if len(tokens) >= max_terms:
                break
        return tokens

    def _load_tool_artifact_content(
        self,
        artifact: Dict[str, Any],
        thread_id: str = "",
    ) -> str:
        if not isinstance(artifact, dict):
            return ""
        direct = str(artifact.get("content", "") or "")
        if direct:
            return direct
        store_key = str(artifact.get("content_store_key", "") or "").strip()
        store = getattr(self, "store", None)
        if store_key and store is not None and hasattr(store, "get"):
            try:
                payload = str(store.get(store_key) or "")
                if payload:
                    return payload
            except Exception:
                pass
        artifact_id = str(artifact.get("id", "") or "").strip()
        backup_key = f"tool_artifact:{thread_id or 'default'}:{artifact_id}" if artifact_id else ""
        if backup_key and store is not None and hasattr(store, "get"):
            try:
                return str(store.get(backup_key) or "")
            except Exception:
                return ""
        return ""

    @staticmethod
    def _build_hydrated_excerpt(content: str, terms: List[str], limit: int) -> str:
        text = str(content or "")
        if not text:
            return ""
        cap = max(int(limit or 0), 200)
        if len(text) <= cap:
            return text

        lowered = text.lower()
        for term in terms:
            idx = lowered.find(term.lower())
            if idx < 0:
                continue
            left = max(0, idx - int(cap * 0.35))
            right = min(len(text), left + cap)
            excerpt = text[left:right]
            if left > 0:
                excerpt = "...\n" + excerpt
            if right < len(text):
                excerpt = excerpt + "\n..."
            return excerpt

        head_size = int(cap * 0.6)
        tail_size = max(cap - head_size, 120)
        head = text[:head_size]
        tail = text[-tail_size:] if tail_size < len(text) else ""
        if tail:
            return f"{head}\n...\n{tail}"
        return head

    def _build_hydrated_tool_outputs_xml(
        self,
        *,
        state: Dict[str, Any],
        current_focus: str,
        current_task: str,
        overall_goal: str,
    ) -> str:
        artifacts = state.get("tool_artifacts", []) or []
        if not isinstance(artifacts, list) or not artifacts:
            return ""

        latest_intent = self._extract_latest_user_intent(state)
        query_text = " ".join(
            [
                str(current_focus or "").strip(),
                str(current_task or "").strip(),
                str(overall_goal or "").strip(),
                str(latest_intent or "").strip(),
            ]
        ).strip()
        query_terms = self._extract_query_terms(query_text, max_terms=24)

        ledger_by_artifact: Dict[str, Dict[str, Any]] = {}
        for item in state.get("tool_ledger_recent", []) or []:
            if not isinstance(item, dict):
                continue
            artifact_id = str(item.get("artifact_id", "") or "").strip()
            if artifact_id:
                ledger_by_artifact[artifact_id] = dict(item)

        scored: List[Dict[str, Any]] = []
        for recency_idx, artifact in enumerate(reversed(artifacts[-40:]), start=1):
            if not isinstance(artifact, dict):
                continue
            artifact_id = str(artifact.get("id", "") or "").strip()
            tool_name = str(artifact.get("tool", "") or "").strip()
            if not artifact_id:
                continue

            params = artifact.get("params", {}) or {}
            try:
                params_text = json.dumps(params, ensure_ascii=False, sort_keys=True)
            except Exception:
                params_text = str(params)
            searchable = f"{tool_name} {params_text}".lower()

            score = max(0, 15 - recency_idx)
            reasons: List[str] = ["recent"]
            ledger = ledger_by_artifact.get(artifact_id, {})
            status = str(ledger.get("status", "") or "").strip().lower()
            if ledger:
                score += 4
                reasons.append("ledger_linked")
            if status and status != "success":
                score += 4
                reasons.append("error_priority")
            if status == "success":
                score += 1
            match_hits = 0
            for term in query_terms:
                if term and term in searchable:
                    match_hits += 1
            if match_hits:
                score += min(match_hits, 4)
                reasons.append("query_match")
            if tool_name in {"execute_command", "run_in_terminal", "read_terminal", "search_in_files", "load_skill"}:
                score += 1

            scored.append(
                {
                    "score": score,
                    "artifact": artifact,
                    "reasons": reasons,
                    "recency_idx": recency_idx,
                    "status": status,
                }
            )

        if not scored:
            return ""
        scored.sort(key=lambda item: (int(item.get("score", 0)), -int(item.get("recency_idx", 0))), reverse=True)
        max_items = max(int(getattr(self, "HYDRATED_TOOL_OUTPUT_MAX_ITEMS", 3) or 3), 1)
        snippet_limit = max(int(getattr(self, "HYDRATED_TOOL_OUTPUT_SNIPPET_CHARS", 1200) or 1200), 200)
        thread_id = str(state.get("thread_id", "") or "").strip()

        lines = ["  <hydrated_tool_outputs policy=\"automatic\" source=\"tool_artifacts\">"]
        included = 0
        for item in scored:
            if included >= max_items:
                break
            artifact = dict(item.get("artifact", {}) or {})
            artifact_id = str(artifact.get("id", "") or "").strip()
            if not artifact_id:
                continue
            content = self._load_tool_artifact_content(artifact, thread_id=thread_id)
            if not content:
                continue
            excerpt = self._build_hydrated_excerpt(content, query_terms, snippet_limit)
            if not excerpt.strip():
                continue
            total_chars = len(content)
            total_lines = max(len(content.splitlines()), 1)
            tool_name = str(artifact.get("tool", "") or "").strip()
            reason_text = ",".join(item.get("reasons", []) or [])
            lines.append(
                f'    <artifact id="{self._escape_xml(artifact_id)}" '
                f'tool="{self._escape_xml(tool_name)}" '
                f'score="{int(item.get("score", 0) or 0)}" '
                f'status="{self._escape_xml(str(item.get("status", "") or ""))}" '
                f'total_lines="{total_lines}" total_chars="{total_chars}" '
                f'reason="{self._escape_xml(reason_text)}">'
                f"{self._escape_xml(excerpt)}"
                "</artifact>"
            )
            included += 1

        if included == 0:
            return ""
        lines.append("  </hydrated_tool_outputs>")
        return "\n".join(lines)

    def _load_prompt_fragment(self, *relative_parts: str, fallback: str = "") -> str:
        prompt_path = self._PROMPT_DIR.joinpath(*relative_parts)
        try:
            return prompt_path.read_text(encoding="utf-8").strip()
        except Exception:
            return fallback.strip()

    def _render_markdown_bullets(self, values: Iterable[str], indent: str = "- ") -> str:
        lines: List[str] = []
        for raw in values:
            text = re.sub(r"\s+", " ", str(raw or "")).strip()
            if text:
                lines.append(f"{indent}{text}")
        return "\n".join(lines)

    def _build_modern_system_prompt(self) -> str:
        base_system_prompt = (getattr(self, "LOCKED_SYSTEM_PROMPT", "") or "").strip()
        prompt_parts = [base_system_prompt] if base_system_prompt else []

        runtime_append = (getattr(self, "_runtime_system_prompt_append", "") or "").strip()
        if runtime_append:
            prompt_parts.append(
                "\n".join(
                    [
                        "## Runtime System Extension",
                        "",
                        "The block below is an append-only system extension.",
                        runtime_append,
                    ]
                ).strip()
            )

        os_lower = str(getattr(self, "os_name", "") or "").strip().lower()
        search_policy = (
            "Use structured search tools first; backend selection is runtime-managed."
            if os_lower in {"windows", "win"}
            else "Use structured search tools first; backend selection is runtime-managed."
        )
        prompt_parts.append(
            "\n".join(
                [
                    "## Environment",
                    "",
                    f"- OS: {self.os_name}",
                    f"- Working directory: `{self.working_dir or '.'}`",
                    f"- Search policy: {search_policy}",
                ]
            )
        )

        skills_prompt = (getattr(self, "_skills_prompt_cache", "") or "").strip()
        if skills_prompt:
            prompt_parts.append(skills_prompt)

        tool_prompt = (getattr(self, "_tool_prompts_cache", "") or "").strip()
        if tool_prompt:
            prompt_parts.append(tool_prompt)

        prompt_parts.append(self._load_prompt_fragment("fragments", "tool_selection.md"))

        runtime_tool_names = set(self._runtime_tool_names())
        has_exec_sessions = bool(
            {"exec_command", "launch_interactive_session"} & runtime_tool_names
        )
        has_write_stdin = "write_stdin" in runtime_tool_names
        has_close_session = "close_exec_session" in runtime_tool_names
        has_legacy_terminal = bool(
            {"create_terminal", "read_terminal", "wait_terminal_until"}
            & runtime_tool_names
        )
        interactive_notes: List[str] = ["## Interactive Execution Notes", ""]
        if has_exec_sessions:
            interactive_notes.extend(
                [
                    "- `exec_command` and `launch_interactive_session` return a session id,",
                    "  often shaped like `exec_xxx`.",
                ]
            )
        if has_write_stdin and has_close_session:
            interactive_notes.extend(
                [
                    "- Continue those sessions with `write_stdin` and close them with",
                    "  `close_exec_session`.",
                ]
            )
        elif has_write_stdin:
            interactive_notes.append(
                "- Continue interactive exec sessions with `write_stdin`."
            )
        elif has_close_session:
            interactive_notes.append(
                "- Close interactive exec sessions with `close_exec_session`."
            )
        if has_legacy_terminal:
            interactive_notes.extend(
                [
                    "- Legacy terminal tools such as `read_terminal` and",
                    "  `wait_terminal_until` only accept a named terminal created by",
                    "  `create_terminal`; do not pass `exec_xxx` session ids to them.",
                ]
            )
        interactive_notes.extend(
            [
                "- On Windows, prefer passing `workdir` instead of composing",
                "  `cd /d ...` command prefixes.",
                "- When `<tui_views>` exists in runtime context, prefer it over raw",
                "  audit history when judging the current screen state.",
            ]
        )
        if len(interactive_notes) > 3:
            prompt_parts.append("\n".join(interactive_notes))

        if self.memory_dir:
            memory_path = os.path.join(self.memory_dir, ".klynx", ".memory")
            rel_path = (
                os.path.relpath(memory_path, self.working_dir)
                if self.working_dir
                else ".actoviq/.memory"
            )
            prompt_parts.append(self._load_prompt_fragment("fragments", "memory_mode.md"))
            prompt_parts.append(
                "\n".join(
                    [
                        "### Memory File",
                        "",
                        f"- Memory path: `{rel_path}`",
                        "- Preferred XML shape:",
                        "```xml",
                        "<memory>",
                        '  <entry key="user_preferences">...</entry>',
                        '  <entry key="project_architecture">...</entry>',
                        "</memory>",
                        "```",
                    ]
                )
            )

        if getattr(self, "_tui_guide_loaded", False):
            prompt_parts.append(self._build_modern_tui_guide())

        return "\n\n".join(part for part in prompt_parts if part).strip()

    def _build_modern_tui_guide(self) -> str:
        base = self._load_prompt_fragment("fragments", "tui_mode.md")
        examples = "\n".join(
            [
                "### Key Reference",
                "",
                "- Supported keys: `Enter`, `Tab`, `Escape`, `Backspace`, `Delete`,",
                "  `Space`, arrow keys, `Home`, `End`, `PageUp`, `PageDown`, `Ctrl-C`,",
                "  `Ctrl-D`, and `F1` through `F12`.",
                "- Example input: `i hello Space world Escape`",
                "- Example input: `Escape :wq Enter`",
                "- Example input: `Ctrl-C`",
            ]
        )
        return "\n\n".join(part for part in (base, examples) if part).strip()
    def _get_system_prompt(self) -> str:
        return self._build_modern_system_prompt()

    def _get_tui_guide(self) -> str:
        return self._build_modern_tui_guide()

    def _get_tool_names_prompt(self) -> str:
        """Return a compact comma-separated list of active tool names."""
        if not self.tools:
            return "()"
        names = sorted(self.tools.keys())
        return ", ".join(names)

    def _compact_text_items(self, values: Any, limit: int = 2, max_chars: int = 180) -> List[str]:
        items: List[str] = []
        seen = set()
        if isinstance(values, (str, bytes)):
            iterable = [values]
        elif isinstance(values, (list, tuple, set)):
            iterable = values
        elif values:
            iterable = [values]
        else:
            iterable = []
        for raw in iterable:
            text = re.sub(r"\s+", " ", str(raw or "")).strip()
            if not text or text in seen:
                continue
            if len(text) > max_chars:
                text = text[:max_chars] + "..."
            items.append(text)
            seen.add(text)
            if len(items) >= limit:
                break
        return items

    def _get_env_snapshot(self) -> str:
        """
        (XML)
        """
        # 
        file_tree = self._generate_file_tree(self.working_dir, depth=2)
        
        snapshot = f"""<file_tree>
{file_tree}
</file_tree>"""
        return snapshot
    
    def _generate_file_tree(self, path: str, depth: int = 2, prefix: str = "") -> str:
        """"""
        if depth < 0:
            return ""
        
        try:
            entries = sorted(os.listdir(path))
        except PermissionError:
            return f"{prefix}[]"
        
        lines = []
        # 
        ignore_patterns = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', '.idea', '.vscode'}
        entries = [e for e in entries if e not in ignore_patterns and not e.startswith('.')]
        
        for i, entry in enumerate(entries):
            is_last = (i == len(entries) - 1)
            connector = "鈹斺攢鈹€ " if is_last else "鈹溾攢鈹€ "
            entry_path = os.path.join(path, entry)
            
            if os.path.isdir(entry_path):
                lines.append(f"{prefix}{connector}{entry}/")
                if depth > 0:
                    extension = "    " if is_last else "鈹?  "
                    subtree = self._generate_file_tree(entry_path, depth - 1, prefix + extension)
                    if subtree:
                        lines.append(subtree)
            else:
                lines.append(f"{prefix}{connector}{entry}")
        
        return "\n".join(lines)
    def _is_mimo_model_route(self) -> bool:
        model_obj = getattr(self, "model", None)
        route = str(getattr(model_obj, "model", "") or "").strip().lower()
        return route.startswith("xiaomi_mimo/") or route.startswith("mimo-v2-")

    def _get_recent_history_limit(self, state: Optional[dict] = None) -> int:
        """ recent_history , 8."""
        state = state or {}
        raw_limit = state.get(
            "recent_history_limit",
            getattr(self, "recent_history_limit", self.RECENT_HISTORY_LIMIT),
        )
        try:
            limit = int(raw_limit)
        except Exception:
            limit = self.RECENT_HISTORY_LIMIT
        return max(limit, 1)

    def _get_low_info_terms(self) -> set:
        """."""
        custom_terms = getattr(self, "low_info_terms", None)
        if isinstance(custom_terms, (list, tuple, set)) and custom_terms:
            return {str(term).strip().lower() for term in custom_terms}
        return {term.lower() for term in self.DEFAULT_LOW_INFO_TERMS}

    def _is_low_information_text(self, text: str) -> bool:
        """."""
        normalized = (text or "").strip().lower()
        if not normalized:
            return True
        compact = re.sub(r"\s+", "", normalized)
        low_terms = self._get_low_info_terms()
        return normalized in low_terms or compact in {re.sub(r"\s+", "", t) for t in low_terms}

    @staticmethod
    def _safe_json_loads(payload: Any) -> Any:
        """ JSON ;."""
        if isinstance(payload, (dict, list)):
            return payload
        if not isinstance(payload, str):
            return payload
        raw = payload.strip()
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except Exception:
            return payload

    def _runtime_tool_names(self) -> List[str]:
        """( + )."""
        runtime_tools = set(self.tools.keys()) if getattr(self, "tools", None) else set()
        runtime_tools.add("state_update")
        return sorted(name for name in runtime_tools if name)

    def _collect_native_tool_calls(self, message: AIMessage) -> List[Dict[str, Any]]:
        """
         AIMessage.additional_kwargs.tool_calls  native tool calling .
         provider :
        1) {"tool": "...", "params": {...}}
        2) {"name": "...", "arguments": "..."}
        3) {"function": {"name": "...", "arguments": "..."}}
        """
        calls: List[Dict[str, Any]] = []
        additional = getattr(message, "additional_kwargs", {}) or {}
        raw_calls = additional.get("tool_calls", []) or []
        if not isinstance(raw_calls, list):
            return calls

        for raw_call in raw_calls:
            if not isinstance(raw_call, dict):
                continue

            tool_name = ""
            params: Any = {}
            call_id = ""

            if raw_call.get("tool"):
                tool_name = str(raw_call.get("tool", "")).strip()
                params = raw_call.get("params", {})
                call_id = str(
                    raw_call.get("call_id", "")
                    or raw_call.get("id", "")
                    or raw_call.get("tool_call_id", "")
                    or ""
                ).strip()
            elif isinstance(raw_call.get("function"), dict):
                function_block = raw_call.get("function", {}) or {}
                tool_name = str(function_block.get("name", "")).strip()
                params = function_block.get("arguments", {})
                call_id = str(
                    raw_call.get("call_id", "")
                    or raw_call.get("id", "")
                    or raw_call.get("tool_call_id", "")
                    or function_block.get("call_id", "")
                    or function_block.get("id", "")
                    or function_block.get("tool_call_id", "")
                    or ""
                ).strip()
            elif raw_call.get("name"):
                tool_name = str(raw_call.get("name", "")).strip()
                params = raw_call.get("arguments", {})
                call_id = str(
                    raw_call.get("call_id", "")
                    or raw_call.get("id", "")
                    or raw_call.get("tool_call_id", "")
                    or ""
                ).strip()

            if not tool_name:
                continue

            normalized_params = self._safe_json_loads(params)
            if normalized_params is None:
                normalized_params = {}

            calls.append(
                {
                    "source": "native",
                    "tool": tool_name,
                    "params": normalized_params,
                    "call_id": call_id,
                }
            )
        return calls

    def _collect_assistant_tool_events(self, message: AIMessage, tool_names: Iterable[str]) -> List[Dict[str, Any]]:
        """ assistant native tool calls only."""
        events: List[Dict[str, Any]] = []
        seen = set()

        native_calls = self._collect_native_tool_calls(message)
        _ = tool_names

        for event in native_calls:
            try:
                params_fingerprint = json.dumps(event.get("params", {}), ensure_ascii=False, sort_keys=True)
            except Exception:
                params_fingerprint = str(event.get("params", ""))
            key = (
                event.get("call_id", ""),
                event.get("tool", ""),
                params_fingerprint,
            )
            if key in seen:
                continue
            seen.add(key)
            events.append(event)
        return events

    def _canonicalize_tool_history_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """
        Keep model-visible tool history pairable.

        - Repair orphan tool_result messages by inserting a synthetic assistant tool call.
        - Append interrupted tool_result markers for pending calls without output.
        """
        if not messages:
            return []

        canonical: List[BaseMessage] = []
        pending_calls: List[Dict[str, str]] = []
        pending_by_call_id: Dict[str, Dict[str, str]] = {}

        for message in messages:
            if isinstance(message, AIMessage):
                canonical.append(message)
                for call in self._collect_native_tool_calls(message):
                    tool_name = str(call.get("tool", "") or "").strip()
                    call_id = str(call.get("call_id", "") or "").strip()
                    if not tool_name:
                        continue
                    entry = {
                        "tool": tool_name,
                        "call_id": call_id,
                    }
                    pending_calls.append(entry)
                    if call_id:
                        pending_by_call_id[call_id] = entry
                continue

            if isinstance(message, HumanMessage):
                content = str(getattr(message, "content", "") or "")
                if "<tool_result" in content.lower():
                    tool_name = self._extract_tool_result_name(content) or "unknown"
                    result_call_id = self._extract_tool_result_call_id(content)
                    resolved_call_id = result_call_id
                    if result_call_id and result_call_id in pending_by_call_id:
                        matched_entry = pending_by_call_id.pop(result_call_id)
                        pending_calls = [
                            entry
                            for entry in pending_calls
                            if not (
                                str(entry.get("call_id", "") or "").strip() == result_call_id
                                and str(entry.get("tool", "") or "").strip()
                                == str(matched_entry.get("tool", "") or "").strip()
                            )
                        ]
                        canonical.append(HumanMessage(content=content))
                    elif pending_calls:
                        match_index = -1
                        for idx, entry in enumerate(pending_calls):
                            if str(entry.get("tool", "") or "").strip() == tool_name:
                                match_index = idx
                                break
                        if match_index < 0:
                            match_index = 0
                        matched_entry = pending_calls.pop(match_index)
                        resolved_call_id = str(matched_entry.get("call_id", "") or "").strip() or result_call_id
                        if resolved_call_id:
                            pending_by_call_id.pop(resolved_call_id, None)
                        content = self._ensure_tool_result_call_id(content, resolved_call_id)
                        canonical.append(HumanMessage(content=content))
                    else:
                        synthetic_call_id = result_call_id or f"orphan_call_{len(canonical)}"
                        content = self._ensure_tool_result_call_id(content, synthetic_call_id)
                        canonical.append(
                            AIMessage(
                                content="",
                                additional_kwargs={
                                    "tool_calls": [
                                        {
                                            "tool": tool_name,
                                            "params": {},
                                            "call_id": synthetic_call_id,
                                        }
                                    ]
                                },
                            )
                        )
                        canonical.append(HumanMessage(content=content))
                    continue
                canonical.append(message)
                continue

            canonical.append(message)

        for pending in pending_calls:
            tool_name = str(pending.get("tool", "") or "").strip()
            call_id = str(pending.get("call_id", "") or "").strip()
            safe_tool = self._escape_xml(str(tool_name or "").strip() or "unknown")
            call_attr = f' call_id="{self._escape_xml(call_id)}"' if call_id else ""
            canonical.append(
                HumanMessage(
                    content=(
                        f'<tool_result tool="{safe_tool}" status="interrupted"{call_attr}>'
                        "interrupted: missing tool output for this call"
                        "</tool_result>"
                    )
                )
            )

        return canonical

    @staticmethod
    def _truncate_text(text: str, limit: int) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        if len(text) <= limit:
            return text
        return text[: max(limit - 3, 0)] + "..."

    def _score_file_view_focus(self, path: str, current_focus: str) -> int:
        focus_text = str(current_focus or "").strip().lower()
        normalized_path = str(path or "").strip().lower()
        if not focus_text or not normalized_path:
            return 0
        basename = os.path.basename(normalized_path)
        rel_path = normalized_path.replace("\\", "/")
        score = 0
        if basename and basename in focus_text:
            score += 4
        if rel_path and rel_path in focus_text:
            score += 4
        for segment in [seg for seg in re.split(r"[\\/]+", rel_path) if seg]:
            if len(segment) >= 3 and segment in focus_text:
                score += 1
        return score

    def _select_file_view_paths(
        self,
        file_views: Dict[str, Any],
        active_paths: List[str],
        current_focus: str,
    ) -> List[str]:
        max_files = int(getattr(self, "FILE_VIEW_FILES_MAX", 12) or 12)
        ordered: List[str] = []
        seen = set()

        for path in active_paths or []:
            normalized_path = str(path or "").strip()
            if not normalized_path or normalized_path not in file_views or normalized_path in seen:
                continue
            ordered.append(normalized_path)
            seen.add(normalized_path)

        remaining = sorted(
            (
                str(path or "").strip()
                for path in (file_views or {}).keys()
                if str(path or "").strip() and str(path or "").strip() not in seen
            ),
            key=lambda path: (
                self._score_file_view_focus(path, current_focus),
                int((file_views.get(path, {}) or {}).get("last_updated", 0) or 0),
                int((file_views.get(path, {}) or {}).get("reads", 0) or 0),
                path,
            ),
            reverse=True,
        )
        ordered.extend(remaining)
        return ordered[:max_files]

    def _render_file_views_xml(self, state: Dict[str, Any], current_focus: str) -> str:
        file_views = state.get("file_views", {}) or {}
        if not isinstance(file_views, dict) or not file_views:
            return ""

        active_paths = state.get("active_file_view_paths", []) or []
        selected_paths = self._select_file_view_paths(file_views, active_paths, current_focus)
        if not selected_paths:
            return ""

        max_snippets = int(getattr(self, "FILE_VIEW_SNIPPETS_PER_FILE", 4) or 4)
        file_lines: List[str] = ["  <file_views>"]

        for path in selected_paths:
            meta = file_views.get(path, {}) or {}
            if not isinstance(meta, dict):
                continue
            total_lines = int(meta.get("total_lines", 0) or 0)
            reads = int(meta.get("reads", 0) or 0)
            snippets = meta.get("snippets", []) or []
            if not isinstance(snippets, list):
                snippets = []

            rendered_snippets: List[str] = []
            ordered_snippets = sorted(
                (
                    item
                    for item in snippets
                    if isinstance(item, dict) and str(item.get("content", "") or "").strip()
                ),
                key=lambda item: (
                    int(item.get("updated_at", 0) or 0),
                    int(item.get("end_line", 0) or 0),
                ),
                reverse=True,
            )[:max_snippets]

            for snippet in ordered_snippets:
                start_line = int(snippet.get("start_line", 0) or 0)
                end_line = int(snippet.get("end_line", 0) or 0)
                chunk_ids = ",".join(
                    str(item).strip()
                    for item in (snippet.get("chunk_ids", []) or [])
                    if str(item).strip()
                )
                content_hashes = ",".join(
                    str(item).strip()
                    for item in (snippet.get("content_hashes", []) or [])
                    if str(item).strip()
                )
                rendered_snippets.append(
                    f'      <snippet start_line="{start_line}" end_line="{end_line}" '
                    f'chunk_ids="{self._escape_xml(chunk_ids)}" '
                    f'content_hashes="{self._escape_xml(content_hashes)}">\n'
                    f'{self._escape_xml(str(snippet.get("content", "") or ""))}\n'
                    "      </snippet>"
                )

            if not rendered_snippets:
                continue

            file_lines.append(
                f'    <file path="{self._escape_xml(path)}" total_lines="{total_lines}" reads="{reads}">'
            )
            file_lines.extend(rendered_snippets)
            file_lines.append("    </file>")

        if len(file_lines) == 1:
            return ""
        file_lines.append("  </file_views>")
        return "\n".join(file_lines)

    def _select_tui_view_names(
        self,
        tui_views: Dict[str, Any],
        active_tui_view_names: List[str],
        current_focus: str,
    ) -> List[str]:
        max_sessions = int(getattr(self, "TUI_VIEW_SESSIONS_MAX", 8) or 8)
        ordered: List[str] = []
        seen = set()

        for name in active_tui_view_names or []:
            normalized_name = str(name or "").strip()
            if not normalized_name or normalized_name not in tui_views or normalized_name in seen:
                continue
            ordered.append(normalized_name)
            seen.add(normalized_name)

        focus_text = str(current_focus or "").strip().lower()
        remaining = sorted(
            (
                str(name or "").strip()
                for name in (tui_views or {}).keys()
                if str(name or "").strip() and str(name or "").strip() not in seen
            ),
            key=lambda name: (
                4 if focus_text and str(name).lower() in focus_text else 0,
                int((tui_views.get(name, {}) or {}).get("updated_at", 0) or 0),
                name,
            ),
            reverse=True,
        )
        ordered.extend(remaining)
        return ordered[:max_sessions]

    def _render_tui_views_xml(self, state: Dict[str, Any], current_focus: str) -> str:
        tui_views = state.get("tui_views", {}) or {}
        if not isinstance(tui_views, dict) or not tui_views:
            return ""

        selected_names = self._select_tui_view_names(
            tui_views=tui_views,
            active_tui_view_names=state.get("active_tui_view_names", []) or [],
            current_focus=current_focus,
        )
        if not selected_names:
            return ""

        active_focus = dict(state.get("active_tui_focus", {}) or {})
        last_diff = dict(state.get("last_tui_diff", {}) or {})
        last_region = dict(state.get("last_tui_region", {}) or {})
        last_anchor = dict(state.get("last_tui_anchor_match", {}) or {})
        max_lines = int(getattr(self, "TUI_VIEW_LINES_MAX", 18) or 18)
        xml_lines: List[str] = ["  <tui_views>"]

        for session_name in selected_names:
            meta = tui_views.get(session_name, {}) or {}
            if not isinstance(meta, dict):
                continue
            visible_lines = meta.get("visible_lines", []) or []
            if not isinstance(visible_lines, list):
                visible_lines = []
            last_region_excerpt = meta.get("last_region_excerpt", []) or []
            if not isinstance(last_region_excerpt, list):
                last_region_excerpt = []
            anchor_matches = meta.get("last_anchor_matches", []) or []
            if not isinstance(anchor_matches, list):
                anchor_matches = []
            changed_rows = ",".join(
                str(item) for item in (meta.get("last_changed_rows", []) or []) if str(item).strip()
            )

            xml_lines.append(
                f'    <tui_view session="{self._escape_xml(session_name)}" '
                f'rows="{int(meta.get("rows", 0) or 0)}" cols="{int(meta.get("cols", 0) or 0)}" '
                f'cursor_row="{int(meta.get("cursor_row", 0) or 0)}" cursor_col="{int(meta.get("cursor_col", 0) or 0)}" '
                f'screen_hash="{self._escape_xml(str(meta.get("screen_hash", "") or ""))}" '
                f'last_action="{self._escape_xml(str(meta.get("last_action", "") or ""))}" '
                f'updated_at="{int(meta.get("updated_at", 0) or 0)}" '
                f'changed_rows="{self._escape_xml(changed_rows)}">'
            )

            summary = str(meta.get("summary", "") or "").strip()
            if summary:
                xml_lines.append(f"      <summary>{self._escape_xml(summary)}</summary>")
            if active_focus and str(active_focus.get("session_name", "") or "").strip() == session_name:
                xml_lines.append(
                    f'      <focus event_type="{self._escape_xml(str(active_focus.get("event_type", "") or ""))}" '
                    f'changed_rows="{self._escape_xml(str(active_focus.get("changed_rows", "") or ""))}" '
                    f'match_rows="{self._escape_xml(str(active_focus.get("match_rows", "") or ""))}">'
                    f'{self._escape_xml(str(active_focus.get("summary", "") or ""))}</focus>'
                )
            if last_diff and str(last_diff.get("session_name", "") or "").strip() == session_name:
                xml_lines.append(
                    f'      <last_diff changed_rows="{self._escape_xml(str(last_diff.get("changed_rows", "") or ""))}">'
                    f'{self._escape_xml(str(last_diff.get("summary", "") or ""))}</last_diff>'
                )
            if last_region and str(last_region.get("session_name", "") or "").strip() == session_name:
                xml_lines.append(
                    f'      <last_region changed_rows="{self._escape_xml(str(last_region.get("changed_rows", "") or ""))}">'
                    f'{self._escape_xml(str(last_region.get("summary", "") or ""))}</last_region>'
                )
            if last_anchor and str(last_anchor.get("session_name", "") or "").strip() == session_name:
                xml_lines.append(
                    f'      <last_anchor text="{self._escape_xml(str(last_anchor.get("text", "") or ""))}" '
                    f'match_rows="{self._escape_xml(str(last_anchor.get("match_rows", "") or ""))}">'
                    f'{self._escape_xml(str(last_anchor.get("summary", "") or ""))}</last_anchor>'
                )
            if anchor_matches:
                xml_lines.append("      <anchors>")
                for item in anchor_matches[:6]:
                    xml_lines.append(
                        f'        <match row="{int(item.get("row", 0) or 0)}" text="{self._escape_xml(str(item.get("text", "") or ""))}">'
                        f'{self._escape_xml(str(item.get("excerpt", "") or ""))}</match>'
                    )
                xml_lines.append("      </anchors>")
            if last_region_excerpt:
                xml_lines.append("      <region_excerpt>")
                for item in last_region_excerpt[:max_lines]:
                    xml_lines.append(
                        f'        <line row="{int(item.get("row", 0) or 0)}">{self._escape_xml(str(item.get("text", "") or ""))}</line>'
                    )
                xml_lines.append("      </region_excerpt>")
            if visible_lines:
                xml_lines.append("      <viewport>")
                for item in visible_lines[:max_lines]:
                    xml_lines.append(
                        f'        <line row="{int(item.get("row", 0) or 0)}">{self._escape_xml(str(item.get("text", "") or ""))}</line>'
                    )
                xml_lines.append("      </viewport>")

            xml_lines.append("    </tui_view>")

        if len(xml_lines) == 1:
            return ""
        xml_lines.append("  </tui_views>")
        return "\n".join(xml_lines)

    def _strip_meta_blocks(self, content: str, tool_names: Iterable[str]) -> str:
        """,."""
        text = content or ""
        if not text:
            return ""

        text = re.sub(r"(?is)<thinking>.*?</thinking>", " ", text)
        text = re.sub(r"(?is)<reflection>.*?</reflection>", " ", text)
        text = re.sub(r"(?is)<step>.*?</step>", " ", text)
        text = re.sub(r"(?is)<task_goal>.*?</task_goal>", " ", text)

        for tool_name in tool_names:
            pattern = rf"(?is)<{re.escape(tool_name)}[\s>].*?</{re.escape(tool_name)}>"
            text = re.sub(pattern, " ", text)

        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    def _format_tool_event_text(self, event: Dict[str, Any]) -> str:
        """."""
        tool_name = str(event.get("tool", "")).strip() or "unknown"
        call_id = str(event.get("call_id", "") or "").strip()
        params = event.get("params", {})
        try:
            params_text = json.dumps(params, ensure_ascii=False, sort_keys=True)
        except Exception:
            params_text = str(params)
        source = str(event.get("source", "unknown"))
        call_prefix = f"{call_id} " if call_id else ""
        return f"[{source}] {call_prefix}{tool_name}({params_text})"

    @staticmethod
    def _normalize_compare_text(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip()

    def _is_reasoning_echo_message(
        self,
        message: AIMessage,
        plain_text: str,
        tool_names: Iterable[str],
    ) -> bool:
        additional = getattr(message, "additional_kwargs", {}) or {}
        reasoning_raw = str(additional.get("reasoning_content", "") or "").strip()
        if not reasoning_raw:
            return False

        normalized_plain = self._normalize_compare_text(plain_text)
        if not normalized_plain:
            return False

        normalized_reasoning = self._normalize_compare_text(
            self._strip_meta_blocks(reasoning_raw, tool_names)
        )
        if not normalized_reasoning:
            return False
        return normalized_plain == normalized_reasoning

    def _compress_ai_message(self, message: AIMessage, compression: str = "heavy") -> str:
        """
         AI :
        1. (native/xml )
        2. 
        """
        content = getattr(message, "content", "") or ""
        tool_names = self._runtime_tool_names()
        events = self._collect_assistant_tool_events(message, tool_names)
        plain_text = self._strip_meta_blocks(content, tool_names)
        if plain_text and self._is_reasoning_echo_message(message, plain_text, tool_names):
            plain_text = ""

        lines: List[str] = []
        for event in events:
            lines.append(self._format_tool_event_text(event))

        if plain_text and not self._is_low_information_text(plain_text):
            if compression == "light":
                preview = self._truncate_text(plain_text, self.LIGHT_ASSISTANT_PREVIEW_CHARS)
                if preview:
                    lines.append(f"[assistant] {preview}")
            elif not lines:
                # ,
                preview = self._truncate_text(plain_text, self.HEAVY_ASSISTANT_PREVIEW_CHARS)
                if preview:
                    lines.append(f"[assistant] {preview}")

        if not lines:
            return ""
        return "\n".join(lines)

    def _format_history_message(
        self,
        index: int,
        role: str,
        content: str,
        compression: str = "",
    ) -> str:
        escaped = self._escape_xml(content)
        if compression:
            return (
                f'    <message index="{index}" role="{role}" compression="{compression}">\n'
                f"      {escaped}\n"
                f"    </message>"
            )
        return (
            f'    <message index="{index}" role="{role}">\n'
            f"      {escaped}\n"
            f"    </message>"
        )

    def _first_pending_step_from_plan(
        self,
        task_plan: List[Dict[str, Any]],
        completed_steps: set,
    ) -> str:
        completed = {str(step).strip() for step in (completed_steps or set()) if str(step).strip()}
        for idx, step in enumerate(task_plan or []):
            if isinstance(step, dict):
                step_id = str(step.get("id", "") or f"step_{idx + 1}").strip()
            else:
                step_id = f"step_{idx + 1}"
            if step_id and step_id not in completed:
                return step_id
        return ""

    def _build_context(self, state, include_history: bool = True, emit_stats: bool = True) -> str:
        """
        XML(,)
        
        :
        - ()
        - 
        - ()
        - 
        
        ,
        
        Args:
            state: 
            include_history: 
            emit_stats: 
            
        Returns:
            XML
        """
        # Token
        token_stats = {
            "system_identity": 0,
            "task_goal": 0,
            "task_plan": 0,
            "step_budget": 0,
            "environment": 0,
            "working_directory": 0,
            "progress_summary": 0,
            "read_coverage": 0,
            "evidence_index": 0,
            "file_views": 0,
            "search_hits_index": 0,
            "file_candidates": 0,
            "active_tools": 0,
            "active_skills": 0,
            "patch_summaries": 0,
            "mutation_truth": 0,
            "terminal_events": 0,
            "tui_events": 0,
            "exec_sessions": 0,
            "tui_views": 0,
            "tui_verification": 0,
            "command_verification": 0,
            "convergence_stats": 0,
            "convergence_mode": 0,
            "command_executions": 0,
            "tool_artifacts_index": 0,
            "hydrated_tool_outputs": 0,
            "summary_events": 0,
            "step_checkpoints": 0,
            "archived_history": 0,
            "recent_history": 0,
            "context_summary": 0,
            "document_content": 0,
            "skill_context": 0,
            "minimum_judgement_package": 0,
            "context_slice_budget": 0,
        }
        context_slice_budget = self._build_context_slice_budget(state)
        
        # ()
        existing_context = state.get("context", "")
        current_task = state.get("current_task", "")
        progress = state.get("progress_summary", "")
        
        # XML
        xml_parts = ['<?xml version="1.0" encoding="UTF-8"?>\n<context>']
        
        # 
        overall_goal = state.get("overall_goal", "")
        if overall_goal:
            escaped_goal = self._escape_xml(overall_goal)
            goal_xml = f"""  <overall_goal>
    {escaped_goal}
  </overall_goal>"""
            xml_parts.append(goal_xml)
        
        # 
        if current_task:
            escaped_task = self._escape_xml(current_task)
            task_xml = f"""  <current_task>
    {escaped_task}
  </current_task>"""
            xml_parts.append(task_xml)
            token_stats["task_goal"] = TokenCounter.estimate_tokens(task_xml)

        current_focus = str(state.get("current_focus", "") or current_task or overall_goal).strip()
        if current_focus:
            focus_xml = f"""  <current_focus>
    {self._escape_xml(current_focus)}
  </current_focus>"""
            xml_parts.append(focus_xml)
            token_stats["task_goal"] += TokenCounter.estimate_tokens(focus_xml)

        active_tool_groups = [
            str(item).strip()
            for item in (state.get("active_tool_groups", []) or [])
            if str(item).strip()
        ]
        active_tool_names = [
            str(item).strip()
            for item in (state.get("active_tool_names", []) or [])
            if str(item).strip()
        ]
        if active_tool_names:
            tools_xml = (
                f'  <active_tools groups="{self._escape_xml(",".join(active_tool_groups))}">\n'
                + "\n".join(
                    f'    <tool name="{self._escape_xml(name)}" />'
                    for name in active_tool_names[:60]
                )
                + "\n  </active_tools>"
            )
            xml_parts.append(tools_xml)
            token_stats["active_tools"] = TokenCounter.estimate_tokens(tools_xml)

        active_skill_names = [
            str(item).strip()
            for item in (state.get("active_skill_names", []) or [])
            if str(item).strip()
        ]
        active_skill_paths = [
            str(item).strip()
            for item in (state.get("active_skill_paths", []) or [])
            if str(item).strip()
        ]
        permission_mode = str(state.get("permission_mode", "") or "").strip()
        if permission_mode:
            allow_shell_commands = bool(state.get("allow_shell_commands", True))
            permission_xml = (
                "  <permission>"
                f'<mode name="{self._escape_xml(permission_mode)}" '
                f'shell_allowed="{str(allow_shell_commands).lower()}" />'
                "</permission>"
            )
            xml_parts.append(permission_xml)

        skill_injection_mode = str(state.get("skill_injection_mode", "") or "").strip()
        if skill_injection_mode:
            xml_parts.append(
                "  <skill_injection>"
                f'<mode name="{self._escape_xml(skill_injection_mode)}" />'
                "</skill_injection>"
            )
        skill_registry_digest = str(state.get("skill_registry_digest", "") or "").strip()
        if active_skill_names or skill_registry_digest:
            active_skills_xml = [f'  <active_skills registry_digest="{self._escape_xml(skill_registry_digest)}">']
            for idx, skill_name in enumerate(active_skill_names[:20]):
                skill_path = active_skill_paths[idx] if idx < len(active_skill_paths) else ""
                active_skills_xml.append(
                    f'    <skill name="{self._escape_xml(skill_name)}" path="{self._escape_xml(skill_path)}" />'
                )
            active_skills_xml.append("  </active_skills>")
            rendered = "\n".join(active_skills_xml)
            xml_parts.append(rendered)
            token_stats["active_skills"] = TokenCounter.estimate_tokens(rendered)

        has_new_user_input = bool(state.get("has_new_user_input", False))
        xml_parts.append(
            f"""  <has_new_user_input>{str(has_new_user_input).lower()}</has_new_user_input>"""
        )

        should_plan = bool(state.get("should_plan", False))
        planning_xml = f"""  <planning_mode>{str(should_plan).lower()}</planning_mode>"""
        xml_parts.append(planning_xml)

        task_plan = state.get("task_plan", []) or []
        current_step_id = str(state.get("current_step_id", "") or "").strip()
        completed_steps = {
            str(step_id).strip()
            for step_id in (state.get("completed_steps", []) or [])
            if str(step_id).strip()
        }
        blocked_reason = str(state.get("blocked_reason", "") or "").strip()
        if task_plan or blocked_reason:
            step_lines = []
            for idx, step in enumerate(task_plan):
                if isinstance(step, dict):
                    step_id = str(step.get("id", "") or f"step_{idx+1}").strip()
                    step_title = str(
                        step.get("title", "")
                        or step.get("task", "")
                        or step.get("name", "")
                        or step_id
                    ).strip()
                else:
                    step_id = f"step_{idx+1}"
                    step_title = str(step).strip()
                if not step_title:
                    step_title = step_id

                status = "pending"
                if step_id and step_id in completed_steps:
                    status = "completed"
                elif current_step_id and step_id == current_step_id:
                    status = "active"
                elif not current_step_id and idx == 0 and not completed_steps:
                    status = "active"

                step_lines.append(
                    f'    <step id="{self._escape_xml(step_id)}" status="{status}">{self._escape_xml(step_title)}</step>'
                )

            blocked_xml = ""
            if blocked_reason:
                blocked_xml = f"\n    <blocked_reason>{self._escape_xml(blocked_reason)}</blocked_reason>"
            plan_xml = (
                f'  <task_plan current_step_id="{self._escape_xml(current_step_id)}">\n'
                + "\n".join(step_lines)
                + f"{blocked_xml}\n"
                + "  </task_plan>"
            )
            xml_parts.append(plan_xml)
            token_stats["task_plan"] = TokenCounter.estimate_tokens(plan_xml)

        mjp_xml = self._build_minimum_judgement_package_xml(
            state=state,
            task_plan=task_plan if isinstance(task_plan, list) else [],
            current_step_id=current_step_id,
            completed_steps=completed_steps,
        )
        if mjp_xml:
            self._append_context_section(
                xml_parts=xml_parts,
                token_stats=token_stats,
                section_name="minimum_judgement_package",
                section_xml=mjp_xml,
                budget=None,
                required=True,
            )

        if should_plan:
            step_execution_stats = state.get("step_execution_stats", {}) or {}
            active_step_id = current_step_id or self._first_pending_step_from_plan(task_plan, completed_steps)
            if active_step_id:
                active_stats = {}
                if isinstance(step_execution_stats, dict):
                    active_stats = dict(step_execution_stats.get(active_step_id, {}) or {})
                tools_used = int(active_stats.get("tools", 0) or 0)
                max_tools = max(1, int(state.get("max_tools_per_step", 20) or 20))
                remaining_tools = max(0, max_tools - tools_used)
                max_reads = max(1, int(state.get("max_reads_per_file_per_step", 6) or 6))
                max_retries = max(1, int(state.get("max_retry_per_tool_per_step", 2) or 2))
                budget_xml = (
                    f'  <step_budget step_id="{self._escape_xml(active_step_id)}" '
                    f'max_tools_per_step="{max_tools}" used_tools="{tools_used}" '
                    f'remaining_tools="{remaining_tools}" '
                    f'max_reads_per_file_per_step="{max_reads}" '
                    f'max_retry_per_tool_per_step="{max_retries}" />'
                )
                xml_parts.append(budget_xml)
                token_stats["step_budget"] = TokenCounter.estimate_tokens(budget_xml)
        
        # 
        env_snapshot = state.get("env_snapshot", "")
        if env_snapshot:
            escaped_env = self._escape_xml(env_snapshot)
            env_xml = f"""  <environment>
    {escaped_env}
  </environment>"""
            xml_parts.append(env_xml)
            token_stats["environment"] = TokenCounter.estimate_tokens(env_xml)
        
        # 
        escaped_workdir = self._escape_xml(str(self.working_dir))
        workdir_xml = f"""  <working_directory>
    {escaped_workdir}
  </working_directory>"""
        xml_parts.append(workdir_xml)
        token_stats["working_directory"] = TokenCounter.estimate_tokens(workdir_xml)
        
        # 
        if progress:
            progress_text = str(progress)
            if len(progress_text) > self.PROGRESS_SUMMARY_MAX_CHARS:
                tail = progress_text[-self.PROGRESS_SUMMARY_MAX_CHARS :]
                omitted = len(progress_text) - len(tail)
                progress_text = f"[progress_summary truncated: omitted {omitted} chars]\n{tail}"
            escaped_progress = self._escape_xml(progress_text)
            progress_xml = f"""  <progress_summary>
    {escaped_progress}
  </progress_summary>"""
            xml_parts.append(progress_xml)
            token_stats["progress_summary"] = TokenCounter.estimate_tokens(progress_xml)

        read_coverage = state.get("read_coverage", {}) or {}
        if isinstance(read_coverage, dict) and read_coverage:
            coverage_lines = []
            for path, meta in sorted(read_coverage.items(), key=lambda item: str(item[0]))[:30]:
                if not isinstance(meta, dict):
                    continue
                total_lines = int(meta.get("total_lines", 0) or 0)
                reads = int(meta.get("reads", 0) or 0)
                ranges = []
                for pair in (meta.get("ranges", []) or [])[:20]:
                    if isinstance(pair, (list, tuple)) and len(pair) == 2:
                        try:
                            s = int(pair[0])
                            e = int(pair[1])
                        except Exception:
                            continue
                        if s > 0 and e > 0:
                            ranges.append(f"{s}-{e}")
                ranges_text = ",".join(ranges) if ranges else ""
                coverage_lines.append(
                    f'    <file path="{self._escape_xml(str(path))}" total_lines="{total_lines}" reads="{reads}" ranges="{self._escape_xml(ranges_text)}" />'
                )
            if coverage_lines:
                coverage_xml = "  <read_coverage>\n" + "\n".join(coverage_lines) + "\n  </read_coverage>"
                self._append_context_section(
                    xml_parts=xml_parts,
                    token_stats=token_stats,
                    section_name="read_coverage",
                    section_xml=coverage_xml,
                    budget=context_slice_budget,
                    required=False,
                )

        evidence_index = state.get("evidence_index", []) or []
        if isinstance(evidence_index, list) and evidence_index:
            evidence_lines = []
            for item in evidence_index[-40:]:
                if not isinstance(item, dict):
                    continue
                ev_id = str(item.get("id", "") or "").strip()
                ev_type = str(item.get("type", "") or "").strip()
                path = str(item.get("path", "") or "").strip()
                line = str(item.get("line", "") or "").strip()
                range_text = str(item.get("range", "") or "").strip()
                hit_id = str(item.get("hit_id", "") or "").strip()
                chunk_id = str(item.get("chunk_id", "") or "").strip()
                reason = str(item.get("reason", "") or "").strip()
                score = str(item.get("score", "") or "").strip()
                summary_text = str(item.get("summary", "") or "").strip()
                if len(summary_text) > 180:
                    summary_text = summary_text[:180] + "..."
                evidence_lines.append(
                    f'    <evidence id="{self._escape_xml(ev_id)}" type="{self._escape_xml(ev_type)}" '
                    f'path="{self._escape_xml(path)}" line="{self._escape_xml(line)}" '
                    f'range="{self._escape_xml(range_text)}" hit_id="{self._escape_xml(hit_id)}" '
                    f'chunk_id="{self._escape_xml(chunk_id)}" score="{self._escape_xml(score)}" '
                    f'reason="{self._escape_xml(reason)}">{self._escape_xml(summary_text)}</evidence>'
                )
            if evidence_lines:
                evidence_xml = "  <evidence_index>\n" + "\n".join(evidence_lines) + "\n  </evidence_index>"
                self._append_context_section(
                    xml_parts=xml_parts,
                    token_stats=token_stats,
                    section_name="evidence_index",
                    section_xml=evidence_xml,
                    budget=context_slice_budget,
                    required=False,
                )

        file_views_xml = self._render_file_views_xml(state, current_focus)
        if file_views_xml:
            self._append_context_section(
                xml_parts=xml_parts,
                token_stats=token_stats,
                section_name="file_views",
                section_xml=file_views_xml,
                budget=context_slice_budget,
                required=False,
            )

        search_hits_index = state.get("search_hits_index", []) or []
        if isinstance(search_hits_index, list) and search_hits_index:
            search_hit_lines = []
            backend = str(state.get("last_search_backend", "") or "").strip()
            backend_attr = f' backend="{self._escape_xml(backend)}"' if backend else ""
            for item in search_hits_index[-40:]:
                if not isinstance(item, dict):
                    continue
                hit_id = str(item.get("id", "") or "").strip()
                path = str(item.get("path", "") or "").strip()
                rel_path = str(item.get("rel_path", "") or "").strip()
                line = str(item.get("line", "") or "").strip()
                score = str(item.get("score", "") or "").strip()
                source = str(item.get("source_tool", "") or "").strip()
                command = str(item.get("command", "") or "").strip()
                if len(command) > 140:
                    command = command[:140] + "..."
                summary_text = str(item.get("summary", "") or "").strip()
                if len(summary_text) > 180:
                    summary_text = summary_text[:180] + "..."
                search_hit_lines.append(
                    f'    <search_hit id="{self._escape_xml(hit_id)}" path="{self._escape_xml(path)}" '
                    f'rel_path="{self._escape_xml(rel_path)}" line="{self._escape_xml(line)}" '
                    f'score="{self._escape_xml(score)}" source_tool="{self._escape_xml(source)}" '
                    f'command="{self._escape_xml(command)}">{self._escape_xml(summary_text)}</search_hit>'
                )
            if search_hit_lines:
                search_hits_xml = (
                    f"  <search_hits_index{backend_attr}>\n"
                    + "\n".join(search_hit_lines)
                    + "\n  </search_hits_index>"
                )
                self._append_context_section(
                    xml_parts=xml_parts,
                    token_stats=token_stats,
                    section_name="search_hits_index",
                    section_xml=search_hits_xml,
                    budget=context_slice_budget,
                    required=False,
                )

        file_candidates = state.get("file_candidates", []) or []
        if isinstance(file_candidates, list) and file_candidates:
            candidate_lines = []
            backend = str(state.get("last_search_backend", "") or "").strip()
            backend_attr = f' backend="{self._escape_xml(backend)}"' if backend else ""
            for item in file_candidates[-40:]:
                if not isinstance(item, dict):
                    continue
                candidate_id = str(item.get("id", "") or "").strip()
                path = str(item.get("path", "") or "").strip()
                rel_path = str(item.get("rel_path", "") or "").strip()
                source = str(item.get("source_tool", "") or "").strip()
                command = str(item.get("command", "") or "").strip()
                if len(command) > 140:
                    command = command[:140] + "..."
                candidate_lines.append(
                    f'    <candidate id="{self._escape_xml(candidate_id)}" path="{self._escape_xml(path)}" '
                    f'rel_path="{self._escape_xml(rel_path)}" source_tool="{self._escape_xml(source)}" '
                    f'command="{self._escape_xml(command)}" />'
                )
            if candidate_lines:
                candidates_xml = (
                    f"  <file_candidates{backend_attr}>\n"
                    + "\n".join(candidate_lines)
                    + "\n  </file_candidates>"
                )
                self._append_context_section(
                    xml_parts=xml_parts,
                    token_stats=token_stats,
                    section_name="file_candidates",
                    section_xml=candidates_xml,
                    budget=context_slice_budget,
                    required=False,
                )

        trusted_modified_files = [
            str(item).strip()
            for item in (state.get("trusted_modified_files", []) or [])
            if str(item).strip()
        ]
        last_patch_summaries = state.get("last_patch_summaries", []) or []
        if trusted_modified_files or last_patch_summaries:
            patch_lines = []
            for item in (last_patch_summaries[-12:] if isinstance(last_patch_summaries, list) else []):
                if not isinstance(item, dict):
                    continue
                patch_lines.append(
                    f'    <patch tool="{self._escape_xml(str(item.get("tool", "") or ""))}" '
                    f'files="{self._escape_xml(",".join(str(x).strip() for x in (item.get("files", []) or []) if str(x).strip()))}" '
                    f'summary="{self._escape_xml(str(item.get("summary", "") or ""))}" />'
                )
            trusted_lines = [
                f'    <file path="{self._escape_xml(path)}" />'
                for path in trusted_modified_files[-20:]
            ]
            patch_xml = "  <patch_evidence>\n"
            if trusted_lines:
                patch_xml += "    <trusted_modified_files>\n" + "\n".join(trusted_lines) + "\n    </trusted_modified_files>\n"
            if patch_lines:
                patch_xml += "    <last_patch_summaries>\n" + "\n".join(patch_lines) + "\n    </last_patch_summaries>\n"
            patch_xml += "  </patch_evidence>"
            self._append_context_section(
                xml_parts=xml_parts,
                token_stats=token_stats,
                section_name="patch_summaries",
                section_xml=patch_xml,
                budget=context_slice_budget,
                required=False,
            )

        last_mutation = dict(state.get("last_mutation", {}) or {})
        recent_mutations = state.get("recent_mutations", []) or []
        pending_verification_targets = [
            str(item).strip()
            for item in (state.get("pending_verification_targets", []) or [])
            if str(item).strip()
        ]
        mutation_truth_digest = str(state.get("mutation_truth_digest", "") or "").strip()
        if last_mutation or recent_mutations or pending_verification_targets:
            mutation_lines = [
                f'  <mutation_truth digest="{self._escape_xml(mutation_truth_digest)}">'
            ]
            if last_mutation:
                mutation_lines.append(
                    f'    <last_mutation tool="{self._escape_xml(str(last_mutation.get("tool", "") or ""))}" '
                    f'path="{self._escape_xml(str(last_mutation.get("path", "") or ""))}" '
                    f'status="{self._escape_xml(str(last_mutation.get("status", "") or ""))}" '
                    f'file_changed="{str(bool(last_mutation.get("file_changed", False))).lower()}" '
                    f'error_kind="{self._escape_xml(str(last_mutation.get("error_kind", "") or ""))}">'
                    f'{self._escape_xml(str(last_mutation.get("error_excerpt", "") or "")[:180])}</last_mutation>'
                )
            if pending_verification_targets:
                mutation_lines.append("    <next_verification_targets>")
                mutation_lines.extend(
                    f'      <file path="{self._escape_xml(path)}" />'
                    for path in pending_verification_targets[:3]
                )
                mutation_lines.append("    </next_verification_targets>")
            elif isinstance(recent_mutations, list) and recent_mutations:
                latest_recent = next(
                    (item for item in reversed(recent_mutations) if isinstance(item, dict)),
                    {},
                )
                if latest_recent:
                    mutation_lines.append(
                        f'    <recent tool="{self._escape_xml(str(latest_recent.get("tool", "") or ""))}" '
                        f'path="{self._escape_xml(str(latest_recent.get("path", "") or ""))}" '
                        f'status="{self._escape_xml(str(latest_recent.get("status", "") or ""))}" '
                        f'file_changed="{str(bool(latest_recent.get("file_changed", False))).lower()}" '
                        f'error_kind="{self._escape_xml(str(latest_recent.get("error_kind", "") or ""))}" />'
                    )
            mutation_lines.append(
                "    <rule>Do not claim a file was updated unless mutation_truth or direct read evidence confirms it.</rule>"
            )
            mutation_lines.append(
                "    <rule>Once you have a direct edit target, patch first and only re-read on failure or insufficient evidence.</rule>"
            )
            mutation_lines.append("  </mutation_truth>")
            mutation_xml = "\n".join(mutation_lines)
            self._append_context_section(
                xml_parts=xml_parts,
                token_stats=token_stats,
                section_name="mutation_truth",
                section_xml=mutation_xml,
                budget=context_slice_budget,
                required=False,
            )

        recent_terminal_events = state.get("recent_terminal_events", []) or []
        if isinstance(recent_terminal_events, list) and recent_terminal_events:
            terminal_lines = []
            for item in recent_terminal_events[-12:]:
                if not isinstance(item, dict):
                    continue
                terminal_lines.append(
                    f'    <event type="{self._escape_xml(str(item.get("event_type", "") or ""))}" '
                    f'session="{self._escape_xml(str(item.get("session_name", "") or ""))}" '
                    f'op_id="{self._escape_xml(str(item.get("op_id", "") or ""))}" '
                    f'status="{self._escape_xml(str(item.get("status", "") or ""))}" '
                    f'exit_code="{self._escape_xml(str(item.get("exit_code", "") or ""))}">'
                    f'{self._escape_xml(str(item.get("summary", "") or ""))}</event>'
                )
            if terminal_lines:
                terminal_xml = "  <recent_terminal_events>\n" + "\n".join(terminal_lines) + "\n  </recent_terminal_events>"
                self._append_context_section(
                    xml_parts=xml_parts,
                    token_stats=token_stats,
                    section_name="terminal_events",
                    section_xml=terminal_xml,
                    budget=context_slice_budget,
                    required=False,
                )

        recent_exec_sessions = state.get("recent_exec_sessions", []) or []
        active_exec_session = dict(state.get("active_exec_session", {}) or {})
        last_exec_output = dict(state.get("last_exec_output", {}) or {})
        if isinstance(recent_exec_sessions, list) and recent_exec_sessions:
            exec_lines = []
            for item in recent_exec_sessions[-12:]:
                if not isinstance(item, dict):
                    continue
                exec_lines.append(
                    f'    <session event_type="{self._escape_xml(str(item.get("event_type", "") or ""))}" '
                    f'session_id="{self._escape_xml(str(item.get("session_id", "") or ""))}" '
                    f'status="{self._escape_xml(str(item.get("status", "") or ""))}" '
                    f'transport="{self._escape_xml(str(item.get("transport", "") or ""))}">'
                    f'{self._escape_xml(str(item.get("summary", "") or ""))}</session>'
                )
            if active_exec_session:
                exec_lines.append(
                    f'    <active session_id="{self._escape_xml(str(active_exec_session.get("session_id", "") or ""))}" '
                    f'status="{self._escape_xml(str(active_exec_session.get("status", "") or ""))}" '
                    f'transport="{self._escape_xml(str(active_exec_session.get("transport", "") or ""))}">'
                    f'{self._escape_xml(str(active_exec_session.get("summary", "") or ""))}</active>'
                )
            if last_exec_output:
                exec_lines.append(
                    f'    <last_output session_id="{self._escape_xml(str(last_exec_output.get("session_id", "") or ""))}" '
                    f'status="{self._escape_xml(str(last_exec_output.get("status", "") or ""))}">'
                    f'{self._escape_xml(str(last_exec_output.get("summary", "") or ""))}</last_output>'
                )
            if exec_lines:
                exec_xml = "  <recent_exec_sessions>\n" + "\n".join(exec_lines) + "\n  </recent_exec_sessions>"
                self._append_context_section(
                    xml_parts=xml_parts,
                    token_stats=token_stats,
                    section_name="exec_sessions",
                    section_xml=exec_xml,
                    budget=context_slice_budget,
                    required=False,
                )

        tui_views_xml = self._render_tui_views_xml(state, current_focus)
        if tui_views_xml:
            self._append_context_section(
                xml_parts=xml_parts,
                token_stats=token_stats,
                section_name="tui_views",
                section_xml=tui_views_xml,
                budget=context_slice_budget,
                required=False,
            )

        recent_tui_events = state.get("recent_tui_events", []) or []
        if isinstance(recent_tui_events, list) and recent_tui_events:
            tui_lines = []
            for item in recent_tui_events[-12:]:
                if not isinstance(item, dict):
                    continue
                tui_lines.append(
                    f'    <event type="{self._escape_xml(str(item.get("event_type", "") or ""))}" '
                    f'session="{self._escape_xml(str(item.get("session_name", "") or ""))}" '
                    f'screen_hash="{self._escape_xml(str(item.get("screen_hash", "") or ""))}" '
                    f'changed_rows="{self._escape_xml(str(item.get("changed_rows", "") or ""))}">'
                    f'{self._escape_xml(str(item.get("summary", "") or ""))}</event>'
                )
            if tui_lines:
                tui_xml = "  <recent_tui_events>\n" + "\n".join(tui_lines) + "\n  </recent_tui_events>"
                self._append_context_section(
                    xml_parts=xml_parts,
                    token_stats=token_stats,
                    section_name="tui_events",
                    section_xml=tui_xml,
                    budget=context_slice_budget,
                    required=False,
                )

        tui_verification_targets = state.get("tui_verification_targets", []) or []
        last_tui_verification = dict(state.get("last_tui_verification", {}) or {})
        recent_tui_verifications = state.get("recent_tui_verifications", []) or []
        tui_verification_digest = str(state.get("tui_verification_digest", "") or "").strip()
        if tui_verification_targets or last_tui_verification or recent_tui_verifications:
            verification_lines = [
                f'  <tui_verification digest="{self._escape_xml(tui_verification_digest)}">'
            ]
            primary_tui_target = next(
                (item for item in tui_verification_targets if isinstance(item, dict)),
                {},
            )
            if primary_tui_target:
                verification_lines.append(
                    f'    <target goal="{self._escape_xml(str(primary_tui_target.get("goal", "") or ""))}" '
                    f'pass_condition="{self._escape_xml(str(primary_tui_target.get("pass_condition", "") or ""))}">'
                    f'{self._escape_xml(str(primary_tui_target.get("assertion", "") or ""))}</target>'
                )
            if last_tui_verification:
                verification_lines.append(
                    f'    <last goal="{self._escape_xml(str(last_tui_verification.get("goal", "") or ""))}" '
                    f'status="{self._escape_xml(str(last_tui_verification.get("status", "") or ""))}" '
                    f'tui_progressed="{str(bool(last_tui_verification.get("tui_progressed", False))).lower()}" '
                    f'tool="{self._escape_xml(str(last_tui_verification.get("tool", "") or ""))}">'
                    f'{self._escape_xml(str(last_tui_verification.get("reason", "") or "")[:220])}</last>'
                )
            verification_lines.append(
                "    <rule>Operational TUI progress and semantic bug-fix verification are different signals.</rule>"
            )
            verification_lines.append(
                "    <rule>Use one explicit TUI assertion and collect before/after evidence before drawing a conclusion.</rule>"
            )
            verification_lines.append("  </tui_verification>")
            verification_xml = "\n".join(verification_lines)
            self._append_context_section(
                xml_parts=xml_parts,
                token_stats=token_stats,
                section_name="tui_verification",
                section_xml=verification_xml,
                budget=context_slice_budget,
                required=False,
            )

        command_verification_targets = state.get("command_verification_targets", []) or []
        last_command_verification = dict(state.get("last_command_verification", {}) or {})
        recent_command_verifications = state.get("recent_command_verifications", []) or []
        command_verification_digest = str(state.get("command_verification_digest", "") or "").strip()
        if command_verification_targets or last_command_verification or recent_command_verifications:
            command_lines = [
                f'  <command_verification digest="{self._escape_xml(command_verification_digest)}">'
            ]
            primary_command_target = next(
                (item for item in command_verification_targets if isinstance(item, dict)),
                {},
            )
            if primary_command_target:
                command_lines.append(
                    f'    <target goal="{self._escape_xml(str(primary_command_target.get("goal", "") or ""))}" '
                    f'pass_condition="{self._escape_xml(str(primary_command_target.get("pass_condition", "") or ""))}">'
                    f'{self._escape_xml(str(primary_command_target.get("assertion", "") or ""))}</target>'
                )
            if last_command_verification:
                command_lines.append(
                    f'    <last goal="{self._escape_xml(str(last_command_verification.get("goal", "") or ""))}" '
                    f'status="{self._escape_xml(str(last_command_verification.get("status", "") or ""))}" '
                    f'tool="{self._escape_xml(str(last_command_verification.get("tool", "") or ""))}">'
                    f'{self._escape_xml(str(last_command_verification.get("reason", "") or "")[:220])}</last>'
                )
                for tag_name, field_name in (
                    ("fact", "facts"),
                    ("supported", "supported_hypotheses"),
                    ("disproven", "disproven_hypotheses"),
                ):
                    limit = 2 if tag_name == "fact" else 1
                    for item in self._compact_text_items(
                        last_command_verification.get(field_name, []),
                        limit=limit,
                        max_chars=180,
                    ):
                        text = str(item or "").strip()
                        if text:
                            command_lines.append(
                                f"    <{tag_name}>{self._escape_xml(text[:180])}</{tag_name}>"
                            )
            command_lines.append(
                "    <rule>Operational command success and semantic bug verification are different signals.</rule>"
            )
            command_lines.append(
                "    <rule>Do not reopen disproven hypotheses unless new command evidence contradicts them.</rule>"
            )
            command_lines.append("  </command_verification>")
            command_xml = "\n".join(command_lines)
            self._append_context_section(
                xml_parts=xml_parts,
                token_stats=token_stats,
                section_name="command_verification",
                section_xml=command_xml,
                budget=context_slice_budget,
                required=False,
            )

        convergence_xml = (
            "  <convergence_stats "
            f'dedupe_hits="{int(state.get("tool_dedupe_hits", 0) or 0)}" '
            f'repeated_read_hits="{int(state.get("repeated_read_hits", 0) or 0)}" '
            f'stall_rounds="{int(state.get("stall_rounds", 0) or 0)}" '
            f'tui_stall_rounds="{int(state.get("tui_stall_rounds", 0) or 0)}" '
            f'last_tui_screen_hash="{self._escape_xml(str(state.get("last_tui_screen_hash", "") or ""))}" '
            f'evidence_digest="{self._escape_xml(str(state.get("evidence_digest", "") or ""))}"'
            " />"
        )
        xml_parts.append(convergence_xml)
        token_stats["convergence_stats"] = TokenCounter.estimate_tokens(convergence_xml)

        convergence_mode = str(state.get("convergence_mode", "") or "normal").strip() or "normal"
        convergence_reason = str(state.get("convergence_reason", "") or "").strip()
        next_step_requirements = [
            str(item).strip()
            for item in (state.get("next_step_requirements", []) or [])
            if str(item).strip()
        ]
        if convergence_mode != "normal" or convergence_reason or next_step_requirements:
            mode_lines = [
                f'  <convergence_mode name="{self._escape_xml(convergence_mode)}">',
            ]
            if convergence_reason:
                mode_lines.append(f"    <reason>{self._escape_xml(convergence_reason)}</reason>")
            if next_step_requirements:
                mode_lines.append("    <next_step_requirements>")
                mode_lines.extend(
                    f"      <requirement>{self._escape_xml(item)}</requirement>"
                    for item in next_step_requirements[:2]
                )
                mode_lines.append("    </next_step_requirements>")
            mode_lines.append("  </convergence_mode>")
            convergence_mode_xml = "\n".join(mode_lines)
            self._append_context_section(
                xml_parts=xml_parts,
                token_stats=token_stats,
                section_name="convergence_mode",
                section_xml=convergence_mode_xml,
                budget=context_slice_budget,
                required=False,
            )

        command_executions = state.get("command_executions", []) or []
        if command_executions:
            command_lines = []
            for item in command_executions[-20:]:
                if not isinstance(item, dict):
                    continue
                op_id = str(item.get("op_id", "") or "").strip()
                tool = str(item.get("tool", "") or "").strip()
                status = str(item.get("status", "") or "").strip()
                mode = str(item.get("mode", "") or "").strip()
                exit_code = str(item.get("exit_code", "") or "").strip()
                terminal_name = str(item.get("terminal_name", "") or "").strip()
                command_preview = str(item.get("command", "") or "").strip()
                if len(command_preview) > 100:
                    command_preview = command_preview[:100] + "..."
                poll_spec = item.get("poll_spec", {}) or {}
                poll_tool = str(poll_spec.get("tool", "") or "").strip()
                poll_name = str(poll_spec.get("name", "") or "").strip()
                poll_lines = str(poll_spec.get("lines", "") or "").strip()
                command_lines.append(
                    f'    <command op_id="{self._escape_xml(op_id)}" tool="{self._escape_xml(tool)}" '
                    f'status="{self._escape_xml(status)}" mode="{self._escape_xml(mode)}" '
                    f'exit_code="{self._escape_xml(exit_code)}" terminal_name="{self._escape_xml(terminal_name)}" '
                    f'command="{self._escape_xml(command_preview)}" '
                    f'poll_tool="{self._escape_xml(poll_tool)}" poll_name="{self._escape_xml(poll_name)}" '
                    f'poll_lines="{self._escape_xml(poll_lines)}" />'
                )
            if command_lines:
                command_xml = "  <command_executions>\n" + "\n".join(command_lines) + "\n  </command_executions>"
                self._append_context_section(
                    xml_parts=xml_parts,
                    token_stats=token_stats,
                    section_name="command_executions",
                    section_xml=command_xml,
                    budget=context_slice_budget,
                    required=False,
                )

        tool_artifacts = state.get("tool_artifacts", []) or []
        if tool_artifacts:
            artifact_lines = []
            for item in tool_artifacts[-20:]:
                if not isinstance(item, dict):
                    continue
                artifact_id = str(item.get("id", "") or "").strip()
                if not artifact_id:
                    continue
                tool_name = str(item.get("tool", "") or "unknown").strip()
                size = int(item.get("size", 0) or 0)
                store_key = str(item.get("content_store_key", "") or "").strip()
                key_attr = (
                    f' content_store_key="{self._escape_xml(store_key)}"'
                    if store_key
                    else ""
                )
                artifact_lines.append(
                    f'    <artifact id="{self._escape_xml(artifact_id)}" tool="{self._escape_xml(tool_name)}" size="{size}"{key_attr} />'
                )
            if artifact_lines:
                artifact_xml = "  <tool_artifacts_index>\n" + "\n".join(artifact_lines) + "\n  </tool_artifacts_index>"
                self._append_context_section(
                    xml_parts=xml_parts,
                    token_stats=token_stats,
                    section_name="tool_artifacts_index",
                    section_xml=artifact_xml,
                    budget=context_slice_budget,
                    required=False,
                )

        hydrated_outputs_xml = self._build_hydrated_tool_outputs_xml(
            state=state,
            current_focus=current_focus,
            current_task=str(current_task or ""),
            overall_goal=str(overall_goal or ""),
        )
        if hydrated_outputs_xml:
            self._append_context_section(
                xml_parts=xml_parts,
                token_stats=token_stats,
                section_name="hydrated_tool_outputs",
                section_xml=hydrated_outputs_xml,
                budget=context_slice_budget,
                required=False,
            )

        summary_events = state.get("summary_events", []) or []
        if summary_events:
            event_lines = []
            for item in summary_events[-10:]:
                if not isinstance(item, dict):
                    continue
                event_id = str(item.get("id", "") or "").strip()
                event_type = str(item.get("type", "") or "summary").strip()
                event_time = str(item.get("timestamp", "") or "").strip()
                summary_text = str(item.get("summary", "") or "").strip()
                if len(summary_text) > 180:
                    summary_text = summary_text[:180] + "..."
                event_lines.append(
                    f'    <summary_event id="{self._escape_xml(event_id)}" type="{self._escape_xml(event_type)}" timestamp="{self._escape_xml(event_time)}">{self._escape_xml(summary_text)}</summary_event>'
                )
            if event_lines:
                summary_events_xml = "  <summary_events>\n" + "\n".join(event_lines) + "\n  </summary_events>"
                self._append_context_section(
                    xml_parts=xml_parts,
                    token_stats=token_stats,
                    section_name="summary_events",
                    section_xml=summary_events_xml,
                    budget=context_slice_budget,
                    required=False,
                )

            checkpoint_lines = []
            for item in summary_events:
                if not isinstance(item, dict):
                    continue
                if str(item.get("type", "") or "").strip() != "step_checkpoint":
                    continue
                step_id = str(item.get("step_id", "") or "").strip()
                done_text = str(item.get("done", "") or "").strip()
                risks = item.get("open_risks", []) or []
                if isinstance(risks, list):
                    risks_text = "; ".join(str(x).strip() for x in risks if str(x).strip())
                else:
                    risks_text = str(risks).strip()
                artifacts = item.get("artifacts", []) or []
                if isinstance(artifacts, list):
                    artifacts_text = ",".join(str(x).strip() for x in artifacts if str(x).strip())
                else:
                    artifacts_text = str(artifacts).strip()
                if not step_id:
                    continue
                checkpoint_lines.append(
                    f'    <checkpoint step_id="{self._escape_xml(step_id)}" artifacts="{self._escape_xml(artifacts_text)}" open_risks="{self._escape_xml(risks_text)}">{self._escape_xml(done_text)}</checkpoint>'
                )
            if checkpoint_lines:
                checkpoint_xml = "  <step_checkpoints>\n" + "\n".join(checkpoint_lines[-8:]) + "\n  </step_checkpoints>"
                self._append_context_section(
                    xml_parts=xml_parts,
                    token_stats=token_stats,
                    section_name="step_checkpoints",
                    section_xml=checkpoint_xml,
                    budget=context_slice_budget,
                    required=False,
                )
        
        # (,;)
        context_summary = state.get("context_summary", "")
        previous_history_has_file_views = bool(getattr(self, "_history_has_file_views", False))
        self._history_has_file_views = bool(int(token_stats.get("file_views", 0) or 0) > 0)
        try:
            if include_history:
                history_msgs = state.get("messages", []) or []
                recent_limit = self._get_recent_history_limit(state)

                if context_summary:
                    #  recent_history ,
                    summary_xml = f"""  <context_summary>
    {self._escape_xml(context_summary)}
  </context_summary>"""
                    xml_parts.append(summary_xml)
                    token_stats["context_summary"] = TokenCounter.estimate_tokens(summary_xml)
                    if emit_stats:
                        self._emit("info", "  []  context_summary + recent_history")

                if history_msgs:
                    recent_msgs = history_msgs[-recent_limit:] if recent_limit > 0 else history_msgs
                    archived_msgs = history_msgs[:-len(recent_msgs)] if recent_msgs else history_msgs

                    # :" + "
                    if archived_msgs and not context_summary:
                        archived_xml = self._format_conversation_history_xml(
                            archived_msgs,
                            compression="heavy",
                        )
                        if archived_xml:
                            archived_block = f"""  <archived_history compression="heavy">
{archived_xml}
  </archived_history>"""
                            self._append_context_section(
                                xml_parts=xml_parts,
                                token_stats=token_stats,
                                section_name="archived_history",
                                section_xml=archived_block,
                                budget=context_slice_budget,
                                required=False,
                            )

                    if recent_msgs:
                        recent_xml = self._format_conversation_history_xml(
                            recent_msgs,
                            compression="light",
                            start_index=max(len(history_msgs) - len(recent_msgs), 0),
                        )
                        if recent_xml:
                            full_history_xml = f"""  <recent_history compression="light">
{recent_xml}
  </recent_history>"""
                            self._append_context_section(
                                xml_parts=xml_parts,
                                token_stats=token_stats,
                                section_name="recent_history",
                                section_xml=full_history_xml,
                                budget=context_slice_budget,
                                required=False,
                            )
        finally:
            self._history_has_file_views = previous_history_has_file_views

        # ()
        if existing_context:
            escaped_context = self._escape_xml(existing_context)
            doc_xml = f"""  <document_content>
    {escaped_context}
  </document_content>"""
            self._append_context_section(
                xml_parts=xml_parts,
                token_stats=token_stats,
                section_name="document_content",
                section_xml=doc_xml,
                budget=context_slice_budget,
                required=False,
            )

        #  skills  SKILL.md 
        skill_preload_warnings = [
            str(item).strip()
            for item in (state.get("skill_preload_warnings", []) or [])
            if str(item).strip()
        ]
        if skill_preload_warnings:
            warning_xml = (
                "  <skill_preload_warnings>\n"
                + "\n".join(
                    f"    <warning>{self._escape_xml(item)}</warning>"
                    for item in skill_preload_warnings[:8]
                )
                + "\n  </skill_preload_warnings>"
            )
            xml_parts.append(warning_xml)

        skill_context = state.get("skill_context", "")
        if skill_context:
            escaped_skill_context = self._escape_xml(skill_context)
            skill_xml = f"""  <skill_context>
    {escaped_skill_context}
  </skill_context>"""
            self._append_context_section(
                xml_parts=xml_parts,
                token_stats=token_stats,
                section_name="skill_context",
                section_xml=skill_xml,
                budget=context_slice_budget,
                required=False,
            )

        skipped_slices = context_slice_budget.get("skipped", []) if isinstance(context_slice_budget, dict) else []
        if skipped_slices:
            skipped_lines = []
            for item in skipped_slices[:20]:
                if not isinstance(item, dict):
                    continue
                section_name = str(item.get("section", "") or "").strip()
                reason = str(item.get("reason", "") or "").strip()
                tokens = int(item.get("tokens", 0) or 0)
                if not section_name:
                    continue
                skipped_lines.append(
                    f'    <skipped section="{self._escape_xml(section_name)}" reason="{self._escape_xml(reason)}" tokens="{tokens}" />'
                )
            if skipped_lines:
                budget_xml = (
                    "  <context_slice_budget "
                    f'max_segments="{int(context_slice_budget.get("max_segments", 0) or 0)}" '
                    f'max_segment_tokens="{int(context_slice_budget.get("max_segment_tokens", 0) or 0)}" '
                    f'max_total_optional_tokens="{int(context_slice_budget.get("max_total_tokens", 0) or 0)}" '
                    f'used_segments="{int(context_slice_budget.get("used_segments", 0) or 0)}" '
                    f'used_optional_tokens="{int(context_slice_budget.get("used_tokens", 0) or 0)}">'
                    "\n"
                    + "\n".join(skipped_lines)
                    + "\n  </context_slice_budget>"
                )
                self._append_context_section(
                    xml_parts=xml_parts,
                    token_stats=token_stats,
                    section_name="context_slice_budget",
                    section_xml=budget_xml,
                    budget=None,
                    required=True,
                )
        
        xml_parts.append("</context>")
        context_xml = "\n".join(xml_parts)
        
        # token
        total_tokens = sum(token_stats.values())
        usage_pct = (total_tokens / self.max_context_tokens) * 100
        
        if emit_stats:
            # (, TUI )
            non_zero_stats = {k: v for k, v in token_stats.items() if v > 0}
            self._emit(
                "context_stats",
                f"[] Token: {total_tokens:,} / {self.max_context_tokens:,} ({usage_pct:.1f}%)",
                context_total_tokens=total_tokens,
                context_max_tokens=self.max_context_tokens,
                context_usage_pct=usage_pct,
                context_breakdown=non_zero_stats,
                usage_source="estimate",
            )

            # ()
            if non_zero_stats:
                stats_str = " | ".join([f"{k}: {v:,}" for k, v in non_zero_stats.items()])
                self._emit(
                    "context_stats",
                    f"  : {stats_str}",
                    context_breakdown=non_zero_stats,
                )
            
            # :
            if usage_pct > 80:
                self._emit("warning", "鈿狅笍 : 80%,")
        
        return context_xml
    
    def _escape_xml(self, text: str) -> str:
        """XML"""
        if not text:
            return ""
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        text = text.replace('"', "&quot;")
        text = text.replace("'", "&apos;")
        return text

    def _resolve_tool_output_delivery_mode(self) -> str:
        raw_mode = str(getattr(self, "tool_output_delivery_mode", "full_inline") or "full_inline").strip().lower()
        if raw_mode in {"full_inline", "hybrid", "artifact_first"}:
            return raw_mode
        return "full_inline"

    def _resolve_tool_output_hard_ceiling(self) -> int:
        try:
            raw = int(getattr(self, "tool_output_hard_ceiling_chars", 200000) or 200000)
        except Exception:
            raw = 200000
        return max(4000, raw)

    @staticmethod
    def _extract_tool_result_name(text: str) -> str:
        match = re.search(r'<tool_result\b[^>]*\btool="([^"]+)"', str(text or ""), re.IGNORECASE)
        if not match:
            return ""
        return str(match.group(1) or "").strip()

    @staticmethod
    def _extract_tool_result_attr(text: str, attr_name: str) -> str:
        attr = str(attr_name or "").strip()
        if not attr:
            return ""
        pattern = rf'<tool_result\b[^>]*\b{re.escape(attr)}="([^"]+)"'
        match = re.search(pattern, str(text or ""), re.IGNORECASE)
        if not match:
            return ""
        return str(match.group(1) or "").strip()

    def _extract_tool_result_call_id(self, text: str) -> str:
        return self._extract_tool_result_attr(text, "call_id")

    def _ensure_tool_result_call_id(self, text: str, call_id: str) -> str:
        payload = str(text or "")
        call_id_text = str(call_id or "").strip()
        if not payload or not call_id_text:
            return payload
        if self._extract_tool_result_call_id(payload):
            return payload
        escaped_id = self._escape_xml(call_id_text)
        return re.sub(
            r"(<tool_result\b)([^>]*?)>",
            rf'\1\2 call_id="{escaped_id}">',
            payload,
            count=1,
            flags=re.IGNORECASE,
        )

    def _should_preserve_tool_result_in_history(self, *, tool_name: str, content: str, compression: str) -> bool:
        _ = (tool_name, content, compression)
        # Keep full tool results in model-visible history without text truncation.
        return True
    
    def _format_conversation_history_xml(
        self,
        messages: List[BaseMessage],
        compression: str = "heavy",
        start_index: int = 0,
    ) -> str:
        """
        XML
        
        :
        1. user/tool_result .
        2. assistant  native/xml , compression .
        
        Args:
            messages: 
            compression: "light"  "heavy"
            start_index: ( recent_history)
            
        Returns:
            XML
        """
        if not messages:
            return ""

        normalized_messages = self._canonicalize_tool_history_messages(messages)
        if not normalized_messages:
            return ""

        entries = []
        for idx, msg in enumerate(normalized_messages):
            i = start_index + idx
            if isinstance(msg, HumanMessage):
                content = msg.content if hasattr(msg, 'content') else str(msg)
                stripped = content.strip()
                if not stripped:
                    continue
                lowered = stripped.lower()
                if "<tool_result" in lowered:
                    role = "tool_result"
                    tool_name = self._extract_tool_result_name(stripped)
                    if self._should_preserve_tool_result_in_history(
                        tool_name=tool_name,
                        content=stripped,
                        compression=compression,
                    ):
                        pass
                else:
                    role = "user"
                    if self._is_low_information_text(stripped):
                        continue
                    stripped = self._truncate_text(stripped, 320 if compression == "light" else 200)
                entries.append(self._format_history_message(i, role, stripped, compression))
            elif isinstance(msg, AIMessage):
                compressed = self._compress_ai_message(msg, compression=compression)
                if compressed:
                    entries.append(self._format_history_message(i, "assistant", compressed.strip(), compression))
            else:
                content = msg.content if hasattr(msg, 'content') else str(msg)
                stripped = content.strip()
                if not stripped:
                    continue
                stripped = self._truncate_text(stripped, 200 if compression == "heavy" else 320)
                entries.append(self._format_history_message(i, "system", stripped, compression))
        
        return "\n".join(entries)
    
    def _format_conversation_history(self, messages: List[BaseMessage]) -> str:
        """
        (,)
        
        Args:
            messages: 
            
        Returns:
            
        """
        history_parts = []
        tool_names = self._runtime_tool_names()
        normalized_messages = self._canonicalize_tool_history_messages(messages)

        for msg in normalized_messages:
            if isinstance(msg, HumanMessage):
                content = msg.content if hasattr(msg, 'content') else str(msg)
                role = "Tool Result" if "<tool_result" in str(content).lower() else "User"
                if role == "Tool Result":
                    history_parts.append(f"[{role}]: {str(content)}")
                else:
                    history_parts.append(f"[{role}]: {self._truncate_text(str(content), 600)}")
            elif isinstance(msg, AIMessage):
                compressed = self._compress_ai_message(msg, compression="light")
                if compressed:
                    history_parts.append(f"[Assistant]: {compressed}")
                else:
                    content = msg.content if hasattr(msg, 'content') else str(msg)
                    plain_content = self._strip_meta_blocks(str(content), tool_names)
                    if self._is_reasoning_echo_message(msg, plain_content, tool_names):
                        continue
                    if plain_content and not self._is_low_information_text(str(plain_content)):
                        history_parts.append(f"[Assistant]: {self._truncate_text(str(plain_content), 320)}")
        
        return "\n".join(history_parts)
    
    def _quick_summarize_messages(self, messages: list, max_entries: int = 5) -> str:
        """
        (LLM)
        
        AI,.
        promptAgent.
        
        Args:
            messages: 
            max_entries: 
            
        Returns:
            
        """
        if not messages:
            return ""
        
        summary_parts = []
        # 
        recent = messages[-max_entries * 2:] if len(messages) > max_entries * 2 else messages
        
        for msg in recent:
            if isinstance(msg, HumanMessage):
                content = getattr(msg, 'content', str(msg))
                snippet = self._truncate_text(str(content), 180)
                if not snippet:
                    continue
                if "<tool_result" in snippet.lower():
                    summary_parts.append(f": {snippet}")
                elif not self._is_low_information_text(snippet):
                    summary_parts.append(f": {snippet}")
            elif isinstance(msg, AIMessage):
                compressed = self._compress_ai_message(msg, compression="light")
                if compressed:
                    summary_parts.append(f": {self._truncate_text(compressed, 220)}")
        
        if not summary_parts:
            return ""
        
        return "[]\n" + "\n".join(summary_parts)

    def get_skill_names(self) -> list:
        """ Agent ()."""
        registry = getattr(self, "skill_registry", {}) or {}
        return sorted(registry.keys())



