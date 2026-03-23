"""
Klynx Agent - LangGraph StateGraph
 OODA 
"""

import hashlib
import json
import os
import platform
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from klynx.agent.tools.browser import BrowserManager
from klynx.agent.tools.interactive_exec import InteractiveExecManager
from klynx.agent.mcp_manager import MCPManager
from klynx.agent.tools.terminal import TerminalManager
from klynx.agent.tools import ToolRegistry, get_json_schemas
from klynx.agent.tools.dispatch import ToolDispatchMixin
from klynx.agent.tools.tui import TUIManager
from klynx.agent.tools.web_search import WebSearchTool, is_tavily_configured

from .backend import AgentBackend, LocalAgentBackend, resolve_runtime_paths
from .hooks import AgentHook, AgentHookContext, HookManager, RuntimeTruthHook
from .nodes import NodesMixin
from .prompt_builder import PromptBuilderMixin
from .state import AgentState
from .store import AgentStore, InMemoryAgentStore
from .toolbox import ToolboxRuntime


class KlynxAgent(PromptBuilderMixin, NodesMixin, ToolDispatchMixin, ToolboxRuntime):
    """Klynx Agent main class."""

    DEFAULT_SKILLS_ROOT = os.path.join(
        os.path.expanduser(os.environ.get("KLYNX_HOME", "~/.klynx")),
        "skills",
    )
    DEFAULT_BASIC_SKILLS = ("skill-creator", "skill-installer")

    SYSTEM_TOOLS = {
        "state_update": (
            "Update goal/task/plan/todos state in one tool call. "
            "Call only when at least one task-state field actually changes; "
            "do not repeat identical state_update payloads."
        ),
        "run_subtask": "Execute a bounded action list: run_subtask(title, actions=[{tool, params}, ...]).",
        "parallel_tool_call": "Bundle independent tool calls: parallel_tool_call(calls=[{tool, params}, ...]).",
    }

    CORE_TOOLS = {
        "read_file": ",..",
        "apply_patch": " patch .,,//.",
        "execute_command": "Run short-lived shell commands for build/test/git/orchestration.",
        "list_directory": ",.",
        "search_in_files": ", structured search with hit_id for read_file follow-up.",
    }

    TERMINAL_TOOLS = {
        "create_terminal": "..",
        "run_in_terminal": ".",
        "read_terminal": ".",
        "wait_terminal_until": " pattern.",
        "read_terminal_since_last": ".",
        "run_and_wait": " pattern.",
        "exec_command": ". PTY/pipe , session_id.",
        "write_stdin": " stdin, chars='' .",
        "close_exec_session": ".",
        "check_syntax": ".",
        "launch_interactive_session": "(): exec_command(tty=true), TUI.",
    }

    TUI_TOOLS = {
        "open_tui": " TUI .",
        "read_tui": " TUI .",
        "read_tui_diff": " TUI .",
        "read_tui_region": " TUI .",
        "find_text_in_tui": " TUI .",
        "send_keys": " TUI .",
        "send_keys_and_read": ".",
        "wait_tui_until": " TUI .",
        "close_tui": " TUI .",
        "activate_tui_mode": " TUI : TUI .",
    }

    NETWORK_AND_EXTRA_TOOLS = {
        "web_search": ".,,.",
        "browser_open": " URL.",
        "browser_view": ", selector .",
        "browser_act": ",action  click/type/press/hover.",
        "browser_scroll": ".",
        "browser_screenshot": ".",
        "browser_console_logs": ".",
    }

    SKILL_TOOLS = {
        "load_skill": " SKILL.md . skill .",
    }

    TOOL_GROUPS = {
        "system": SYSTEM_TOOLS,
        "core": CORE_TOOLS,
        "terminal": TERMINAL_TOOLS,
        "tui": TUI_TOOLS,
        "network_and_extra": NETWORK_AND_EXTRA_TOOLS,
        "skills": SKILL_TOOLS,
    }

    CORE_TOOLS.update(
        {
            "execute_command": "Short-lived command execution. Use exec_command for REPL/long-running interactive tasks.",
        }
    )
    TERMINAL_TOOLS.update(
        {
            "read_terminal": " legacy named terminal .name  create_terminal, exec_xxx session_id.",
            "wait_terminal_until": " legacy named terminal  pattern.name  create_terminal.",
            "read_terminal_since_last": " legacy named terminal .name  create_terminal.",
            "run_and_wait": " legacy named terminal  pattern.name  create_terminal.",
            "exec_command": ". REPL,shell,,--mode tui/textual/curses; exec_xxx session_id, write_stdin / close_exec_session.",
            "write_stdin": " stdin, chars='' .session_id  exec_command / launch_interactive_session.",
            "close_exec_session": ".session_id  exec_command / launch_interactive_session.",
            "launch_interactive_session": "(), exec_command(tty=true); TUI.",
        }
    )
    TUI_TOOLS.update(
        {
            "open_tui": " TUI , TUI guide,.",
            "read_tui": " TUI ;.",
            "read_tui_diff": " TUI , before/after .",
            "read_tui_region": " TUI ,.",
            "find_text_in_tui": " TUI ,.",
            "send_keys": " TUI , changed_rows  after excerpt.",
            "send_keys_and_read": ", send_keys + read_tui.",
            "wait_tui_until": " TUI , hash_change .",
            "activate_tui_mode": " TUI ;open_tui .",
        }
    )

    TOOL_GROUP_LABELS = {
        "system": "",
        "core": "",
        "terminal": "",
        "tui": "",
        "network_and_extra": "",
        "skills": "",
    }

    BASE_TOOLS: Dict[str, str] = {}
    for _tool_group in TOOL_GROUPS.values():
        BASE_TOOLS.update(_tool_group)

    SKILL_MARKER_RE = re.compile(r"\$([A-Za-z0-9._-]+)")
    SKILL_PATH_RE = re.compile(r"([A-Za-z]:[^\s\"'<>]*SKILL\.md|[~/\.][^\s\"'<>]*SKILL\.md)", re.IGNORECASE)
    SKILL_SUMMARY_STEPS_MAX = 8
    SKILL_SUMMARY_ITEMS_MAX = 6
    SKILL_CONTEXT_INLINE_CHAR_LIMIT = 6000


    def __init__(
        self,
        working_dir: str = ".",
        model=None,
        max_iterations: Optional[int] = None,
        memory_dir: str = "",
        load_project_docs: bool = True,
        os_name: str = "windows",
        browser_headless: bool = False,
        tool_call_mode: str = "native",
        tool_protocol_mode: Optional[str] = None,
        skills: Optional[List[Any]] = None,
        skills_root: str = "",
        checkpointer: Optional[Any] = None,
        permission_mode: str = "workspace",
        tool_virtual_root: str = "",
        allow_shell_commands: bool = True,
        skill_injection_mode: str = "hybrid",
        max_tools_per_step: int = 20,
        max_reads_per_file_per_step: int = 6,
        max_retry_per_tool_per_step: int = 2,
        tui_stall_threshold: int = 3,
        full_tui_echo: bool = False,
        tool_output_delivery_mode: str = "full_inline",
        tool_output_hard_ceiling_chars: int = 200000,
        backend: Optional[AgentBackend] = None,
        store: Optional[AgentStore] = None,
        hooks: Optional[List[AgentHook]] = None,
    ):
        """
         Klynx Agent
        
        Args:
            working_dir: 
            model: LangChain ( max_context_tokens )
            max_iterations: .None/<=0 ().
            memory_dir: Agent (.klynx/.rules/.memory ),
            load_project_docs:  KLYNX.md , True
            os_name: , Agent  (: windows, linux, macos)
            browser_headless: , False ()
            tool_call_mode: protocol selector (native only in current runtime).
            tool_protocol_mode: protocol selector (native only in current runtime).
            skills: Agent .:
                    - ["skill-name", "path/to/skill"]
                    - [("name", "path", "description")]
                    - [{"name": "...", "path": "...", "description": "..."}]
            skills_root: ,
            checkpointer:  LangGraph checkpointer. MemorySaver.
            permission_mode:  "workspace" (sandbox)  "global".
            tool_virtual_root: ( working_dir).
            allow_shell_commands:  execute_command .
            skill_injection_mode: skill : "preload" / "tool" / "hybrid".
            max_tools_per_step: .
            max_reads_per_file_per_step: .
            max_retry_per_tool_per_step: .
            tui_stall_threshold: TUI (,).
            full_tui_echo:  TUI ( False,+artifact).
            tool_output_delivery_mode: tool ; full_inline / hybrid / artifact_first.
            tool_output_hard_ceiling_chars: tool  fallback  char .
        """
        self.backend: AgentBackend = backend or LocalAgentBackend()
        resolved_paths = resolve_runtime_paths(
            self.backend,
            working_dir=working_dir,
            memory_dir=memory_dir,
            skills_root=skills_root,
            tool_virtual_root=tool_virtual_root,
        )
        self.working_dir = resolved_paths["working_dir"]
        self.model = model
        # None or <=0 means unlimited iterations.
        if max_iterations is None:
            self.max_iterations = None
        else:
            try:
                parsed_max_iterations = int(max_iterations)
                self.max_iterations = parsed_max_iterations if parsed_max_iterations > 0 else None
            except Exception:
                self.max_iterations = None
        self.memory_dir = resolved_paths["memory_dir"]
        self.load_project_docs = load_project_docs
        self.os_name = os_name
        # Plan19 hard cutover: runtime protocol is native-only.
        self.tool_protocol_mode = "native"
        self.tool_call_mode = "native"
        self.force_native_tool_protocol = True
        self.store: AgentStore = store or InMemoryAgentStore()
        runtime_hooks = list(hooks or [])
        runtime_hooks.append(RuntimeTruthHook())
        self.hook_manager = HookManager(runtime_hooks)
        self.max_tools_per_step = max(1, int(max_tools_per_step or 1))
        self.max_reads_per_file_per_step = max(1, int(max_reads_per_file_per_step or 1))
        self.max_retry_per_tool_per_step = max(1, int(max_retry_per_tool_per_step or 1))
        self.tui_stall_threshold = max(0, int(tui_stall_threshold or 0))
        self.full_tui_echo = bool(full_tui_echo)
        self.tool_output_delivery_mode = self._normalize_tool_output_delivery_mode(
            tool_output_delivery_mode
        )
        try:
            parsed_hard_ceiling = int(tool_output_hard_ceiling_chars)
        except Exception:
            parsed_hard_ceiling = 200000
        self.tool_output_hard_ceiling_chars = max(4000, parsed_hard_ceiling)
        
        if self.model and hasattr(self.model, "model"):
            pass
        
        # 
        self.interactive_exec_manager = InteractiveExecManager(self.working_dir)
        self.terminal_manager = TerminalManager(
            self.working_dir,
            interactive_exec_manager=self.interactive_exec_manager,
        )
        # TUI 
        self.tui_manager = TUIManager(
            self.working_dir,
            interactive_exec_manager=self.interactive_exec_manager,
        )
        # 
        self.web_search_tool = WebSearchTool()
        # 
        self.browser_manager = BrowserManager(headless=browser_headless)
        # , 128k
        self.max_context_tokens = getattr(model, 'max_context_tokens', 128000)
        
        # MCP (Model Context Protocol) 
        self.mcp_manager = MCPManager()
        
        # 
        self.streaming_callback = None
        
        # :( -> )( -> callable)
        self.tools: Dict[str, str] = {}
        self.external_tool_funcs: Dict[str, callable] = {}
        self.external_tool_registry: Dict[str, Dict[str, Any]] = {}
        self.external_tool_groups: Dict[str, List[str]] = {}
        self.active_tool_groups: List[str] = []
        self._default_active_tool_groups: List[str] = ["system", "core"]
        self._auto_sync_skill_tool: bool = True

        # 
        self._tool_prompts_cache: str = ""
        
        # Native Tool Calling( Function Calling) JSON Schema 
        self._json_schemas: list = []

        # :
        resolved_skills_root = resolved_paths["skills_root"]
        if resolved_skills_root:
            self.skills_root = resolved_skills_root
        else:
            self.skills_root = os.path.abspath(os.path.expanduser(self.DEFAULT_SKILLS_ROOT))
        self.builtin_skills_root = str(
            (Path(__file__).resolve().parent.parent / "skills" / "system").resolve()
        )
        self.basic_skills_enabled = True
        self.skill_registry: Dict[str, Dict[str, str]] = {}
        self.skill_registry_cache_by_cwd: Dict[str, Dict[str, Dict[str, str]]] = {}
        self.skill_registry_digest: str = ""
        self._skills_prompt_cache: str = ""
        self._skill_body_cache: Dict[str, str] = {}
        self.skill_injection_mode = self._normalize_skill_injection_mode(skill_injection_mode)
        
        # Pending rollback selection keyed by thread_id.
        self._pending_rollback_by_thread: Dict[str, Dict[str, Any]] = {}
        self._rollback_result_by_thread: Dict[str, Dict[str, Any]] = {}
        self._rollback_lock = threading.Lock()
        self.rollback_journal_root = os.path.join(
            self.memory_dir or self.working_dir,
            ".klynx",
            "rollback_journal",
        )
        
        # ()
        self._event_buffer: Deque[Dict[str, Any]] = deque()
        self._event_signal = threading.Event()
        self.allow_shell_commands = bool(allow_shell_commands)
        self.permission_mode = self._normalize_permission_mode(permission_mode)
        self._workspace_virtual_root = os.path.abspath(
            str(resolved_paths.get("tool_virtual_root") or self.working_dir)
        )
        
        # 
        ToolRegistry.set_working_dir(self.working_dir)
        self._apply_permission(
            mode=self.permission_mode,
            allow_shell_commands=self.allow_shell_commands,
            refresh_prompts=False,
        )
        ToolRegistry.configure_rollback(journal_root=self.rollback_journal_root)

        # ( basic_skills("off") )
        self._load_builtin_skills()

        # ()
        if skills:
            if isinstance(skills, str):
                self.add_skills(skills)
            else:
                self.add_skills(*skills)

        #  system + core; terminal/tui/network .
        self._load_default_tool_groups()
        self._apply_default_tool_policy()

        # 
        self.workflow = self._build_graph()
        self.memory = checkpointer if checkpointer is not None else MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)


    def _load_default_tool_groups(self):
        """."""
        self.add_tools(*[f"group:{name}" for name in self._default_active_tool_groups])
        return self

    def _apply_default_tool_policy(self):
        """
        Apply default tool policy for lightweight startup prompts.

        - Keep `list_directory` available in default tools; it uses terminal-first backend internally.
        """
        return self

    def _toolbox_backend(self):
        return self

    def _normalize_permission_mode(self, mode: str) -> str:
        normalized = str(mode or "").strip().lower()
        if normalized in {"workspace", "sandbox"}:
            return "workspace"
        if normalized in {"global", "all"}:
            return "global"
        return "workspace"

    def _normalize_skill_injection_mode(self, mode: str) -> str:
        normalized = str(mode or "").strip().lower()
        if normalized in {"preload", "tool", "hybrid"}:
            return normalized
        return "hybrid"

    def _normalize_tool_output_delivery_mode(self, mode: str) -> str:
        normalized = str(mode or "").strip().lower()
        if normalized in {"full_inline", "hybrid", "artifact_first"}:
            return normalized
        return "full_inline"

    def _effective_tool_virtual_root(self) -> str:
        if self.permission_mode == "workspace":
            return self._workspace_virtual_root
        return ""

    def _apply_permission(
        self,
        mode: str,
        allow_shell_commands: Optional[bool] = None,
        *,
        refresh_prompts: bool = True,
    ) -> None:
        self.permission_mode = self._normalize_permission_mode(mode)
        if allow_shell_commands is not None:
            self.allow_shell_commands = bool(allow_shell_commands)
        ToolRegistry.configure_security(
            virtual_root=self._effective_tool_virtual_root(),
            allow_shell_commands=self.allow_shell_commands,
        )
        if refresh_prompts:
            self._refresh_tool_prompts()

    def set_sandbox(self, enabled: bool = True):
        mode = "workspace" if bool(enabled) else "global"
        return self.set_permission(mode)

    def set_permission(self, mode: str, allow_shell_commands: Optional[bool] = None):
        raw_mode = str(mode or "").strip().lower()
        if raw_mode not in {"workspace", "global"}:
            raise ValueError("set_permission only accepts 'workspace' or 'global'.")
        self._apply_permission(mode, allow_shell_commands=allow_shell_commands, refresh_prompts=True)
        self._emit(
            "info",
            (
                f"[Permission] mode={self.permission_mode}, "
                f"allow_shell_commands={self.allow_shell_commands}, "
                f"virtual_root={self._effective_tool_virtual_root() or '(global)'}"
            ),
        )
        return self

    def get_permission(self) -> Dict[str, Any]:
        effective_root = ""
        try:
            effective_root = str(getattr(ToolRegistry, "virtual_root", "") or "")
        except Exception:
            effective_root = ""
        return {
            "mode": self.permission_mode,
            "allow_shell_commands": bool(self.allow_shell_commands),
            "workspace_virtual_root": str(self._workspace_virtual_root or ""),
            "effective_virtual_root": effective_root,
        }

    def set_skill_injection_mode(self, mode: str):
        raw_mode = str(mode or "").strip().lower()
        if raw_mode not in {"preload", "tool", "hybrid"}:
            raise ValueError("skill_injection_mode must be one of: preload, tool, hybrid.")
        self.skill_injection_mode = self._normalize_skill_injection_mode(mode)
        self._refresh_skills_prompt()
        if self.tools:
            self._sync_load_skill_tool()
        self._emit("info", f"[Skills] injection_mode={self.skill_injection_mode}")
        return self

    def _model_route_name(self) -> str:
        model = getattr(self, "model", None)
        route = str(getattr(model, "model", "") or "").strip().lower()
        return route

    def _is_xiaomi_mimo_model(self) -> bool:
        route = self._model_route_name()
        return route.startswith("xiaomi_mimo/") or route.startswith("mimo-v2-")

    def _should_force_native_tool_protocol(self) -> bool:
        return True

    def _is_web_search_available(self) -> bool:
        explicit = str(getattr(self.web_search_tool, "_explicit_api_key", "") or "").strip()
        return is_tavily_configured(explicit)

    def _get_builtin_tool_groups(self) -> Dict[str, Dict[str, str]]:
        groups: Dict[str, Dict[str, str]] = {
            name: dict(items) for name, items in self.TOOL_GROUPS.items()
        }

        network_tools = dict(groups.get("network_and_extra", {}))
        if not self._is_web_search_available():
            network_tools.pop("web_search", None)
        groups["network_and_extra"] = network_tools
        return groups

    def _get_builtin_tools(self) -> Dict[str, str]:
        builtins: Dict[str, str] = {}
        for group in self._get_builtin_tool_groups().values():
            builtins.update(group)
        return builtins

    def _normalize_tool_group_name(self, group_name: str) -> str:
        normalized = str(group_name or "").strip().lower()
        return normalized.replace("-", "_")

    def _iter_group_tool_names(self, group_name: str) -> List[str]:
        normalized = self._normalize_tool_group_name(group_name)
        group = self._get_all_tool_groups().get(normalized, {})
        return list(group.keys())

    def _get_all_tool_groups(self) -> Dict[str, Dict[str, str]]:
        groups: Dict[str, Dict[str, str]] = self._get_builtin_tool_groups()
        for group_name, tool_names in self.external_tool_groups.items():
            external_items: Dict[str, str] = {}
            for tool_name in tool_names:
                meta = dict(self.external_tool_registry.get(tool_name, {}) or {})
                description = str(meta.get("description", "") or "").strip()
                if description:
                    external_items[tool_name] = description
            if external_items:
                groups[group_name] = external_items
        return groups

    def _update_active_tool_groups(self):
        active_groups: List[str] = []
        active_names = set(self.tools.keys())
        for group_name, group_tools in self._get_all_tool_groups().items():
            if any(name in active_names for name in group_tools.keys()):
                active_groups.append(group_name)
        self.active_tool_groups = active_groups

    def _normalize_external_tool_group_name(self, group_name: str) -> str:
        normalized = self._normalize_tool_group_name(group_name)
        if not normalized:
            raise ValueError("External tool group name cannot be empty.")
        if normalized in self.TOOL_GROUPS:
            raise ValueError(f"External tool group conflicts with built-in group: {normalized}")
        return normalized

    def _normalize_external_tool_group_spec(
        self,
        item: Any,
    ) -> Optional[Tuple[str, List[Any], bool]]:
        if not isinstance(item, dict):
            return None
        tools = item.get("tools", item.get("items", item.get("members")))
        group_name = item.get("group", item.get("group_name", ""))
        if not isinstance(group_name, str) or not isinstance(tools, (list, tuple)):
            return None
        load_value = item.get("load", True)
        if isinstance(load_value, bool):
            load_now = load_value
        else:
            load_now = str(load_value).strip().lower() not in {"0", "false", "no", "off"}
        return str(group_name).strip(), list(tools), load_now

    def _load_external_tool(self, tool_name: str) -> str:
        normalized_name = str(tool_name or "").strip()
        meta = dict(self.external_tool_registry.get(normalized_name, {}) or {})
        if not meta:
            raise ValueError(f"Unknown external tool: {normalized_name}")
        self.tools[normalized_name] = str(meta.get("description", "") or f"External tool: {normalized_name}")
        self.external_tool_funcs[normalized_name] = meta.get("func")
        return normalized_name

    def _normalize_external_tool_spec(
        self,
        item: Any,
    ) -> Optional[Tuple[str, Callable[..., Any], str]]:
        func: Optional[Callable[..., Any]] = None
        tool_name = ""
        description = ""

        if isinstance(item, dict):
            candidate = item.get("func", item.get("callable", item.get("tool")))
            if callable(candidate):
                func = candidate
                tool_name = str(
                    item.get("name", item.get("tool_name", getattr(candidate, "__name__", ""))) or ""
                ).strip()
                description = str(
                    item.get("description", item.get("desc", getattr(candidate, "__doc__", ""))) or ""
                ).strip()
        elif isinstance(item, (tuple, list)):
            if len(item) == 2 and callable(item[0]) and isinstance(item[1], str):
                func = item[0]
                tool_name = str(getattr(func, "__name__", "") or "").strip()
                description = str(item[1] or "").strip()
            elif (
                len(item) == 3
                and isinstance(item[0], str)
                and callable(item[1])
                and isinstance(item[2], str)
            ):
                tool_name = str(item[0] or "").strip()
                func = item[1]
                description = str(item[2] or "").strip()

        if func is None:
            return None

        if not tool_name:
            tool_name = str(getattr(func, "__name__", "") or "").strip()
        if not tool_name:
            raise ValueError("External tool must provide a name or use a callable with a usable __name__.")

        if not description:
            description = str(getattr(func, "__doc__", "") or "").strip()
        if not description:
            description = f"External tool: {tool_name}"

        return tool_name, func, description

    def _register_external_tool(
        self,
        tool_name: str,
        func: Callable[..., Any],
        description: str,
        *,
        load_now: bool = True,
    ) -> str:
        normalized_name = str(tool_name or "").strip()
        if not normalized_name:
            raise ValueError("External tool name cannot be empty.")
        if normalized_name in self.BASE_TOOLS:
            raise ValueError(f"External tool name conflicts with built-in tool: {normalized_name}")

        desc = str(description or "").strip() or f"External tool: {normalized_name}"
        self.external_tool_registry[normalized_name] = {
            "func": func,
            "description": desc,
        }
        if load_now:
            self.tools[normalized_name] = desc
            self.external_tool_funcs[normalized_name] = func
        return normalized_name

    def register_tool_group(self, group_name: str, *tool_specs: Any, load: bool = False):
        """Register a reusable external tool group and optionally load it immediately."""
        normalized_group = self._normalize_external_tool_group_name(group_name)
        if not tool_specs:
            raise ValueError("register_tool_group requires at least one external tool spec.")

        registered_names: List[str] = []
        for item in tool_specs:
            external_spec = self._normalize_external_tool_spec(item)
            if external_spec is None:
                raise ValueError(
                    "register_tool_group only accepts external-tool dicts, "
                    "or (callable, description)/(name, callable, description) specs."
                )
            tool_name, func, description = external_spec
            registered_name = self._register_external_tool(
                tool_name,
                func,
                description,
                load_now=load,
            )
            if registered_name not in registered_names:
                registered_names.append(registered_name)

        self.external_tool_groups[normalized_group] = registered_names
        self._update_active_tool_groups()
        if load:
            self._refresh_tool_prompts()

        self._emit(
            "info",
            f"[]  {normalized_group}: {', '.join(registered_names)}",
        )
        return self

    def clear_tools(self):
        """Disable all currently active tools, including loaded external tools."""
        removed_count = len(self.tools)
        self.tools.clear()
        self.external_tool_funcs.clear()
        self.active_tool_groups = []
        self._auto_sync_skill_tool = False
        self._refresh_tool_prompts()
        self._emit("info", f"[]  ({removed_count})")
        return self

    def disable_tools(self):
        """Alias of clear_tools()."""
        return self.clear_tools()

    def _infer_skill_scope(self, skill_dir: Path, source: str) -> str:
        source_norm = str(source or "").strip().lower()
        if source_norm in {"builtin", "system"}:
            return "system"
        try:
            resolved_dir = skill_dir.resolve()
        except Exception:
            resolved_dir = skill_dir

        repo_root = (Path(self.working_dir) / ".klynx" / "skills").resolve()
        user_root = Path(self.skills_root).resolve()
        try:
            resolved_dir.relative_to(repo_root)
            return "repo"
        except Exception:
            pass
        try:
            resolved_dir.relative_to(user_root)
            return "user"
        except Exception:
            pass
        if source_norm == "memory":
            return "repo"
        return "extra"

    def _scope_sort_key(self, scope: str) -> Tuple[int, str]:
        scope_norm = str(scope or "").strip().lower()
        order = {"repo": 0, "extra": 1, "user": 2, "system": 3}
        return (order.get(scope_norm, 9), scope_norm)

    def _refresh_skill_registry_cache(self):
        cwd_key = os.path.abspath(self.working_dir)
        ordered_items = sorted(
            self.skill_registry.items(),
            key=lambda item: (self._scope_sort_key(item[1].get("scope", "")), str(item[0]).lower()),
        )
        self.skill_registry = {name: meta for name, meta in ordered_items}
        self.skill_registry_cache_by_cwd[cwd_key] = {
            name: dict(meta) for name, meta in self.skill_registry.items()
        }
        if ordered_items:
            serialized = json.dumps(
                [
                    {
                        "name": name,
                        "scope": meta.get("scope", ""),
                        "path": meta.get("skill_md_path", ""),
                    }
                    for name, meta in ordered_items
                ],
                ensure_ascii=False,
                sort_keys=True,
            )
            self.skill_registry_digest = hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:16]
        else:
            self.skill_registry_digest = ""

    def _refresh_tool_prompts(self):
        if not self.tools:
            self._tool_prompts_cache = ""
            self._json_schemas = []
            return

        external_descs = {
            name: desc for name, desc in self.tools.items() if name in self.external_tool_funcs
        }
        self._json_schemas = get_json_schemas(
            list(self.tools.keys()),
            external_tools=external_descs,
            external_tool_funcs=self.external_tool_funcs,
        )

        self._update_active_tool_groups()

        label_map = {
            "system": "System Tools",
            "core": "Core Tools",
            "terminal": "Terminal Tools",
            "tui": "TUI Tools",
            "network_and_extra": "Network and Browser Tools",
            "skills": "Skills",
        }
        all_groups = self._get_all_tool_groups()
        grouped: Dict[str, List[str]] = {}
        ungrouped: List[str] = []
        for tool_name in sorted(self.tools.keys()):
            placed = False
            for group_name in self.active_tool_groups:
                names = set(all_groups.get(group_name, {}).keys())
                if tool_name in names:
                    category = label_map.get(group_name, f"External Group: {group_name}")
                    grouped.setdefault(category, []).append(tool_name)
                    placed = True
                    break
            if not placed:
                ungrouped.append(tool_name)

        lines: List[str] = [
            "## Active Tools",
            "",
            "- Use tool schemas as the primary source of per-tool arguments.",
            f"- Active protocol mode: `{self.tool_protocol_mode}`",
            (
                f"- Permission mode: `{self.permission_mode}` "
                f"(shell={str(bool(self.allow_shell_commands)).lower()}, "
                f"virtual_root={self._effective_tool_virtual_root() or '(global)'})"
            ),
        ]
        if self.active_tool_groups:
            lines.append(
                "- Active tool groups: "
                + ", ".join(f"`{name}`" for name in self.active_tool_groups)
            )
        lines.append("")
        lines.append("### Tool Families")
        for category, tool_names in grouped.items():
            if tool_names:
                joined = ", ".join(f"`{name}`" for name in tool_names)
                lines.append(f"- {category}: {joined}")
        if ungrouped:
            joined = ", ".join(f"`{name}`" for name in sorted(ungrouped))
            lines.append(f"- Other Tools: {joined}")
        self._tool_prompts_cache = "\n".join(lines).strip()
        return

    def _find_skill_md_path(self, skill_dir: Path) -> Optional[Path]:
        """ SKILL.md ()."""
        if not skill_dir.exists() or not skill_dir.is_dir():
            return None

        direct = skill_dir / "SKILL.md"
        if direct.exists() and direct.is_file():
            return direct

        for child in skill_dir.iterdir():
            if child.is_file() and child.name.lower() == "skill.md":
                return child
        return None

    def _extract_skill_frontmatter(self, content: str) -> Dict[str, str]:
        """ SKILL.md frontmatter  name  description."""
        text = (content or "").lstrip("\ufeff")
        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            return {}

        end_idx = None
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                end_idx = i
                break
        if end_idx is None:
            return {}

        frontmatter = "\n".join(lines[1:end_idx])
        name_match = re.search(r"(?m)^name:\s*(.+?)\s*$", frontmatter)
        desc_match = re.search(r"(?m)^description:\s*(.+?)\s*$", frontmatter)

        parsed: Dict[str, str] = {}
        if name_match:
            parsed["name"] = name_match.group(1).strip().strip("'\"")
        if desc_match:
            parsed["description"] = desc_match.group(1).strip().strip("'\"")
        return parsed

    def _resolve_skill_dir(self, skill_ref: str) -> Path:
        """., SKILL.md ."""
        if not isinstance(skill_ref, str) or not skill_ref.strip():
            raise ValueError("")

        raw_ref = skill_ref.strip()
        path_ref = Path(raw_ref).expanduser()
        candidate_dirs = []

        if path_ref.is_absolute():
            candidate_dirs.append(path_ref)
        else:
            candidate_dirs.append(Path(self.working_dir) / path_ref)
            candidate_dirs.append(Path(self.skills_root) / path_ref)

        for candidate in candidate_dirs:
            if candidate.is_file() and candidate.name.lower() == "skill.md":
                return candidate.parent.resolve()
            if candidate.is_dir() and self._find_skill_md_path(candidate):
                return candidate.resolve()

        # : skills_root ( .system/skill-creator)
        skills_root_path = Path(self.skills_root)
        if skills_root_path.exists() and skills_root_path.is_dir():
            for found in skills_root_path.rglob("*"):
                if not found.is_dir() or found.name != raw_ref:
                    continue
                if self._find_skill_md_path(found):
                    return found.resolve()

        # : package 
        builtin_root_path = Path(self.builtin_skills_root)
        if builtin_root_path.exists() and builtin_root_path.is_dir():
            for found in builtin_root_path.rglob("*"):
                if not found.is_dir() or found.name != raw_ref:
                    continue
                if self._find_skill_md_path(found):
                    return found.resolve()

        raise FileNotFoundError(f" SKILL.md: {skill_ref}")

    def _iter_skill_specs(self, *skills):
        """
         add_skills .

        :
        1) "skill-name" / "path/to/skill"
        2) ("name", "path") / ("name", "path", "description")
        3) {"name": "...", "path": "...", "description": "...", "source": "..."}
        4) add_skills("name", "path", "description")  # 
        """
        if len(skills) == 1 and isinstance(skills[0], (list, tuple, set)):
            skills = tuple(skills[0])

        if not skills:
            raise ValueError("add_skills ")

        # :add_skills(name, path, description)
        if (
            len(skills) in (2, 3)
            and all(isinstance(item, str) for item in skills)
            and (
                os.path.exists(os.path.expanduser(skills[1]))
                or any(ch in skills[1] for ch in ("/", "\\", ":"))
            )
        ):
            maybe_name = skills[0].strip()
            maybe_path = skills[1].strip()
            maybe_desc = skills[2].strip() if len(skills) == 3 else ""
            return [
                {
                    "name": maybe_name,
                    "path": maybe_path,
                    "description": maybe_desc,
                    "source": "external",
                }
            ]

        normalized = []
        for item in skills:
            if isinstance(item, str):
                normalized.append(
                    {
                        "name": "",
                        "path": item,
                        "description": "",
                        "source": "external",
                    }
                )
                continue

            if isinstance(item, (tuple, list)):
                if len(item) not in (2, 3):
                    raise ValueError(f": {item}")
                if not all(isinstance(x, str) for x in item):
                    raise ValueError(f": {item}")
                normalized.append(
                    {
                        "name": item[0].strip(),
                        "path": item[1].strip(),
                        "description": item[2].strip() if len(item) == 3 else "",
                        "source": "external",
                    }
                )
                continue

            if isinstance(item, dict):
                raw_name = str(item.get("name", "") or "").strip()
                raw_path = str(item.get("path", "") or item.get("skill", "") or "").strip()
                if not raw_path:
                    raise ValueError(f" path: {item}")
                raw_desc = str(item.get("description", "") or "").strip()
                raw_source = str(item.get("source", "external") or "external").strip()
                normalized.append(
                    {
                        "name": raw_name,
                        "path": raw_path,
                        "description": raw_desc,
                        "source": raw_source,
                    }
                )
                continue

            raise ValueError(f": {item}")

        return normalized

    def _discover_skill_dirs(self, root_dir: Path) -> List[Path]:
        """ SKILL.md ."""
        if not root_dir.exists() or not root_dir.is_dir():
            return []

        found_dirs = set()
        for md_name in ("SKILL.md", "skill.md"):
            for md_path in root_dir.rglob(md_name):
                if md_path.is_file():
                    found_dirs.add(md_path.parent.resolve())
        return sorted(found_dirs)

    def _sync_load_skill_tool(self):
        """ load_skill ."""
        if not getattr(self, "_auto_sync_skill_tool", True):
            self.tools.pop("load_skill", None)
            self._update_active_tool_groups()
            self._refresh_tool_prompts()
            return
        if self.skill_injection_mode == "preload":
            self.tools.pop("load_skill", None)
        elif self.skill_registry:
            if "load_skill" not in self.tools:
                self.tools["load_skill"] = self.BASE_TOOLS["load_skill"]
        else:
            self.tools.pop("load_skill", None)
        self._update_active_tool_groups()
        self._refresh_tool_prompts()

    def _load_builtin_skills(self):
        """(skill-creator / skill-installer)."""
        if not self.basic_skills_enabled:
            return self

        specs = []
        for skill_name in self.DEFAULT_BASIC_SKILLS:
            skill_dir = Path(self.builtin_skills_root) / skill_name
            if skill_dir.exists() and skill_dir.is_dir():
                specs.append(
                    {
                        "name": skill_name,
                        "path": str(skill_dir),
                        "description": "",
                        "source": "builtin",
                    }
                )
            else:
                self._emit("warning", f"[Skills] : {skill_name} ({skill_dir})")

        if specs:
            self.add_skills(specs)
        return self

    def _load_memory_skills(self):
        """ memory_dir/.klynx/skills ."""
        if not self.memory_dir:
            return self

        skills_root = Path(self.memory_dir) / ".klynx" / "skills"
        if not skills_root.exists() or not skills_root.is_dir():
            return self

        existing_md_paths = {
            str(item.get("skill_md_path", ""))
            for item in self.skill_registry.values()
            if item.get("skill_md_path")
        }
        specs = []
        for skill_dir in self._discover_skill_dirs(skills_root):
            skill_md = self._find_skill_md_path(skill_dir)
            if not skill_md:
                continue
            resolved_md = str(skill_md.resolve())
            if resolved_md in existing_md_paths:
                continue
            specs.append(
                {
                    "name": "",
                    "path": str(skill_dir),
                    "description": "",
                    "source": "memory",
                }
            )

        if specs:
            self.add_skills(specs)
            self._emit("info", f"[Skills]  .klynx/skills  {len(specs)} ")
        return self
    def _refresh_skills_prompt(self):
        if not self.skill_registry:
            self._skills_prompt_cache = ""
            return

        lines = [
            "## Available Skills",
            "",
            "- Only use installed skills listed below.",
            f"- Skill injection mode: `{self.skill_injection_mode}`.",
        ]
        if self.skill_injection_mode == "preload":
            lines.append("- Skills are preloaded from user input hints when possible.")
            lines.append("- Do not call `load_skill`; rely on `<skill_context>` and file tools.")
        elif self.skill_injection_mode == "hybrid":
            lines.append("- Runtime may preload skills from user input hints.")
            lines.append("- Prefer preloaded `<skill_context>`. Use `load_skill` only if context is missing.")
        else:
            lines.append("- Call `load_skill` when the task clearly needs a skill workflow.")
        lines.append("- After a skill is available, follow its `SKILL.md` steps, scripts, and references.")
        lines.append("")
        for name in sorted(self.skill_registry.keys()):
            item = self.skill_registry[name]
            source = str(item.get("source", "external") or "external").strip()
            scope = str(item.get("scope", source) or source).strip()
            desc = str(item.get("description", "") or "").strip() or "No description provided."
            lines.append(f"- `{name}` ({source}, scope={scope}): {desc}")

        self._skills_prompt_cache = "\n".join(lines).strip()

    def add_skills(self, *skills):
        """
         Agent .

        :
        1) 
        2) SKILL.md 
        3) ( skills_root )
        4) (name, path) / (name, path, description)
        5) add_skills(name, path, description) 
        """
        specs = self._iter_skill_specs(*skills)
        installed_names = []

        for spec in specs:
            skill_ref = spec.get("path", "")
            alias_name = spec.get("name", "").strip()
            alias_desc = spec.get("description", "").strip()
            source = spec.get("source", "external")

            skill_dir = self._resolve_skill_dir(skill_ref)
            skill_md = self._find_skill_md_path(skill_dir)
            if skill_md is None:
                raise FileNotFoundError(f" SKILL.md: {skill_dir}")

            try:
                with open(skill_md, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception as e:
                raise RuntimeError(f": {skill_md} ({e})") from e

            frontmatter = self._extract_skill_frontmatter(content)
            skill_name = (alias_name or frontmatter.get("name") or skill_dir.name).strip()
            description = (alias_desc or frontmatter.get("description") or "").strip()
            scope = self._infer_skill_scope(skill_dir, source)

            existing = self.skill_registry.get(skill_name)
            if existing and existing.get("skill_md_path") != str(skill_md.resolve()):
                existing_scope = str(existing.get("scope", "") or "").strip()
                if self._scope_sort_key(scope) < self._scope_sort_key(existing_scope):
                    self._emit(
                        "info",
                        f"[Skills]  scope : {skill_name} {existing_scope} -> {scope}",
                    )
                else:
                    self._emit(
                        "info",
                        f"[Skills] : {skill_name} ({scope})",
                    )
                    continue

            self.skill_registry[skill_name] = {
                "name": skill_name,
                "description": description,
                "skill_dir": str(skill_dir),
                "skill_md_path": str(skill_md.resolve()),
                "source": source,
                "scope": scope,
            }
            installed_names.append(skill_name)
            self._emit("info", f"[Skills] : {skill_name} ({skill_md}) [{scope}]")

        self._refresh_skill_registry_cache()
        self._sync_load_skill_tool()
        self._refresh_skills_prompt()
        if installed_names:
            self._emit("info", f"[Skills]  {len(installed_names)} ")
        return self

    def basic_skills(self, mode: str = "on"):
        """
        (skill-creator / skill-installer).

        Args:
            mode: "on"  "off"
        """
        normalized = str(mode or "").strip().lower()
        if normalized in ("on", "true", "1", "enable", "enabled"):
            self.basic_skills_enabled = True
            self._load_builtin_skills()
            return self

        if normalized in ("off", "false", "0", "disable", "disabled"):
            self.basic_skills_enabled = False
            removed = []
            for name, meta in list(self.skill_registry.items()):
                if meta.get("source") == "builtin":
                    self.skill_registry.pop(name, None)
                    self._skill_body_cache.pop(name, None)
                    removed.append(name)
            self._refresh_skill_registry_cache()
            self._sync_load_skill_tool()
            self._refresh_skills_prompt()
            if removed:
                self._emit("info", f"[Skills] : {', '.join(sorted(removed))}")
            return self

        raise ValueError("basic_skills  'on'  'off'")
    def get_skill_markdown(self, skill_name: str) -> Dict[str, Any]:
        """Load SKILL.md and return both full content and a compact summary."""
        if not isinstance(skill_name, str) or not skill_name.strip():
            available = ", ".join(sorted(self.skill_registry.keys())) or "()"
            return {
                "ok": False,
                "error": f".: {available}",
            }

        normalized = skill_name.strip()
        match_name = None
        if normalized in self.skill_registry:
            match_name = normalized
        else:
            for name in self.skill_registry.keys():
                if name.lower() == normalized.lower():
                    match_name = name
                    break

        if not match_name:
            available = ", ".join(sorted(self.skill_registry.keys())) or "()"
            return {
                "ok": False,
                "error": f": {normalized}.: {available}",
            }

        meta = self.skill_registry[match_name]
        if match_name in self._skill_body_cache:
            content = self._skill_body_cache[match_name]
        else:
            try:
                with open(meta["skill_md_path"], "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().strip()
                self._skill_body_cache[match_name] = content
            except Exception as e:
                return {
                    "ok": False,
                    "error": f" SKILL.md : {e}",
                }

        skill_outline = self._extract_skill_outline(content)
        content_digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16] if content else ""
        return {
            "ok": True,
            "name": match_name,
            "description": meta.get("description", ""),
            "skill_dir": meta.get("skill_dir", ""),
            "skill_md_path": meta.get("skill_md_path", ""),
            "content": content,
            "content_digest": content_digest,
            "when_to_use": str(skill_outline.get("when_to_use", "") or ""),
            "steps": list(skill_outline.get("steps", []) or []),
            "scripts": list(skill_outline.get("scripts", []) or []),
            "references": list(skill_outline.get("references", []) or []),
        }

    @staticmethod
    def _strip_skill_frontmatter(content: str) -> str:
        text = str(content or "")
        if not text.startswith("---"):
            return text
        end = text.find("\n---", 3)
        if end < 0:
            return text
        return text[end + 4 :].strip()

    def _extract_skill_outline(self, content: str) -> Dict[str, Any]:
        body = self._strip_skill_frontmatter(content)
        lines = [str(item or "").strip() for item in body.splitlines()]
        when_to_use = ""
        steps: List[str] = []
        scripts: List[str] = []
        references: List[str] = []

        for line in lines:
            if not line or line.startswith("#"):
                continue
            if not when_to_use and not re.match(r"^[-*]\s+", line) and not re.match(r"^\d+[\.)]\s+", line):
                when_to_use = line
            step_match = re.match(r"^\d+[\.)]\s+(.*)$", line)
            if step_match:
                step_text = str(step_match.group(1) or "").strip()
                if step_text:
                    steps.append(step_text)
            if "scripts/" in line or line.endswith(".py") or line.endswith(".sh"):
                scripts.append(line)
            if "references/" in line or line.startswith("http://") or line.startswith("https://"):
                references.append(line)

        def _uniq(values: List[str], limit: int) -> List[str]:
            out: List[str] = []
            seen = set()
            for item in values:
                text = str(item or "").strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                out.append(text)
                if len(out) >= limit:
                    break
            return out

        return {
            "when_to_use": when_to_use,
            "steps": _uniq(steps, self.SKILL_SUMMARY_STEPS_MAX),
            "scripts": _uniq(scripts, self.SKILL_SUMMARY_ITEMS_MAX),
            "references": _uniq(references, self.SKILL_SUMMARY_ITEMS_MAX),
        }

    def _append_skill_context_block(self, skill_context: str, block: str, digest: str = "") -> str:
        context = str(skill_context or "").strip()
        payload = str(block or "").strip()
        digest_text = str(digest or "").strip()
        if not payload:
            return context
        if digest_text and f"digest: {digest_text}" in context:
            return context
        return f"{context}\n\n{payload}".strip() if context else payload

    def _canonical_skill_name(self, skill_name: str) -> str:
        normalized = str(skill_name or "").strip()
        if not normalized:
            return ""
        if normalized in self.skill_registry:
            return normalized
        normalized_lower = normalized.lower()
        for name in self.skill_registry.keys():
            if str(name).lower() == normalized_lower:
                return name
        return ""

    def _skill_name_from_path(self, skill_path: str) -> str:
        raw = str(skill_path or "").strip().lstrip("@")
        if not raw:
            return ""
        try:
            candidate = Path(os.path.expanduser(raw))
            if candidate.is_dir():
                skill_md = self._find_skill_md_path(candidate)
                if skill_md:
                    resolved = str(skill_md.resolve())
                else:
                    resolved = str(candidate.resolve())
            else:
                resolved = str(candidate.resolve())
        except Exception:
            resolved = os.path.abspath(os.path.expanduser(raw))

        normalized_target = resolved.replace("\\", "/").lower()
        for name, meta in self.skill_registry.items():
            registry_path = str(meta.get("skill_md_path", "") or "").replace("\\", "/").lower()
            if registry_path and registry_path == normalized_target:
                return name
        return ""

    def _collect_skill_refs_from_text(self, text: str) -> List[str]:
        if not self.skill_registry:
            return []
        content = str(text or "")
        if not content.strip():
            return []
        refs: List[str] = []
        seen: Set[str] = set()

        for marker_name in self.SKILL_MARKER_RE.findall(content):
            canonical = self._canonical_skill_name(marker_name)
            if canonical and canonical not in seen:
                refs.append(canonical)
                seen.add(canonical)

        for raw_path in self.SKILL_PATH_RE.findall(content):
            canonical = self._skill_name_from_path(raw_path)
            if canonical and canonical not in seen:
                refs.append(canonical)
                seen.add(canonical)

        for name in sorted(self.skill_registry.keys(), key=len, reverse=True):
            if name in seen:
                continue
            pattern = rf"(?<![A-Za-z0-9._-]){re.escape(name)}(?![A-Za-z0-9._-])"
            if re.search(pattern, content, flags=re.IGNORECASE):
                refs.append(name)
                seen.add(name)
        return refs
    def _build_skill_context_block(
        self,
        skill_result: Dict[str, Any],
        source: str = "tool",
        include_full: bool = False,
    ) -> str:
        canonical_name = str(skill_result.get("name", "") or "").strip()
        source_text = str(source or "tool").strip().lower()
        digest = str(skill_result.get("content_digest", "") or "").strip()
        when_to_use = str(skill_result.get("when_to_use", "") or "").strip()
        steps = [
            str(item).strip()
            for item in (skill_result.get("steps", []) or [])
            if str(item).strip()
        ][: self.SKILL_SUMMARY_STEPS_MAX]
        scripts = [
            str(item).strip()
            for item in (skill_result.get("scripts", []) or [])
            if str(item).strip()
        ][: self.SKILL_SUMMARY_ITEMS_MAX]
        references = [
            str(item).strip()
            for item in (skill_result.get("references", []) or [])
            if str(item).strip()
        ][: self.SKILL_SUMMARY_ITEMS_MAX]

        lines = [
            f"[SKILL] {canonical_name}",
            f"source: {source_text}",
            f"digest: {digest}",
            f"skill_dir: {skill_result.get('skill_dir', '')}",
            f"skill_md: {skill_result.get('skill_md_path', '')}",
            f"description: {skill_result.get('description', '')}",
        ]
        if when_to_use:
            lines.append(f"when_to_use: {when_to_use}")
        if steps:
            lines.append("steps:")
            lines.extend(f"- {item}" for item in steps)
        if scripts:
            lines.append("scripts:")
            lines.extend(f"- {item}" for item in scripts)
        if references:
            lines.append("references:")
            lines.extend(f"- {item}" for item in references)

        if include_full:
            lines.extend(
                [
                    "----- SKILL.md BEGIN -----",
                    str(skill_result.get("content", "") or ""),
                    "----- SKILL.md END -----",
                ]
            )
        return "\n".join(lines).strip()

    def get_skill_paths_for_names(self, skill_names: List[str]) -> List[str]:
        paths: List[str] = []
        for skill_name in list(skill_names or []):
            canonical = self._canonical_skill_name(skill_name)
            if not canonical:
                continue
            meta = self.skill_registry.get(canonical, {})
            skill_path = str(meta.get("skill_md_path", "") or meta.get("skill_dir", "") or "").strip()
            if skill_path:
                paths.append(skill_path)
        return paths

    def preload_skills_for_input(
        self,
        user_input: str,
        loaded_skill_names: Optional[List[str]] = None,
        skill_context: str = "",
    ) -> Tuple[List[str], str, List[str]]:
        names = [str(item).strip() for item in (loaded_skill_names or []) if str(item).strip()]
        context = str(skill_context or "").strip()
        warnings: List[str] = []

        if self.skill_injection_mode == "tool" or not self.skill_registry:
            return names, context, warnings

        references = self._collect_skill_refs_from_text(user_input)
        for ref in references:
            canonical = self._canonical_skill_name(ref)
            if not canonical or canonical in names:
                continue
            skill_result = self.get_skill_markdown(canonical)
            if not skill_result.get("ok", False):
                warnings.append(str(skill_result.get("error", f"Failed to preload skill: {canonical}")))
                continue
            names.append(canonical)
            block = self._build_skill_context_block(skill_result, source="preload", include_full=False)
            context = self._append_skill_context_block(
                context,
                block,
                str(skill_result.get("content_digest", "") or ""),
            )
        return names, context, warnings

    def add_tools(self, *args):
        """
        Add built-in tool groups, specific built-in tools, or custom external tools.

        Supported forms:
        1. agent.add_tools("all")
        2. agent.add_tools("group:core", "group:terminal", "group:tui")
        3. agent.add_tools("read_file", "apply_patch")
        4. agent.add_tools("none")
        5. agent.add_tools(my_func, "Tool description")
        6. agent.add_tools({"name": "my_tool", "func": my_func, "description": "..."})
        7. agent.add_tools(("my_tool", my_func, "Tool description"))
        8. agent.register_tool_group("my_group", {...}, {...}); agent.add_tools("group:my_group")
        9. agent.add_tools({"group": "my_group", "tools": [{...}, {...}], "load": True})
        """
        if not args:
            raise ValueError("add_tools requires at least one argument.")

        if len(args) == 2 and callable(args[0]) and isinstance(args[1], str):
            tool_name = self._register_external_tool(
                str(getattr(args[0], "__name__", "") or "").strip(),
                args[0],
                args[1],
            )
            self._emit("info", f"[] : {tool_name} - {self.tools[tool_name]}")
            self._update_active_tool_groups()
            self._refresh_tool_prompts()
            return self

        loaded_groups: List[str] = []
        loaded_names: List[str] = []
        loaded_external: List[str] = []
        cleared_all = False
        builtin_tools = self._get_builtin_tools()

        for item in args:
            external_group_spec = self._normalize_external_tool_group_spec(item)
            if external_group_spec is not None:
                group_name, tool_specs, load_now = external_group_spec
                self.register_tool_group(group_name, *tool_specs, load=load_now)
                if load_now:
                    loaded_groups.append(self._normalize_external_tool_group_name(group_name))
                continue

            external_spec = self._normalize_external_tool_spec(item)
            if external_spec is not None:
                tool_name, func, description = external_spec
                self._register_external_tool(tool_name, func, description)
                loaded_external.append(tool_name)
                continue

            if not isinstance(item, str):
                raise ValueError(
                    "add_tools only accepts tool-name strings, external-tool dicts, "
                    "or (callable, description)/(name, callable, description) specs."
                )

            name = str(item or "").strip()
            normalized_name = name.lower()

            if normalized_name in {"none", "off", "disable_all"}:
                self.tools.clear()
                self.external_tool_funcs.clear()
                self.active_tool_groups = []
                self._auto_sync_skill_tool = False
                loaded_groups.clear()
                loaded_names.clear()
                loaded_external.clear()
                cleared_all = True
                continue

            if normalized_name == "all":
                self.tools.update(builtin_tools)
                self._auto_sync_skill_tool = True
                loaded_groups = list(self._get_builtin_tool_groups().keys())
                continue

            if name.startswith("group:"):
                group_name = self._normalize_tool_group_name(name.split(":", 1)[1])
                all_groups = self._get_all_tool_groups()
                if group_name not in all_groups:
                    raise ValueError(
                        f"Unknown tool group: {group_name}. "
                        f"Available groups: {', '.join(sorted(all_groups.keys()))}"
                    )
                for tool_name in self._iter_group_tool_names(group_name):
                    if tool_name in builtin_tools:
                        self.tools[tool_name] = builtin_tools[tool_name]
                    else:
                        self._load_external_tool(tool_name)
                if group_name == "skills":
                    self._auto_sync_skill_tool = True
                loaded_groups.append(group_name)
                continue

            if name not in builtin_tools:
                if normalized_name == "web_search" and not self._is_web_search_available():
                    raise ValueError(
                        "web_search requires Tavily API key. "
                        "Use set_tavily_api('tvly-...') or set TAVILY_API_KEY."
                    )
                raise ValueError(
                    f"Unknown built-in tool: {name}. "
                    f"Available tools: {', '.join(builtin_tools.keys())}"
                )
            self.tools[name] = builtin_tools[name]
            if name == "load_skill":
                self._auto_sync_skill_tool = True
            loaded_names.append(name)

        self._update_active_tool_groups()
        if self.tools:
            self._sync_load_skill_tool()
        else:
            self._refresh_tool_prompts()

        loaded_parts: List[str] = []
        if cleared_all:
            loaded_parts.append("")
        if loaded_groups:
            loaded_parts.append(f" {', '.join(loaded_groups)}")
        if loaded_names:
            loaded_parts.append(f" {', '.join(loaded_names)}")
        if loaded_external:
            loaded_parts.append(f" {', '.join(loaded_external)}")

        summary = ";".join(loaded_parts) if loaded_parts else ", ".join(str(arg) for arg in args)
        self._emit("info", f"[] : {summary}")
        if self.tools:
            self._refresh_tool_prompts()
        return self


    def add_mcp(self, mcp_json_path: str):
        """
         MCP (Model Context Protocol) Server,
         Tools  Agent .
        """
        self._emit("info", f"[MCP]  MCP : {mcp_json_path}")
        mcp_tools = self.mcp_manager.load_from_json(mcp_json_path)
        if mcp_tools:
            # MCP tools are returned as LangChain StructuredTools
            for tool in mcp_tools:
                self.add_tools(tool, tool.description)
            self._emit("info", f"[MCP]  {len(mcp_tools)}  MCP ")
        else:
            self._emit("warning", f"[MCP]  {mcp_json_path} ")
        return self


    def _build_graph(self) -> StateGraph:
        """Graph orchestration is defined by concrete agent classes in agents.py."""
        raise NotImplementedError("_build_graph must be implemented in libs/klynx/klynx/agent/agents.py")


    def _emit(self, event_type: str, content: str, **kwargs):
        """( print)"""
        self._event_buffer.append({"type": event_type, "content": content, **kwargs})
        try:
            self._event_signal.set()
        except Exception:
            pass

    def add_hook(self, hook: AgentHook):
        """Register one runtime hook."""
        self.hook_manager.add_hook(hook)
        return self

    def set_hooks(self, hooks: Optional[List[AgentHook]]):
        """Replace all runtime hooks."""
        self.hook_manager.set_hooks(hooks)
        return self

    def clear_hooks(self):
        """Remove all runtime hooks."""
        self.hook_manager.clear()
        return self

    def _build_hook_context(self, state: Dict[str, Any], iteration: int, stage: str) -> AgentHookContext:
        return AgentHookContext(
            state=dict(state),
            iteration=iteration,
            thread_id=str(state.get("thread_id", "") or ""),
            working_dir=self.working_dir,
            metadata={"stage": stage},
        )

    def _run_before_prompt_hooks(self, state: Dict[str, Any], iteration: int, messages: List[Any]) -> Dict[str, Any]:
        if not self.hook_manager.hooks:
            return {"messages": messages, "state": {}}
        context = self._build_hook_context(state, iteration, stage="before_prompt")
        try:
            return self.hook_manager.run_before_prompt(context, messages)
        except Exception as exc:
            self._emit("warning", f"[Hook] before_prompt failed: {exc}")
            return {"messages": messages, "state": {}}

    def _run_after_model_hooks(self, state: Dict[str, Any], iteration: int, model_output: Dict[str, Any]) -> Dict[str, Any]:
        if not self.hook_manager.hooks:
            return model_output
        context = self._build_hook_context(state, iteration, stage="after_model")
        try:
            return self.hook_manager.run_after_model(context, model_output)
        except Exception as exc:
            self._emit("warning", f"[Hook] after_model failed: {exc}")
            return model_output

    def _run_after_tools_hooks(
        self,
        state: Dict[str, Any],
        iteration: int,
        tool_result: Dict[str, Any],
        executed_tools: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not self.hook_manager.hooks:
            return tool_result
        context = self._build_hook_context(state, iteration, stage="after_tools")
        try:
            return self.hook_manager.run_after_tools(context, tool_result, executed_tools)
        except Exception as exc:
            self._emit("warning", f"[Hook] after_tools failed: {exc}")
            return tool_result


    def ask(self, message: str, system_prompt: str = None, thread_id: str = "default"):
        """Ask flow is defined by concrete agent classes in agents.py."""
        raise NotImplementedError("ask must be implemented in libs/klynx/klynx/agent/agents.py")


    def invoke(
        self,
        task: str,
        thread_id: str = "default",
        thinking_context: bool = False,
        system_prompt_append: str = "",
    ):
        """Invoke flow is defined by concrete agent classes in agents.py."""
        raise NotImplementedError("invoke must be implemented in libs/klynx/klynx/agent/agents.py")

    def _normalize_thread_id(self, thread_id: str = "default") -> str:
        normalized = str(thread_id or "").strip()
        return normalized or "default"

    def _build_run_config(
        self,
        thread_id: str = "default",
        recursion_limit: Optional[int] = None,
        include_pending_rollback: bool = True,
    ) -> Dict[str, Any]:
        normalized_thread_id = self._normalize_thread_id(thread_id)
        config: Dict[str, Any] = {"configurable": {"thread_id": normalized_thread_id}}
        if include_pending_rollback:
            pending = self.get_pending_rollback(normalized_thread_id)
            checkpoint_id = str(pending.get("checkpoint_id", "") or "").strip()
            if checkpoint_id:
                config["configurable"]["checkpoint_id"] = checkpoint_id
        if recursion_limit is not None:
            config["recursion_limit"] = int(recursion_limit)
        return config

    def get_pending_rollback(self, thread_id: str = "default") -> Dict[str, Any]:
        normalized_thread_id = self._normalize_thread_id(thread_id)
        with self._rollback_lock:
            value = dict(self._pending_rollback_by_thread.get(normalized_thread_id, {}) or {})
        return value

    def _set_pending_rollback(
        self,
        *,
        thread_id: str,
        checkpoint_id: str,
        raw_index: int,
        display_index: int,
        once: bool = True,
        with_files: bool = False,
        with_git: bool = False,
    ) -> Dict[str, Any]:
        payload = {
            "thread_id": self._normalize_thread_id(thread_id),
            "checkpoint_id": str(checkpoint_id or "").strip(),
            "raw_index": int(raw_index),
            "display_index": int(display_index),
            "once": bool(once),
            "with_files": bool(with_files),
            "with_git": bool(with_git),
            "created_at": int(time.time()),
        }
        with self._rollback_lock:
            self._pending_rollback_by_thread[payload["thread_id"]] = dict(payload)
        return payload

    def _consume_pending_rollback(
        self,
        thread_id: str = "default",
        expected_checkpoint_id: str = "",
    ) -> bool:
        normalized_thread_id = self._normalize_thread_id(thread_id)
        expected = str(expected_checkpoint_id or "").strip()
        with self._rollback_lock:
            current = dict(self._pending_rollback_by_thread.get(normalized_thread_id, {}) or {})
            if not current:
                return False
            current_checkpoint = str(current.get("checkpoint_id", "") or "").strip()
            if expected and current_checkpoint != expected:
                return False
            if not bool(current.get("once", True)):
                return False
            self._pending_rollback_by_thread.pop(normalized_thread_id, None)
        return True

    def cancel_rollback(self, thread_id: str = "default") -> bool:
        normalized_thread_id = self._normalize_thread_id(thread_id)
        with self._rollback_lock:
            existed = normalized_thread_id in self._pending_rollback_by_thread
            self._pending_rollback_by_thread.pop(normalized_thread_id, None)
        if existed:
            self._emit("info", f"[Rollback] cleared pending rollback for thread={normalized_thread_id}")
        return existed

    def get_last_rollback_result(self, thread_id: str = "default") -> Dict[str, Any]:
        normalized_thread_id = self._normalize_thread_id(thread_id)
        with self._rollback_lock:
            return dict(self._rollback_result_by_thread.get(normalized_thread_id, {}) or {})

    def _save_last_rollback_result(self, thread_id: str, result: Dict[str, Any]) -> None:
        normalized_thread_id = self._normalize_thread_id(thread_id)
        with self._rollback_lock:
            self._rollback_result_by_thread[normalized_thread_id] = dict(result or {})

    def _resolve_thread_checkpoint_id(self, thread_id: str = "default") -> str:
        config = self._build_run_config(
            thread_id=thread_id,
            recursion_limit=None,
            include_pending_rollback=False,
        )
        try:
            current_state = self.app.get_state(config)
            state_config = dict(getattr(current_state, "config", {}) or {}) if current_state else {}
            configurable = dict(state_config.get("configurable", {}) or {})
            return str(configurable.get("checkpoint_id", "") or "").strip()
        except Exception:
            return ""

    def _collect_rollback_checkpoint_ids(self, thread_id: str, target_checkpoint_id: str) -> List[str]:
        normalized_thread_id = self._normalize_thread_id(thread_id)
        target = str(target_checkpoint_id or "").strip()
        if not target:
            return []
        config = self._build_run_config(
            thread_id=normalized_thread_id,
            recursion_limit=None,
            include_pending_rollback=False,
        )
        try:
            snapshots = list(self.app.get_state_history(config))
        except Exception:
            return []
        target_raw_index: Optional[int] = None
        checkpoint_rows: List[Tuple[int, str]] = []
        for raw_index, snapshot in enumerate(snapshots):
            state_config = dict(getattr(snapshot, "config", {}) or {})
            configurable = dict(state_config.get("configurable", {}) or {})
            checkpoint_id = str(configurable.get("checkpoint_id", "") or "").strip()
            if not checkpoint_id:
                continue
            checkpoint_rows.append((raw_index, checkpoint_id))
            if checkpoint_id == target and target_raw_index is None:
                target_raw_index = raw_index
        if target_raw_index is None:
            return []
        selected: List[str] = []
        seen = set()
        for raw_index, checkpoint_id in checkpoint_rows:
            if raw_index <= target_raw_index and checkpoint_id not in seen:
                selected.append(checkpoint_id)
                seen.add(checkpoint_id)
        return selected


    def compact_context(self, thread_id: str = "default") -> tuple:
        """
        
        
        Returns:
            (status_message: str, summary_text: str or None)
        """
        config = self._build_run_config(
            thread_id=thread_id,
            recursion_limit=None,
            include_pending_rollback=False,
        )
        current_state = self.app.get_state(config)
        
        if not current_state or not current_state.values:
            return ("", None)
            
        state = current_state.values
        messages = state.get("messages", [])
        
        if not messages:
            return (",", None)
            
        # 
        try:
            self._emit("info", "[] ...")
            result = self._summarize_context(state)
            
            if result:
                # 
                self.app.update_state(config, result)
                summary = result.get("context_summary", "")
                return (f".: {len(summary)} ", summary)
            else:
                return ("", None)
        except Exception as e:
            return (f": {e}", None)


    def get_context(self, thread_id: str = "default", checkpoint_id: str = "") -> dict:
        """
        
        
         LangGraph MemorySaver  thread_id ,
        ,Token.
        
        Args:
            thread_id:  ID
            
        Returns:
            .,.
            : "messages", "overall_goal", "total_tokens", "iteration_count" .
        """
        config = self._build_run_config(
            thread_id=thread_id,
            recursion_limit=None,
            include_pending_rollback=False,
        )
        checkpoint_text = str(checkpoint_id or "").strip()
        if checkpoint_text:
            config["configurable"]["checkpoint_id"] = checkpoint_text
        current_state = self.app.get_state(config)
        return current_state.values if current_state else {}


    def get_history(self, thread_id: str = "default", limit: int = 20) -> list:
        """
        
        
        Agent(,),
        Agent.
        
        Args:
            thread_id:  ID
            limit: 
            
        Returns:
            , index, checkpoint_id, action_summary 
        """
        config = self._build_run_config(
            thread_id=thread_id,
            recursion_limit=None,
            include_pending_rollback=False,
        )
        
        history = []
        prev_msg_count = 0
        
        try:
            snapshots = list(self.app.get_state_history(config))
            
            for idx, state_snapshot in enumerate(snapshots):
                if idx >= limit:
                    break
                
                values = state_snapshot.values
                messages = values.get("messages", [])
                checkpoint_id = state_snapshot.config["configurable"].get("checkpoint_id", "")
                next_nodes = list(state_snapshot.next) if state_snapshot.next else []
                node_name = next_nodes[0] if next_nodes else "()"
                
                # 
                action_summary = self._extract_action_summary(messages, prev_msg_count)
                prev_msg_count = len(messages)
                
                history.append({
                    "index": idx,
                    "checkpoint_id": checkpoint_id,
                    "node": node_name,
                    "message_count": len(messages),
                    "iteration": values.get("iteration_count", 0),
                    "action": action_summary,
                    "task_completed": values.get("task_completed", False),
                    "progress": values.get("progress_summary", "")  
                })
        except Exception as e:
            self._emit("error", f"[] : {e}")
        
        for i, item in enumerate(history):
            item["display_index"] = i
        
        return history


    def _extract_action_summary(self, messages: list, prev_count: int) -> str:
        """
        
        
        Args:
            messages: 
            prev_count: 
            
        Returns:
            
        """
        if not messages:
            return ""
        
        # 
        new_messages = messages[prev_count:] if prev_count < len(messages) else messages[-1:]
        
        actions = []
        for msg in new_messages:
            content = getattr(msg, 'content', str(msg))
            
            if isinstance(msg, HumanMessage):
                # 
                if '<tool_result' in content:
                    import re
                    tool_match = re.search(r'tool="(\w+)"', content)
                    if tool_match:
                        tool_name = tool_match.group(1)
                        # 
                        if 'success' in content.lower():
                            actions.append(f"鉁?{tool_name}")
                        elif 'error' in content.lower():
                            actions.append(f"鉁?{tool_name}")
                        else:
                            actions.append(f"鈫?{tool_name}")
                elif content.startswith('[---'):
                    actions.append("鈹€鈹€鈹€  鈹€鈹€鈹€")
                else:
                    # 
                    snippet = content[:60] + "..." if len(content) > 60 else content
                    actions.append(f": {snippet}")
            elif isinstance(msg, AIMessage):
                # AI
                import re
                tool_tags = [
                    'read_file', 'apply_patch', 'execute_command', 'search_in_files',
                    'list_directory',
                    'load_skill', 'state_update'
                ]
                found_tools = []
                for tag in tool_tags:
                    if f'<{tag}' in content or f'<{tag}>' in content:
                        found_tools.append(tag)
                
                if found_tools:
                    actions.append(f"Agent: {', '.join(found_tools)}")
        
        return " | ".join(actions) if actions else ""


    def rollback(
        self,
        thread_id: str = "default",
        target_index: int = None,
        once: bool = True,
        with_files: bool = False,
        with_git: bool = False,
    ) -> bool:
        """
        
        
         config, stream .
         update_state().
        
        Args:
            thread_id:  ID
            target_index:  get_state_history 
            
        Returns:
            
        """
        normalized_thread_id = self._normalize_thread_id(thread_id)
        config = self._build_run_config(
            thread_id=normalized_thread_id,
            recursion_limit=None,
            include_pending_rollback=False,
        )
        
        try:
            snapshots = list(self.app.get_state_history(config))
            
            if not snapshots:
                self._emit("info", "[] ")
                return False
            
            if target_index is None:
                raw_target_index = 1 if len(snapshots) > 1 else 0
                display_target_index = raw_target_index
            else:
                display_target_index = int(target_index)
                if display_target_index < 0 or display_target_index >= len(snapshots):
                    self._emit("error", f"[Rollback] index {display_target_index} out of range ({len(snapshots)})")
                    self._save_last_rollback_result(
                        normalized_thread_id,
                        {
                            "ok": False,
                            "thread_id": normalized_thread_id,
                            "error": "index out of range",
                            "display_index": display_target_index,
                        },
                    )
                    return False
                raw_target_index = display_target_index

            target_state = snapshots[raw_target_index]
            target_values = target_state.values

            target_config = dict(getattr(target_state, "config", {}) or {})
            target_configurable = dict(target_config.get("configurable", {}) or {})
            checkpoint_id = str(target_configurable.get("checkpoint_id", "") or "").strip()
            if not checkpoint_id:
                self._emit("error", "[Rollback] checkpoint_id missing on selected snapshot")
                self._save_last_rollback_result(
                    normalized_thread_id,
                    {
                        "ok": False,
                        "thread_id": normalized_thread_id,
                        "display_index": display_target_index,
                        "raw_index": raw_target_index,
                        "error": "missing checkpoint_id",
                    },
                )
                return False

            rollback_checkpoint_ids: List[str] = []
            workspace_report: Dict[str, Any] = {}
            git_restore_report: Dict[str, Any] = {}
            if with_files or with_git:
                rollback_checkpoint_ids = self._collect_rollback_checkpoint_ids(
                    normalized_thread_id,
                    checkpoint_id,
                )
            if with_git:
                git_restore_report = ToolRegistry.rollback_workspace_with_git(
                    thread_id=normalized_thread_id,
                    target_checkpoint_id=checkpoint_id,
                )
                if not bool(git_restore_report.get("ok", False)):
                    self._emit("warning", "[Rollback] git restore completed with warnings")
            if with_files:
                workspace_report = ToolRegistry.rollback_workspace(
                    thread_id=normalized_thread_id,
                    rollback_checkpoint_ids=rollback_checkpoint_ids,
                )
                if not bool(workspace_report.get("ok", False)):
                    self._emit("warning", "[Rollback] workspace restore completed with warnings")

            pending_payload = self._set_pending_rollback(
                thread_id=normalized_thread_id,
                checkpoint_id=checkpoint_id,
                raw_index=raw_target_index,
                display_index=display_target_index,
                once=once,
                with_files=with_files,
                with_git=with_git,
            )

            msgs = target_values.get("messages", [])
            progress = target_values.get("progress_summary", "")

            self._emit(
                "info",
                "[Rollback] selected "
                f"display_index={display_target_index}, checkpoint_id={checkpoint_id[:12]}, once={bool(once)}",
            )
            self._emit("info", f"  : {len(msgs)}")
            self._emit("info", f"  : {target_values.get('iteration_count', 0)}")
            if progress:
                # 
                progress_lines = progress.strip().split('\n')
                for line in progress_lines[-3:]:
                    self._emit("info", f"  {line}")

            result_payload = {
                "ok": True,
                "thread_id": normalized_thread_id,
                "display_index": display_target_index,
                "raw_index": raw_target_index,
                "checkpoint_id": checkpoint_id,
                "once": bool(once),
                "with_files": bool(with_files),
                "with_git": bool(with_git),
                "message_count": len(msgs),
                "iteration": int(target_values.get("iteration_count", 0) or 0),
                "workspace_restore": workspace_report,
                "git_restore": git_restore_report,
                "pending": pending_payload,
            }
            self._save_last_rollback_result(normalized_thread_id, result_payload)
            return True
            
        except Exception as e:
            self._emit("error", f"[Rollback] failed: {e}")
            self._save_last_rollback_result(
                normalized_thread_id,
                {
                    "ok": False,
                    "thread_id": normalized_thread_id,
                    "error": str(e),
                },
            )
            import traceback
            traceback.print_exc()
            return False


    def run_terminal_agent_stream(
        self,
        task: str,
        thread_id: str,
        system_prompt_append: str = "",
    ) -> dict:
        """Run invoke() and print streaming events in terminal."""
        from klynx.agent.package import run_terminal_agent_stream
        return run_terminal_agent_stream(
            self,
            task,
            thread_id,
            system_prompt_append=system_prompt_append,
        )


    def run_terminal_ask_stream(self, message: str, system_prompt: str = None, thread_id: str = "default") -> str:
        """ agent.ask(),,"""
        from klynx.agent.package import run_terminal_ask_stream
        return run_terminal_ask_stream(self, message, system_prompt, thread_id)



GraphKlynxAgent = KlynxAgent


def create_agent(working_dir: str = ".", model=None, max_iterations: Optional[int] = None,
                 memory_dir: str = "", load_project_docs: bool = True,
                 os_name: str = platform.system(), browser_headless: bool = False,
                 append_system_prompt: str = "", skills: Optional[List[Any]] = None,
                 tool_protocol_mode: Optional[str] = None, tool_call_mode: str = "native",
                 skills_root: str = "", checkpointer: Optional[Any] = None,
                 permission_mode: str = "workspace",
                 tool_virtual_root: str = "", allow_shell_commands: bool = True,
                 skill_injection_mode: str = "hybrid",
                 max_tools_per_step: int = 20,
                 max_reads_per_file_per_step: int = 6,
                 max_retry_per_tool_per_step: int = 2,
                 tui_stall_threshold: int = 3,
                 full_tui_echo: bool = False,
                 tool_output_delivery_mode: str = "full_inline",
                 tool_output_hard_ceiling_chars: int = 200000,
                 backend: Optional[AgentBackend] = None,
                 store: Optional[AgentStore] = None,
                 hooks: Optional[List[AgentHook]] = None,
                 mode: str = "react",
                 enable_subagent: bool = False,
                 fast_model: Any = None,
                 thinking_model: Any = None,
                 small_self_loop_max: int = 10,
                 context_warn_ratio: float = 0.80,
                 context_hard_ratio: float = 0.92) -> KlynxAgent:
    """Create default Klynx agent instance."""
    from .agents import resolve_agent_class
    import inspect

    agent_class = resolve_agent_class(mode)
    requested_kwargs = {
        "working_dir": working_dir,
        "model": model,
        "max_iterations": max_iterations,
        "memory_dir": memory_dir,
        "load_project_docs": load_project_docs,
        "os_name": os_name,
        "browser_headless": browser_headless,
        "append_system_prompt": append_system_prompt,
        "skills": skills,
        "tool_protocol_mode": tool_protocol_mode,
        "tool_call_mode": tool_call_mode,
        "skills_root": skills_root,
        "checkpointer": checkpointer,
        "permission_mode": permission_mode,
        "tool_virtual_root": tool_virtual_root,
        "allow_shell_commands": allow_shell_commands,
        "skill_injection_mode": skill_injection_mode,
        "max_tools_per_step": max_tools_per_step,
        "max_reads_per_file_per_step": max_reads_per_file_per_step,
        "max_retry_per_tool_per_step": max_retry_per_tool_per_step,
        "tui_stall_threshold": tui_stall_threshold,
        "full_tui_echo": full_tui_echo,
        "tool_output_delivery_mode": tool_output_delivery_mode,
        "tool_output_hard_ceiling_chars": tool_output_hard_ceiling_chars,
        "backend": backend,
        "store": store,
        "hooks": hooks,
        "enable_subagent": enable_subagent,
        "fast_model": fast_model,
        "thinking_model": thinking_model,
        "small_self_loop_max": small_self_loop_max,
        "context_warn_ratio": context_warn_ratio,
        "context_hard_ratio": context_hard_ratio,
    }
    accepted_keys = set()
    for cls in agent_class.mro():
        init_fn = cls.__dict__.get("__init__")
        if init_fn is None:
            continue
        try:
            signature = inspect.signature(init_fn)
        except (TypeError, ValueError):
            continue
        for name, param in signature.parameters.items():
            if name == "self":
                continue
            if param.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }:
                continue
            accepted_keys.add(name)
    supported = {
        key: value for key, value in requested_kwargs.items() if key in accepted_keys
    }
    return agent_class(**supported)









