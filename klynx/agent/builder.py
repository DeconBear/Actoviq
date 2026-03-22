"""Builder APIs for composable Klynx runtimes."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .graph import create_agent
from .state import AgentState
from .subgraphs import get_builtin_subgraph_registry, stream_ask
from .toolbox import ToolboxRuntime


NodeHandler = Callable[["ComposableAgentRuntime", Dict[str, Any]], Any]


@dataclass
class _NodeSpec:
    name: str
    handler: NodeHandler
    is_builtin: bool = False
    builtin_alias: str = ""


class ComposableAgentRuntime(ToolboxRuntime):
    """Runtime built from `KlynxGraphBuilder`."""

    def __init__(
        self,
        *,
        name: str,
        nodes: Dict[str, _NodeSpec],
        edges: Dict[str, List[str]],
        conditional_edges: Dict[str, Tuple[Callable[..., Any], Dict[str, str]]],
        entry_point: str,
        finish_point: str,
        runtime_config: Dict[str, Any],
    ) -> None:
        self.name = name
        self._nodes = dict(nodes)
        self._edges = {key: list(values) for key, values in edges.items()}
        self._conditional_edges = {
            key: (value[0], dict(value[1])) for key, value in conditional_edges.items()
        }
        self._entry_point = entry_point
        self._finish_point = finish_point
        self._runtime_config = dict(runtime_config)
        self._loop_agent = None
        self._last_done_event: Dict[str, Any] = {}

    def _toolbox_backend(self):
        return self._ensure_loop_agent()

    def _ensure_loop_agent(self):
        if self._loop_agent is not None:
            return self._loop_agent

        signature = inspect.signature(create_agent)
        supported = {
            name: value
            for name, value in self._runtime_config.items()
            if name in signature.parameters
        }
        self._loop_agent = create_agent(**supported)

        mcp_json_path = str(self._runtime_config.get("mcp_json_path", "") or "").strip()
        if mcp_json_path:
            add_mcp = getattr(self._loop_agent, "add_mcp", None)
            if callable(add_mcp):
                add_mcp(mcp_json_path)
        return self._loop_agent

    def _call_router(self, router: Callable[..., Any], payload: Dict[str, Any]) -> str:
        try:
            result = router(payload)
        except TypeError:
            result = router(payload, self)
        return str(result or "").strip()

    def _split_node_result(
        self,
        result: Any,
    ) -> Tuple[Iterable[dict], Dict[str, Any]]:
        if isinstance(result, dict):
            if "events" in result:
                events = result.get("events", [])
                payload_patch = dict(result.get("payload", {}) or {})
                return events, payload_patch
            return [], dict(result)
        if result is None:
            return [], {}
        if isinstance(result, Iterable) and not isinstance(result, (str, bytes, dict)):
            return result, {}
        return [], {}

    def _pick_next_node(self, current: str, payload: Dict[str, Any]) -> str:
        if current in self._conditional_edges:
            router, mapping = self._conditional_edges[current]
            route = self._call_router(router, payload)
            target = mapping.get(route, "")
            if str(target).strip().lower() in {"", "end", "__end__", "none"}:
                return ""
            return str(target)

        options = list(self._edges.get(current, []) or [])
        return str(options[0]) if options else ""

    def _emit_fallback_done(self, payload: Dict[str, Any]) -> dict:
        return {
            "type": "done",
            "content": "",
            "answer": str(payload.get("answer", "") or ""),
            "iteration_count": int(payload.get("iteration_count", 0) or 0),
            "total_tokens": int(payload.get("total_tokens", 0) or 0),
            "prompt_tokens": int(payload.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(payload.get("completion_tokens", 0) or 0),
        }

    def invoke(
        self,
        task: str,
        thread_id: str = "default",
        **kwargs: Any,
    ) -> Iterable[dict]:
        if not self._entry_point:
            raise ValueError("Builder runtime has no entry point.")

        payload: Dict[str, Any] = {
            "task": str(task or ""),
            "thread_id": str(thread_id or "default"),
            "invoke_kwargs": dict(kwargs or {}),
        }
        self._last_done_event = {}

        current = self._entry_point
        steps = 0
        max_steps = max(1, int(payload.get("max_steps", 256) or 256))

        while current:
            steps += 1
            if steps > max_steps:
                yield {"type": "error", "content": "Builder graph exceeded max_steps safety limit."}
                break

            spec = self._nodes.get(current)
            if spec is None:
                yield {"type": "error", "content": f"Unknown node in runtime graph: {current}"}
                break

            try:
                result = spec.handler(self, payload)
            except Exception as exc:
                yield {"type": "error", "content": f"Node '{current}' failed: {exc}"}
                break

            events, payload_patch = self._split_node_result(result)
            for event in events:
                if not isinstance(event, dict):
                    event = {"type": "answer", "content": str(event)}
                if event.get("type") == "done":
                    self._last_done_event = dict(event)
                yield event

            if payload_patch:
                payload.update(payload_patch)

            if self._finish_point and current == self._finish_point:
                break
            current = self._pick_next_node(current, payload)

        if not self._last_done_event:
            yield self._emit_fallback_done(payload)

    def ask(
        self,
        message: str,
        thread_id: str = "default",
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Iterable[dict]:
        _ = kwargs
        agent = self._ensure_loop_agent()
        return stream_ask(
            agent,
            message,
            system_prompt=system_prompt,
            thread_id=thread_id,
        )

    def get_context(self, thread_id: str = "default") -> dict:
        if self._loop_agent is None:
            return {}
        getter = getattr(self._loop_agent, "get_context", None)
        if not callable(getter):
            return {}
        return dict(getter(thread_id=thread_id) or {})

    def compact_context(self, thread_id: str = "default"):
        agent = self._ensure_loop_agent()
        compact = getattr(agent, "compact_context", None)
        if not callable(compact):
            return ("Context compaction is not supported by this runtime.", "")
        try:
            return compact(thread_id)
        except TypeError:
            return compact(thread_id=thread_id)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        agent = self._ensure_loop_agent()
        return getattr(agent, name)


class KlynxGraphBuilder:
    """Graph builder for composable Klynx runtime orchestration."""

    def __init__(self, *, state_type=AgentState, name: str = "klynx_graph") -> None:
        self.state_type = state_type
        self.name = str(name or "klynx_graph")
        self._nodes: Dict[str, _NodeSpec] = {}
        self._edges: Dict[str, List[str]] = {}
        self._conditional_edges: Dict[str, Tuple[Callable[..., Any], Dict[str, str]]] = {}
        self._entry_point = ""
        self._finish_point = ""
        self._runtime_defaults: Dict[str, Any] = {}
        self._tool_specs: List[Any] = []
        self._builtin_registry = get_builtin_subgraph_registry()

    def _resolve_builtin_node_alias(self, alias: str) -> Optional[NodeHandler]:
        normalized = str(alias or "").strip().lower()
        return self._builtin_registry.get(normalized)

    def add_node(
        self,
        name: str,
        node: Optional[Any] = None,
        *,
        kind: str = "node",
    ):
        _ = kind
        node_name = str(name or "").strip()
        if not node_name:
            raise ValueError("node name cannot be empty")

        if node is None:
            builtin = self._resolve_builtin_node_alias(node_name)
            if builtin is None:
                raise ValueError(
                    f"Unknown built-in node alias: {node_name}. "
                    f"Available: {', '.join(sorted(self._builtin_registry.keys()))}"
                )
            self._nodes[node_name] = _NodeSpec(
                name=node_name,
                handler=builtin,
                is_builtin=True,
                builtin_alias=node_name.lower(),
            )
            if not self._entry_point:
                self._entry_point = node_name
            return self

        if isinstance(node, str):
            builtin = self._resolve_builtin_node_alias(node)
            if builtin is None:
                raise ValueError(
                    f"Unknown built-in node alias: {node}. "
                    f"Available: {', '.join(sorted(self._builtin_registry.keys()))}"
                )
            self._nodes[node_name] = _NodeSpec(
                name=node_name,
                handler=builtin,
                is_builtin=True,
                builtin_alias=str(node).strip().lower(),
            )
            if not self._entry_point:
                self._entry_point = node_name
            return self

        if not callable(node):
            raise ValueError("node must be callable, built-in alias string, or None")

        def _wrapped(runtime: ComposableAgentRuntime, payload: Dict[str, Any]):
            try:
                return node(runtime, payload)
            except TypeError:
                return node(payload)

        self._nodes[node_name] = _NodeSpec(name=node_name, handler=_wrapped)
        if not self._entry_point:
            self._entry_point = node_name
        return self

    def add_edge(self, src: str, dst: str):
        src_name = str(src or "").strip()
        dst_name = str(dst or "").strip()
        if not src_name or not dst_name:
            raise ValueError("edge src and dst cannot be empty")
        self._edges.setdefault(src_name, []).append(dst_name)
        return self

    def add_edges(self, edges: Iterable[Tuple[str, str]]):
        for src, dst in edges:
            self.add_edge(src, dst)
        return self

    def add_conditional_edges(
        self,
        src: str,
        router: Callable[..., Any],
        mapping: Dict[str, str],
    ):
        src_name = str(src or "").strip()
        if not src_name:
            raise ValueError("conditional edge source cannot be empty")
        if not callable(router):
            raise ValueError("router must be callable")
        self._conditional_edges[src_name] = (router, dict(mapping or {}))
        return self

    def set_entry_point(self, name: str):
        node_name = str(name or "").strip()
        if not node_name:
            raise ValueError("entry point cannot be empty")
        self._entry_point = node_name
        return self

    def set_finish_point(self, name: str):
        self._finish_point = str(name or "").strip()
        return self

    def with_runtime_defaults(self, **kwargs: Any):
        self._runtime_defaults.update(dict(kwargs or {}))
        return self

    def _ensure_klynx_loop_entry(self) -> None:
        if "klynx_loop" not in self._nodes:
            self.add_node("klynx_loop")
        if not self._entry_point:
            self._entry_point = "klynx_loop"

    def react(self, **runtime_defaults: Any):
        """
        Configure this builder for the default ReAct runtime.

        This is a convenience preset equivalent to:
        - add_node("klynx_loop")
        - set entry point to "klynx_loop" (when unset)
        - runtime default `mode="react"`
        """
        self._ensure_klynx_loop_entry()
        defaults = {"mode": "react"}
        defaults.update(dict(runtime_defaults or {}))
        self.with_runtime_defaults(**defaults)
        return self

    def request(self, **runtime_defaults: Any):
        """
        Configure this builder for RequestOrchestrator runtime.

        This is a convenience preset equivalent to:
        - add_node("klynx_loop")
        - set entry point to "klynx_loop" (when unset)
        - runtime default `mode="request_orchestrator"`
        - if `model` is not provided, fallback to `fast_model` or
          `thinking_model` as baseline model for ask/fallback paths.
        """
        self._ensure_klynx_loop_entry()
        defaults = {"mode": "request_orchestrator"}
        defaults.update(dict(runtime_defaults or {}))
        if defaults.get("model") is None:
            if defaults.get("fast_model") is not None:
                defaults["model"] = defaults.get("fast_model")
            elif defaults.get("thinking_model") is not None:
                defaults["model"] = defaults.get("thinking_model")
        self.with_runtime_defaults(**defaults)
        return self

    def ask(self, **runtime_defaults: Any):
        """
        Configure this builder for ask-first usage.

        Ask streaming itself uses runtime.ask(...), but this preset keeps
        a consistent builder entry setup and defaults to `mode="react"`.
        """
        self._ensure_klynx_loop_entry()
        defaults = {"mode": "react"}
        defaults.update(dict(runtime_defaults or {}))
        self.with_runtime_defaults(**defaults)
        return self

    def with_tools(self, *tool_specs: Any):
        self._tool_specs.extend(list(tool_specs or []))
        return self

    def build(self, **kwargs: Any) -> ComposableAgentRuntime:
        if not self._nodes:
            raise ValueError("builder has no nodes")

        runtime_config = dict(self._runtime_defaults)
        runtime_config.update(dict(kwargs or {}))
        entry_point = self._entry_point
        if not entry_point:
            entry_point = next(iter(self._nodes.keys()))

        runtime = ComposableAgentRuntime(
            name=self.name,
            nodes=self._nodes,
            edges=self._edges,
            conditional_edges=self._conditional_edges,
            entry_point=entry_point,
            finish_point=self._finish_point,
            runtime_config=runtime_config,
        )
        if self._tool_specs:
            runtime.add_tools(*self._tool_specs)
        return runtime


def create_builder(
    *,
    state_type=AgentState,
    name: str = "klynx_graph",
) -> KlynxGraphBuilder:
    return KlynxGraphBuilder(state_type=state_type, name=name)


__all__ = [
    "ComposableAgentRuntime",
    "KlynxGraphBuilder",
    "create_builder",
]
