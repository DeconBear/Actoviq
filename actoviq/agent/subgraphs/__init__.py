"""Built-in subgraph registry."""

from __future__ import annotations

from typing import Callable, Dict

from .ask import build_ask_messages, stream_ask, stream_model_answer
from .actoviq_loop import build_actoviq_initial_state, build_actoviq_loop_subgraph, run_actoviq_loop_node
from .react_once import (
    act_once_node,
    build_react_once_subgraph,
    emit_react_once_done,
    run_react_once_node,
    think_once_node,
)

BuiltinNodeHandler = Callable[..., object]


def get_builtin_subgraph_registry() -> Dict[str, BuiltinNodeHandler]:
    return {
        "actoviq_loop": run_actoviq_loop_node,
        "react_once": run_react_once_node,
    }


__all__ = [
    "BuiltinNodeHandler",
    "get_builtin_subgraph_registry",
    "build_actoviq_initial_state",
    "build_actoviq_loop_subgraph",
    "run_actoviq_loop_node",
    "build_react_once_subgraph",
    "run_react_once_node",
    "think_once_node",
    "act_once_node",
    "emit_react_once_done",
    "stream_ask",
    "build_ask_messages",
    "stream_model_answer",
]

