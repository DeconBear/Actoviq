from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class AgentBackend(Protocol):
    """Path/backend abstraction for runtime directories."""

    def resolve_working_dir(self, configured: str) -> str:
        ...

    def resolve_memory_dir(self, configured: str) -> str:
        ...

    def resolve_skills_root(self, configured: str) -> str:
        ...

    def resolve_tool_virtual_root(self, configured: str, working_dir: str) -> str:
        ...


@dataclass
class LocalAgentBackend:
    """Default local-filesystem backend."""

    working_dir: str = "."
    memory_dir: str = ""
    skills_root: str = ""
    tool_virtual_root: str = ""

    def _abs_or_empty(self, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        return str(Path(text).expanduser().resolve())

    def resolve_working_dir(self, configured: str) -> str:
        candidate = configured or self.working_dir or "."
        return self._abs_or_empty(candidate) or str(Path(".").resolve())

    def resolve_memory_dir(self, configured: str) -> str:
        return self._abs_or_empty(configured or self.memory_dir)

    def resolve_skills_root(self, configured: str) -> str:
        return self._abs_or_empty(configured or self.skills_root)

    def resolve_tool_virtual_root(self, configured: str, working_dir: str) -> str:
        candidate = configured or self.tool_virtual_root or working_dir
        return self._abs_or_empty(candidate)


def resolve_runtime_paths(
    backend: Optional[AgentBackend],
    *,
    working_dir: str,
    memory_dir: str,
    skills_root: str,
    tool_virtual_root: str,
) -> dict:
    """Resolve runtime paths via backend or direct values."""
    if backend is None:
        backend = LocalAgentBackend()

    resolved_working_dir = backend.resolve_working_dir(working_dir)
    resolved_memory_dir = backend.resolve_memory_dir(memory_dir)
    resolved_skills_root = backend.resolve_skills_root(skills_root)
    resolved_tool_virtual_root = backend.resolve_tool_virtual_root(
        tool_virtual_root,
        resolved_working_dir,
    )

    return {
        "working_dir": os.path.abspath(resolved_working_dir),
        "memory_dir": os.path.abspath(resolved_memory_dir) if resolved_memory_dir else "",
        "skills_root": os.path.abspath(resolved_skills_root) if resolved_skills_root else "",
        "tool_virtual_root": os.path.abspath(resolved_tool_virtual_root)
        if resolved_tool_virtual_root
        else os.path.abspath(resolved_working_dir),
    }
