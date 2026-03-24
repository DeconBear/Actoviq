"""Shared toolbox facade for runtimes that delegate tool operations."""

from __future__ import annotations

from typing import Any


class ToolboxRuntime:
    """Provide a stable tool-management surface for composable runtimes."""

    def _toolbox_backend(self) -> Any:
        raise NotImplementedError("_toolbox_backend must be implemented by subclasses")

    def add_tools(self, *args: Any):
        backend = self._toolbox_backend()
        backend.add_tools(*args)
        return self

    def register_tool_group(self, group_name: str, *tool_specs: Any, load: bool = False):
        backend = self._toolbox_backend()
        backend.register_tool_group(group_name, *tool_specs, load=load)
        return self

    def clear_tools(self):
        backend = self._toolbox_backend()
        backend.clear_tools()
        return self

    def disable_tools(self):
        return self.clear_tools()

    def _refresh_tool_prompts(self):
        backend = self._toolbox_backend()
        refresher = getattr(backend, "_refresh_tool_prompts", None)
        if callable(refresher):
            refresher()
        return self

    def set_sandbox(self, enabled: bool = True):
        backend = self._toolbox_backend()
        setter = getattr(backend, "set_sandbox", None)
        if not callable(setter):
            raise AttributeError("set_sandbox is not supported by this runtime")
        setter(enabled=enabled)
        return self

    def set_permission(self, mode: str, allow_shell_commands: Any = None):
        backend = self._toolbox_backend()
        setter = getattr(backend, "set_permission", None)
        if not callable(setter):
            raise AttributeError("set_permission is not supported by this runtime")
        setter(mode=mode, allow_shell_commands=allow_shell_commands)
        return self

    def get_permission(self):
        backend = self._toolbox_backend()
        getter = getattr(backend, "get_permission", None)
        if not callable(getter):
            raise AttributeError("get_permission is not supported by this runtime")
        return getter()
