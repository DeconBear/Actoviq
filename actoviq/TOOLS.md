# Actoviq Tools Snapshot

This document lists built-in tools, tool groups, and the key tool guidance injected into the model.

## Source of Truth

- `libs/actoviq/actoviq/agent/graph.py`
- `libs/actoviq/actoviq/agent/tools/registry.py`
- `libs/actoviq/actoviq/agent/prompt_builder.py`
- `libs/actoviq/actoviq/agent/prompts/fragments/tool_selection.md`
- `libs/actoviq/actoviq/agent/prompts/system_base.md`

## Default Loading

- Default active groups: `system`, `core`
- `load_skill` is dynamic:
  - `skill_injection_mode=preload`: hidden
  - `skill_injection_mode=hybrid/tool`: available when skill registry exists

## Prompted Tool Guidance (LLM-visible)

1. Dynamic active-tools block from `_refresh_tool_prompts()`
2. Static tool policy from `prompts/fragments/tool_selection.md`
3. Interactive execution notes (conditionally injected)
4. `<active_tools ...>` index in structured context XML
5. Function-calling schemas from `get_json_schemas(...)`

## Built-in Tool Groups

Total groups: **6**

### Group: `system`

| Tool | Schema Description |
|---|---|
| `state_update` | Unified planning/task state update tool. Use only when task state has actually changed. |
| `run_subtask` | Run a bounded sequential action list inside one subtask. Each action must be `{tool, params}`. Nested `run_subtask/state_update/parallel_tool_call` is disallowed. |
| `parallel_tool_call` | Bundle independent tool calls into one wrapper call. Only use when calls have no prerequisite dependency, do not write the same target, and failures do not affect each other. |

### Group: `core`

| Tool | Schema Description |
|---|---|
| `read_file` | Use for precise, bounded file reads after target narrowing (line range, offset/limit, hit_id). Avoid whole-repository discovery with this tool. |
| `apply_patch` | Apply a structured minimal patch once you have a direct edit target; exact context matching required. |
| `execute_command` | Use for short-lived build/test/git/runtime orchestration and one-shot verification commands. Avoid REPL/long-running foreground sessions. |
| `list_directory` | Structured directory discovery (terminal-backed listing with depth control). |
| `search_in_files` | Structured grep+glob search with `hit_id` handoff to `read_file`. |

### Group: `terminal`

| Tool | Schema Description |
|---|---|
| `create_terminal` | Create a legacy named terminal session. |
| `run_in_terminal` | Run command in legacy named terminal. |
| `read_terminal` | Read output from a legacy terminal created by `create_terminal`. |
| `wait_terminal_until` | Wait on a legacy terminal until exit or pattern match. |
| `read_terminal_since_last` | Read incremental output from a legacy terminal. |
| `run_and_wait` | Run command in legacy terminal and wait for completion/pattern. |
| `exec_command` | Start interactive terminal session (returns `exec_xxx` session id). |
| `write_stdin` | Write input to an interactive `exec_command` session. |
| `close_exec_session` | Close an interactive `exec_command` session. |
| `check_syntax` | Syntax check helper. |
| `launch_interactive_session` | Shortcut for interactive session (`exec_command(tty=true)`). |

### Group: `tui`

| Tool | Schema Description |
|---|---|
| `open_tui` | Start full-screen TUI and return first-screen excerpt. |
| `read_tui` | Read TUI screen snapshot. |
| `read_tui_diff` | Read semantic diff since last TUI observation. |
| `read_tui_region` | Read a focused row range from current TUI screen. |
| `find_text_in_tui` | Find anchor text in current TUI screen. |
| `send_keys` | Send keys to TUI and return short post-action excerpt when available. |
| `send_keys_and_read` | Send keys and immediately return changed rows plus after excerpt. |
| `wait_tui_until` | Wait until text appears or screen hash changes. |
| `close_tui` | Close TUI session. |
| `activate_tui_mode` | Explicitly enable TUI interaction guidance. |

### Group: `network_and_extra`

| Tool | Schema Description |
|---|---|
| `web_search` | Web search (requires configured provider key). |
| `browser_open` | Open URL in browser tool. |
| `browser_view` | Read page/selector content. |
| `browser_act` | Browser actions such as click/type/press/hover. |
| `browser_scroll` | Scroll page. |
| `browser_screenshot` | Capture browser screenshot. |
| `browser_console_logs` | Read browser console logs. |

### Group: `skills`

| Tool | Schema Description |
|---|---|
| `load_skill` | Load `SKILL.md` content into current context on demand. |

## Notes

- `write_to_file`, `replace_in_file`, `create_directory`, `preview_file`, and group `extended_fs` are removed from built-in tool groups and schemas.
- Runtime enforces active-tool execution; inactive tools are rejected.


