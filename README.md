# Actoviq Python SDK for Agent Development

**Actoviq: Intelligence in Action**

**Language / 语言**: [English](./README.md) | [简体中文](./README-zh.md)

Actoviq is a Python framework for building coding agents with a default `think -> act -> feedback` loop.  
It supports both:

- A ready-to-use default agent runtime
- A composable graph builder API for custom orchestration

### Install (PyPI)

```bash
pip install -U actoviq
```

Optional browser tooling (needed only if your tools use browser automation):

```bash
playwright install chromium
```

### Quick Start

#### 1) Set your model API key

Example (OpenAI):

```bash
export OPENAI_API_KEY="sk-..."
```

#### 2) Create a model and runtime via builder

```python
from actoviq import create_builder, setup_model

model = setup_model("gpt-4o")
builder = create_builder(name="quickstart").react()
agent = builder.build(
    working_dir=".",
    model=model,
    max_iterations=20,
    load_project_docs=False,
)
```

#### 3) Stream a task and read the final answer

```python
for event in agent.invoke(task="Summarize this repository architecture", thread_id="demo"):
    if event.get("type") == "done":
        print(event.get("answer", ""))
```

### API Exports

```python
from actoviq import (
    create_agent,
    create_builder,
    KlynxAgent,
    KlynxGraphBuilder,
    ComposableAgentRuntime,
    setup_model,
    list_models,
    set_tavily_api,
    is_tavily_configured,
    run_terminal_agent_stream,
    run_terminal_ask_stream,
)
```

### Usage Tutorial

#### Tutorial 1: Direct Q&A mode (builder.ask preset)

```python
from actoviq import create_builder, setup_model

model = setup_model("gpt-5.3")
builder = create_builder(name="ask-demo").ask()
agent = builder.build(working_dir=".", model=model)

for event in agent.ask("Explain the key modules in this codebase", thread_id="ask-demo"):
    if event.get("type") == "done":
        print(event.get("answer", ""))
```

#### Tutorial 2: Build a composable runtime with `create_builder`

```python
from actoviq import create_builder, setup_model

model = setup_model("gpt-4o")

builder = create_builder(name="demo_builder")
builder.react()

runtime = builder.build(
    working_dir=".",
    model=model,
    max_iterations=12,
)

for event in runtime.invoke(task="Find TODOs and propose fixes", thread_id="builder-demo"):
    if event.get("type") == "done":
        print(event.get("answer", ""))
```

#### Tutorial 3: Add custom nodes after the default loop

```python
from actoviq import create_builder, setup_model

model = setup_model("gpt-4o")

def post_process(runtime, payload):
    return [{"type": "summary", "content": "Post-processing completed."}]

builder = create_builder(name="pipeline")
builder.react()
builder.add_node("post_process", post_process)
builder.add_edge("actoviq_loop", "post_process")

runtime = builder.build(working_dir=".", model=model)
for event in runtime.invoke(task="Refactor this module", thread_id="pipeline-demo"):
    print(event)
```

#### Tutorial 4: Manage toolsets dynamically

Both the default agent and builder runtime support tool mutation:

```python
runtime.add_tools("group:core")
runtime.add_tools("group:terminal")
runtime.add_tools("group:tui")
runtime.add_tools("group:network_and_extra")
runtime.add_tools("none")
```

### Built-in Tool Groups

Actoviq has 6 built-in tool groups:

- `system`: `state_update`, `run_subtask`, `parallel_tool_call`
- `core`: `read_file`, `apply_patch`, `execute_command`, `list_directory`, `search_in_files`
- `terminal`: `create_terminal`, `run_in_terminal`, `read_terminal`, `wait_terminal_until`, `read_terminal_since_last`, `run_and_wait`, `exec_command`, `write_stdin`, `close_exec_session`, `check_syntax`, `launch_interactive_session`
- `tui`: `open_tui`, `read_tui`, `read_tui_diff`, `read_tui_region`, `find_text_in_tui`, `send_keys`, `send_keys_and_read`, `wait_tui_until`, `close_tui`, `activate_tui_mode`
- `network_and_extra`: `web_search`, `browser_open`, `browser_view`, `browser_act`, `browser_scroll`, `browser_screenshot`, `browser_console_logs`
- `skills`: `load_skill`

Default loading behavior:

- Default agent startup loads `group:system` and `group:core`.
- `load_skill` availability is controlled by `skill_injection_mode`: `preload` hides it, `hybrid` and `tool` expose it.

#### Tutorial 5: Permission modes (`workspace` / `global`)

```python
agent = create_builder(name="perm").react().build(working_dir=".", model=model)

# default: workspace sandbox
print(agent.get_permission())

# global mode
agent.set_permission("global")

# back to workspace sandbox
agent.set_sandbox(True)
```

#### Tutorial 6: Enable web search tools

```python
from actoviq import set_tavily_api, is_tavily_configured

set_tavily_api("tvly-...")
print(is_tavily_configured())  # True
```

If Tavily API key is not configured, `web_search` is removed from tool groups and JSON schemas.

#### Tutorial 7: Skill injection modes

```python
agent = create_builder(name="skills").react().build(
    working_dir=".",
    model=model,
    skill_injection_mode="hybrid",  # preload | tool | hybrid
)
```

- `preload`: preload `SKILL.md` from user input hints, hide `load_skill`.
- `tool`: disable preload, rely on `load_skill`.
- `hybrid` (default): preload first, keep `load_skill` fallback.

#### Tutorial 8: Roll back to a checkpoint

You can inspect checkpoint history for a thread and select a rollback target.  
Rollback is one-shot by default: the next `invoke/ask` resumes from the selected checkpoint.

```python
from actoviq import create_builder, setup_model

model = setup_model("gpt-4o")
agent = create_builder(name="rollback").react().build(
    working_dir=".",
    model=model,
    load_project_docs=False,
)

thread_id = "rollback-demo"
list(agent.invoke("task A", thread_id=thread_id))
list(agent.invoke("task B", thread_id=thread_id))

history = agent.get_history(thread_id=thread_id, limit=20)
for item in history:
    print(
        item["display_index"],
        item["checkpoint_id"][:12],
        item["iteration"],
        item["action"],
    )

# Roll back to a selected display index (latest first).
agent.rollback(thread_id=thread_id, target_index=1)

# Optional: include tool-managed file restore.
# agent.rollback(thread_id=thread_id, target_index=1, with_files=True)

# Next run resumes from rollback checkpoint.
list(agent.invoke("task C after rollback", thread_id=thread_id))

# Optional: clear a pending rollback if needed.
agent.cancel_rollback(thread_id=thread_id)
```

Notes:
- Checkpoint rollback restores agent session state, not all external side effects.
- `with_files=True` restores file edits recorded through mutation tools (`apply_patch`).
- Shell commands that mutate files outside these tools are out of scope for automatic restore.

### Event Model

`invoke(...)` and `ask(...)` produce event dictionaries. Common types include:

- `token`
- `reasoning_token`
- `tool_exec`
- `tool_result`
- `warning`
- `error`
- `done`

A `done` event usually contains the final answer and token metrics.

### Model Setup Notes

`setup_model(...)` supports both alias and provider/model usage:

```python
setup_model("gpt-4o")
setup_model("openai", "gpt-4o")
setup_model("deepseek", "deepseek-chat")
```

Use `list_models()` to inspect available aliases.

### Terminal Helpers

For terminal-only usage, you can use helper stream runners:

- `run_terminal_agent_stream(...)`
- `run_terminal_ask_stream(...)`

### Related Package

If you want a full command-line and TUI experience, install:

```bash
pip install -U actoviq-cli
```



