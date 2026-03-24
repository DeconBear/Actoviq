# Actoviq Python SDK for Agent Development

**Actoviq: Intelligence in Action**

**Language / 语言**: [English](./README.md) | [简体中文](./README-zh.md)

Actoviq 是一个用于构建代码智能体的 Python 框架，默认采用 `think -> act -> feedback` 循环。  
它支持：

- 开箱即用的默认 Agent 运行时
- 可组合的图构建 API（用于自定义编排）

### 安装（PyPI）

```bash
pip install -U actoviq
```

可选浏览器能力（仅在工具需要浏览器自动化时使用）：

```bash
playwright install chromium
```

### 快速开始

#### 1）设置模型 API Key

示例（OpenAI）：

```bash
export OPENAI_API_KEY="sk-..."
```

#### 2）通过 builder 创建模型与运行时

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

#### 3）流式执行任务并读取最终答案

```python
for event in agent.invoke(task="Summarize this repository architecture", thread_id="demo"):
    if event.get("type") == "done":
        print(event.get("answer", ""))
```

### API 导出

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

### 使用教程

#### 教程 1：直接问答模式（`builder.ask` 预设）

```python
from actoviq import create_builder, setup_model

model = setup_model("gpt-5.3")
builder = create_builder(name="ask-demo").ask()
agent = builder.build(working_dir=".", model=model)

for event in agent.ask("Explain the key modules in this codebase", thread_id="ask-demo"):
    if event.get("type") == "done":
        print(event.get("answer", ""))
```

#### 教程 2：使用 `create_builder` 构建可组合运行时

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

#### 教程 3：在默认循环后追加自定义节点

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

#### 教程 4：动态管理工具集

默认 Agent 与 builder 运行时都支持工具集合增删：

```python
runtime.add_tools("group:core")
runtime.add_tools("group:terminal")
runtime.add_tools("group:tui")
runtime.add_tools("group:network_and_extra")
runtime.add_tools("none")
```

### 内置工具组

Actoviq 内置 6 个工具组：

- `system`: `state_update`, `run_subtask`, `parallel_tool_call`
- `core`: `read_file`, `apply_patch`, `execute_command`, `list_directory`, `search_in_files`
- `terminal`: `create_terminal`, `run_in_terminal`, `read_terminal`, `wait_terminal_until`, `read_terminal_since_last`, `run_and_wait`, `exec_command`, `write_stdin`, `close_exec_session`, `check_syntax`, `launch_interactive_session`
- `tui`: `open_tui`, `read_tui`, `read_tui_diff`, `read_tui_region`, `find_text_in_tui`, `send_keys`, `send_keys_and_read`, `wait_tui_until`, `close_tui`, `activate_tui_mode`
- `network_and_extra`: `web_search`, `browser_open`, `browser_view`, `browser_act`, `browser_scroll`, `browser_screenshot`, `browser_console_logs`
- `skills`: `load_skill`

默认加载行为：

- 默认启动时加载 `group:system` 与 `group:core`
- `load_skill` 是否可见由 `skill_injection_mode` 控制：`preload` 隐藏，`hybrid` 与 `tool` 暴露

#### 教程 5：权限模式（`workspace` / `global`）

```python
agent = create_builder(name="perm").react().build(working_dir=".", model=model)

# 默认: workspace 沙箱
print(agent.get_permission())

# 切换到 global
agent.set_permission("global")

# 切回 workspace 沙箱
agent.set_sandbox(True)
```

#### 教程 6：启用 Web Search 工具

```python
from actoviq import set_tavily_api, is_tavily_configured

set_tavily_api("tvly-...")
print(is_tavily_configured())  # True
```

若未配置 Tavily API Key，`web_search` 会从工具组和 JSON schema 中移除。

#### 教程 7：技能注入模式

```python
agent = create_builder(name="skills").react().build(
    working_dir=".",
    model=model,
    skill_injection_mode="hybrid",  # preload | tool | hybrid
)
```

- `preload`：根据用户输入预加载 `SKILL.md`，隐藏 `load_skill`
- `tool`：关闭预加载，仅通过 `load_skill`
- `hybrid`（默认）：先预加载，保留 `load_skill` 兜底

#### 教程 8：回滚到检查点

你可以查看线程的检查点历史并选择一个目标回滚。  
默认是一次性回滚：下一次 `invoke/ask` 会从目标检查点继续。

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

# 回滚到指定展示序号（最新优先）
agent.rollback(thread_id=thread_id, target_index=1)

# 可选：同时恢复由工具记录的文件变更
# agent.rollback(thread_id=thread_id, target_index=1, with_files=True)

# 下一轮会从回滚后的状态继续
list(agent.invoke("task C after rollback", thread_id=thread_id))

# 可选：取消待执行回滚
agent.cancel_rollback(thread_id=thread_id)
```

说明：
- 检查点回滚恢复的是 Agent 会话状态，不保证恢复所有外部副作用
- `with_files=True` 会恢复通过变更工具（如 `apply_patch`）记录的文件修改
- 对于工具之外的 Shell 改动，不在自动恢复范围内

### 事件模型

`invoke(...)` 与 `ask(...)` 会输出事件字典，常见事件类型包括：

- `token`
- `reasoning_token`
- `tool_exec`
- `tool_result`
- `warning`
- `error`
- `done`

`done` 事件通常包含最终答案与 token 统计信息。

### 模型配置说明

`setup_model(...)` 同时支持 alias 和 provider/model 两种调用方式：

```python
setup_model("gpt-4o")
setup_model("openai", "gpt-4o")
setup_model("deepseek", "deepseek-chat")
```

可通过 `list_models()` 查看可用别名。

### 终端辅助函数

仅终端场景可使用：

- `run_terminal_agent_stream(...)`
- `run_terminal_ask_stream(...)`

### 相关包

如果你需要完整的命令行与 TUI 体验，可安装：

```bash
pip install -U actoviq-cli
```


