"""
Basic Actoviq library example.

This example uses klynx as an installed library and keeps the default
minimal tool surface (`system` + `core`).

Examples:
1) Interactive mode:
   python tutorials/examples/basic_agent_example.py
2) One-shot mode:
   python tutorials/examples/basic_agent_example.py --task "List files in current directory"
"""

from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

from actoviq import create_agent, setup_model



DEFAULT_PROVIDER = "deepseek"
DEFAULT_MODEL = "deepseek-chat"
PROVIDER_API_ENV = {
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "kimi": "MOONSHOT_API_KEY",
}


def _build_agent(
    working_dir: str,
    provider: str,
    model_name: str,
    max_iterations: int,
    api_key_env: str | None = None,
):
    env_name = api_key_env or PROVIDER_API_ENV.get(provider, "DEEPSEEK_API_KEY")
    api_key = os.getenv(env_name, "")
    if not api_key:
        raise ValueError(f"Missing API key: set {env_name}.")

    model = setup_model(provider, model_name, api_key)
    agent = create_agent(
        working_dir=working_dir,
        model=model,
        max_iterations=max_iterations,
        memory_dir=working_dir,
        load_project_docs=False,
    )

    mcp_config = Path(working_dir) / ".klynx" / "mcp_servers.json"
    if mcp_config.exists():
        agent.add_mcp(str(mcp_config))
    return agent


def _stream_once(agent, task: str, thread_id: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    printed_reasoning_header = False
    printed_answer_header = False
    streamed_reasoning = False
    streamed_answer = False

    for event in agent.invoke(task=task, thread_id=thread_id):
        etype = event.get("type", "")
        content = event.get("content", "")

        if etype == "done":
            result = event
        elif etype == "reasoning_token":
            if not printed_reasoning_header:
                print("\n[Reasoning]")
                printed_reasoning_header = True
            print(content, end="", flush=True)
            streamed_reasoning = True
        elif etype == "token":
            if not printed_answer_header:
                print("\n[Answer]")
                printed_answer_header = True
            print(content, end="", flush=True)
            streamed_answer = True
        elif etype == "reasoning" and not streamed_reasoning:
            print(f"\n[Reasoning]\n{content}")
        elif etype == "answer" and not streamed_answer:
            print(f"\n[Answer]\n{content}")
        elif etype in {"summary", "tool_exec", "tool_result", "tool_calls", "warning", "error"}:
            print(content)
        elif etype == "iteration":
            print(f"\n{content}")

    if printed_reasoning_header or printed_answer_header:
        print("")
    return result


def run_one_shot(agent, task: str, thread_id: str) -> None:
    print(f"\n[Task] {task}")
    print("-" * 60)
    result = _stream_once(agent, task, thread_id)
    print("-" * 60)
    print(f"[Completed] {result.get('task_completed', False)}")
    print(f"[Iterations] {result.get('iteration_count', 0)}")
    print(f"[Tokens] {result.get('total_tokens', 0)}")


def run_interactive(agent) -> None:
    print("\n" + "=" * 60)
    print("Actoviq Basic Library Example")
    print("Commands: exit | quit | clear | context")
    print("=" * 60)

    thread_id = str(uuid.uuid4())[:8]
    round_count = 0
    total_tokens = 0

    while True:
        try:
            user_input = input(f"[Round {round_count + 1}] > ").strip()
            if user_input.lower() in ("exit", "quit", "q"):
                print(f"\n[Stats] rounds={round_count}, tokens={total_tokens}")
                break
            if not user_input:
                continue
            if user_input.lower() == "clear":
                thread_id = str(uuid.uuid4())[:8]
                round_count = 0
                total_tokens = 0
                print(f"[System] context cleared, new thread={thread_id}\n")
                continue
            if user_input.lower() == "context":
                state_values = agent.get_context(thread_id)
                if not state_values:
                    print(f"[Context] thread={thread_id}, empty\n")
                    continue
                msgs = state_values.get("messages", [])
                approx_tokens = 0
                try:
                    from actoviq.agent.context_manager import TokenCounter

                    approx_tokens = TokenCounter.count_message_tokens(msgs) if msgs else 0
                except Exception:
                    char_count = sum(
                        len(m.content)
                        for m in msgs
                        if hasattr(m, "content") and isinstance(m.content, str)
                    )
                    approx_tokens = char_count // 2
                print(
                    f"[Context] thread={thread_id}, messages={len(msgs)}, "
                    f"approx_tokens={approx_tokens}\n"
                )
                continue

            print("-" * 60)
            result = _stream_once(agent, user_input, thread_id)
            round_count += 1
            round_tokens = int(result.get("total_tokens", 0) or 0)
            total_tokens += round_tokens
            print("-" * 60)
            print(f"[Round End] completed={result.get('task_completed', False)}")
            print(
                f"[Usage] iterations={result.get('iteration_count', 0)}, "
                f"round_tokens={round_tokens}, total_tokens={total_tokens}\n"
            )
        except KeyboardInterrupt:
            print(f"\n[Interrupted] rounds={round_count}, tokens={total_tokens}")
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Basic Actoviq library example.")
    parser.add_argument("--task", default="", help="Run one-shot task and exit.")
    parser.add_argument("--working-dir", default=os.getcwd(), help="Agent working directory.")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, help="Model provider.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name.")
    parser.add_argument("--max-iterations", type=int, default=50, help="Agent max iterations.")
    parser.add_argument("--api-key-env", default="", help="API key environment variable name.")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    agent = _build_agent(
        working_dir=args.working_dir,
        provider=args.provider,
        model_name=args.model,
        max_iterations=args.max_iterations,
        api_key_env=args.api_key_env or None,
    )

    thread_id = str(uuid.uuid4())[:8]
    if args.task:
        run_one_shot(agent, args.task, thread_id)
    else:
        run_interactive(agent)


if __name__ == "__main__":
    main()


