"""
Actoviq agent orchestrations.

Define one subclass per concrete agent to customize behavior.

`LOCKED_SYSTEM_PROMPT` is framework-owned and should not be overridden for the
main agent loop. Prompt extension is append-only via constructor/invoke params.
"""

from pathlib import Path
import os

from langgraph.graph import END, StateGraph

from ...graph import GraphKlynxAgent
from ...state import AgentState
from ...subgraphs import build_actoviq_initial_state, stream_ask


def _load_prompt_asset(*relative_parts: str, fallback: str = "") -> str:
    prompt_path = Path(__file__).resolve().parents[2].joinpath(*relative_parts)
    try:
        return prompt_path.read_text(encoding="utf-8").strip()
    except Exception:
        return fallback.strip()


class KlynxGeneralAgent(GraphKlynxAgent):
    """Generic Actoviq orchestration with the baseline system prompt."""

    LOCKED_SYSTEM_PROMPT = _load_prompt_asset(
        "prompts",
        "system_base.md",
        fallback=(
            "# Role\n\n"
            "You are Actoviq, a task-oriented software engineering agent."
        ),
    )

    ASK_SYSTEM_PROMPT = _load_prompt_asset(
        "prompts",
        "ask_base.md",
        fallback=(
            "You are Actoviq Ask Assistant. "
            "Answer the user directly and concisely."
        ),
    )

    def __init__(self, *args, append_system_prompt: str = "", **kwargs):
        """
        Method 1: inject appended prompt via constructor.
        """
        super().__init__(*args, **kwargs)
        self._init_system_prompt_append = (append_system_prompt or "").strip()
        self._runtime_system_prompt_append = ""

    def _inject_system_prompt_node(self, state: AgentState) -> dict:
        """
        Inject user-appendable system prompt extension.

        The core agent prompt stays locked. Users can append extra instructions
        in two ways:
        1) constructor `append_system_prompt`
        2) per-call `invoke(..., system_prompt_append=...)`
        """
        init_append = (getattr(self, "_init_system_prompt_append", "") or "").strip()
        call_append = (state.get("system_prompt_append", "") or "").strip()

        merged_parts = [p for p in (init_append, call_append) if p]
        merged_append = "\n\n".join(merged_parts).strip()

        self._runtime_system_prompt_append = merged_append
        if merged_append:
            self._emit("info", "[Prompt] ")
        return {"system_prompt_append": merged_append}

    def _build_graph(self) -> StateGraph:
        """
         LangGraph .
        :init -> inject_system_prompt -> load_context -> agent -> act -> feedback
        -> agent/end().
        """
        workflow = StateGraph(AgentState)

        workflow.add_node("init", self._init_node)
        workflow.add_node("inject_system_prompt", self._inject_system_prompt_node)
        workflow.add_node("load_context", self._load_context_node)
        workflow.add_node("agent", self._model_inference_node)
        workflow.add_node("act", self._act_node)
        workflow.add_node("feedback", self._feedback_node)

        workflow.set_entry_point("init")
        workflow.add_edge("init", "inject_system_prompt")
        workflow.add_edge("inject_system_prompt", "load_context")
        workflow.add_edge("load_context", "agent")
        workflow.add_edge("agent", "act")
        workflow.add_edge("act", "feedback")
        workflow.add_conditional_edges(
            "feedback",
            self._should_continue,
            {
                "agent": "agent",
                "tools": "act",
                "end_direct": END,
            },
        )
        return workflow

    def ask(self, message: str, system_prompt: str = None, thread_id: str = "default"):
        return stream_ask(
            self,
            message,
            system_prompt=system_prompt,
            thread_id=thread_id,
        )

    def invoke(
        self,
        task: str,
        thread_id: str = "default",
        thinking_context: bool = False,
        system_prompt_append: str = "",
    ):
        """
        ,().
        """
        state_config = self._build_run_config(
            thread_id=thread_id,
            recursion_limit=None,
            include_pending_rollback=False,
        )
        config = self._build_run_config(
            thread_id=thread_id,
            recursion_limit=2000,
            include_pending_rollback=True,
        )
        pending_rollback = self.get_pending_rollback(thread_id=thread_id)
        pending_checkpoint_id = str(pending_rollback.get("checkpoint_id", "") or "").strip()
        pending_once = bool(pending_rollback.get("once", True))
        rollback_consumed = [False]

        initial_state = build_actoviq_initial_state(
            self,
            task,
            thread_id=thread_id,
            thinking_context=thinking_context,
            system_prompt_append=system_prompt_append,
        )

        self._event_buffer.clear()
        if hasattr(self, "_event_signal"):
            self._event_signal.clear()

        if not hasattr(self, "_cancel_event"):
            import threading

            self._cancel_event = threading.Event()
        else:
            self._cancel_event.clear()

        import threading
        graph_done = threading.Event()
        graph_error = [None]

        def _run_graph():
            try:
                for _ in self.app.stream(initial_state, config=config):
                    if pending_checkpoint_id and pending_once and not rollback_consumed[0]:
                        rollback_consumed[0] = self._consume_pending_rollback(
                            thread_id=thread_id,
                            expected_checkpoint_id=pending_checkpoint_id,
                        )
                    if (
                        getattr(self, "_cancel_event", None)
                        and self._cancel_event.is_set()
                    ):
                        self._emit(
                            "warning", "[System] Task interrupted by user (Ctrl+C)."
                        )
                        break
                if pending_checkpoint_id and pending_once and not rollback_consumed[0]:
                    rollback_consumed[0] = self._consume_pending_rollback(
                        thread_id=thread_id,
                        expected_checkpoint_id=pending_checkpoint_id,
                    )
            except Exception as e:
                import traceback

                traceback.print_exc()
                graph_error[0] = e
            finally:
                graph_done.set()

        thread = threading.Thread(target=_run_graph, daemon=True)
        thread.start()

        poll_interval_s = 0.01
        try:
            poll_interval_s = max(
                0.005,
                float(os.getenv("KLYNX_EVENT_POLL_INTERVAL_S", "0.01") or "0.01"),
            )
        except Exception:
            poll_interval_s = 0.01

        while not graph_done.is_set() or self._event_buffer:
            if (
                getattr(self, "_cancel_event", None)
                and self._cancel_event.is_set()
                and not graph_done.is_set()
            ):
                break
            if self._event_buffer:
                yield self._event_buffer.popleft()
                continue
            if graph_done.is_set():
                break
            event_signal = getattr(self, "_event_signal", None)
            if event_signal is not None:
                event_signal.wait(poll_interval_s)
                event_signal.clear()
            else:
                import time

                time.sleep(min(0.005, poll_interval_s))

        if graph_error[0]:
            self._emit("error", f"[System Error] {str(graph_error[0])}")

        while self._event_buffer:
            yield self._event_buffer.popleft()

        if getattr(self, "_cancel_event", None) and self._cancel_event.is_set():
            yield {"type": "warning", "content": "\n[] "}

            current_state = self.app.get_state(state_config)
            values = current_state.values if current_state else {}

            prompt_tokens = int(values.get("prompt_tokens", 0) or 0)
            completion_tokens = int(values.get("completion_tokens", 0) or 0)
            total_tokens = int(values.get("total_tokens", 0) or 0)
            if total_tokens <= 0 and (prompt_tokens > 0 or completion_tokens > 0):
                total_tokens = prompt_tokens + completion_tokens

            yield {
                "type": "done",
                "content": "cancelled",
                "iteration_count": values.get("iteration_count", 0),
                "task_completed": False,
                "total_tokens": total_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
            return

        final_state = self.app.get_state(state_config)
        values = final_state.values if final_state else {}
        final_completed = bool(values.get("task_completed", False))
        needs_user_confirmation = bool(values.get("needs_user_confirmation", False))
        if (
            not final_completed
            and values.get("ended_without_tools", False)
            and not needs_user_confirmation
        ):
            final_completed = True
        yield {
            "type": "done",
            "content": "",
            "iteration_count": values.get("iteration_count", 0),
            "task_completed": final_completed,
            "total_tokens": values.get("total_tokens", 0),
            "prompt_tokens": values.get("prompt_tokens", 0),
            "completion_tokens": values.get("completion_tokens", 0),
        }


class KlynxAgent(KlynxGeneralAgent):
    """Public default Actoviq agent."""

    pass

