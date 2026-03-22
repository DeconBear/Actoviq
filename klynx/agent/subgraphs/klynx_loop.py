"""Built-in reusable Klynx loop subgraph helpers."""

from __future__ import annotations

from typing import Any, Dict, Iterable

from langchain_core.messages import HumanMessage


def build_klynx_initial_state(
    agent,
    task: str,
    *,
    thread_id: str = "default",
    thinking_context: bool = False,
    system_prompt_append: str = "",
) -> Dict[str, Any]:
    preloaded_skill_names = []
    preloaded_skill_context = ""
    skill_preload_warnings = []
    preload_skills = getattr(agent, "preload_skills_for_input", None)
    if callable(preload_skills):
        try:
            preloaded_skill_names, preloaded_skill_context, skill_preload_warnings = preload_skills(task)
        except Exception:
            preloaded_skill_names, preloaded_skill_context, skill_preload_warnings = [], "", []

    active_skill_paths = []
    skill_path_getter = getattr(agent, "get_skill_paths_for_names", None)
    if callable(skill_path_getter):
        try:
            active_skill_paths = list(skill_path_getter(preloaded_skill_names) or [])
        except Exception:
            active_skill_paths = []

    return {
        "messages": [HumanMessage(content=task)],
        "thread_id": thread_id,
        "env_snapshot": "",
        "pending_tool_calls": [],
        "iteration_count": 0,
        "working_dir": agent.working_dir,
        "task_completed": False,
        "last_action": "",
        "empty_response_count": 0,
        "stall_rounds": 0,
        "stall_round_threshold": 3,
        "tui_stall_rounds": 0,
        "tui_stall_threshold": int(getattr(agent, "tui_stall_threshold", 3) or 3),
        "last_tui_screen_hash": "",
        "tui_last_status_tokens": [],
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "max_context_tokens": getattr(agent, "max_context_tokens", 128000),
        "task_type": "",
        "overall_goal": task,
        "current_focus": task,
        "initial_user_request": task,
        "has_new_user_input": True,
        "current_task": "",
        "context_summary": "",
        "context_summarized": False,
        "context": "",
        "reasoning_content": "",
        "klynx_docs": "",
        "project_rules": "",
        "progress_summary": "",
        "user_input": task,
        "ended_without_tools": False,
        "command_executions": [],
        "needs_user_confirmation": False,
        "clarification_question": "",
        "apply_patch_failure_streak": 0,
        "loaded_skill_names": preloaded_skill_names,
        "skill_context": preloaded_skill_context,
        "skill_preload_warnings": skill_preload_warnings,
        "tui_guide_loaded": False,
        "active_tool_groups": list(getattr(agent, "active_tool_groups", []) or []),
        "active_tool_names": sorted(list(getattr(agent, "tools", {}).keys())),
        "active_skill_names": list(preloaded_skill_names),
        "active_skill_paths": active_skill_paths,
        "skill_registry_digest": str(getattr(agent, "skill_registry_digest", "") or ""),
        "skill_injection_mode": str(getattr(agent, "skill_injection_mode", "hybrid") or "hybrid"),
        "permission_mode": str(getattr(agent, "permission_mode", "workspace") or "workspace"),
        "allow_shell_commands": bool(getattr(agent, "allow_shell_commands", True)),
        "thinking_context": thinking_context,
        "system_prompt_append": (system_prompt_append or "").strip(),
        "tool_protocol_mode": str(getattr(agent, "tool_protocol_mode", "native")),
        "dedupe_tools": True,
        "tool_dedupe_window": 3,
        "soft_loop_confirmation_enabled": True,
        "soft_doom_loop_threshold": 3,
        "soft_repeated_read_threshold": 3,
        "soft_reads_per_path_threshold": 5,
        "tool_call_history": [],
        "tool_ledger_recent": [],
        "tool_reuse_cache": {},
        "tool_dedupe_hits": 0,
        "repeated_read_hits": 0,
        "read_file_failure_streaks": {},
        "read_coverage": {},
        "file_views": {},
        "active_file_view_paths": [],
        "last_read_chunks": [],
        "last_read_fingerprint": "",
        "evidence_index": [],
        "evidence_digest": "",
        "search_hits_index": [],
        "last_search_backend": "",
        "file_candidates": [],
        "trusted_modified_files": [],
        "last_patch_summaries": [],
        "last_mutation": {},
        "recent_mutations": [],
        "pending_verification_targets": [],
        "mutation_truth_digest": "",
        "recent_terminal_events": [],
        "recent_tui_events": [],
        "recent_exec_sessions": [],
        "tui_views": {},
        "active_tui_view_names": [],
        "last_tui_snapshots": [],
        "tui_verification_targets": [],
        "last_tui_verification": {},
        "recent_tui_verifications": [],
        "tui_verification_digest": "",
        "command_verification_targets": [],
        "last_command_verification": {},
        "recent_command_verifications": [],
        "command_verification_digest": "",
        "active_terminal_op": {},
        "active_tui_focus": {},
        "active_exec_session": {},
        "last_terminal_delta": {},
        "last_terminal_failure": {},
        "last_exec_output": {},
        "last_tui_diff": {},
        "last_tui_region": {},
        "last_tui_anchor_match": {},
        "should_plan": False,
        "task_plan": [],
        "current_step_id": "",
        "completed_steps": [],
        "blocked_reason": "",
        "convergence_mode": "normal",
        "convergence_reason": "",
        "next_step_requirements": [],
        "max_tools_per_step": int(getattr(agent, "max_tools_per_step", 20) or 20),
        "max_reads_per_file_per_step": int(
            getattr(agent, "max_reads_per_file_per_step", 6) or 6
        ),
        "max_retry_per_tool_per_step": int(
            getattr(agent, "max_retry_per_tool_per_step", 2) or 2
        ),
        "step_execution_stats": {},
        "tool_artifacts": [],
        "summary_events": [],
        "subtask_history": [],
    }


def run_klynx_loop_node(runtime, payload: Dict[str, Any]) -> Iterable[dict]:
    agent = runtime._ensure_loop_agent()
    task = str(payload.get("task", "") or "")
    thread_id = str(payload.get("thread_id", "default") or "default")
    invoke_kwargs = dict(payload.get("invoke_kwargs", {}) or {})
    return agent.invoke(task=task, thread_id=thread_id, **invoke_kwargs)


def build_klynx_loop_subgraph(builder, *, node_name: str = "klynx_loop"):
    builder.add_node(node_name)
    return builder


