"""Agent state schema for LangGraph runtime."""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Global state object shared by graph nodes."""

    # Conversation
    messages: Annotated[List[BaseMessage], add_messages]
    thread_id: str

    # Runtime/task execution
    env_snapshot: str
    pending_tool_calls: List[Dict[str, Any]]
    iteration_count: int
    working_dir: str
    permission_mode: str
    allow_shell_commands: bool
    task_completed: bool
    last_action: str
    empty_response_count: int
    stall_rounds: int
    stall_round_threshold: int
    tui_stall_rounds: int
    tui_stall_threshold: int
    last_tui_screen_hash: str
    tui_last_status_tokens: List[str]

    # Token usage
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int

    # Task goals
    overall_goal: str
    current_focus: str
    initial_user_request: str
    has_new_user_input: bool
    current_task: str
    tool_protocol_mode: str

    # Context
    context_summary: str
    max_context_tokens: int
    progress_summary: str
    task_type: str
    context: str
    user_input: str
    reasoning_content: str
    context_summarized: bool

    # Project/docs
    klynx_docs: str
    project_rules: str

    # Completion / convergence
    ended_without_tools: bool
    command_executions: List[Dict[str, Any]]
    needs_user_confirmation: bool
    clarification_question: str
    apply_patch_failure_streak: int

    # Skills / TUI
    loaded_skill_names: List[str]
    skill_context: str
    skill_preload_warnings: List[str]
    skill_injection_mode: str
    tui_guide_loaded: bool
    active_tool_groups: List[str]
    active_tool_names: List[str]
    active_skill_names: List[str]
    active_skill_paths: List[str]
    skill_registry_digest: str

    # Prompt control
    thinking_context: bool
    system_prompt_append: str

    # Convergence / dedupe
    dedupe_tools: bool
    tool_dedupe_window: int
    soft_loop_confirmation_enabled: bool
    soft_doom_loop_threshold: int
    soft_repeated_read_threshold: int
    soft_reads_per_path_threshold: int
    tool_call_history: List[Dict[str, Any]]
    tool_ledger_recent: List[Dict[str, Any]]
    tool_reuse_cache: Dict[str, Dict[str, Any]]
    tool_dedupe_hits: int
    repeated_read_hits: int
    read_file_failure_streaks: Dict[str, int]

    # Read coverage / evidence
    read_coverage: Dict[str, Any]
    file_views: Dict[str, Any]
    active_file_view_paths: List[str]
    last_read_chunks: List[Dict[str, Any]]
    last_read_fingerprint: str
    evidence_index: List[Dict[str, Any]]
    evidence_digest: str
    search_hits_index: List[Dict[str, Any]]
    last_search_backend: str
    file_candidates: List[Dict[str, Any]]
    trusted_modified_files: List[str]
    last_patch_summaries: List[Dict[str, Any]]
    last_mutation: Dict[str, Any]
    recent_mutations: List[Dict[str, Any]]
    pending_verification_targets: List[str]
    mutation_truth_digest: str
    recent_terminal_events: List[Dict[str, Any]]
    recent_tui_events: List[Dict[str, Any]]
    recent_exec_sessions: List[Dict[str, Any]]
    tui_views: Dict[str, Any]
    active_tui_view_names: List[str]
    last_tui_snapshots: List[Dict[str, Any]]
    tui_verification_targets: List[Dict[str, Any]]
    last_tui_verification: Dict[str, Any]
    recent_tui_verifications: List[Dict[str, Any]]
    tui_verification_digest: str
    command_verification_targets: List[Dict[str, Any]]
    last_command_verification: Dict[str, Any]
    recent_command_verifications: List[Dict[str, Any]]
    command_verification_digest: str
    active_terminal_op: Dict[str, Any]
    active_tui_focus: Dict[str, Any]
    active_exec_session: Dict[str, Any]
    last_terminal_delta: Dict[str, Any]
    last_terminal_failure: Dict[str, Any]
    last_exec_output: Dict[str, Any]
    last_tui_diff: Dict[str, Any]
    last_tui_region: Dict[str, Any]
    last_tui_anchor_match: Dict[str, Any]

    # Planning state
    should_plan: bool
    task_plan: List[Dict[str, Any]]
    current_step_id: str
    completed_steps: List[str]
    blocked_reason: str
    convergence_mode: str
    convergence_reason: str
    next_step_requirements: List[str]
    max_tools_per_step: int
    max_reads_per_file_per_step: int
    max_retry_per_tool_per_step: int
    step_execution_stats: Dict[str, Any]

    # Tool output artifacts
    tool_artifacts: List[Dict[str, Any]]

    # Recoverable summaries and subtask records
    summary_events: List[Dict[str, Any]]
    subtask_history: List[Dict[str, Any]]


