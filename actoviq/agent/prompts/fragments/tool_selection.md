## Tool Selection

- Use `execute_command` for short-lived search, build, test, git inspection,
  and verification commands.
- Use `list_directory` for structured directory discovery when you need
  machine-readable path/type/depth output.
- Use `search_in_files` for structured grep+glob style code search, including
  `hit_id`-driven follow-up reads.
- Use `read_file` only after narrowing the target. Read the smallest relevant
  slice.
- Use `apply_patch` for edits. After a patch failure, treat `hunk_mismatch` as
  an exact-context problem, not a syntax problem. Read the exact slice first,
  keep stable context lines, and remember that `@@` line numbers are
  separators only.
- Use `exec_command` or `launch_interactive_session` for REPLs, long-running
  foreground programs, and interactive shells.
- For interactive session follow-up, use only the continuation/close tools
  that appear in the active tool list.
- Use TUI tools only when screen-level observation is genuinely required.
- Prefer preloaded skill context when available; use `load_skill` only as fallback in hybrid/tool modes.
