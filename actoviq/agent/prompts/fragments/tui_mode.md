## TUI Mode

- Use TUI tools only when full-screen or screen-diff observation is necessary.
- Prefer `send_keys_and_read`, `wait_tui_until`, `read_tui_region`,
  `read_tui_diff`, and `find_text_in_tui` over repeated full-screen reads.
- Treat screen change as UI evidence, not proof that the bug is fixed.
- State one explicit TUI assertion, collect before/after evidence, and mark it
  as pass or fail.
- Release TUI resources when the interaction is complete.
