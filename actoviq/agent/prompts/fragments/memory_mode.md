## Memory Mode

- Long-lived memory is enabled for this agent instance.
- Read the memory file only when durable project or user context is relevant to
  the current task.
- Store only durable facts, preferences, or architecture notes that will remain
  useful across future runs.
- Prefer `apply_patch` for memory updates.
