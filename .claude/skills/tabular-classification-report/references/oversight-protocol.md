# Oversight log protocol

`<RUN>/oversight_log.md` is the audit trail of who decided what. It is the raw material for
the reflection, so write entries as they happen, not afterwards.

```markdown
# Oversight log - <run name>
- Invocation: /tabular-classification-report <args>
- Date: YYYY-MM-DD; model: <model id>; interface: <harness + version>
- Human policy: <e.g. "human approves the design up front; checkpoints delegated to reviewer agents; human accepts final report">

## [stage] short title
- Actor: human | agent | reviewer-agent | script
- Input: what was looked at (file, numbers)
- Decision / finding: ...
- Response: accepted -> change made (file, field) | rejected -> reason
```

Rules
- Every checkpoint produces at least one entry, even if the outcome is "no change".
- Record re-runs with the reason and the CV evidence that triggered them.
- Record anything you were tempted to change after seeing test results, and that you did not.
- Never edit earlier entries; append corrections.
