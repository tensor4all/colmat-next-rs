Closes #<!-- issue number; claim it (assign yourself) before you start -->

## Summary

<!-- One decision per pull request. Behaviour and performance changes go in separate PRs. -->

## Verification

<!-- At least one fenced block with a real command and its real output (the `body` check).
Every number says the command, host and commit. Unmeasured claims are written as
"Not measured: <what> — <why>". -->

```text
$ cargo test --workspace --profile ci
```

<!-- Line-initial, one line per model that wrote code, e.g. `Model: anthropic/claude-opus-5-5`.
Use `Model: none/human` if no AI was used. A PR without it is converted to draft. -->
Model:
