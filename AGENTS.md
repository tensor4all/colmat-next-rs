# AGENTS.md

Before acting, read the latest shared tensor4all agent rules:
`https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/index.md`.
If the network is unavailable, use the sibling checkout `../tensor4all-agent-rules/rules/index.md`.
Load only the rule files the task needs. If neither source is available, continue and state that
the shared rules were unavailable. `rules/common/provenance.md` applies whenever code is written
while referencing colmat-rs, faer, LAPACK, Eigen or any other third-party code.

The contribution workflow (issues, the PR body, gates, review, versions) is in
[`CONTRIBUTING.md`](CONTRIBUTING.md), which points to the unified policy in `tensor4all-meta`.

## Project overview

**Status: trial.** This repository is a trial of building the next-generation colmat design as a
new codebase. Whether development continues here or inside
[colmat-rs](https://github.com/tensor4all/colmat-rs) (the tensor4all fork of faer) has not been
decided; that decision belongs to the colmat maintainers. Nothing in this repository overrides
[#2](https://github.com/tensor4all/colmat-next-rs/issues/2) or `tensor4all-meta`
`docs/design/next-generation-dense-linear-algebra.md` (status: design under review), which hold the
goals, the candidate crate layout (`colmat-traits`, `colmat-core`, `colmat`, `colmat-xprec`) and
the decisions taken so far.

When you write in this repository, do not present a choice made here as a decision of the colmat
maintainers. Say "proposed" or "in this trial", and cite the source of anything stated as decided.

There is no code yet. Do not start an implementation unless an issue asks for it. Statements in
#2 about colmat-rs describe colmat-rs at the commit they cite, not this repository.

Working rules of this trial: [`REPOSITORY_RULES.md`](REPOSITORY_RULES.md). Design notes:
[`docs/design/`](docs/design/README.md).

## Checks

CI (`.github/workflows/ci.yml`) skips the Rust jobs while there is no `Cargo.toml`, and runs them
from the first crate on. Once a crate exists, the pre-push checklist is:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --profile ci
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features
cargo deny check bans
```
