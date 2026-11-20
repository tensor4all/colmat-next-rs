# AGENTS.md

Before acting, read the latest shared tensor4all agent rules:
`https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/index.md`.
If the network is unavailable, use the sibling checkout `../tensor4all-agent-rules/rules/index.md`.
Load only the rule files the task needs. If neither source is available, continue and state that
the shared rules were unavailable. `rules/common/provenance.md` applies whenever code is written
while referencing colmat-rs, faer, LAPACK, Eigen or any other third-party code.

The contribution workflow (issues, the PR body, gates, review, versions) is in
[`CONTRIBUTING.md`](CONTRIBUTING.md), which points to the unified policy in `tensor4all-meta`.

## Proportionate safety and validation

This is scientific-computing software, not a security product or trust boundary. Preserve Rust
memory safety, aliasing soundness, numerical correctness and reproducibility. Do not add security
machinery against a malicious maintainer, checkout, local build tool or CI runner unless a task
introduces that untrusted boundary. Every added validation step names the correctness failure or
operational mistake it prevents. This does not relax the checks required before unsafe memory
access or mutable aliasing.

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

Crates live under `crates/` (the first is `colmat-traits`, #6). Do not start an implementation unless
an issue asks for it. Statements in #2 about colmat-rs describe colmat-rs at the commit they cite,
not this repository.

Working rules of this trial: [`REPOSITORY_RULES.md`](REPOSITORY_RULES.md). Design notes:
[`docs/design/`](docs/design/README.md).

## Checks

CI (`.github/workflows/ci.yml`) runs the Rust gates on the toolchain pinned in `rust-toolchain.toml`
(a `RUSTUP_TOOLCHAIN` set in your shell overrides that file; unset it or set it to the pinned
version). The pre-push checklist is:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --profile ci
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features
cargo deny check bans
```

## Branches and pull requests

- Branch from the latest `main`: `git fetch origin && git checkout -b <branch> origin/main`. Never
  branch from stale local state or another feature branch unless that is intended.
- Work expected to produce several commits accumulates on one working branch and reaches `main` in
  one pull request once the batch is complete and verified locally. Do not open a pull request per
  commit or per correction: every push to an open pull request reruns CI.
- Use a git worktree under `.worktrees/` (ignored) for parallel work in one checkout.
- `main` is protected: no direct pushes, no force pushes. The required checks are `ci-gate` (all
  Rust gates and the CI budget, `.github/workflows/ci.yml`) and `body / body` (the PR description's
  `## Verification`, `.github/workflows/pr-verification.yml`). When you add or rename a CI job, add
  it to `ci-gate`'s `needs` in the same change, or it stops gating merges.
- Create pull requests to `main` with `gh pr create`. A draft stays a draft until a maintainer
  says otherwise. A ready pull request enables auto-merge:
  `gh pr merge --auto --squash --delete-branch`.
- No AI-generated analysis reports as standalone files in pull requests; link a work log under
  `docs/worklogs/` instead when one is needed.
