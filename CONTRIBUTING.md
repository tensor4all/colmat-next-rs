<!-- unified contribution policy: ../../docs/contributing/rust-repositories.md in tensor4all/tensor4all-meta -->
# Contributing to colmat-next-rs

The rules that apply to every Rust repository in `tensor4all` are in one place:
**[tensor4all-meta · docs/contributing/rust-repositories.md](https://github.com/tensor4all/tensor4all-meta/blob/main/docs/contributing/rust-repositories.md)**.
Read that first — it covers the issue-first workflow, what a pull request body has to contain, red-first tests, the
gates every repository runs, the review requirement, and the version policy.

This file only records what is specific to `colmat-next-rs`:

## Local facts

- **MSRV**: 1.89 (`rust-version` in the workspace `Cargo.toml`, matching colmat-rs). The `msrv` job in
  `.github/workflows/ci.yml` checks it with `RUSTUP_TOOLCHAIN`, because `rust-toolchain.toml` pins the lint
  and test toolchain (1.96.0). `rustfmt.toml` is the stable subset of colmat-rs's settings.
- **Feature combinations CI builds**: default features (for `colmat-core`, `std` and `threadteam`), and
  `--all-features` for clippy and rustdoc; `colmat-core` also with no features (`cargo check`) and with `std` only
  (clippy over all targets), in the `deny` job (#20).
- **Platforms claimed**: `ubuntu-latest` and `macos-latest` (tests); rustdoc, clippy and `deny` run on Linux only.
  `ubuntu-latest` runs per PR; macOS is validated nightly, and on demand via `workflow_dispatch`, not per PR:
  The `schedule` trigger of `ci.yml` (and `workflow_dispatch`) adds the macOS rows; they are not run per PR, because macOS runners are costly and limited to five concurrent jobs across the organization. A failing scheduled run files an issue. The `budget` job does not apply to the scheduled run.
- **Local gate**: `cargo test --workspace --profile ci` once a crate exists.
- **Coverage**: `.github/workflows/coverage.yml` measures line coverage (`cargo llvm-cov`, `ubuntu-latest`, doctests
  not measured) weekly and on demand, and fails below a floor of 58.9 % of lines (#37). It does not gate merges:
  it is not part of `ci-gate`. `docs/coverage.md` has the command, the baseline and the floor.
- **CI time budget**: the push/PR pipeline has one total budget (`CI_BUDGET_MINUTES` in `.github/workflows/ci.yml`).
  The `budget` job fails when it is exceeded and lists the slowest jobs. Tests too slow for it are named `heavy_*`,
  marked `#[ignore = "heavy: ..."]`, and run nightly from `.github/workflows/heavy-tests.yml`, which files (or
  comments on) an issue titled "Scheduled heavy tests are failing" when it fails.

## Anything here that departs from the shared rules

None. Miri runs weekly (Sunday 18:00 UTC, #31) over the lib tests and doctests of `colmat-traits` and `colmat-core`
(`heavy-tests.yml`), once under Stacked Borrows and once under Tree Borrows (#16); tests that build a threadteam pool
are skipped under Miri, which does not support the foreign functions the pool calls (#20). It ran nightly until #31:
the two jobs had become 47 of the 51 runner-minutes of a night. What weekly costs: a Miri failure caused outside the
unsafe code, such as a dependency update or a test that grows (as in #21), now surfaces up to a week later instead of
the next morning. Two things cover the rest. A pull request that adds or changes unsafe code runs Miri locally (both
aliasing models, the commands in `heavy-tests.yml`) and puts the result in its `## Verification`. And Miri can be run
on a branch before it merges: `gh workflow run heavy-tests.yml --ref <branch>` runs every job of the workflow. Kani proves the harnesses of
`crates/colmat-core/src/kani_proofs.rs` nightly (#17, a trial: `cargo kani -p colmat-core`, kani-verifier 0.68.0), on an arm64 runner because Kani cannot build `pulp`'s x86-64 global assembly (#21). None of these runs in the
push/PR pipeline: they need nightly toolchains and exceed the CI budget. The property tests (`prop_*`) run 256 cases per PR and 65 536 nightly
(`PROPTEST_CASES`). Mutation testing (`cargo mutants -p colmat-traits --profile ci`) runs weekly and on demand
(`mutants.yml`) and does not gate merges; a surviving mutant gets a test, or an exclusion with its reason.
`colmat-core` is mutated in the same workflow, sharded, and only reported for now (#12).
