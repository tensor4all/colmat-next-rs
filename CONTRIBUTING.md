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
- **Feature combinations CI builds**: default features, and `--all-features` for clippy and rustdoc.
- **Platforms claimed**: `ubuntu-latest` and `macos-latest` (tests); rustdoc, clippy and `deny` run on Linux only.
- **Local gate**: `cargo test --workspace --profile ci` once a crate exists.
- **CI time budget**: the push/PR pipeline has one total budget (`CI_BUDGET_MINUTES` in `.github/workflows/ci.yml`).
  The `budget` job fails when it is exceeded and lists the slowest jobs. Tests too slow for it are named `heavy_*`,
  marked `#[ignore = "heavy: ..."]`, and run nightly from `.github/workflows/heavy-tests.yml`, which files (or
  comments on) an issue titled "Scheduled heavy tests are failing" when it fails.

## Anything here that departs from the shared rules

None. Miri runs nightly over `colmat-traits`'s lib tests (`heavy-tests.yml`), not in the push/PR pipeline, because it
needs nightly and exceeds the CI budget. The property tests (`prop_*`) run 256 cases per PR and 65 536 nightly
(`PROPTEST_CASES`). Mutation testing (`cargo mutants -p colmat-traits --profile ci`) runs weekly and on demand
(`mutants.yml`) and does not gate merges; a surviving mutant gets a test, or an exclusion with its reason.
