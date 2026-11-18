<!-- unified contribution policy: ../../docs/contributing/rust-repositories.md in tensor4all/tensor4all-meta -->
# Contributing to colmat-next-rs

The rules that apply to every Rust repository in `tensor4all` are in one place:
**[tensor4all-meta · docs/contributing/rust-repositories.md](https://github.com/tensor4all/tensor4all-meta/blob/main/docs/contributing/rust-repositories.md)**.
Read that first — it covers the issue-first workflow, what a pull request body has to contain, red-first tests, the
gates every repository runs, the review requirement, and the version policy.

This file only records what is specific to `colmat-next-rs`:

## Local facts

- **MSRV**: none yet; there is no crate. The first crate declares `rust-version` in the workspace `Cargo.toml` and
  the `msrv` job in `.github/workflows/ci.yml` is set to the same toolchain in that change (colmat-rs declares
  1.89.0; the job's 1.89 is an initial value). The same change adds `rust-toolchain.toml` and `rustfmt.toml`
  following colmat-rs.
- **Feature combinations CI builds**: default features, and `--all-features` for clippy and rustdoc.
- **Platforms claimed**: `ubuntu-latest` and `macos-latest` (tests); rustdoc, clippy and `deny` run on Linux only.
- **Local gate**: `cargo test --workspace --profile ci` once a crate exists.
- **CI time budget**: the push/PR pipeline has one total budget (`CI_BUDGET_MINUTES` in `.github/workflows/ci.yml`).
  The `budget` job fails when it is exceeded and lists the slowest jobs. Tests too slow for it are named `heavy_*`,
  marked `#[ignore = "heavy: ..."]`, and run nightly from `.github/workflows/heavy-tests.yml`, which files (or
  comments on) an issue titled "Scheduled heavy tests are failing" when it fails.

## Anything here that departs from the shared rules

Until the first crate lands, the Rust gates (`fmt`, `clippy`, `test`, `msrv`, `docs`) have nothing to check and CI
skips them; the `rust` job reports whether a `Cargo.toml` is present. That is the state of an empty repository, not
an exemption: the gates run from the first crate on.
