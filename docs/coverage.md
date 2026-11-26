# Coverage

`.github/workflows/coverage.yml` measures the line coverage of the whole workspace with
`cargo llvm-cov` on `ubuntu-latest`, weekly (Sunday 20:00 UTC) and on demand
(`gh workflow run coverage.yml`), and on a pull request that changes the workflow. It reports the
number in the job summary and does not gate merges (#37).

## The plan (#37)

1. **Now: report only.** No `--fail-under-lines`. The job establishes the baseline on the CI host,
   which compiles other paths than an aarch64 machine does (below).
2. **Then: a gate.** Once the baseline is stable, the job moves under `ci-gate` in `ci.yml` with the
   floor set at the measured CI value. Raising the floor is free; lowering it needs a maintainer's
   decision stated in the pull request, and changing the toolchain or the `cargo-llvm-cov` pin means
   re-measuring it in the same pull request.

## The baseline

| Commit | Host | Lines | Regions | Functions |
|---|---|---:|---:|---:|
| `34a7288` (`4cc55ae` with #38) | `ubuntu-latest`, x86-64 ([run](https://github.com/tensor4all/colmat-next-rs/actions/runs/45756603179)) | 58.96 % (8484 of 14389) | 60.22 % | 52.06 % |
| `4cc55ae` | Apple M1 Pro, aarch64 (local, not the CI host) | 59.19 % (8445 of 14268) | 60.50 % | 52.09 % |

The `ubuntu-latest` rows are added from the job's summaries. On a pull request the commit is the
merge commit GitHub tests, not the branch head. The aarch64 row is for orientation only: the floor
will come from `ubuntu-latest`. On `ubuntu-latest` the tests take about 7 minutes on one thread
(the whole job about 8).

**A fixed seed.** The job sets `PROPTEST_RNG_SEED=20261120`, the seed of `mutants.yml`, and keeps
`PROPTEST_CASES` at its default (#37). Without it the number moved by up to 5 lines between runs of
one commit on `ubuntu-latest`. Locally the cause was the property tests' random inputs: one line
(`matown.rs:579`, the early `return` of `resize_with`) was covered in 8 of 9 runs, and with the
seed three runs gave identical results. The seed is part of the measurement, not of the tests: the
`test` job of `ci.yml` and the nightly 65 536-case run still draw new inputs. The floor is chosen
only after several `ubuntu-latest` runs with the seed give the same number.

## Reproduce it locally

```sh
rustup component add llvm-tools-preview --toolchain 1.96.0   # required by cargo-llvm-cov
cargo install cargo-llvm-cov --version 0.9.1 --locked
PROPTEST_RNG_SEED=20261120 cargo llvm-cov --workspace --all-targets --locked --profile ci \
  --summary-only -- --test-threads=1
```

Unset a `RUSTUP_TOOLCHAIN` exported in your shell first, so that `rust-toolchain.toml` selects
1.96.0. The run takes about 3 minutes on an M1 Pro once the instrumented build is cached.

- Drop `--summary-only` for the per-file table, add `--show-missing-lines` for the uncovered lines,
  or use `--html` and open `target/llvm-cov/html/index.html`.
- Only one instrumented run can use `target/llvm-cov-target` at a time.

**Why `--test-threads=1`.** With the default threads, the tests of one binary run on threads of one
process and contend on the shared coverage counters. On `main` `63b75da`, the four
`matmul_matches_the_reference_*` tests then took about 1240 s together, against 14.7 s for one
alone, and the whole run took 21 minutes; with one thread it took 182 s, with the same counts.

## What is not measured

- **Doctests.** `cargo llvm-cov` measures them only on nightly (`--doctests`), and the toolchain is
  the pinned stable one. The `test` job of `ci.yml` still runs them. View types whose examples are
  doctests read low for this reason (`diag/*`, `mat/mat_index.rs`, `col/col_index.rs`,
  `row/row_index.rs`).
- **The other host's paths.** On `ubuntu-latest`, `t4a-private-gemm-x86` and the x86-64 SIMD paths
  count, and the NEON and Apple SME paths do not. `macos-latest` still runs the tests in `ci.yml`.
- **`#[ignore]` tests** do not run, so code only they reach counts as uncovered.
- **Examples and benches** are built by `--all-targets` but not run.

Nothing is excluded with `--ignore-filename-regex`.
