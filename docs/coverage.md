# Coverage

`.github/workflows/coverage.yml` measures the line coverage of the whole workspace with
`cargo llvm-cov` on `ubuntu-latest`, weekly (Sunday 20:00 UTC) and on demand
(`gh workflow run coverage.yml`), and on a pull request that changes the workflow. It reports the
number in the job summary and fails below a floor of 58.9 % of lines. It does not gate merges: it
is not part of `ci-gate` (#37).

## The plan (#37)

1. **Done: report only.** The job established the baseline on the CI host, which compiles other
   paths than an aarch64 machine does (below): 8484 or 8488 lines, by the runner's instruction set.
2. **Now: a floor in this job.** `--fail-under-lines 58.9`, set by the maintainer in #37 after
   three separate runs with the seed: 8476 of 14389 lines is the lowest count that passes (8475 is
   58.899 %). It sits 8 lines below the lower baseline (8484, an AVX-512F runner), so a line that a
   runner happens to miss does not fail the job for a reason unrelated to the change. If a month of runs never goes below 8484, the floor
   rises to 58.95 % (8483 lines) in a separate change. A run below 8483 is a real signal: its
   `lcov.info` is compared with a passing run's and the diff goes to #37, rather than the floor
   going down. Raising the floor is free; lowering it needs a maintainer's decision stated in the
   pull request, and changing the toolchain or the `cargo-llvm-cov` pin means re-measuring it in
   the same pull request.
3. **Not decided: a merge gate.** Moving the job under `ci-gate` in `ci.yml` would run it on every
   pull request (6-8 minutes on `ubuntu-latest`, against a 15-minute budget for the whole
   pipeline). That is a separate decision (#37).

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
`test` job of `ci.yml` and the nightly 65 536-case run still draw new inputs. The floor was set
only after several `ubuntu-latest` runs with the seed gave one of two numbers (below).

**The seed is not enough on `ubuntu-latest`.** Three attempts of one run with the seed, on one
merge commit, covered 8484, 8484 and 8488 lines
([run](https://github.com/tensor4all/colmat-next-rs/actions/runs/45784127160); #37). The seed
stays: it removes one known source. To find the lines that still move, the job uploads the
per-line report as the artifact `lcov-<run id>-<attempt>` (`lcov.info`, kept 14 days). Compare
separate runs (`gh workflow run coverage.yml`, or the weekly ones), not attempts of one run:
on #48, re-running all jobs left only the second attempt's artifact. Download each run's file
(`gh run download <run id>`) and compare two runs whose numbers differ line by line:

```sh
cov() { awk -F'[:,]' '/^SF:/{f=$2} /^DA:/{print f":"$2" "($3>0)}' "$1" | sort; }
diff <(cov a/lcov.info) <(cov b/lcov.info)   # file:line, 1 if covered
```

**The lines that move: the runner's instruction set.** Two runs on #48 covered 8488 lines
([run](https://github.com/tensor4all/colmat-next-rs/actions/runs/45946634249), attempt 2) and
8484 lines ([run](https://github.com/tensor4all/colmat-next-rs/actions/runs/45948372777)); the
Rust code of the two merge commits is the same. They differ in three CPU feature checks and in
nothing else:

| Lines | What they are | 8488 run | 8484 run |
|---|---|---|---|
| `linalg/matmul/native.rs:53` | `x86_instr`, the AVX-512F branch | no | yes |
| `linalg/matmul/native.rs:54-55, 57` | `x86_instr`, the AVX2 and FMA branch | yes | no |
| `utils/cost.rs:114` | `HostClass::detect_uncached`, the AVX-512F branch | no | yes |
| `utils/cost.rs:115-116, 118` | `HostClass::detect_uncached`, the AVX2 and FMA branch | yes | no |
| `utils/cost.rs:1066` | the test `detection_is_stable_and_the_host_row_is_its_class`, its AVX-512F `assert_eq!` | no | yes |
| `utils/cost.rs:1067` | the same test, the line after that `if` | yes | no |

So `ubuntu-latest` runners come with and without AVX-512F. An AVX2 runner covers 8488 lines and
an AVX-512F runner 8484, 4 fewer. The lines are not pool, scheduling or timing code. The 8483 of an
earlier unseeded run is consistent with an AVX-512F runner that also missed the property-test
line above, which the seed fixes; that run kept no per-line report, so this is not verified.
Three further separate runs on `main` `ad2f3a3`
([1](https://github.com/tensor4all/colmat-next-rs/actions/runs/45952923049),
[2](https://github.com/tensor4all/colmat-next-rs/actions/runs/45953801850),
[3](https://github.com/tensor4all/colmat-next-rs/actions/runs/45954446121)) covered 8484, 8488 and
8484 lines, on AVX-512F, AVX2 and AVX-512F runners. The floor is set below the lower number (#37):
excluding the checks from the measurement would need a file split or a nightly-only attribute for
four lines, and would hide that the pool has two runner classes, which this table records.

The numbers here are the job summary's, the same as the sums of the `LF:`/`LH:` records of
`lcov.info` (14389 lines). Counting its `DA:` records instead gives 8243 or 8247 of 14041: in 29
files `LF:` is a few lines more than the file's `DA:` records, for a reason not looked into.
`--fail-under-lines` uses the summary's count, so compare runs by that count.

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
- The job runs the tests once with `--no-report`, then writes both reports from that run:
  `cargo llvm-cov report --profile ci --lcov --output-path lcov.info` and
  `cargo llvm-cov report --profile ci --summary-only --fail-under-lines 58.9`. The floor is for
  `ubuntu-latest`; an aarch64 machine compiles other paths and gives another number.
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
