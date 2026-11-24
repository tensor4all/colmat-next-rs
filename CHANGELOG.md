# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the crates follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Releases are tagged
`<crate>-v<version>`.

## [Unreleased]

### Changed

- CI (#31): the two Miri jobs of `heavy-tests.yml` run weekly (Sunday 18:00 UTC) instead of
  nightly; `heavy` and `kani` stay nightly, and `workflow_dispatch` still runs every job. The
  failure report runs after both schedules. `CONTRIBUTING.md` states what this costs.
- `colmat-core` (#28): a product on a team cuts its output into the grid of
  `utils::partition::tile_grid`, as colmat-rs's `gemm_tiled` did, instead of columns or rows only;
  a tall product no longer packs the whole `lhs` on every participant (`f64` `4096 x 64 x 64` on
  width 2: 1.26x colmat-rs before, 0.99x after). Above the size of `depth_split`, the
  column-major matrix-vector product on a team splits its depth instead of its rows, with the
  partial sums in the caller's scratch (`matmul_scratch` now asks for them). Its result then
  rounds differently from `Par::Seq`, the same on every call of the same width (design record §5,
  "Rounding on a team").

### Fixed

- `colmat-traits` (#26): on NEON, `NativeSimd`'s full pair loads and stores (the registers of
  `Complex<f32>`/`Complex<f64>`) are one `vld2q`/`vst2q` on memory instead of a deinterleave
  through a stack copy. The path is chosen from the register's size (16 bytes, NEON enabled at
  compile time), not the instruction set's identity; Miri and other targets keep the previous
  path. On an Apple M1 Pro the complex axpy now runs at the speed of the hand-written NEON loop
  (it took 1.43-1.48x), and `c64`/`c32` column-major matrix-vector products went from 1.58-1.76x
  colmat-rs `71337cdc` to 1.09-1.16x, rank-one updates from 1.32-1.42x to 0.88x.

- CI (#21): the nightly `kani` job runs on `ubuntu-24.04-arm`. On x86-64 it could not build:
  `pulp`'s build script emits global assembly there, which Kani does not support.
- Tests (#21): under Miri, `lower_area_parts_keeps_every_part_non_empty` skips the 2048 and 4096
  blocks and the 1024 block with `align = 1` (about 28k instead of 2.8M `lower_area_split` calls),
  so the nightly Miri jobs finish within their 60-minute limit.
- `colmat-traits` (#10): complex `sqrt` no longer loses the smaller component when `|im| ≪ |re|`
  (Kahan's method), and complex `recip` no longer flushes to zero for `|z|` near the overflow
  threshold. Both defects were inherited from colmat-rs.
- `colmat-core` (#14): the soundness defects inherited from colmat-rs and recorded in #12.
  `Mat::resize_with` and `Mat::truncate` drop each element once when a closure or a `drop` panics;
  the triangular `zip!` iterations stay in bounds for every shape and layout, and the upper
  triangle of a wide matrix with reversed rows no longer skips the columns `j >= nrows`; casts
  between permutation shapes panic for `One` instead of reinterpreting its zero-sized index;
  `Array`'s range indexing is checked. API changes: `ZipEq::new_unchecked` is an `unsafe fn`, and
  `Array::as_mut` returns a slice borrowed from the array.

### Added

- `colmat-core` and `colmat-traits` (#23): the matrix product and its kernel hook (P1).
  - `colmat-traits`: `ComplexField::Kernels` names a scalar's kernel provider (S2):
    `GenericKernels` or `NativeKernels`.
  - `colmat-core`:
    - `kernels::Gemm<T>` is the hook and `KernelField` the bound the drivers ask for;
    - `linalg::matmul` has `matmul_with_scratch`, `matmul_scratch`, `matmul_with_conj` and
      `matmul` (the last two allocate the scratch per call, as their rustdoc says), with the
      driver ported from colmat-rs `71337cdc`;
    - `utils::simd` is ported onto `SimdLayout`/`SimdField`, and `Accum` comes with it.
  - The generic kernels (matrix-vector, rank-one, the vertical and horizontal micro-kernels)
    run for every scalar and split the output, never the depth, on a team (changed by #28).
  - The native kernels of `f32`, `f64`, `c32` and `c64`: `nano-gemm` up to `16^3`
    multiply-adds, then `t4a-gemm`, or `t4a-private-gemm-x86` on x86-64 with AVX2/FMA or
    AVX-512F. `apple-sme` is a default feature with runtime detection; AMX is not enabled.
  - Known gap: the packed native products allocate on a thread's first call (`t4a-gemm` also
    above L2), until tensor4all/t4a-gemm#21 and tensor4all/t4a-private-gemm-x86#14.
    `matmul_with_scratch`'s rustdoc and the no-allocation test say so.
  - Work log: `docs/worklogs/2026-11-23-colmat-core-matmul.md`.

- `colmat-core` (#20): the execution layer, ported from colmat-rs `71337cdc` and restructured. `Par`
  and `ParShape`; the team skeletons (`utils::thread`: `join_raw`, `join_sized`, `split`,
  `with_width`, `for_each`, `for_each_with_par`) and the route onto a threadteam team
  (`utils::team`, feature `threadteam`, default on); the cost model (`utils::cost`, the width rules
  public); the static partitions (`utils::partition`, public, on every target); the caller-owned
  workspace (`workspace`: `temp_mat_*`, the `@alloc` arms of `mat!`/`col!`/`row!`, `stack_mat!`)
  and the in-place permutations. The item loops deal a `utils::thread::Producer` of this crate
  instead of a rayon iterator; `MatRef::par_col_chunks`/`par_row_chunks` and their `_mut` forms
  return one. `layout::copy_across_layouts` is public and takes `Par`. A call that takes scratch
  checks the stack before it writes; an integration test counts allocations across the team.
  Kani harnesses for the split arithmetic. Work log:
  `docs/worklogs/2026-11-21-colmat-core-execution-layer.md`.
- `colmat-traits` (#20): `CostWeight` and `ComplexField::COST` (S3, optional, default `f64`'s), the
  scalar's weight against `f64` for compute- and memory-bound kernel families, which replaces the
  identity test of colmat-rs's `Rate::of`.

- `colmat-core` (#17, trial): Kani proof harnesses (`src/kani_proofs.rs`) for the row ranges of the
  triangular `zip!` iterations (every size), the index casts and permutation invariant, and
  `MatRef::split_at`; a nightly `kani` job in `heavy-tests.yml`. The triangular row ranges are now
  the pure functions `zip::triangular_lower_ncols`/`triangular_lower_rows` (the upper triangle uses
  them with the transposed layout), and the crate's `concat!` macro is defined last in `lib.rs`, so it
  no longer shadows `core::concat!` inside the crate. Work log:
  `docs/worklogs/2026-11-21-kani-trial.md`.

- CI (#16): the nightly Miri run covers the doctests too and runs a second time under Tree Borrows.
  `colmat-core` gains layout-exhaustive tests (`src/exhaustive_tests.rs`): every shape up to a small
  size in 16 strided layouts, for the view reads, mutable splits and submatrices, the layout copies
  and two-operand `zip!`, checking each element and the padding around the view.

- `colmat-core` (#12): matrix views (`Mat`, `MatRef`, `MatMut`, `Col`, `Row`, `Diag`, `Perm`), `zip!`
  iteration and serial layout copies, ported from colmat-rs without rayon and without the `linalg`
  dependency. Every unsafe block carries a `// SAFETY:` comment; property tests check the views
  against a reference offset model. The review found soundness defects inherited from colmat-rs,
  recorded in `docs/worklogs/2026-11-20-colmat-core-views-port.md` and fixed in #14.
- `NOTICE`: the provenance and licence of the code ported from colmat-rs and faer.

- Repository scaffolding: agent and contribution rules, CI, and the design-record layout.
- `colmat-traits` (#6): `ComplexField`/`RealField` with scalar arithmetic only, `Precision`
  (`eps`, `safe_min`, `safe_max`), and SIMD providers split from the scalar trait (`SimdLayout`,
  `SimdField`, `SimdPairs`; `NoSimd`, `NativeSimd`, `ComplexSimd`). Ported from colmat-rs `faer-traits`.
- `colmat-traits` tests (#8): property tests (proptest) for every provider and for the scaled complex functions, an
  explicit AVX2/FMA pass on x86-64, and weekly mutation testing. They found two defects inherited from colmat-rs
  (`sqrt` cancellation, `recip` underflow near overflow), recorded in `docs/worklogs/2026-11-20-colmat-traits-test-strength.md`.
- Workspace setup: MSRV 1.89, pinned toolchain 1.96.0, rustfmt and lint configuration, Miri in the
  nightly heavy tests.
