# colmat-core execution layer: port, restructuring and checks (#20)

A record of the second part of step 2 of the design record (§7): the execution layer of
`colmat-core` (`Par`, the team skeletons, the cost model, the partitions, the workspace), ported
from colmat-rs `71337cdc` (#20). Everything here describes this trial; nothing is a decision of the
colmat maintainers.

## What was ported

| colmat-rs `faer/src` at `71337cdc` | Here | Notes |
|---|---|---|
| `lib.rs` 1016-1194: `Par`, `ParShape` | `par.rs`, re-exported at the root and in the prelude | With the `compile_fail` doctests |
| `utils/mod.rs` 11-291: `thread` | `utils::thread` | Item loops on a `Producer` (below) |
| `utils/team.rs` 1-450 | `utils::team` (feature `threadteam`) | Counters test-only |
| `utils/team.rs` 443-496: `tile_grid`, `grain_range` | `utils::partition` | Public |
| `utils/partition.rs` | `utils::partition` | Public, every target (debt K6) |
| `utils/cost.rs` 1-512, 615-1033 | `utils::cost` | Rates by weight, width rules public |
| `linalg/mod.rs` 29-73, 602-770: `temp_mat_*` | `workspace` | Capacity checked at entry |
| `linalg/mod.rs` 216-277: parallel split of `copy_across_layouts` | `layout` | Public, takes `Par` |
| `perm/mod.rs` 295-370: in-place permutations | `perm` | Restored (#12 removed them) |
| `lib.rs` 305-320 and the `@alloc` arms of `mat!`/`col!`/`row!` | `lib.rs` | `stack_mat!` exported |

Not ported, and where it goes:

| Item | Where | Why |
|---|---|---|
| x86 kernel adapter (`team::kernel`, `job_tables`, `x86_kernel`), `end_of_job` (`vzeroupper`), GEMM counters, `cost::gemm_ns_per_mac` | core-3 | They belong to the matrix product and its kernels |
| `cost::block_apply_estimate`, `PanelTeam`, `lookahead_width` | The drivers (`colmat`) | Factorization models; no caller here |
| `par-trace` probes (`cost::trace`) | Dropped for now | A probe feature with process-wide state; can return with a driver that needs it |
| `column_major_copy`, `copy_columns_into` | The drivers | `column_major_copy` allocates a `Vec` inside a driver (debt D3) |
| `par_*_iter`, `par_*_partition` of the views and vectors | Later, on demand | The chunk forms cover the copy; the others had callers only in `linalg` |
| `StackReq` helpers of `matown.rs` (#12's list) | Nothing left | The ported `matown.rs` uses `StackReq` only to record its own allocation layout |

## Restructured, not copied

- **One feature.** `Par::Team` wraps `threadteam::TeamExec` behind `colmat-core`'s `threadteam`
  feature, on by default; rayon arrives only through threadteam (`deny.toml`). The CI step of #12
  that forbade `threadteam` in `colmat-core` now checks `colmat-traits` and `colmat-core` without
  the feature, and the `deny` job also builds `colmat-core` without `std` and with `std` only.
- **No rayon traits in the dealing (debt K7).** `utils::thread::Producer` (length, `split_at`,
  walk in order, `zip_eq`) replaces rayon's `Producer`/`IndexedParallelIterator`. `team::deal` is
  the colmat-rs dealing unchanged otherwise: up to 64 blocks of at most 256 bytes each go through one
  fork from an inline array, larger splits halve recursively, and nothing is collected. Ranges, the
  chunks of a view (`MatRef::par_col_chunks`, `par_row_chunks` and their `_mut` forms: the colmat-rs
  names and chunk boundaries) and zipped pairs are producers. The item loops now work without the
  parallel feature too (`Par::Seq` walks the producer in order).
- **No scalar identity in the cost model (debts K2, K4).** `Rate::of<T>` was
  `if IS_NATIVE_F32 { real / 2 } else if IS_NATIVE_C32 { cplx / 2 } else if IS_REAL { real } else { cplx }`.
  It is now `(if IS_REAL { real } else { cplx }) * weight`, the weight being `T::COST`
  (`colmat-traits`, `CostWeight { compute, memory }`, default `1, 1`; `f32` and `Complex<T>`'s
  component `0.5, 0.5`). Each family carries a `Bound`: compute for the products, block
  reflectors, triangular inverse and divide and conquer solves; memory for the swaps,
  matrix-vector products, the unblocked factorization steps, the rank-one apply and the sparse
  product. This classification is proposed here; it only changes anything for a scalar whose two
  weights differ, and no such scalar exists yet. `Rates::gemm_single` and `ns_per_mac` are gone:
  the `f32` product rate is half the `f64` one, which on `x86_64-avx512` is 0.03205 ns per
  multiply-add instead of the measured 0.0320 (0.16 percent). Every other native rate is unchanged.
- **Width rules public.** `par_width`, `join_pays`, `halves_pays`, `par_pays_time`,
  `ParThreshold::region_width`, ... are public, as design record §5 asks for the cost queries.
- **`copy_across_layouts` public** with a `Par` argument and `T: Clone + Send + Sync`. Like
  colmat-rs, the split uses every participant that gets a tile and is not gated by the cost model:
  a small copy on a team pays a section's dispatch. Whether to gate it (a memory-bound `copy` rate)
  is left for when a driver calls it on small shapes.
- **Workspace.** `temp_mat_uninit`/`temp_mat_zeroed` check `MemStack::can_hold` first and panic
  with the requirement in the message, so a function that takes its temporary before it writes
  (the in-place permutations) leaves its output unchanged on a short stack (tested).
  `temp_mat_zeroed` no longer makes a first, discarded allocation. The in-place permutations copy
  into the uninitialised temporary by raw writes (by assignment only when `T` needs drop, in which
  case the temporary was filled), so no reference to an uninitialised element is formed.

## Checks

- **Tests.** The `thread`, `team`, `cost` and `partition` tests of colmat-rs that need no
  factorization or product, adapted (`execution_tests.rs`, `utils/cost.rs`,
  `utils/partition.rs`); producer and chunk tests; the in-place permutations against the
  out-of-place ones with a buffer of exactly the queried size; a short stack leaving the output
  unchanged; the `@alloc` arms and `stack_mat!`. The layout-exhaustive test of #16 gains the
  parallel copy on teams of 2 to 4, all 16 x 16 layout pairs and 8 sizes, padding checked.
- **No allocation** (`tests/no_alloc.rs`): a counting global allocator counts the allocations of the
  test thread and of the team members (marked by the pool's `on_start`) across each call: the
  parallel copy, a team loop over mutable chunks, and both in-place permutations, at first use and
  on repeated calls of growing shape, with `Par::Seq` and teams of 2 and 4. All zero. Two things the
  test had to learn: counting every thread of the process is not reliable (the test harness's own
  thread allocates at times), and a threadteam member allocates three times (48 bytes and less) the
  first time it goes idle after its pool starts it, independently of any call (threadteam 0.6.0,
  measured with an idle team: 3 allocations in the first 30 ms, none in the next five rounds). The
  test lets the members pass that point before it counts. Whether that set-up should happen when the
  pool starts instead is a question for threadteam, not raised there yet.
- **Miri** (lib and doc tests of `colmat-traits` and `colmat-core`, nightly, `-Zmiri-disable-isolation`,
  `PROPTEST_CASES=16`): Stacked Borrows 155 passed, 0 failed, 31 ignored. Tests that build a threadteam pool are
  skipped under Miri: the pool calls foreign functions Miri does not support
  (`pthread_set_qos_class_self_np` on macOS). `Exec::serial` (a team of one) runs, so the
  `Par::Team` code paths of width one are covered, the parallel dealing is not. The unsafe code
  of the dealing (`Slot`, `Piece`, `Once`) is therefore covered by the normal tests and the review
  of their `// SAFETY:` comments only. Tree Borrows (`-Zmiri-tree-borrows`): the same 155 passed,
  0 failed, 31 ignored.
- **Kani** (`kani_proofs.rs`, four new harnesses; all nine run in about 57 s locally):

  | Harness | Claim | Bound | Time | Deliberate mutation caught |
  |---|---|---|---|---|
  | `par_split_indices_cuts_consecutive_blocks` | Consecutive blocks covering `0..n`, lengths differing by at most one | `n <= 1024`, 16 parts | 14.9 s | `rem +` dropped from the block start: overflow and coverage fail |
  | `deal_halves_are_non_empty` | The recursive halving gives each half at least one participant and one item | team 256, items 4096 | 4.0 s | Same mutation: subtraction overflow |
  | `copy_tasks_are_whole_tiles_and_non_empty` | Whole tiles, `1 <= tasks <= degree`, the last task non-empty (the chunk producer has exactly `n_tasks` items) | 4096 lines, team 64 | 3.4 s | Chunk rounded down instead of up: `tasks <= degree` fails |
  | `grain_range_cuts_consecutive_ranges` | Consecutive ranges covering `0..len`, inner edges on the grain | 1024, 16 parts, both grains | 7.9 s | — |

  The first run also found that `copy_tasks` returns `(0, 0)` for zero lines (harmless: the copy
  splits only for more than one task); the harness now states it. Kani's reach ended at nonlinear
  arithmetic: none of the four harnesses finished in 240 s with unbounded `usize` inputs (symbolic
  division), and `lower_area_split` (`u128` products and `isqrt`) did not finish in 500 s even at
  `16 x 16`; it is covered by its ported unit tests. No unsafe block of this step relies on these
  functions (the splits go through checked `split_at`), so the harnesses check correctness of the
  cuts rather than memory safety.

## Native non-regression

Apple M1 Pro (8 performance and 2 efficiency cores), release build, median of 31 batches of at
least 2 ms; colmat-rs `71337cdc` against this branch in one binary. A local probe, not a committed
benchmark.

| Case | Sizes | Ratio new / old |
|---|---|---|
| `permute_rows_in_place`, `permute_cols_in_place`, `transpose().to_owned()`, f64, f32, c64, c32 | 64, 256, 1024 | 0.96 to 1.02 in 34 of 36 cases; 0.56 (f64 rows, 64: 2.1 against 3.8 us) and 1.08 (f32 cols, 1024) |
| team `for_each` over zipped row chunks, identical item work, widths 1, 2, 4 | 256, 1000, 2000, 2048 | 0.99 to 1.02 in most cases, outliers both ways (0.59, 0.63, 1.12) |
| the same, n = 1024 | widths 1, 2, 4 | 1.18 to 1.26 |

The n = 1024 cases are not the dealing: the item loop alone, without `for_each`, shows the same gap,
and the same raw-pointer loop over the two libraries' buffers differs by a factor of 0.80 (a
transpose with a power-of-two stride is sensitive to where its buffers lie); rebuilding the probe
moved the item-loop ratio from 1.39 to 0.90. Width 8 is not compared: on this machine it lands on
efficiency cores and varies run to run on both sides.

The parallel `copy_across_layouts` against its own serial path, f64, transposing copy: 0.52 to 0.61
on 2, 0.26 to 0.34 on 4 participants, for n from 256 to 2048.

## What a colmat-rs patch would look like (design record §2, principle 2)

- K7: a `Producer` trait in `utils/thread`, implemented by the views' chunk types, and `deal`
  generic over it; the `par_*` methods keep their names and change their return type (a breaking
  change). Not prepared.
- K2/K4: `const COST: CostWeight` with a default on `faer-traits`' `ComplexField` and `Rate::of`
  as above; the `IS_NATIVE_*` constants stay for the other dispatch sites (debt K1). Not prepared.
