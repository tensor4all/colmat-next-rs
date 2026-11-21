# Kani trial: proofs for the index arithmetic of colmat-core (#17)

This records the trial proposed in #17: Kani proof harnesses for a few claims that colmat-core's
`// SAFETY:` comments rely on. It ran as a nightly job that does not gate merges, and its result
decides whether to continue. Everything here describes this trial; nothing is a decision of the
colmat maintainers.

Tools: kani-verifier 0.68.0, CBMC 6.11.0, on an Apple M1 Pro (aarch64).

## Changes made for Kani

- **The `concat!` macro moved to the end of `lib.rs`.** Kani replaces `assert!` with a macro that
  expands to a bare `concat!`. The crate's own `concat!` (matrix concatenation, a `macro_rules!`
  ported from faer) was in textual scope for every module and shadowed `core::concat!`, so the crate
  did not build under Kani. Defined last, it shadows nothing. It is still exported as
  `colmat_core::concat!`, so the API is unchanged.
- **The row ranges of the triangular `zip!` iterations are pure functions**
  (`zip::triangular_lower_ncols`, `zip::triangular_lower_rows`, `zip::transposed`). The 16 loops of
  the two `_with_index` iterations and the match of `for_each_mat_triangular_lower` now call them.
  The upper triangle uses the lower triangle's ranges with the transposed layout. In colmat-rs,
  where the four upper arms had their own formulas, that is where the `ReverseRows` column bound
  went wrong (#14). The layout-exhaustive tests (#16) and Miri pass unchanged.
- **`unexpected_cfgs`** now accepts `cfg(kani)`, which gates `src/kani_proofs.rs`.

## Harnesses

| Harness | Claim | Bound | Time | Catches (deliberate mutation) |
|---|---|---|---|---|
| `triangular_ranges_are_in_bounds_and_cover_exactly_the_triangle` | For each layout, both triangles and both diagonal modes: `start <= end <= m` in each iterated column; every visited index maps onto the triangle; every element of the triangle is visited | none: every `usize` dimension (the claim has no loop) | 10.3 s | `TransposeReverseRows` column count `min(m, n)` (the `ReverseRows` skip of #14) → `j < ncols` fails. `ReverseRows` with `0..n` and `m - j - strict` (the colmat-rs#255 underflow) → subtraction overflow. `None` with `0..n` and unclamped `start` (the contract-only case of #14) → `start <= end` fails |
| `cast_idx_slice_reads_the_entries_back` | `cast_idx_slice` returns the same entries for `usize` and `Dim` shapes | 3 entries | 0.5 s | — |
| `cast_idx_slice_panics_for_one` (`should_panic`) | For `One`, the cast panics | 2 entries | 0.02 s | Assertion removed → `dereference failure: pointer invalid`, `misaligned pointer` (the zero-sized read of #14) |
| `perm_new_checked_implies_the_permutation_invariant` | What `new_checked` checks (`forward[i] < n`, `inverse[forward[i]] == i`) implies the whole invariant (every entry `< n`, each the other's inverse); the bound arrays read in bounds | `n = 3`, all entry values | 1.6 s | — |
| `split_at_parts_read_the_parent` | The four parts of `MatRef::split_at` read the parent's elements, with every read in the buffer | `m, n <= 3`, both major orders, reversed rows or not, every split point and index | 8.9 s | `bot_right` at `(row, 0)` → `got == a[(i, j)]` fails |

All five run in about 26 s after the build (`cargo kani -p colmat-core`).

Three one-sided mutations passed verification: dropping the clamp of `None`'s `start`, dropping the
clamp of `ReverseRows`'s `end`, or widening the `ReverseRows` column count to `n`. Each one alone is
not a defect: the column bound and the clamp each keep the ranges in bounds without the other, and
Kani proves that equivalence for every size. Only the combinations colmat-rs had fail.

## Compared with the tests

- **Triangular ranges.** The layout-exhaustive test caught the `ReverseRows` column skip at
  `2 x 3`, but only up to `6 x 6`. The harness covers every size. Its index maps (`to_original`,
  `to_view`) restate the `from_dyn_idx` calls of `zip.rs` by hand. The link between them and the
  real iteration is checked by the exhaustive tests, not by Kani.
- **Casts and permutations.** Miri already reports the zero-sized read. Kani's addition is the
  implication from `new_checked`'s check to the invariant, for every value of a 3-entry permutation.
  It is small but not covered by any test.
- **`split_at`.** Kani verifies the real view code here (`from_raw_parts`, `reverse_rows`,
  `split_at`, indexing): the `generativity` and `equator` machinery did not get in its way. The
  exhaustive tests cover the same at sizes up to `5 x 5` and 16 layouts, so Kani adds no coverage
  beyond the bound here.

Kani found no defect in the current code. Every defect it detects above was introduced on purpose
or had already been fixed in #14.

## Cost

- **Maintenance.** `kani_proofs.rs` is about 190 lines. The refactor of `zip.rs` is a code change
  in its own right, and it removed four hand-written copies of the upper-triangle formulas.
- **CI.** `cargo install kani-verifier` plus `cargo kani setup` (a pinned nightly and CBMC), then
  about 30 s of proofs. The job runs nightly in `heavy-tests.yml` rather than weekly as #17
  proposed: it is short, and a nightly job reaches the existing failure report.
- **Toolchain.** Kani builds with its own nightly (`nightly-2026-08-21` for 0.68.0), separate from
  the pinned 1.96.0.

## Proposal (in this trial)

Keep the job, with these harnesses, until core-2. Kani can prove the index arithmetic of `Par`'s
splits and of the GEMM packing (core-3) for every size, which the exhaustive tests cannot. The
proposal is to add a harness only where an unsafe block relies on arithmetic that can be factored
out as a pure function. Revisit at core-3, where SIMD kernels are likely out of Kani's reach. Drop
the job if no harness has caught anything the exhaustive tests did not by then. The factored range
functions stay either way.
