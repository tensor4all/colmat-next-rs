# colmat-core views: port, unsafe review and defects found (#12)

A record of the port of colmat-rs's view layer into `crates/colmat-core` (#12), of how its unsafe
code was documented, and of the soundness defects that documentation found. Everything here
describes this trial; nothing is a decision of the colmat maintainers.

## What was ported

From colmat-rs `faer/src` at `71337cdc`: `mat/`, `col/`, `row/`, `diag/`, `perm/`,
`linalg/zip.rs` (now `colmat_core::zip`), `utils/bound.rs`, the shape, stride and index traits
of `lib.rs`, the `PartialEq` impls of `linalg/mat_ops.rs`, and the serial layout copies of
`linalg/mod.rs` (now `colmat_core::layout`). Removed or deferred:

| Item | Where it goes | Why |
|---|---|---|
| `par_*` iterators of the views (49 `#[cfg(feature = "rayon")]` items) | Not ported; parallel traversal returns through `Par` (core-2) | Debt K7 |
| Parallel path of `copy_across_layouts`, `column_major_copy`; the public copy entry | core-2 | Needs `Par`; the serial `copy_across_layouts` is `pub(crate)` so that the public name is first published with its `Par` argument |
| `permute_{rows,cols}_in_place{,_scratch}` | core-2 | Needs the workspace contract (`MemStack`) |
| `norm_*`, `sum`, `determinant`, `kron` on the views | `colmat` (step 3) | Depend on `linalg` |
| `mat!`/`col!`/`row!` `@alloc` arms, `stack_mat!` | core-2 | Use `temp_mat_*` |
| `hacks::hijack_debug` | Dropped | `Debug` of a row now prints `T`'s own `Debug` |
| Dead after the above: `SyncCell`, `as_type` (×3), `Conj::apply_rt` | Dropped | Unused |

`Mat == Mat` with equal row counts and different column counts now returns `false`; colmat-rs
panics in `as_shape` (`linalg/mat_ops.rs:71-98` checks only `nrows`).

## Unsafe documentation

The workspace denies `unsafe_op_in_unsafe_fn`, `clippy::undocumented_unsafe_blocks` and
`clippy::missing_safety_doc`. colmat-rs has none of these on, and the ported files had no
`// SAFETY:` comments. Steps:

1. `cargo fix` wrapped the bodies of `unsafe fn` (131 sites) in one `unsafe { }` each.
2. Every block and `unsafe impl` (about 400 sites) got a `// SAFETY:` comment, under the policy
   agreed for #12: one block per justification, not per operation; the view invariant stated once
   (crate docs, "The view invariant") and referred to; owned-matrix and permutation invariants
   stated once near their types; `clippy::multiple_unsafe_ops_per_block` off.
3. Sites whose precondition could not be justified were marked `TODO(review)` instead of given a
   justification, and each was then checked with Miri (below).

Doc-only fixes made on the way: `MatRef::from_raw_parts` and `MatUninitMut::from_raw_parts_mut`
now state that the pointer is non-null even for an empty view (the body calls
`NonNull::new_unchecked`; colmat-rs only implied it); several `# Safety` sections that stood above
the summary line were reordered; a corrupted doc comment on `MatMut::bind` was restored to
`#[doc(hidden)]`.

## Defects found

Each was reproduced under Miri (`cargo +nightly miri test --release`, Stacked Borrows) in a scratch
crate against this port. The code involved is unchanged from colmat-rs `71337cdc` (checked by
reading the colmat-rs source; the probes were not run against colmat-rs itself, nor against
upstream faer).

| # | Where (colmat-core) | Input (safe code unless noted) | Result |
|---|---|---|---|
| 1 | `Mat::resize_with` (`mat/matown.rs`, `noalias_annotate`, `init_with`) | `Mat::from_fn(2, 1, ..)` of a `Drop` type, then `resize_with(4, 1, f)` where `f` panics at `i == 3`, under `catch_unwind` | The panic guard covers rows from 0 instead of from the old row count: rows the matrix still owns are dropped, then dropped again by the matrix; new rows leak. Miri: UB (retag of a dropped location) |
| 2 | `Mat::truncate` (`DropMat::drop`, `truncate`) | `1 × 3` matrix whose element 1 panics on its first drop, `truncate(1, 1)` under `catch_unwind` | Element 1 dropped three times (`[1, 1, 0, 1, 2]`): the drop guard does not advance past a panicking column, and `truncate` updates the dimensions only after the drop |
| 3 | `Perm`/`PermRef` with `N = One` (`perm/`, `Shape::cast_idx_slice` in `lib.rs`) | `PermRef::<u32, One>::new_checked(&[Zero], &[Zero], One)` | `cast_idx_slice` assumes `Idx<N, I>` has `I`'s layout; for `One` it is the zero-sized `Zero`. Miri: UB (4-byte read from a 0-byte allocation). `Perm::into_shape(One)` rebuilds a `Box` with the wrong layout |
| 4 | `for_each_triangular_upper_with_index`, transposed-and-reversed operand (`zip.rs`) | `a = Mat::zeros(2, 4)`, `b = Mat::zeros(4, 2)`, `zip!(a.as_mut().transpose_mut().reverse_cols_mut(), &mut b).for_each_triangular_upper_with_index(Diag::Include, ..)` | `m - j - strict` underflows for `j + strict > m`; in release the loop indexes out of bounds. Miri: UB (out-of-bounds offset) |
| 5 | `ZipEq::new_unchecked` (`zip.rs`) | `ZipEq::new_unchecked(zip!(&mut a), zip!(&mut b))` with `a` `8 × 8`, `b` `1 × 1` | A safe function that only `debug_assert!`s matching shapes, while the iteration relies on them. Miri: UB (out-of-bounds read) |
| 6 | `Array::as_mut` (`utils/bound.rs`) | Two calls on one `Array`, writes through both | Returns `&'a mut [T]` with a caller-chosen `'a`, not tied to `&mut self`. Miri: UB (aliasing `&mut`) |
| 7 | `Index<Range<IdxInc<'n>>>` for `Array` (`utils/bound.rs`) | `&a[n.idx_inc(2)..n.idx_inc(1)]` | Without debug assertions, `start <= end` is not checked before `get_unchecked`. Miri: UB (`unchecked_sub` overflow) |

Also marked, not reachable as UB in the cases tried:

- `zip.rs`, lower-triangle `None` arm and upper-triangle `Transpose` arm: for wide (resp. tall)
  matrices the contiguous branch calls `get_slice_unchecked` with an out-of-range start and a
  wrapped length, which breaks its contract; the loop range is then empty, so nothing is read.
- `utils/bound.rs`: the `<= I::Signed::MAX` clause of the `Idx`/`IdxInc` invariant is not enforced
  by `Dim::new`/`Dim::with`; no unsafe code was found that depends on it.

These are left as `TODO(review)` comments in this port; their handling is decided separately.

## Tests

- The 34 unit tests and 31 doctests of the ported modules (doctests renamed `colmat::` →
  `colmat_core::`).
- New in `src/tests.rs`: property tests of every view operation against a reference offset model
  (shapes 0–8, strides −12..12 including 0 for read-only views, padded and reversed layouts for
  mutable ones), disjointness and coverage of mutable splits, `zip!` across layouts, owned-matrix
  growth and truncation, the tiled layout copy, and `permute_rows` against its forward and inverse
  arrays; unit tests for the copy thresholds and the triangular tiled copy.

## Miri and proptest

Under Miri, proptest needs two settings: `-Zmiri-disable-isolation` (it reads the working directory
to locate its failure-persistence file) and `PROPTEST_RNG_ALGORITHM=xs` (its default ChaCha RNG uses
NEON or SSE intrinsics that Miri does not implement on every target). The nightly Miri job sets both.
This applied to `colmat-traits`'s property tests already; the nightly job had not run them yet.
