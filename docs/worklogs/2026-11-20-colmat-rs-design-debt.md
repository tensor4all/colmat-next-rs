# colmat-rs design debt against the goals of #2

A read-only review of colmat-rs at `71337cdc` (origin/main, 2026-11-20). It asks one question: which
parts of colmat-rs's structure make the extension points of
[#2](https://github.com/tensor4all/colmat-next-rs/issues/2) hard to add, and which parts already fit.
It feeds [`docs/design/trial-architecture.md`](../design/trial-architecture.md). It judges structure
against #2's goals, not code quality in general. Many of these choices were reasonable for faer's
goals.

Paths are under `faer/src/` unless they start with `faer-traits/`. "Structural" means the fix
changes a public trait or signature. "Local" means a patch inside colmat-rs can fix it.

## Scalar and trait layer

| ID | Finding | Evidence | Kind |
|---|---|---|---|
| T1 | One trait holds scalar arithmetic and the SIMD surface. Of `ComplexField`'s 56 required functions, 37 are `simd_*`, and its associated types include the SIMD context, vector, mask and index types. A scalar cannot exist without the SIMD surface, and one scalar cannot have a second kernel set. (Also counted in #2, comment of 2026-11-13.) | `faer-traits/src/lib.rs:1257-1290` | Structural |
| T2 | The default load, store, mask-load and `iota` hooks branch on `Unit::IS_NATIVE_F32/F64` and otherwise `panic!()`. Every SIMD scalar must be built from `f32` or `f64` lanes. | `faer-traits/src/lib.rs:1521-1616` | Structural |
| T3 | The SIMD backend is fixed to `pulp` in every signature. | `faer-traits/src/lib.rs:1179-1192` | Structural |
| T4 | No precision contract beyond `epsilon_impl`/`nbits_impl`. The accuracy policy (for example, whether reciprocal-multiply is allowed) is expressed through `IS_NATIVE_*` and comments. | `faer-traits/src/lib.rs:1632-1633`; `linalg/pfaffian/mod.rs:276-287` | Structural |
| T5 | The `fx128` SIMD multiply-add uses `qd::simd::add_estimate`, which is weaker than the scalar path's addition. Nothing in the type states this. | `faer-traits/src/lib.rs:5483` | Local |

## Kernel dispatch, cost model, threading

| ID | Finding | Evidence | Kind |
|---|---|---|---|
| K1 | GEMM kernel selection branches on `IS_NATIVE_F64/C64/F32/C32`. Every other type goes to the generic `matmul_vertical`/`matmul_horizontal` path. There is no hook through which another crate could supply a kernel. Across `faer/src`, `IS_NATIVE_` occurs 75 times in 16 files (reductions, triangular solve, LU, Cholesky, pfaffian, cost). | `linalg/matmul/mod.rs:1567-1595` | Structural |
| K2 | `IS_NATIVE_*` are `#[doc(hidden)]` constants of a safe trait. A downstream type that sets one to `true` reaches `transmute_copy` to the native type, which is undefined behaviour reachable from safe code. | `faer-traits/src/lib.rs:1272-1279`; `linalg/matmul/mod.rs:1396-1411` | Structural (soundness) |
| K3 | Type identity is decided in three ways: `IS_NATIVE_*`, `TypeId` (`x86_kernel`), and `size_of` ratios. The x86 feature-detection ladder is repeated across matmul, triangular, team and cost (25 `is_x86_feature_detected` sites outside tests) instead of a single cached selection. | `utils/team.rs:498-512`; `linalg/matmul/mod.rs:46-53` | Local |
| K4 | `Rate::of<T>` keys rates by (host class, family, real or complex). Any non-native scalar takes the `f64`/`c64` rate (#2 body, "Gap"). | `utils/cost.rs:261-273` | Structural |
| K5 | Block sizes and recursion thresholds are literals in drivers and ignore the scalar and the kernel: for example `F64_RECURSION_THRESHOLD`, `LLT_BLOCK_SIZE`, the QR block-size ladder, and the `n <= 16` base cases in `triangular.rs`. | `linalg/lu/partial_pivoting/factor.rs:768`; `linalg/cholesky/llt/factor.rs:70`; `linalg/qr/no_pivoting/factor.rs:85-129` | Local |
| K6 | The equal-area split is used only on x86-64 with the private x86 kernel, and is `allow(dead_code)` elsewhere. Other targets use binary recursion. | `utils/partition.rs:15-25`; `linalg/matmul/triangular.rs` | Local |
| K7 | Rayon remains a mechanism: its producer traits are the item-dealing interface of the team helpers, and the matrix view types expose rayon iterators. | `utils/mod.rs:148-241`; `mat/matmut.rs` (30 `rayon` references) | Structural for the views |

## Drivers and API

| ID | Finding | Evidence | Kind |
|---|---|---|---|
| D1 | Each driver has its own SIMD panel leaf (pivot search plus rank-1/2 update): partial-pivot LU, full-pivot LU, Bunch–Kaufman and LDLT. There is no shared leaf interface. | `linalg/lu/partial_pivoting/factor.rs:23-43`; `linalg/lu/full_pivoting/factor.rs:104-400`; `linalg/cholesky/bunch_kaufman/factor.rs` | Structural |
| D2 | The building blocks an alternative driver would reuse are `pub(crate)`: `lu_in_place_recursion`, `cholesky_block_left_looking`, `qr_in_place_packed`, `copy_columns_into`, `split_lu_in_place`, and the cost functions. There is no driver selection: each method is wired to one driver. | `linalg/lu/partial_pivoting/factor.rs:504`; `linalg/cholesky/ldlt/factor.rs:615`; `linalg/qr/no_pivoting/factor.rs:744`; `linalg/mod.rs:375,433` | Structural |
| D3 | Drivers described as in-place allocate on the heap outside the `MemStack`: a column-major copy in LU and column-pivoted QR, and a leaf buffer in LU. The scratch query does not see the layout. | `linalg/lu/partial_pivoting/factor.rs:501,907`; `linalg/qr/col_pivoting/factor.rs:482` | Structural |
| D4 | The solver traits take no scratch. Wrappers allocate a `MemBuffer` per call, so iterative refinement allocates on every step. | `linalg/solvers.rs:568,605` | Structural |
| D5 | Owned-result wrappers do serial set-up: `Llt::new` zero-fills with `Mat::zeros`, then copies the triangle, then factors. SVD and EVD wrappers zero-fill outputs that the driver overwrites. Only `PartialPivLu::factor_into` uses parallel copies and buffer reuse. | `linalg/solvers.rs:990-1000,1323-1370` | Local |
| D6 | Some tuning lives in the owned wrappers (for example a small-tall rule in `Qr::new`), so in-place callers and alternative drivers get different defaults. | `linalg/solvers.rs:1537-1551` | Local |

## Crate and module structure

| ID | Finding | Evidence | Kind |
|---|---|---|---|
| C1 | One crate, `colmat` (about 143k lines), holds dense, sparse, matrix-free operators, stats, io and the parallel runtime. Dense operators call into sparse (`linalg/mat_ops.rs:3076`), and `mat` calls `linalg::zip`. | `lib.rs:690-710` | Structural |
| C2 | `fx128 = qd::Quad` and its `ComplexField` impl live in `colmat-traits`, which depends on `qd`. A separate extended-precision crate needs this moved out. | `faer-traits/src/lib.rs:5189-5708`; `faer-traits/Cargo.toml` | Structural |
| C3 | Divergence from upstream faer is large: since the last sync base (`0ecd846`), `faer/src` and `faer-traits/src` changed in about 195 files (+49.9k/−5.9k lines), and `Par<'team>` changed every compute signature. Upstream changes to `linalg` now conflict on signatures. (The sync cost is an inference from this.) | `PATCHSET.md`; `git diff --shortstat 0ecd846 71337cdc` | — |

## What already fits #2 and is worth porting

- `Par<'team>` (exclusive, reborrowed, split into sub-teams) and the `Copy` `ParShape` used by
  scratch queries; `with_width`; the `Fork` adapter that keeps external kernels on the team
  (`lib.rs:1082-1099`; `utils/team.rs:541-560`).
- `*_scratch → StackReq` with a caller-owned `MemStack`, and the no-alloc integration tests.
- `HostClass → Rates` with named kernel families and documented provenance, the single `pays_at`
  rule, and `ParThreshold::{Derived, Fixed, Never}` (`utils/cost.rs`).
- `partition::lower_area_split`/`lower_area_parts` and `tile_grid`.
- `utils/simd.rs` (scalar-generic head/body/tail iteration over branded dimensions) and
  `SimdCapabilities` as a capability gate.
- `Auto`/`Spec` params with `#[non_exhaustive]`; SVD composing QR params; LQ as transposed QR;
  `factor_into` buffer reuse.

## Conclusion for this trial

The extension points of #2 (P1–P3 hooks, per-(scalar, kernel) cost, `Precision`, alternative
drivers, `colmat-xprec`) collide with T1–T4, K1–K2, K4, D1–D4 and C1–C2, which are structural in
colmat-rs. The pieces in the previous section match #2 and can be ported. The trial therefore ports
selectively and restructures at those seams (`trial-architecture.md` §6). This does not decide
whether colmat-rs should change; that is for its maintainers.

## Limits

- One snapshot (`71337cdc`); a module-level review, not a line-by-line audit. Evidence lines point
  to that commit and will drift.
- Line counts and occurrence counts come from text search and include comments.
- No measurement is part of this record.
