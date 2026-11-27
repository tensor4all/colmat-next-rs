# Trial architecture

**Status: proposed in this trial.** Nothing here is a decision of the colmat maintainers. The design
references remain [#2](https://github.com/tensor4all/colmat-next-rs/issues/2) and `tensor4all-meta`
[`docs/design/next-generation-dense-linear-algebra.md`](https://github.com/tensor4all/tensor4all-meta/blob/b4294ec/docs/design/next-generation-dense-linear-algebra.md)
(cited below as "meta", at `b4294ec`; status: design under review). Where this record and those
sources disagree, they win. Each row below says whether it restates a source or is proposed here.

The colmat-rs findings this record relies on are in
[`docs/worklogs/2026-11-20-colmat-rs-design-debt.md`](../worklogs/2026-11-20-colmat-rs-design-debt.md)
(colmat-rs at `71337cdc`).

## 1. Position

- This repository carries out the `colmat-core` split of #2 ("Rough order", step 3) and the steps
  that depend on it as a new codebase: code is ported from colmat-rs where its design fits, and
  restructured where it does not. Proposed.
- Whether development continues here or in colmat-rs is for the colmat maintainers (README,
  AGENTS.md). Results from here can also be offered to colmat-rs as patches; whether colmat-rs takes
  them is for its main maintainer. Either way, the work of this trial happens in this repository.
- The audit found that the extension points #2 asks for (per-kernel selection, a per-(scalar,
  kernel) cost model, a precision contract, alternative drivers) cannot be added to colmat-rs at the
  edges: each needs edits inside matmul dispatch, the scalar trait or the drivers. That is the
  technical case for restructuring rather than wrapping colmat-rs as a dependency.

## 2. Principles (proposed)

1. **Port selectively, with provenance.** Pieces whose design already matches #2 are ported and keep
   faer's copyright notice and MIT text, with repository, path and commit
   ([`REPOSITORY_RULES.md`](../../REPOSITORY_RULES.md), rule 1). Pieces that carry the debts in the
   audit are restructured before or while porting, not copied and fixed later.
2. **Back-portable units.** Kernels, cost rows and team skeletons depend on as little as possible,
   so that each could also be written as a colmat-rs patch. A pull request that adds such a unit
   says in its work log what the colmat-rs patch would look like. It does not prepare the patch.
3. **One-way dependencies, checked.** Lower crates never name higher ones. Two changes must stay
   inside one crate each: adding a scalar or kernel must not touch a driver, and adding a driver
   must not touch a kernel. CI enforces the dependency rules that `cargo tree` can see (meta §7a:
   no direct `rayon`/`threadteam` or `std::thread` below `colmat-core`).
4. **Native non-regression is a gate.** Every boundary keeps the default fast paths of `f64`,
   `c64`, `c32` and `f32`, measured against a pinned colmat-rs baseline, serial and team (meta
   §7a "Native non-regression", §11 acceptance checks; #2, last comment). Restated from the sources.
5. **Small first, seams first.** The first slice implements little, but places the trait
   boundaries where the full design needs them. Moving a boundary later is the expensive change.

## 3. What this trial takes from #2 and meta

| Topic | Source | How this trial applies it |
|---|---|---|
| Goals: coarse team-aware primitives, per-scalar cost, `Df64`, `Par` everywhere, caller-owned workspace | #2 body; meta §1 | Restated |
| CPU only; no stream, async or batch abstractions | #2 scope; meta §9 | Restated (`REPOSITORY_RULES.md` rule 3) |
| AoS storage, SoA in registers, conversion at load/store and at packing | #2 comment 2026-11-12; meta §6 | Restated |
| Primitive kinds P1–P10; only P1–P3 need a per-scalar override | #2 comment 2026-11-13; meta §7 | Restated |
| Scalar layers S0 (arithmetic, precision) required; S1 SIMD, S2 kernels, S3 cost optional | #2 comment 2026-11-13; meta §7a | Restated |
| No master trait over all level 3; no BLAS `[ss][op]` naming; find hooks from existing prologues | Markus Wallerberger in #2; meta §7a | Restated. Kernel hooks are per kind and separately selectable |
| CPU features chosen once outside the loop; no per-element or per-call trampoline | #2 comments 2026-11-13; meta §7a | Restated |
| Precision contract `eps`, `safe_min`, `safe_max`; values under review | #2; meta §5 | Restated. Values not chosen here |
| Rates per (host class × scalar × kernel family); block sizes from the same model | #2 body; meta §8 | Restated. §5 proposes that the selected kernel supplies its row |
| Crate layout `colmat-traits` → `colmat-core` → `colmat`, `colmat-xprec` | #2 body; meta §7a | Adopted as the trial layout (§4) |
| Review boundaries with Markus and the maintainers before implementing the redesign | meta §11 | The first slice is a trial implementation that feeds that review; its boundaries stay "proposed" until recorded in meta |

## 4. Crate layout in this trial

```
colmat-traits   S0 arithmetic + Precision, S1 SimdField (optional), no SIMD in the scalar trait
      |
colmat-core     views (Mat/Col/Row/Diag/Perm), zip and layout copies, Par/ParShape,
                team skeletons, cost model, P1-P3 hooks with generic and native kernels,
                generic P4-P8, workspace contract
     / \
colmat   colmat-xprec    factorisations | Df64 scalar support and Df64 kernels
```

- Sparse, matrix-free operators, stats and io are out of the first scope. They would sit beside or
  above `colmat` and must not be depended on by dense code. Proposed.
- `colmat-traits` does not depend on `qd` or `xprec`. Extended-precision types attach from
  `colmat-xprec` (how is an open question, meta §7a "Attaching `Df64`"). Proposed.
- `colmat-core` re-exports `Par` and the skeletons; `colmat-xprec` uses only those re-exports
  (#2 body).

## 5. Boundaries proposed for the first slice

These answer meta §11 questions only for the purpose of building the trial. Each stays proposed
until recorded in meta.

- **Scalar.** `ComplexField` keeps only S0. The `simd_*` surface moves to a provider that the
  scalar names in `ComplexField::Simd`, split into data movement (`SimdLayout`) and arithmetic
  (`SimdField`), following meta §7a's classification; real providers also load and store pairs
  (`SimdPairs`), from which the complex provider is derived. A scalar-only type names `NoSimd`, a
  one-lane provider built on S0, so it runs P4–P8 as scalar loops. An associated provider is used
  because stable Rust cannot specialise on an optional trait; the type writes one line for it.
  `RealField` keeps colmat-rs's `num_traits::{Num, NumAssign}` bounds, which `num_complex`'s
  division needs. Implemented in `colmat-traits` (#6).
- **Capabilities, not identity.** No dispatch on `IS_NATIVE_*`, `TypeId` or `size_of`. A fast path
  is reached through a kernel provider. Where a provider asserts layout equivalence with a native
  type (for reinterpretation), that assertion is an `unsafe trait`, so safe code cannot reach a
  transmute through a wrong constant. Accuracy policy (for example reciprocal-multiply) is a
  separate named item, not a side effect of "native".
- **Kernel hooks.** One trait per hook, starting with P1 (`gemm` with a triangular-output variant).
  P2 and P3 first go through P1 inside generic drivers; a separate hook is added only if
  measurement shows the base case matters (meta §7a table). Each hook has a generic fallback. The
  selection mechanism (associated provider on the scalar, or an explicit provider argument) is open
  in meta §11. The trial prototypes the associated provider with a default, because #2's body
  chose static binding and operators need a default, and keeps the explicit form possible for
  adapter crates. Implemented for P1 (#23). The scalar names its provider in
  `ComplexField::Kernels` (`colmat-traits`). The hook `Gemm<T>` lives in `colmat-core`, next to
  the views its contract is written in, and drivers ask for `KernelField`. Stable Rust has no
  associated-type defaults, so the "default" is a line each scalar writes: `GenericKernels` (every
  hook for every scalar) or `NativeKernels` (`f32`, `f64`, `c32`, `c64`). The matrix product's
  driver keeps the empty, vector and rank-one cases on the generic kernels and hands the rest to
  the provider. The triangular-output variant is not in #23.
- **Cost.** A hook's provider declares its rate per kernel family. The cost model reads the row of
  the selected (host class, scalar, kernel), with a ratio to `f64` as fallback. Block sizes and
  recursion crossovers are derived from the same row, not from literals in drivers. The fallback
  ratio is implemented in `colmat-traits` (#20): the scalar states a `CostWeight` (one weight for
  compute-bound and one for memory-bound families, meta §8) as `ComplexField::COST`, an associated
  constant whose default is `f64`'s, so a new scalar writes nothing unless it wants a different
  price; `f32` and `c32` declare half. Each family of the rate table is tagged compute- or
  memory-bound. The per-kernel rows come with the kernel hooks.
- **Precision.** `Precision { eps, safe_min, safe_max }` is required; values are the type's own.
  A kernel that changes the accumulation algorithm (packed Dot2, Ozaki-type split) documents its
  error bound and valid range against the generic operation it replaces (meta §5, §7).
- **Workspace.** Meta §6 "Caller-owned workspace" as written: query and execution agree on the
  path, no heap or thread-local allocation during execution, capacity errors before outputs change.
  Solvers take scratch, so iterative refinement does not allocate per step. Until meta §11
  question 3 settles the error type, an insufficient stack panics at entry, before any output is
  written (#20). The packed native products (`t4a-gemm`, and `t4a-private-gemm-x86` on x86-64)
  take their packing memory from the caller's stack in `matmul_with_scratch` (#34; it was the
  exception of #23 until their `gemm_with_stack` entries). Proposed in this trial (#34): the
  allocating entries `matmul` and `matmul_with_conj` let the provider find its memory itself
  (`Gemm::gemm_alloc`), which for these kernels means their thread-locals, so a small product
  does not pay for sizing and allocating a stack on every call. On x86-64,
  `t4a-private-gemm-x86` asks for about twice the L3 size per participant whatever the shape;
  that request reaches only a caller of `matmul_with_scratch`, and the rustdoc of
  `matmul_with_scratch` and `matmul_scratch` states it.
- **Execution.** `Par::Team` wraps a threadteam token, behind `colmat-core`'s `threadteam` feature
  (meta §7a's layout puts `threadteam` under `colmat-core`). The team skeletons deal a
  `colmat-core` trait (`utils::thread::Producer`) rather than rayon's producer (debt K7), so the
  view types and the skeletons name no rayon item (#20).
- **Rounding on a team (#28).** A product on a team splits its output and computes every element
  as `Par::Seq` does, so the result is bitwise the same, with one exception. A column-major
  matrix-vector product above the size of `depth_split` (a depth of at least 1024, and a matrix
  larger than 8 MiB or columns of at most 512 bytes, measured on an Apple M1 Pro) splits its
  depth, as colmat-rs did: each participant sums its block of columns, and the partial sums are
  added in order. That result rounds differently from `Par::Seq`, within the same error bound,
  and is the same on every call of the same width, since the partition and the order of the
  addition depend on the shape and the width only. The partial sums (`nrows x width`) come from
  the caller's scratch: `matmul_scratch` asks for them when `par` is a team and the shape takes
  the split. The packed native products call their kernel once per tile, so the bitwise equality
  also needs the kernel to group the depth of every element the same way whatever the tile's
  shape: `t4a-gemm` cut it by the output's shape until tensor4all/t4a-gemm#25, which made `f64`
  `257 x 3 x 513` differ between `Par::Seq` and a team (#45). For `t4a-private-gemm-x86` this is
  read from the code but not tested (tensor4all/t4a-private-gemm-x86#24).
- **Drivers.** Factorisations call primitives and shared leaves only. Panel leaves (pivot search
  plus rank-1/2 update), layout copies and the cost queries are public items of `colmat-core` or
  `colmat`, so an alternative driver or an algorithm outside the crate can reuse them. An
  alternative driver is selected through the driver's params, not a new top-level method.

## 6. What is ported, restructured or left

| colmat-rs piece (`71337cdc`) | Treatment | Why |
|---|---|---|
| `Par`/`ParShape`, `with_width`, `Fork` adapter for external kernels | Port | Matches #2 goal 4 |
| `StackReq`/`MemStack` convention, no-alloc tests | Port, tighten to meta §6 | Base of the workspace contract |
| `HostClass`, `Rates`, `pays_at`, `ParThreshold` | Port, rekey by kernel | Rule is right; key is too coarse |
| `partition::lower_area_split`, `tile_grid` | Port, use on every target | Today gated to x86 |
| `utils/simd.rs`, `SimdCapabilities` | Port onto `SimdField` | Scalar-generic already |
| `mat/`, `col/`, `row/`, `diag/`, `perm/` with `zip` and layout copies | Port without rayon in the view types | #2 checked the dependency; rayon is the remaining coupling |
| `ComplexField`/`RealField` | Restructure (S0/S1 split, `Precision`) | Debt T1–T4 in the work log |
| `matmul` dispatch, x86 detection ladders | Restructure behind P1 hooks | Debt K1–K3 |
| Factorisation drivers | Port per driver after leaves are shared and literals moved to the cost model | Debt D1–D4 |
| Owned-result wrappers | Rewrite on parallel copy primitives | Debt D5 |
| `fx128` in `colmat-traits` | Leave; extended precision attaches from `colmat-xprec` | Debt C2 |
| Sparse, operator, stats, io, ffi | Leave out of the first scope | Not needed by the first users |

## 7. First slice and order (proposed)

1. `colmat-traits`: S0, `Precision`, `SimdField`; `f64`, `f32` and the generic complex.
2. `colmat-core`: views, `Par`, skeletons, cost model, workspace; the P1 hook with the native
   kernels and a generic kernel. Gate: native non-regression (principle 4).
3. `colmat-xprec`: `Df64` support and a packed P1 kernel, with the generic kernel as the portable
   default. x86 SIMD kernels are later work.
4. `colmat`: tall thin QR first (the shape of sparse-ir-rs's SVE workload), then the drivers it
   needs.

Each step is an issue and a pull request in this repository.

## 8. Alternatives considered

- **Depend on colmat-rs and add only new pieces.** Rejected: kernel selection, cost keys and
  driver leaves are internal to colmat-rs, so a new kernel could not reach existing
  factorisations.
- **Write everything from scratch.** Rejected: the token, workspace convention, cost rule and
  partition helpers already match #2 and have tests.
- **One `Kernels<T>` trait over all level-3 operations** (#2 body sketch). Not adopted, following
  Markus Wallerberger's review in #2 and meta §7a.

## 9. Open

- Meta §11 questions 1–4 (hooks and selection, scalar and precision contracts, workspace API,
  exception handling) remain open; this trial's choices in §5 are inputs to that review.
- Licences of `qd` and `xprec` for anything ported or depended on from them: to be checked before
  `colmat-xprec` starts.
- Final crate names and publishing belong to the maintainers. The licence is MIT (#29).
