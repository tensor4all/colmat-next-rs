# Kernel selection and the tuning harness

**Status: proposed in this trial (2026-11-28).** This is a design for review in #44, #45 and #46.
§11 (the backend boundary, #60) records decisions the maintainer took on 2026-11-28 and revises
level 1 of §3. §12 lists what a port must carry over from colmat-rs; §11.8 was corrected
by that audit.
It is not a decision of the colmat maintainers. It does not change any code. Where it relies on
colmat-rs, it cites the commit or the work log.

## 1. What this record answers

- **#45:** where the choice of a kernel for a BLAS-like primitive is made, and what the call must
  carry so that measurements can later change that choice without changing the factorisations
  that call the primitive.
- **#44:** how measurements become a selection rule. The rule is decided per primitive, per size
  and per team width (the maintainer's correction in #44: the BLAS primitive, not the LAPACK
  routine). It is not a build feature.
- **#46:** the harness that produces those measurements. It is built and validated on colmat-rs
  first, then rebuilt for this repository.
- The mechanism question raised in review: **are kernels reached through function pointers?**
  §3 answers it.

## 2. What recent colmat-rs work shows

| Evidence (colmat-rs) | What it implies for selection |
|---|---|
| The `is_x86_feature_detected!` ladders (AVX-512 → AVX2+FMA → scalar) are repeated in `matmul/mod.rs`, `matmul/triangular.rs`, `matmul/internal/mod.rs`, `utils/team.rs` and `utils/cost.rs` at `9155fa1`. This repository repeats the ladder again in `linalg/matmul/native.rs` (`x86_instr`). | The host is probed in many places, and nothing guarantees that the cost model and the kernel saw the same answer. The probe should run once per process, and every consumer should read its result. |
| `t4a-gemm` already chooses its kernel once per process. `init_gemm_fn` stores a function pointer in an `AtomicPtr`, and `get_gemm_fn` reads it on later calls (`t4a-gemm-common` 0.20.4, `gemm.rs` around line 1291). The `t4a-private-gemm-x86` kernel tables (`F64_SIMD512x4`, ...) are arrays of kernels. | Below the primitive, kernels are already reached through function pointers resolved at run time. The open question is only the layer above them. |
| The SME route needed an lldb hit-count script to find out which kernel ran (work log `2026-11-22-sme-route-180`). The `x86-v4` isolation needed a patched `pulp` that skips V4 when an environment variable is set (`2026-11-26-x86-v4-isolation-297`). Build features cannot separate the arms, because Cargo unifies them. | Selection must be **observable**: which kernel ran should be readable without a debugger. It must also be **overridable at run time** for experiments. Neither is possible while the choice is buried inside each dependency. |
| A small product on a team was 0.48x AOCL because of the kernel's own job grid, and was fixed by tiling the output above the kernel with an 8 ms serial-work ceiling (`2026-11-21-gemm-team-tile-split`, colmat-rs#262). The ceiling is set by a cache-aliasing cliff at `512^3`, not by a smooth crossover. | The choice includes the schedule (tile grid or the kernel's own grid) as well as the kernel. Thresholds can be cliffs, so they must come from measurements, not from a fitted formula. |
| The SME gate inside `t4a-gemm` sees the shape of each sequential call. On a team that tiles the product, that shape is the tile, not the whole product (`2026-11-22-sme-route-180`, "The gate sees the shape of each sequential call"). | **Confirmed at the dispatch level** (2026-11-28, on an M2, by calling `split_tiles`, `par_width_time` on real teams of 2–8 and `sme_thresholds::wants_sme` at the pinned `t4a-gemm` `29b5b46`; no SME hardware): a team reaches a different kernel than `Par::Seq` in 1,405 of 3,753 tiled f64 cases and 1,510 of 3,351 tiled f32 cases (m, n in 16–512, k in 64–512). 110 f64 and 91 f32 cases mix SME and NEON tiles in one product. Example: f64 `16x128x128` is SME as one call, but its two `16x64` tiles are NEON (`m*n` below 2048). Whether the rounding differs needs an M4 or later. Choosing on the tile is right for speed, since NEON does win on small calls, so this record weakens the rounding rule instead of overriding the gate (§5.3, §5.5). |
| `x86-v4` as a build feature cost a whole campaign to answer one yes/no question, and the answer was "no gain on average, a gain for some families and sizes" (#44). | Selection is per (primitive, dtype, size, width) at run time. It is not per build. |

## 3. Decision: two levels of dispatch (proposed)

**Level 1, type level: static.** The scalar names its provider (`ComplexField::Kernels`), and the
hook traits (`Gemm<T>`, ...) are implemented for the provider and the scalar together. This is
the design of #23. **Revised by §11:** the provider becomes a backend value `B: Backend<T>` that
the caller passes, no longer tied to the scalar; the rest of this paragraph holds for it
unchanged. It answers "which implementations exist for `T`". It
costs nothing at run time, keeps `transmute` out of safe code, and lets `colmat-xprec` attach
`Df64` kernels without a registry. **Function pointers are not used here.** A generic scalar never
pays for an indirect call.

**Level 2, host and call level: a kernel table resolved once, and a plan per call.** Inside a
provider that has several implementations for one type (`NativeKernels` today, `colmat-xprec`
later), the choice among them goes through two things:

1. **A `KernelTable` per (provider, scalar), built once per process.** The CPU is probed once (ISA,
   SME, cache sizes, host class) and the candidate kernels are stored as function pointers in a
   `OnceLock`. The table also holds the selection thresholds that apply to this host class (§6).
   **This is the function-pointer layer.** It is how `t4a-gemm` already works, lifted one layer up
   so that one table covers every backend crate.
2. **A `Plan` per primitive call, computed from the table, the whole call's shape, its layout and
   the team width.** The plan fixes the kernel, the blocking (`kc` included), the schedule (tile
   grid or the kernel's own grid, depth split or none) and the scratch size. Scratch query,
   cost estimate and execution all take the same plan, so they cannot disagree. Tiles and
   sub-calls receive the plan and do not choose again.

Why not function pointers everywhere: the indirect call itself is cheap (a few nanoseconds, not
measured here) next to the 20 µs dispatch floor of a team region. But it blocks inlining, and the
generic path and the small-product path (`nano-gemm` below `16^3` multiply-adds) depend on
inlining. These paths stay statically dispatched. The table holds only kernels whose calls are long
enough that an indirect call is not measurable.

Why not a run-time plug-in registry (`dyn Kernels`, kernels registered by users): #2 chose static
binding, and no consumer needs to swap kernels at run time. The table is internal to the provider.
Overrides exist for experiments (§5.6), not as an extension API. A third party replaces the whole
backend statically, by type (§11), not by registering kernels.

## 4. Sketch

The names are provisional. The point is the split between choosing and running.

```rust
/// One BLAS-like primitive of a provider for `T`, split into choosing and running.
pub trait Primitive<T: ComplexField> {
    /// The operation's description: shape, layout class, flags (conj, side, uplo, diag, accum).
    type Call: Copy;
    /// The provider's choice for that call: kernel, blocking, schedule. Small, `Copy`, `Debug`.
    type Plan: Copy + core::fmt::Debug;

    /// Pure function of (host table, call, team width). Deterministic: the same inputs on the
    /// same host give the same plan on every call and on every thread.
    fn plan(call: &Self::Call, width: usize) -> Self::Plan;
    fn scratch(plan: &Self::Plan) -> StackReq;
    fn estimate(plan: &Self::Plan) -> Estimate; // serial ns and useful width, for drivers
    /// Runs exactly `plan`. Panics before writing when the views do not match `plan`'s call.
    fn run(plan: &Self::Plan, args: /* views, alpha, ... */, par: Par<'_>, stack: &mut MemStack);
}
```

- `Gemm<T>` of #23 becomes the first `Primitive`. `gemm_scratch(m, n, k, par)` turns into
  `scratch(&plan(call, width))`, and `ns_per_mac` turns into `estimate`.
- `NativeKernels` implements `plan` by reading its `KernelTable`. `GenericKernels` has a single
  candidate, and its `plan` is a constant.
- The public entries (`matmul`, `matmul_with_scratch`, ...) stay one call each. They plan, check
  the stack, and run. A driver that needs the plan for its own decisions (panel width, look-ahead
  split) calls `plan` itself and passes the plan down.

## 5. The selection contract (#45)

### 5.1 The primitive owns selection

The primitive chooses at its entry point. It may use the scalar, `m/n/k`, the actual strides and
layout class (column-major, row-major, strided), the operation flags, the team width it is given,
and the host table. A caller does not name a kernel. A driver can only pass a team narrower than
its own, or a plan that it got from `plan`.

### 5.2 One plan serves query, estimate and execution

`scratch`, `estimate` and `run` take the plan, never the shape alone. This is meta §6 "query and
execution agree on the path", made structural. The workspace query of a driver is the sum of the
`scratch` of the plans it will run.

### 5.3 Tiles may choose their own kernel; the plan records it

**Speed comes first (maintainer's decision, 2026-11-28).** A schedule that cuts the call into
tiles lets each tile take the kernel that is fastest for the tile's shape. On SME hosts, small
tiles therefore go to NEON even when the whole call would go to SME (§2). The plan computes the
tile grid and the kernel of each tile shape (a grid has at most four tile shapes), so the choice
is still made once per call, is visible in the trace, and is a deterministic function of the
call and the width. A tile does not need to pass the backend's gate again, but it may when the
backend offers no entry that takes a chosen kernel.

This replaces the earlier rule that a team result is bitwise equal to `Par::Seq` (trial design
§5, "Rounding on a team"; #28). The `kc` fix of #45 and t4a-gemm#25 is kept. It removes one
source of variation between widths at no cost, but the contract no longer depends on it.

### 5.4 Validate before the kernel

`run` checks that the views match the plan's call: shape, layout class, and that `dst` is not
aliased. A native kernel's invariants (alignment, minimum stride) are part of the plan's layout
class, so a safe entry cannot reach a kernel with views it does not accept. A plan is `Copy` and
cannot be forged into something that is unsafe to run. `run` asserts the parts of the call that
the plan depends on.

### 5.5 Numerical equivalence

- **Guaranteed: run-to-run reproducibility.** The same binary on the same host class, with the
  same call (shape, layout, flags) and the same team width, gives a bitwise identical result on
  every call. The plan is a deterministic function of exactly these inputs, and no schedule
  depends on timing (no dynamic work stealing, no reduction in completion order).
- **Not guaranteed: equality across widths.** `Par::Seq` and a team, or two team widths, may
  round differently, within the error bound documented for the primitive. Today this happens
  when a tile takes another kernel (§5.3) or when the column-major matrix-vector product splits
  its depth (#28).
- A caller that needs bitwise reproducibility, such as sphaira-rs for a fixed rank count, must
  therefore also fix the team width of each rank. A width fixed by the caller is honoured
  exactly: the cost model may narrow a region only as a function of the call and the width it
  was given, so the result is still a function of the inputs above.
- Different hosts, and overrides (§5.6), may round differently within the same bound.
- Tie-breaking that a driver depends on (pivot search) is part of the primitive's contract and is
  identical across all kernels of the table. colmat-rs#105 (lane-order ties) is the defect this
  rule prevents.

### 5.6 Observability and override

- `Plan: Debug`. A thread-local, opt-in trace (feature `trace-plans`, off by default and with no
  cost when off) records the plans that ran. The harness and the probes read the kernel from
  there, so neither lldb nor a patched `pulp` is needed.
- `COLMAT_FORCE_KERNEL=<primitive>:<kernel>` (feature `tuning`, off by default) makes `plan`
  return the named candidate whenever it is valid for the call. This is the run-time arm switch
  that `x86-v4` lacked. Without the feature, the variable is ignored and the code is not compiled.

### 5.7 Drivers compose primitives

Drivers own the dependencies between operations. They use `estimate` of the plans to choose a
panel width or to split the team for look-ahead. They do not see kernels. Fused sweeps (block
reflector apply, rotation sequences) are primitives of their own, so their data reuse is not lost
at the boundary.

## 6. Thresholds: generated, keyed, validated (#44)

- **Key:** `(host class, primitive, dtype, layout class, size bucket, width) → kernel and
  schedule`. Host class is the probed tuple (ISA level, SME, L2 and L3 size class, vendor), the
  successor of colmat-rs's `HostClass`.
- **Generated, not typed.** A script turns a saved sweep into
  `crates/colmat-core/src/tuning/<host-class>.rs` (a `const` table) with a header that names the
  sweep's manifest. The derivation rule is the one already used on the result pages: switch away
  from the default kernel only when the ratio clears both spreads and a minimum gain (proposed:
  5%), and merge neighbouring buckets that share the same winner.
- **A default that needs no data.** Every key that is not measured falls back to the current
  rule of the provider (today's `native_gemm`). A table can only add exceptions to it, so a
  missing or partial table is never a correctness problem.
- **Holdout.** A table derived on one host is checked on a second host with another
  microarchitecture before it is used there. Disagreements are reported, not averaged away.
- **LAPACK cells validate.** The `cholesky`, `lu`, `qr_factor` and `eigh` cells of
  colmat-benchmark are not used to derive thresholds. They check them: a routine's ratio should
  follow from the primitives' ratios weighted by its blocking. A large miss means a primitive is
  missing from the table or a blocking does not behave as assumed (#44, second comment).
- **Below the threshold is a valid entry.** "Below `m = 64`, keep the narrow kernel" is a
  decision. It is not a gap.
- No install-time or run-time autotuning (ATLAS style). A table is a reviewed source file, so a
  result can be reproduced from the commit alone.

## 7. Walkthroughs

**GEMM.** `plan` reads `(m, n, k)`, the layout and the width:

- `m·n·k ≤ 16^3`: `nano-gemm`, static, no table lookup.
- On a team and below the tile ceiling (8 ms of serial work in colmat-rs): a balanced output tile
  grid, with the kernel and `kc` chosen for the whole call.
- Above the ceiling: the kernel's own grid, which packs the shared panels once.
- On SME hosts: each tile shape gets SME or NEON by the backend's size gate, and the plan
  records which (§5.3).

Exact-size caller scratch is `scratch(&plan)`. On x86-64 that removes the "twice L3 whatever the
shape" request of `t4a-private-gemm-x86` as soon as its plan can carry a shape-sized buffer. That
needs a change in that crate and is not proposed here. Skinny products and strided views are
layout classes of the call, so they are separate keys and do not need separate entry points.

**TRSM and LU.** `trsm`'s plan chooses how the right-hand side is staged (copied by leading
dimension or used in place, colmat-rs `2026-11-05-trsm-aliased-rhs`) and the GEMM plan of its
trailing updates. The LU driver asks for the panel's plan and the trailing update's plan. It
uses their estimates to decide one-panel look-ahead (colmat-rs#239) and how to split the team
between panel and update. It never names a kernel.

**Cholesky.** The panel (a leaf primitive), `herk`/`syrk` on the triangle and `trsm` are three
primitives, each with its own plan. That is why the campaign needs `syrk`/`herk` and `trsm` as
separate families (#44). With plan tracing on, the benchmark records which kernel, ISA and
effective width ran for each of the three, so a Cholesky cell can be explained from its parts.

## 8. The tuning harness (#46)

### 8.1 Stage 1: on colmat-rs

- **Where:** a new runner in `colmat-benchmark`, separate from the publication suites. It reuses
  the existing pinning, host records, the `pins/` mechanism and the paired-arm machinery
  (`pair_feature_arms.py` pairs by `(family, dtype, order, shape, threads)` and works for any
  family). It is not a crate feature of the library.
- **What it calls:** colmat-rs's public `matmul` with a held team of forced width, and
  `t4a-private-gemm-x86`'s public `call_microkernel` for the register-tile stage. The microkernel
  probe of colmat-rs#278 is the template for that stage. colmat-rs has no plan layer, so a
  candidate that colmat-rs cannot select from outside (tile split or job grid, `kc`) is reached
  through a candidate build. That build is a git revision, recorded in the manifest. Nothing is
  added to colmat-rs's API for the harness.
- **First grid:** f64 GEMM on one pinned x86-64 host (evo, Zen 5). Square and skinny shapes chosen
  around the known cliffs (`480^3`/`512^3`/`544^3`, `n = 32..192` on teams). Widths 1, 2, 4 and 8.
  F and C order. Strides equal to the shape and padded.
- **Done when stage 1 is valid** (from #46): a second run reproduces the first within the observed
  spread, and the saved data finds at least one crossover again without being told where it is.
  The tile-split ceiling of colmat-rs#262 is the obvious check.

### 8.2 The workflow

1. **Declare.** A YAML grid lists candidates, timed cells, the reference result, pinning,
   repetitions and the noise rule. Nothing is timed that is not declared.
2. **Screen register tiles** (optional per primitive): the generator's `MR x NR` and unroll
   variants, bounded by the register budget, screened with `llvm-mca` and then timed with
   `call_microkernel` at the four cache residencies of colmat-rs#278.
3. **Measure the whole primitive**: `MC/KC/NC`, packing or staging, tile grid or kernel grid, and
   forced widths. Each candidate is checked against the reference before it is timed. A wrong
   result removes the candidate from the cell and is reported.
4. **Save everything**: one JSONL record per sample, keyed by primitive, dtype, `m/n/k`, strides,
   flags, candidate, requested and effective width, ISA that ran, toolchain, source revisions,
   scratch bytes, and the plan when the build traces plans. The manifest holds the grid, the host
   record and the binary's sha256, as in `gemm-microkernel-bench.sh`.
5. **Derive** the table with the rule of §6, and print the cells that decided each breakpoint.
6. **Validate** on the holdout host and against the LAPACK cells.

### 8.3 Stage 2: on this repository

Once stage 1 has shown that it finds real crossovers, the same runner targets this repository.
Here candidates are switched with `COLMAT_FORCE_KERNEL` instead of candidate builds, and the
kernel that ran is read from the plan trace instead of lldb. Thresholds are measured again on this
repository's own kernels. colmat-rs's table is the baseline that the new table has to beat. It is
not copied, because it encodes the limits of the old blocking (#44, first comment).

## 9. Alternatives considered

- **Function pointers at both levels** (an OpenBLAS `gotoblas_t`-style struct for every scalar).
  Rejected: generic and extended-precision scalars would pay for an indirect call and lose
  inlining, and the static provider of #23 already answers the type question at zero cost.
- **Static dispatch only, with ISA chosen by `cfg` or build features.** Rejected: this is the
  `x86-v4` situation. Cargo unifies the features, one binary cannot serve two hosts, and arms
  cannot be compared in one process.
- **Leaving selection inside each backend crate** (today: `t4a-gemm`'s `AtomicPtr`,
  `t4a-private-gemm-x86`'s probes, the ladder in `native.rs`). Rejected: three probes, no single
  record of what ran, and a per-tile SME choice that nothing outside `t4a-gemm` can see or
  predict (§2).
- **Choosing per call without a plan object** (a lookup at the start of `gemm`, as today).
  Rejected: scratch query and execution would compute the choice separately and can drift, and
  sub-calls cannot inherit the choice.
- **Run-time autotuning.** Rejected for reproducibility (§6).

## 10. Open questions and first steps

Open:

- Whether `Primitive` is one trait with associated `Call`/`Plan`, or each hook (`Gemm`, `Trsm`,
  ...) defines its own `plan`/`scratch`/`run` triple. The latter is closer to #23. Either way, it
  is a question for #2 and meta §11 question 1.
- Host-class granularity: whether the L2/L3 size class belongs in the key or only scales the
  blocking. This needs the holdout data.
- Whether `t4a-gemm` and `t4a-private-gemm-x86` should expose entries that take a chosen kernel
  without re-running their own gates. §5.3 no longer requires them. They would let the plan
  decide instead of copying the backend's gate, so the two cannot drift. Proposing those entries
  is a change in those crates.
- Whether an opt-in "same result at every width" mode is worth offering later (one kernel and one
  `kc` per call, no depth split), for callers that cannot fix their width. No consumer has asked
  for it.

First steps, each its own issue and PR:

1. Apply the weakened contract to the code: the rustdoc of `matmul_with_scratch` ("Rounding on a
   team"), and the test `team_result_is_bitwise_seq`. Split that test into a same-width repeat
   (bitwise) and a cross-width comparison (within the error bound). Run it once on an M4 or
   later, where the cross-width part exercises mixed SME and NEON tiles.
2. Probe the host once: one `HostInfo` in `colmat-core`, read by `native.rs` and by the cost
   model. Nothing else changes.
3. Turn `Gemm<T>` into a plan, scratch and run triple, with `trace-plans`. No new kernels or
   thresholds.
4. Stage 1 of the harness in `colmat-benchmark` (§8.1).
5. `tuning` and `COLMAT_FORCE_KERNEL`, then the first generated table for GEMM on evo, checked on a
   second host.

Step 1 landed as #53 (#57) and step 2 as #55 (#58). Step 3 is now done as part of the backend split (§11.6, #60).

## 11. The backend boundary (#60)

### 11.1 Decisions (maintainer, 2026-11-28)

- **A third party can replace the backend of the primitives.** The LAPACK-level routines take the
  backend as a parameter. The backend trait is defined outside `colmat-core`, and the LAPACK part
  is a generic crate of its own.
- **Level-2 operations belong to the backend**, next to level-3. Panel factorisations spend most
  of their time in level-2: the LU, QR and Cholesky panels, and above all the `latrd`/`labrd`
  panels of tridiagonal and bidiagonal reduction, where level-2 is about half the work.
- **No per-primitive replacement.** A backend implements the whole trait. One that supplies only
  some primitives holds the native backend and delegates the rest in plain Rust. No combinator.
- **Vendor libraries (MKL, Accelerate, ...) are not a target.** The third parties in view are
  other native kernels: `colmat-xprec`, kernels for other CPUs (SVE, RISC-V), research kernels,
  and the tuning harness.
- **Fused reduction steps are backend primitives** (§11.8). The one-step kernels of tridiagonal,
  bidiagonal and Hessenberg reduction stay fused, so the trailing matrix is read once per step.
  They are not rebuilt from separate rank-2 update and matrix-vector calls. The same holds for
  the other fused kernels of colmat-rs listed in §11.9.

### 11.2 Why level 1 of §3 changes

`ComplexField::Kernels` ties the backend to the scalar. `colmat-traits` fixes it to
`NativeKernels` for `f64`, so by the orphan rule nobody outside can give `f64` another GEMM
without inventing a new scalar type. Passing the backend as a value removes that tie. A value
rather than a bare type also lets the backend carry its state: the `KernelTable`, the host
record and the `tuning` override (§3, §5.6). In §4 `plan`, `scratch`, `estimate` and `run`
therefore take `&self`.

```rust
pub trait Backend<T: ComplexField>: Send + Sync {
    // level-3: gemm, herk/syrk, trsm, trmm, swap_rows, block reflector apply, ...
    // level-2: gemv, ger, symmetric/Hermitian matrix-vector and rank-2 update, trsv, ...
    // fused: one step of tridiagonal, bidiagonal and Hessenberg reduction (§11.8)
    // each as a plan, scratch, estimate and run family (§4)
}

pub fn cholesky<T: ComplexField, B: Backend<T>>(
    backend: &B, a: MatMut<'_, T>, par: Par<'_>, stack: &mut MemStack,
) -> Result<(), CholeskyError>;
```

Dispatch stays static: `colmat-lapack` is monomorphised per backend, and a binary normally holds
one. A `dyn` backend is out of scope.

### 11.3 Crates (names provisional)

| crate | contents |
|---|---|
| `colmat-traits` | scalars, as today, without `Kernels` |
| `colmat-base` | views, `Par`/`ParShape`, the team operations (`with_team`, `run`, `for_each`, `for_each_child`), `MemStack`, partition helpers, generic level-1, cost-model helpers |
| `colmat-primitives` | `Backend<T>`, its contract, `GenericBackend` (reference and fallback), and a conformance suite that third parties run in their own CI |
| `colmat-kernels` | `NativeBackend`: `HostInfo`, `KernelTable`, the generated tables (§6), the `t4a-gemm`, `t4a-private-gemm-x86` and `nano-gemm` kernels. The x86 prefetch experiments (colmat-rs#253 E3/E4), deferred to this repository, land here |
| `colmat-lapack` | Cholesky, LU, QR, eigh, SVD, generic over `B` |
| `colmat` | facade; picks the default backend per scalar |

The trait takes safe views, not raw pointers and strides. A wrong backend then gives wrong
numbers, never undefined behaviour, and a backend that calls foreign code keeps its `unsafe`
inside. This is why the views move below the trait crate. The layout changes the candidate
crates of #2 and of meta `next-generation-dense-linear-algebra.md`, which are updated when #60
lands.

### 11.4 Parallelism

- **The tools** (`Par`, the team operations) live in `colmat-base` and are shared.
  `Par::Team(TeamExec)` is the authority over the reserved members, so a backend can only use
  the members it is given. Starting threads of its own is forbidden by the contract; the type
  system cannot prevent it.
- **Inside a primitive** the backend decides the width, the tile grid and any depth split, through
  its plan. It may use fewer members than it is given (§5.1).
- **Across primitives** `colmat-lapack` decides: look-ahead, panel and update overlap, recursion,
  divide and conquer. It splits the team with `for_each_child`, hands each child to a backend
  call, and chooses widths from the backends' `estimate` (§5.7). A nested call sees only its
  child team.
- **Reproducibility** (§5.5): both layers decide from the call, the width and the host class
  only. Claiming disjoint output tiles through a counter is fine, since no tile's result depends
  on which member ran it. A reduction split across members (depth split, norms) has a fixed order.

### 11.5 Contract and stability

The contract of §5 is the backend's contract, written in `colmat-primitives` and checked by the
conformance suite: the result and its error bound, run-to-run reproducibility, the pivot
tie-breaking, running on the given members only, and allocating nothing beyond the planned
scratch. The trait makes no stability promise until Cholesky, LU and QR run on it (stage 4 of
#54). Adding a primitive before then breaks third-party backends, and that is accepted.

### 11.6 Steps

1. Move the views, `Par` and the team operations from `colmat-core` into `colmat-base`. Pure move.
2. `colmat-primitives` with `Backend<T>` holding GEMM only, as a plan, scratch and run triple
   (absorbs §10 step 3). `GenericBackend` and `NativeBackend` implement it, and
   `ComplexField::Kernels` goes away.
3. The conformance suite, run against both backends.
4. Further primitives join the trait as #54 reaches them, level-2 and level-3 alike. The fused
   reduction steps (§11.8) join with eigh and SVD (#54 stage 5). The fused kernels of §11.9 join
   with the driver that needs them: the partial-pivoting leaf with LU, the small reflector
   applies with QR, the rotation sequence with eigh and SVD.
5. `colmat-lapack` appears with the first driver (Cholesky, #54 stage 2).

### 11.7 Open

- Whether level-1 (`nrm2`, `dot`, `iamax`, `scal`, `axpy`) joins the backend too. Proposed: it
  stays generic in `colmat-base` until the harness shows a panel limited by it. This covers
  level-1 called on its own only. A pivot search or a norm update that a factorisation runs
  together with its update is part of a fused primitive (§11.9), never a separate `iamax` or
  `nrm2` call.
- How the facade picks the default backend of a scalar (for example a trait on the scalar,
  implemented in `colmat` and in `colmat-xprec`).
- The block-size query of the drivers (`nb` per primitive, the `ILAENV` equivalent) as a backend
  method.
- The first open question of §10 (one `Primitive` trait or one family per primitive) is now a
  question about the methods of `Backend<T>`.

### 11.8 Fused reduction steps (maintainer, 2026-11-28)

colmat-rs reduces a self-adjoint matrix to tridiagonal form one column at a time, without the
`latrd` panel and `her2k` update of LAPACK's `sytrd` (`faer/src/linalg/evd/tridiag.rs` at
`45a5be2`, `tridiag_fused_op`). Each step runs one kernel that does two things on the trailing
matrix `A`:

1. the rank-2 update of the previous step, `A ← A − u·yᴴ − z·uᴴ` (an `her2`);
2. the matrix-vector products of the next step, `z = A·u` and `y = Aᴴ·v` (an `hemv`).

The updated `A` feeds the products while it is still in registers. Once `A` no longer fits in
cache, both halves are limited by memory traffic:

| | traffic over the trailing matrix per step |
|---|---|
| fused (colmat-rs) | one read and one write |
| separate `her2`, then `hemv` | one read and one write, then one more read |

So splitting the kernel costs about 1.3–1.5 times in that phase for large `n`. This is an
estimate from the traffic count, not a measurement. Inside cache the difference is small.

**Corrected 2026-11-28 (§12 audit).** The first version of this section said that the
bidiagonal and Hessenberg reductions are fused one-step reductions throughout. They are not.
At `45a5be2` the fused step covers only their small sizes:

| reduction | fused one-step kernel | blocked route |
|---|---|---|
| tridiagonal (`evd/tridiag.rs`) | `tridiag_fused_op`, every size | none |
| bidiagonal (`svd/bidiag.rs`) | `bidiag_fused_op`, `min(m, n) ≤ BLOCK_FLOOR` (128), and the tail of the blocked loop | `bidiag_blocked_phase`: `dlabrd`-style panels of `BLOCK_SIZE` (16) with X/Y corrections, panel rows stored transposed and contiguous, one trailing product of width `2·ib`, the factor upgrade skipped for values only |
| Hessenberg (`evd/hessenberg.rs`) | `hessenberg_fused_op` inside `hessenberg_rearranged_unblocked`, `n² < blocking_threshold` (`256²`) | `hessenberg_gqvdg_blocked` |

The SVD gains of October and November are in the blocked bidiagonal phase: bidiagonalisation
512, 8 threads, 35.2 → 19.5 ms (`docs/worklogs/2026-10-23-blocked-bidiag.md`), and `svd_values`
f64 511 at 0.884/0.774/0.809 of the previous main at 1/4/8 threads
(`2026-11-14-svd-mkl-targets.md`). A port must keep both routes.

Decision:

- **Each fused step is a backend primitive of its own:** the tridiagonal step, the bidiagonal step
  and the Hessenberg step. Each has its plan, scratch, estimate and run, like any other primitive.
  This is §5.7 ("fused sweeps are primitives of their own") applied to the reductions.
- `hemv`/`symv` and `her2`/`syr2` stay in the backend as separate level-2 primitives for their other
  callers. The reductions call the fused primitive, not the pair.
- **The blocked routes are drivers in `colmat-lapack`.** The bidiagonal blocked phase and the
  blocked Hessenberg reduction are ported as colmat-rs runs them, with their panel layout (rows
  stored transposed) and their thresholds (§11.11). Their panels call level-2 primitives and their
  trailing updates call GEMM. Only the small-size step is the fused primitive.
- For the tridiagonal reduction, the blocked LAPACK variant (`latrd`, then `her2k`) is a later
  candidate, compared with the fused step by the harness. Which one wins should depend on `n` and
  the cache sizes.
- The fused step's contract includes its rounding: it is not required to round like `her2`
  followed by `hemv`. Only the error bound of the reduction is guaranteed, as in §5.5.

### 11.9 The other fused kernels (maintainer, 2026-11-28)

A survey of colmat-rs at `45a5be2` (`faer/src/linalg`) found seven more families of kernels that do
in one pass what separate level-1/2 calls would do in two or more. Each becomes a backend
primitive, or part of one, with the driver that needs it (§11.6 step 4).

| # | colmat-rs kernel | one pass does | split into | expected cost of splitting |
|---|---|---|---|---|
| 1 | `lu/partial_pivoting/factor.rs`: `leaf_step_simd`, `lu_in_place_unblocked_simd` | scale the pivot column, rank-1 update of the leaf's columns, search for the next pivot | `scal`, `ger`, `iamax` | one more read of the leaf column; the leaf is narrow and usually in cache, so small to moderate |
| 2 | `lu/full_pivoting/factor.rs`: `update_and_best_in_col_simd`, `update_and_best_in_mat_simd` | rank-1 update of the Schur complement and the search for its largest entry; the row swap, done one column ahead inside the same sweep; in the transposed form, the scaling of the pivot row | `ger`, then an argmax over the matrix, and a separate `swap_rows` | one more read of the whole trailing matrix per step: large. A separate swap also brings back the dip at power-of-two leading dimensions that the fused swap removed (8 threads, 512: the fused version takes 0.342 of the time, `2026-11-19-lu-eigh-fullpiv-shapes.md`) |
| 3 | `qr/col_pivoting/factor.rs`: `update_mat_and_dot_simd` | update of the trailing block and of the column norms used for the next pivot | `ger`, then the norms | one more read of the trailing block per step: large |
| 4 | `cholesky/bunch_kaufman/factor.rs` (the step marked "the rank-1 or rank-2 update fused with the pivot search") | rank-1 or rank-2 update and the pivot search | update, then the search | one more read of the trailing block per step: large |
| 5 | `householder.rs`: `apply_reflector_small`, `apply_reflector_pair_small` (used by the small path of `qr/no_pivoting/factor.rs` and by `qr/no_pivoting/unitary.rs`) | per column, the inner product with the reflector and the update; the pair version applies two reflectors in one pass | `gemv`, then `ger`; a pair as two single reflectors | up to twice the reads of the target columns |
| 6 | `evd/tridiag_evd.rs`: `apply_rotation_sequence` | the whole rotation sequence of one implicit QR sweep, with a strip of rows kept in registers, so each column is loaded and stored once per sweep | one rotation at a time | **measured** in colmat-rs#234: at f64 `n = 128`, one thread, the rotation-at-a-time loop took about 1.2 ms of a 2.4 ms `eigh` |
| 7 | `cholesky/llt/update.rs`, `cholesky/ldlt/update.rs`: `rank_update_step_simd` | one step of updating a Cholesky factor by a low-rank term | separate updates of the column and the workspace | moderate |

Only #6 is a measurement. The others are estimates from the number of passes over memory.

Consequences:

- **Pivot search fused with the update (#1–#4) is a backend primitive**, one for each pivoting
  scheme: the partial-pivoting leaf, full pivoting, column-pivoting QR and Bunch-Kaufman. The
  tie-breaking that drivers depend on (§5.5, colmat-rs#105) is part of these primitives' contract.
  This is why §11.7 keeps generic level-1 for standalone calls only.
- **Small reflector applies (#5)**, single and pair, are primitives next to the block reflector
  apply of §5.7. The rotation sequence (#6) is the rotation-sequence primitive §5.7 already names.
- **Cholesky rank update (#7)** joins when factor updates are ported. It is not on the #54 path.
- Not affected: standalone reductions (`reductions/norm_max.rs`, `reductions/sum.rs`) are level-1
  on their own. The multishift sweeps of Schur and QZ accumulate rotations into a small orthogonal
  matrix and apply it with GEMM, so they lose nothing at the primitive boundary.
- As in §11.8, a fused primitive guarantees the error bound of its operation, not the rounding of
  the split sequence.

### 11.10 Contract additions from the §12 audit

- **Clean vector state on return.** A backend's x86 kernels return with the upper halves of the
  vector registers cleared (`vzeroupper`), and a team job ends the same way. colmat-rs does this in
  `team::end_of_job` (`2026-11-19-eigh-dirty-upper-state.md`: `eigh` f64, 4 and 8 threads, 0.59–0.65
  of the time without it). This repository has not ported it yet: `utils/simd.rs` and
  `utils/team.rs` say it waits for the x86 kernels, which already run on teams here.
- **The compact-WY factor build is a primitive.** `upgrade_householder_factor` (building `T` of
  `I − V T Vᴴ`) joins the backend next to the block reflector apply, with its own plan: the
  parallel floor checked once, and inner products on the calling thread for a basis of at most 8
  columns. Without the floor a threaded 128x128 QR measured 4.2–4.6 times slower than sequential
  (`2026-10-14-compact-factor-region-floor.md`).
- **Scratch may depend on strides.** A driver that copies a row-major input into a column-major
  buffer (QR, column-pivoting QR, partial-pivoting LU, `Llt::new`, `eigh` vectors) needs a buffer
  whose size depends on the layout, not only on the shape. colmat-rs allocates it on the heap
  because its scratch query sees the shape only. Here the query takes the layout class of the
  call, so the buffer comes from the caller's scratch. Without the copy, column-pivoting QR 256²
  in C order took 58.96 ms instead of 1.52 ms (`2026-10-15-row-major-adaptation-lu-colpiv.md`).
- **Per-thread buffers become plan scratch.** The GEMM job tables and the TRSM staging buffer of
  colmat-rs are per-thread buffers grown on demand. Under §11.5 they are part of the plan's
  scratch.
- **Uninitialised destinations.** A product with `Accum::Replace` never reads `dst`; colmat-rs uses
  this to write into `MaybeUninit` storage. The safe view trait cannot express it, so the backend
  offers a separate entry for an uninitialised destination, or the driver zero-fills. Open; low
  impact.

### 11.11 Where driver thresholds and rates live

Drivers own constants that no primitive's `estimate` covers: the divide-and-conquer join rates,
`secular_roots`, the repair rates, `lu_swap`, the LU look-ahead gate, `PANEL_TEAM`, the switch to
divide and conquer on a team (`tridiag_evd_threshold` 16), the SVD divide-and-conquer threshold
(40), the f64 LU leaf width (8), the QR block width rule, the small-QR gates, `BLOCK_FLOOR`, the
Hessenberg `blocking_threshold`, and the tile-copy gates. They move with the driver, as constants
of `colmat-lapack` keyed like §6, and later as rows of the generated tables (§6) with the driver
as the "primitive" of the key. None of them is dropped as "a literal in a driver" during a port.

## 12. Carry-over from colmat-rs

**Principle (proposed in this trial).** The algorithms of colmat-rs are ported as they are. The
backend boundary of §11 is drawn so that every one of them can be ported unchanged: where an
algorithm fuses work across what would be two primitives, the fused work is one primitive
(§11.8, §11.9); where it depends on a driver-level detail, the driver keeps that detail (§11.11).
A port that cannot keep an optimization listed here says so in its PR, with a measurement.

The table lists the fused kernels of §11.8 and §11.9, and the performance changes of colmat-rs
whose work logs are dated October and November 2026 and that are in the code at `45a5be2`. It
was compiled on 2026-11-28 from those work logs, `PATCHSET.md` and the code. Work logs are in
colmat-rs `docs/worklogs/`. Reverted or rejected experiments are left out.

| # | what | work log(s) | home in colmat-next |
|---|---|---|---|
| 1 | Blocked bidiagonalisation above 128: `dlabrd` panels of 16, transposed panel rows, one trailing product of width `2·ib`, no factor upgrade for values only | 10-23-blocked-bidiag, 11-14-svd-mkl-targets | driver (§11.8) |
| 2 | Unblocked bidiagonal step at ≤128 with the row and previous reflector gathered contiguously | 10-23-svd-64-and-tall | fused primitive (§11.8) |
| 3 | 64-byte column padding of the SVD and Pfaffian work matrices at strides that are multiples of 1 KiB (`work_rows`) | 10-23-blocked-bidiag, 10-31-pfaffian-176 | driver workspace |
| 4 | dqds for values-only SVD, with the backed-off shift, `mul_add_impl` and the cached minimum | 10-23-dqds-values-only, 10-23-dqds-shift, 11-14-svd-mkl-targets | driver |
| 5 | SVD divide-and-conquer threshold 128 → 40 | 11-12-svd-dc-threshold | driver constant (§11.11) |
| 6 | Tridiagonal fused step | — (§11.8) | fused primitive |
| 7 | Rotation sequence of the tridiagonal QR sweep, in chunks of 8, and `sqrt(1+t²)` in `make_givens` | 11-19-lu-eigh-fullpiv-shapes | fused primitive (§11.9 #6); `make_givens` in the driver |
| 8 | Divide-and-conquer EVD on a team: secular roots split by width, weight and vector repair in parallel, switch threshold 16 on a team | 11-11-eigh-secular-root-split, 11-14-eigh-parallel-repair, 11-10-eigh-vs-mkl-small-sizes, 11-12-eigh-vs-mkl-evo2 | driver and its rates (§11.11) |
| 9 | Values-only `eigh` through `sterf` (draft experiment) | 11-28-sa-evd-values-sterf-287 | driver |
| 10 | Row-major `eigh` vectors computed column-major, copied once | 11-10-row-major-evd-u-and-cholesky | driver, stride-dependent scratch (§11.10) |
| 11 | Blocked Hessenberg reduction above `256²` | — (code) | driver (§11.8) |
| 12 | Partial-pivoting LU leaf: scaling, update and next pivot search in one pass | — (§11.9 #1) | fused primitive |
| 13 | f64 LU leaf width 16 → 8 | 11-05-lu-leaf-width | driver constant (§11.11) |
| 14 | LU one-panel look-ahead on teams of 4 or more | 11-19-lu-lookahead-panel | driver (§7) |
| 15 | LU trailing-update and swap parallel gates | 10-14-lu-trailing-update-floor, 10-19-lu-work-floor-units | base cost helpers, driver rates |
| 16 | TRSM right-hand side staged into a non-aliasing stride | 11-05-trsm-aliased-rhs | trsm plan (§7), plan scratch (§11.10) |
| 17 | Full-pivoting LU: update, argmax and row swap in one sweep | 11-19-lu-eigh-fullpiv-shapes | fused primitive (§11.9 #2) |
| 18 | Full-pivoting LU on row-major input through the transposed form | 11-09-full-piv-lu-row-major-dual | layout class of the fused primitive |
| 19 | Partial-pivoting LU reusing the factor buffer (`factor_into`) | 11-18-lu-factor-buffer | facade |
| 20 | Column-pivoting QR: trailing update and norm update in one pass | — (§11.9 #3) | fused primitive |
| 21 | Bunch-Kaufman: update fused with the pivot search; candidates from scratch | 10-29-no-region-allocation | fused primitive (§11.9 #4) |
| 22 | Row-major QR, column-pivoting QR and LU through a tiled copy to column-major | 10-15-qr-row-major-layout, 10-15-row-major-adaptation-lu-colpiv | driver, stride-dependent scratch (§11.10) |
| 23 | `Llt::new` on row-major input: tiled triangular copy and its gates (power-of-two destination and source strides) | 11-06-row-major-triangular-copy, 11-10-power-of-two-column-stride, 11-12-tiled-copy-source-stride | base layout copy with its gates |
| 24 | Small QR and thin `Q` through single and paired reflector applies, with their size gates | 11-22-small-qr-level2 | fused primitives (§11.9 #5), gates in the driver |
| 25 | Compact-WY factor build: one floor check, inner products for ≤8 columns | 10-14-compact-factor-region-floor, 10-23-svd-64-and-tall | primitive (§11.10) |
| 26 | Block reflector apply: the column split takes the whole thread budget | 10-18-qr-block-apply-splits | block reflector apply plan |
| 27 | QR look-ahead on wide teams (`lookahead_width`, `PANEL_TEAM` 0.68) | 10-30-qr-lookahead | driver and its constants |
| 28 | QR block width 16 on squares of 129–640 at 8 threads or more | 11-14-svd-mkl-targets | driver nb rule (§11.11) |
| 29 | LQ with staged column-major panels and a row-major temporary in the block apply | 11-10-lq-staged-panels | driver; the temporary's layout is chosen by the block reflector apply's plan |
| 30 | Cholesky/LDLT trailing update split into equal-area column ranges, with the part-count cap | 10-30-cholesky-area-split, 11-02-equal-area-partition-cap | herk/syrk plan, base partition |
| 31 | Diagonal-scaled, indexed product of LDLT (`spicy_matmul`) on the kernel's team grid | 10-26-team-gemm-executor | to decide: a primitive next to herk, or a driver over GEMM |
| 32 | Cholesky rank update step | — (§11.9 #7) | fused primitive |
| 33 | Pfaffian: logical pivot relabelling, shared compact panel, two GEMMs per tile and one store of both skew triangles, panel 24 | 10-31-pfaffian, 10-31-pfaffian-byte-traffic, 11-02-pfaffian-supply | driver, not yet on the #54 list |
| 34 | Real Schur small-n sweep with zipped reflector loops | 11-27-lahqr-sweep-285 | driver, not yet on the #54 list |
| 35 | Parallel gates in every region: run when the serial estimate pays three times the dispatch cost, at the widest width that pays, rates keyed by host class | 10-26-dispatch-cost-floors, 10-26-remaining-gates, 10-29-width-per-region | base cost helpers and `estimate` (§5, §11.4) |
| 36 | Join costs of divide-and-conquer SVD/EVD and `triangular_inverse` | 10-26-qr128-and-join-gates | driver rates (§11.11) |
| 37 | Team regions allocate nothing (inline child slots, typed join slots) | 10-29-no-region-allocation | base team operations |
| 38 | `vzeroupper` at the end of each team job | 11-19-eigh-dirty-upper-state | contract (§11.10); **not yet ported** |
| 39 | Small native GEMM given threads only when it pays | 10-18-small-gemm-thread-floor | gemm plan width |
| 40 | x86 GEMM on a team through the kernel's own job grid (`gemm_with`) | 10-26-team-gemm-executor | gemm plan (§7) |
| 41 | GEMM job tables from a per-thread buffer | 11-04-gemm-team-job-tables | plan scratch (§11.10) |
| 42 | Even tile grid for small GEMM on a team below 8 ms of serial work | 11-21-gemm-team-tile-split | gemm plan (§7); ported |
| 43 | Overwrite-only product with `m = 1` through the matrix-vector branch | 11-03-matmul-uninit-thin | gemm plan / gemv |
| 44 | Product into uninitialised storage | 10-29-matmul-uninit | open (§11.10) |

Each driver PR that ports a routine lists the rows of this table it carries and shows the
measurement for each one it changes.
