# Kernel selection and the tuning harness

**Status: proposed in this trial (2026-11-28).** This is a design for review in #44, #45 and #46.
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
| The SME gate inside `t4a-gemm` sees the shape of each sequential call. On a team that tiles the product, that shape is the tile, not the whole product (`2026-11-22-sme-route-180`, "The gate sees the shape of each sequential call"). | **Hypothesis, not tested:** on an SME host, `Par::Seq` and a team can reach different kernels for the same product. That breaks this repository's rule that a team result is bitwise equal to `Par::Seq` (§5, "Rounding on a team"). It is the same class of defect as the `kc` difference fixed through #45 and t4a-gemm#25. A kernel choice made once on the whole call and passed down to the tiles removes the defect by construction. |
| `x86-v4` as a build feature cost a whole campaign to answer one yes/no question, and the answer was "no gain on average, a gain for some families and sizes" (#44). | Selection is per (primitive, dtype, size, width) at run time. It is not per build. |

## 3. Decision: two levels of dispatch (proposed)

**Level 1, type level: static.** The scalar names its provider (`ComplexField::Kernels`), and the
hook traits (`Gemm<T>`, ...) are implemented for the provider and the scalar together. This is
the design of #23 and it stays as it is. It answers "which implementations exist for `T`". It
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
Overrides exist for experiments (§5.6), not as an extension API.

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

### 5.3 Selection is made on the whole call, never on a tile

A schedule that cuts the call into tiles or depth blocks gives every piece the **same kernel and
the same depth blocking**. The rounding of an element then does not depend on the tile it fell
in. This generalises the `kc` fix of #45 and t4a-gemm#25. It also closes the SME hypothesis of §2,
which a test should confirm on an M4 or M5 before the plan lands (§10). The one documented
exception stays explicit in the plan: the column-major matrix-vector depth split (§5 of the trial
design, #28).

### 5.4 Validate before the kernel

`run` checks that the views match the plan's call: shape, layout class, and that `dst` is not
aliased. A native kernel's invariants (alignment, minimum stride) are part of the plan's layout
class, so a safe entry cannot reach a kernel with views it does not accept. A plan is `Copy` and
cannot be forged into something that is unsafe to run. `run` asserts the parts of the call that
the plan depends on.

### 5.5 Numerical equivalence

- Same plan → bitwise the same result, at every team width the plan allows (§5.3).
- Different plans (another host, another width that changes the kernel, an override) may round
  differently, within the error bound documented for the primitive.
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
- On SME hosts: the SME or NEON choice is made once for the whole call (§5.3).

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
  record of what ran, and the per-tile SME choice of §2.
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
- Whether `t4a-gemm` and `t4a-private-gemm-x86` expose entries that take a chosen kernel and `kc`
  without re-running their own gates. §5.3 needs that for SME. Proposing those entries is a change
  in those crates.

First steps, each its own issue and PR:

1. A test on an SME host (M4 or M5): a team result of f32/f64 at shapes around the SME
   thresholds compared bitwise with `Par::Seq`. This confirms or refutes the hypothesis of §2.
2. Probe the host once: one `HostInfo` in `colmat-core`, read by `native.rs` and by the cost
   model. Nothing else changes.
3. Turn `Gemm<T>` into a plan, scratch and run triple, with `trace-plans`. No new kernels or
   thresholds.
4. Stage 1 of the harness in `colmat-benchmark` (§8.1).
5. `tuning` and `COLMAT_FORCE_KERNEL`, then the first generated table for GEMM on evo, checked on a
   second host.
