# colmat-core matrix product: the P1 hook, its kernels and checks (#23)

A record of step 2's last part in the design record (§7): the matrix product of `colmat-core`, its
kernel hook (P1) and the generic and native kernels behind it, ported from colmat-rs `71337cdc`
(`faer/src/linalg/matmul/mod.rs`, `faer/src/utils/simd.rs`). Everything here describes this
trial; nothing is a decision of the colmat maintainers. The answers to #23's five questions and
the constraints attached to them are in #23 (shinaoka, 2026-11-22 and 2026-11-23).

## What was built

| Item | Where | From colmat-rs |
|---|---|---|
| `ComplexField::Kernels`, `GenericKernels`, `NativeKernels` | `colmat-traits` (`scalar.rs`, `kernels.rs`) | New; replaces `IS_NATIVE_*` (debts K1, K2) |
| `Gemm<T>`, `KernelField` | `colmat-core::kernels` | New |
| `matmul_with_scratch`, `matmul_scratch`, `matmul_with_conj`, `matmul`, the driver | `linalg::matmul` | `matmul_imp`, `matmul`, `matmul_with_conj`, `precondition` |
| Matrix-vector (both layouts), rank one, dot product, vertical and horizontal micro-kernels | `linalg::matmul::generic` | `matvec_*`, `rank_update`, `dot`, `matmul_vertical`, `matmul_horizontal` |
| `nano-gemm`, `t4a-gemm`, `t4a-private-gemm-x86` calls | `linalg::matmul::native` | The `gemm_call!` block |
| `SimdCtx` and its body/tail walk | `utils::simd` | `utils/simd.rs` without `new_align` |
| `Accum` | `lib.rs` | `lib.rs` 958 |

Not ported in this step: the triangular products (`triangular.rs`, out of #23's scope), the
uninitialised-output entry (`matmul_with_conj_uninit`), and the x86 team adapter (`team::kernel`,
`job_tables`, `gemm_team`, `vzeroupper` handling). A team runs one serial kernel call per
participant here (below).

## Choices, and why

- **The hook is a trait per provider and scalar.** `Gemm<T>` is implemented for `GenericKernels`
  for every `T` and for `NativeKernels` for the four native types only. A pointer is therefore
  handed only to the kernel of the type it points to, and a scalar cannot reach a native kernel
  by setting a constant. Drivers ask for `KernelField` (`ComplexField<Kernels: Gemm<Self>>`,
  blanket-implemented). Stable Rust has no associated-type defaults, so each scalar writes one
  line for its provider.
- **The driver keeps the shapes that need no packing.** The empty, matrix-vector and rank-one
  cases run the generic kernels for every scalar before the hook is called. A provider that
  supplies only a packed product still gets those paths.
- **Parallel regions split the output, never the depth.** colmat-rs's column-major
  matrix-vector product split the depth and kept the partial sums in a thread-local
  (`with_partials`). The workspace rule of the design record (§5) forbids that thread-local.
  Splitting the rows (matrix-vector), the columns (rank one) or the columns then the rows
  (product) needs no buffer. It also computes every element in the order of one thread, so a
  team's result is bitwise that of `Par::Seq`, which a test checks. The cost: a matrix-vector
  product with few rows and a long depth gets fewer participants. The width comes from the cost
  model at the provider's `ns_per_mac`.
- **Native products run one serial kernel call per participant.** `t4a-gemm` runs with
  `Parallelism::None`, `t4a-private-gemm-x86` with one thread, each on its part of the output.
  Neither kernel's own parallelism is used. shinaoka asked for `Parallelism::None` only.
- **One micro-kernel shape for every scalar (2 registers x 2 columns).** colmat-rs derived the
  shape from the size of a scalar's unit, which the scalars of this trial do not have.
- **Under Miri, the generic kernels stand in for the native ones.** Miri cannot run
  `nano-gemm`'s inline assembly or the packed kernels' intrinsics, and this way Miri checks the
  unsafe loads and stores of the generic kernels.

## The allocation exception

`t4a-gemm` keeps its packing memory in thread-locals (an L2-sized slab and a pack buffer kept up
to L2), and `t4a-private-gemm-x86` keeps L3-sized buffers. Neither takes scratch from the
caller. So the packed native products allocate on a thread's first call, and `t4a-gemm` also on
every call whose pack exceeds L2. The fix belongs in those crates: tensor4all/t4a-gemm#21 and
tensor4all/t4a-private-gemm-x86#14 propose a `gemm_scratch` / `gemm_with_stack` pair. A
thread-local of our own was ruled out in #23. shinaoka chose to land core-3 first, with the gap
stated in four places:

- `matmul_with_scratch`'s rustdoc ("Allocation");
- the no-allocation test:
  - the generic, `nano-gemm`, matrix-vector and rank-one paths are counted from the first call,
    on `Par::Seq` and teams of 2 and 4;
  - the packed native path is counted after one warm-up;
  - its first-call case is `#[ignore]` with both links;
  - on x86-64 the warm-up case is ignored too, with the second link;
- the design record (§5, "Workspace");
- the pull request.

`matmul_scratch` stays empty for these products until the entries exist.

## Correctness checks

- The reference tests run on `f32`, `f64`, `c32` and `c64`. They cover:
  - one in three layout triples of `dst`, `lhs` and `rhs` (16 layouts each: column- and
    row-major, unit or doubled inner stride, each axis forward or reversed);
  - both conjugations and both accumulation modes;
  - shapes that reach every path.

  Each element is held to the stated bound: `(k + 2)·eps·|α|·Σ|a||b| + eps·(|dst| + |ref|)`,
  doubled for `f64`/`c64` (whose reference rounds like the product) and for complex types. The
  reference sums in `f64`; no double-double precision is claimed. The padding around every view
  is checked, and `Replace` starts `dst` as NaN, so a read of it would show.
- No BLAS-derived test was ported (#23, answer (4)).
- Under Miri, each layout of each operand appears once per shape, for `f64` and `c64`. `f32` and
  `c32` run the same kernels at another lane count, which the `utils::simd` walk test covers
  under Miri. Measured locally (Apple M1 Pro), the product tests took 600 s under Stacked Borrows
  before this cut; the totals after it are in the pull request.
- x86-64 builds and lints locally (`--target x86_64-apple-darwin`). Its tests run in CI only,
  as answer (5) of #23 asked.

## Performance (non-regression)

Apple M1 Pro, Rust 1.96.0, single thread, `Accum::Add`, against colmat-rs `71337cdc`. Median of
21 batches of at least 5 ms each; inputs are built before timing. The ratio is colmat-core over
colmat-rs.

- Products from `32^3` to `512^3`, both layouts, all four types: 0.95–1.09. These run the same
  kernels as colmat-rs.
- Real matrix-vector and rank-one products, and the dot-product-shaped complex paths:
  0.91–1.06.
- Complex column-major matrix-vector products: 1.60–1.75 (`c64` at `4096` is 0.76). Complex
  rank-one updates: 1.36–1.38. colmat-rs kept complex registers interleaved. Here the complex
  provider splits them, and its pair loads and stores go through a stack copy, so an axpy-shaped
  loop takes 1.4x the direct NEON code: #26. Unrolling four columns per pass made the small sizes
  faster but `n = 4096` 2–3x slower, so it was not kept.
- Products up to `8^3`: up to 10 ns more per call (0.85–1.53 on 20–140 ns), the driver's checks.

The first numbers of this work were taken with Rust 1.90.0. That compiler did not remove the
complex provider's stack copies, and showed 2.2–3.3x on the complex paths. Measure with the
pinned toolchain.

No x86-64 measurement: there is no x86-64 host (as in #20).
