# colmat-core: the packing memory of the native products from the caller (#34)

A record of #34: the packed native products (`t4a-gemm`, and `t4a-private-gemm-x86` on
x86-64) take their packing memory from the caller's stack in `matmul_with_scratch`. This closes
the exception that #23 left open. The upstream entries are `gemm_with_stack` and its request:
tensor4all/t4a-gemm#22 and tensor4all/t4a-private-gemm-x86#20, #21. The choices below follow
shinaoka's answers on #34 (2026-11-24). Everything here describes this trial.

Setup: Apple M1 Pro, Rust 1.96.0, `f64`, column-major, `Accum::Replace`. Times are medians of 15
batches, in µs per call. x86-64 was compiled and clippy-checked (`--target x86_64-apple-darwin`),
not run: there is no x86-64 host here.

## Two paths, chosen by the entry

A first prototype took the memory from the stack on every entry. That cost `matmul` 24–28 % at
`20^3` and at `64^3` on a team of four. Two things caused it:

- `matmul` sized the stack twice per call;
- it allocated a fresh buffer on every call, which the kernels' thread-locals had avoided.

On x86-64 that buffer would also have been about twice the L3 size per participant, for every
product above `16^3`.

So `Gemm` gains a provided method, `gemm_alloc`. It is the product with memory the provider finds
itself, and the default allocates `gemm_scratch` and calls `gemm`.

- `matmul` and `matmul_with_conj` call `gemm_alloc`, and allocate only the driver's own scratch
  (the partial sums of the depth split, #28).
- `matmul_with_scratch` calls `gemm`.
- `NativeKernels` implements the two methods with the kernels' allocating entries and with their
  `gemm_with_stack` entries. Both go through one function, so their paths stay the same.

The x86-64 request (`gemm_stack_bytes`, shape-independent) thus reaches only a caller of
`matmul_with_scratch`, whose rustdoc states it (#34, question 2). How the entry chooses the path
is a change to the shape of `Gemm`, and it is open in the pull request.

## Sizing the stack on a team

`split_product` cuts the stack into one part per tile.

- **A part holds the largest request among the grid's tile shapes** (at most four), taken at its
  unaligned size, since a part may start anywhere. The largest tile alone would not do, because
  `t4a-gemm`'s `gemm_scratch` is not monotone in the shape for `f32`. At `k = 4096`, `4 x 156`
  asks 5.50 MiB and `4 x 176` asks 5.25 MiB. On a grid over `m` and `n`, 170 steps go down, by up
  to 7.7 %. `f64` and `c64` had none.
- **The query cannot know which width the run picks.** It covers each of `1..=degree`, or 1 alone
  when no width pays (the same cheapest-cost rejection as `par_width_time`).
- **The query evaluates each distinct tile shape once** over all the widths. A request costs
  60–230 ns, and the grids of neighbouring widths share tiles. On a team of four this halved the
  query: at `64^3` it went from 593 to 289 ns, and at `256^3` and above from 2766 to 1156 ns.

## Every shape reaches the general product

`matmul_scratch` used to leave out the kernel's request for the vector and `k == 1` shapes. Two
kinds of call still fall through to the general product, and the query does not see the layout:

- a strided vector;
- a rank-one product whose `dst` is neither column- nor row-major.

Two team tests caught this. The query now always includes the general product.

## Cost

| Case | `main` `matmul` | now `matmul` | `main` `matmul_with_scratch` | now `matmul_with_scratch` | `matmul_scratch` |
|---|---:|---:|---:|---:|---:|
| `Seq`, 20³ | 0.565 | 0.572 | 0.561 | 0.608 (+8 %) | 15 KiB |
| `Seq`, 64³ | 12.54 | 12.58 | 12.53 | 12.63 | 96 KiB |
| `Seq`, 256³ | 719.0 | 719.7 | 719.2 (one run; the other 738.9) | 713.3 | 1.25 MiB |
| team of 4, 20³ | 0.589 | 0.582 | 0.579 | 0.631 (+9 %) | 15 KiB |
| team of 4, 64³ | 5.29 | 5.28 | 5.25 | 5.75 (+10 %) | 240 KiB |
| team of 4, 256³ | 201.0 | 202.5 | 200.8 | 204–209 | 3 MiB |
| team of 4, 1024³ | 12201 | 12212 | 12196 | 12177 | 33 MiB |

`main` is `b62d2b4`; each column is the mean of two runs.

What `matmul_with_scratch` still pays at small sizes:

- the query, once, for its entry check (58 ns on `Seq`);
- on a team, the part size at the width it runs (up to four requests).

`matmul` is back to `main`'s times.

## Verification

- `cargo test --workspace --profile ci`, with and without `--all-features`, on aarch64.
- Both no-allocation tests of the native product pass with the `#[ignore]` removed: after one
  warm-up and from the first call on, on `Seq` and teams of 2 and 4. They include a
  `700 x 600 x 500` case above L2, whose pack `t4a-gemm` used to allocate on every call.
- `team_result_is_bitwise_seq` also runs `matmul_with_scratch` on the team, so both paths are
  compared with `Par::Seq`.

## Remaining

- Both kernels are pinned by git revision (`t4a-gemm` `29b5b46`, `t4a-private-gemm-x86`
  `d518bab`) in the workspace `[patch.crates-io]`, with a TODO to move to their crates.io
  releases.
- A shape-dependent request in `t4a-private-gemm-x86` would shrink the x86-64 stack. It is not
  part of this change.
