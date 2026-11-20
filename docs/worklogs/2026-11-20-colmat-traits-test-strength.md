# colmat-traits: property tests and mutation testing (#8)

## What was added and why

| Tool | Failure it prevents | Where it runs |
|---|---|---|
| proptest, provider properties (`simd/tests/props.rs`) | Lane-wise results that differ from the scalar operation on inputs the hand-picked tests miss: NaN, ±0, ±inf, subnormals, random partial masks | PR pipeline (256 cases), nightly heavy tests (65 536 cases) |
| proptest, scalar properties (`scalar/tests`) | Wrong results in the scaling branches of complex `abs`, `recip`, `sqrt`, which only extreme exponents reach | Same |
| Explicit `pulp::x86::V3` pass in every provider check | The AVX2 pair shuffle going untested when detection picks another path | Every x86-64 host with AVX2/FMA (the ubuntu CI runners) |
| cargo-mutants (`.github/workflows/mutants.yml`) | Tests that execute code without checking it | Weekly and on demand; not a merge gate |

Comparisons follow what IEEE arithmetic guarantees. Operations that are exact lane by lane (add,
sub, neg, conj, `abs1`, `abs_max`, real `mul`, `mul_real`, `mul_pow2`, comparisons, select, masked
memory access) are compared bit for bit, with any NaN matching any NaN. A real `mul_add` (and
`conj_mul_add`, `abs2_add`, complex `abs2`) must equal either the fused or the unfused result.
Complex products and reductions are checked on bounded finite inputs against an `f64` reference,
within `8 eps` of the operand scale (`2 n eps` for sums) plus the same multiples of the subnormal
spacing. Property tests are skipped under Miri, whose lane keeps the fixed tests.

## Mutation testing

Before the new tests, cargo-mutants 27.1.0 on `colmat-traits` reported 607 mutants: 248 caught,
122 missed, 237 unviable. The missed ones were in the scaling branches of complex
`recip` (40) and `abs` (10), the ported index helpers (41), `Complex` `is_nan`/`is_finite`,
`max`/`min` ties, `absmax`, `digits` with an invalid `eps`, the default `mul_add_impl`, and the
one-lane `NoSimd` index operations. After the tests in this change: 609 mutants, 372 caught,
0 missed, 237 unviable. No mutant needed an exclusion.

## Defects found (inherited from colmat-rs)

Both are present unchanged in colmat-rs `faer-traits` at `71337cdc` (same results on the same
inputs). Their property tests are committed with `#[ignore]` and the reason, so that the fix turns
them back on; the fix changes numerical results and is left to its own change.

1. **`sqrt` loses the imaginary part when `|im| ≪ re`.** The imaginary part is computed as
   `sqrt((|z| - re) / 2)`, and `|z| - re` cancels. `sqrt(1 + 1e-8 i)` returns imaginary part
   `0` instead of `5e-9`; for `re = 4.77e299, im = 2.39e297` the imaginary part has relative error
   about `1e-11`. The usual remedy computes the small part as `im / (2 t)` from the large one
   `t = sqrt((|z| + |re|) / 2)`. (`prop_sqrt_squares_back`)
2. **`recip` flushes to zero near the overflow threshold.** For `|z|` above about `0.5 · MAX`,
   the big-value branch scales by `sqrt(safe_min)` only, the squared magnitude overflows, and the
   result is `0` instead of a subnormal: `recip(f64::MAX)` gives `0` instead of `5.56e-309`, and
   `recip(0 + 3e38 i)` in `f32` gives `0` instead of `-3.3e-39 i`. `complex_abs` scales the same
   range by `sqrt(safe_min) · eps` and is not affected. (`prop_recip_f32`; the `f64` property
   reaches this range only rarely)

A third observation is a convention, not a defect: `sqrt(x - 0i)` returns `+0` as imaginary
part, where C99 `csqrt` returns `-0`. It is left as colmat-rs has it.

## Remaining

- The x86-64 V3 pass runs only where the host has AVX2/FMA; no x86 host was available locally.
- AVX-512 (`x86-v4`) is not enabled in this crate and not tested.
- Mutation results are for this crate's code only; `pulp`'s kernels are trusted.
