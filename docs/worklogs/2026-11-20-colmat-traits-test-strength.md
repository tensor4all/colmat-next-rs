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
one-lane `NoSimd` index operations.

Property tests draw new inputs on every run, so a mutant near a rarely drawn input is caught in
one run and missed in the next. The weekly job therefore fixes the seed and raises the case count
(`PROPTEST_RNG_SEED=20261120`, `PROPTEST_CASES=4096`). With that setting and the tests of this
change, 17 mutants survived, all at branch thresholds or dead guards. Each was checked:

- 16 are equivalent and are excluded in `.cargo/mutants.toml` with the reason: `>`/`>=` and
  `<`/`<=` at the power-of-two scaling thresholds of complex `abs`/`recip` (the neighbouring
  branch gives a bit-identical result at the threshold), the `sum < 0` clamp of `sqrt` (cannot
  hold), the empty-range normalisation in `split_pair_range`, and `>`/`>=` in `NoSimd::abs_max`.
- 1 is not equivalent: `im < 0` → `im <= 0` in `sqrt` changes the sign of the zero imaginary part
  for `im = -0`. The current convention is pinned by `sqrt_of_negative_zero_imaginary`.

Final run: 591 mutants, 354 caught, 0 missed, 237 unviable.

## Defects found (inherited from colmat-rs)

Both are present unchanged in colmat-rs `faer-traits` at `71337cdc` (same results on the same
inputs). The properties stay active on the inputs where the functions are correct; the defective regimes
have their own tests, committed with `#[ignore]` and the reason, so that the fix turns them on.
The fix changes numerical results and is left to its own change.

1. **`sqrt` loses the imaginary part when `|im| ≪ re`.** The imaginary part is computed as
   `sqrt((|z| - re) / 2)`, and `|z| - re` cancels. `sqrt(1 + 1e-8 i)` returns imaginary part
   `0` instead of `5e-9`; for `re = 4.77e299, im = 2.39e297` the imaginary part has relative error
   about `1e-11`. The usual remedy computes the small part as `im / (2 t)` from the large one
   `t = sqrt((|z| + |re|) / 2)`. (Ignored: `prop_sqrt_squares_back`. Active:
   `prop_sqrt_large_component`, which checks `t` and the signs.)
2. **`recip` flushes to zero near the overflow threshold.** For `|z|` above about `0.5 · MAX`,
   the big-value branch scales by `sqrt(safe_min)` only, the squared magnitude overflows, and the
   result is `0` instead of a subnormal: `recip(f64::MAX)` gives `0` instead of `5.56e-309`, and
   `recip(0 + 3e38 i)` in `f32` gives `0` instead of `-3.3e-39 i`. `complex_abs` scales the same
   range by `sqrt(safe_min) · eps` and is not affected. (Ignored: `prop_recip_near_overflow`.
   Active: `prop_recip_f64`/`prop_recip_f32` on parts up to `2^1022` / `2^126`.)

A third observation is a convention, not a defect: `sqrt(x - 0i)` returns `+0` as imaginary
part, where C99 `csqrt` returns `-0`. It is left as colmat-rs has it and pinned by a test.

## Remaining

- The x86-64 V3 pass runs only where the host has AVX2/FMA; no x86 host was available locally.
- AVX-512 (`x86-v4`) is not enabled in this crate and not tested.
- Mutation results are for this crate's code only; `pulp`'s kernels are trusted.
