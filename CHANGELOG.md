# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the crates follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Releases are tagged
`<crate>-v<version>`.

## [Unreleased]

### Fixed

- `colmat-traits` (#10): complex `sqrt` no longer loses the smaller component when `|im| ≪ |re|`
  (Kahan's method), and complex `recip` no longer flushes to zero for `|z|` near the overflow
  threshold. Both defects were inherited from colmat-rs.
- `colmat-core` (#14): the soundness defects inherited from colmat-rs and recorded in #12.
  `Mat::resize_with` and `Mat::truncate` drop each element once when a closure or a `drop` panics;
  the triangular `zip!` iterations stay in bounds for every shape and layout, and the upper
  triangle of a wide matrix with reversed rows no longer skips the columns `j >= nrows`; casts
  between permutation shapes panic for `One` instead of reinterpreting its zero-sized index;
  `Array`'s range indexing is checked. API changes: `ZipEq::new_unchecked` is an `unsafe fn`, and
  `Array::as_mut` returns a slice borrowed from the array.

### Added

- `colmat-core` (#17, trial): Kani proof harnesses (`src/kani_proofs.rs`) for the row ranges of the
  triangular `zip!` iterations (every size), the index casts and permutation invariant, and
  `MatRef::split_at`; a nightly `kani` job in `heavy-tests.yml`. The triangular row ranges are now
  the pure functions `zip::triangular_lower_ncols`/`triangular_lower_rows` (the upper triangle uses
  them with the transposed layout), and the crate's `concat!` macro is defined last in `lib.rs`, so it
  no longer shadows `core::concat!` inside the crate. Work log:
  `docs/worklogs/2026-11-21-kani-trial.md`.

- CI (#16): the nightly Miri run covers the doctests too and runs a second time under Tree Borrows.
  `colmat-core` gains layout-exhaustive tests (`src/exhaustive_tests.rs`): every shape up to a small
  size in 16 strided layouts, for the view reads, mutable splits and submatrices, the layout copies
  and two-operand `zip!`, checking each element and the padding around the view.

- `colmat-core` (#12): matrix views (`Mat`, `MatRef`, `MatMut`, `Col`, `Row`, `Diag`, `Perm`), `zip!`
  iteration and serial layout copies, ported from colmat-rs without rayon and without the `linalg`
  dependency. Every unsafe block carries a `// SAFETY:` comment; property tests check the views
  against a reference offset model. The review found soundness defects inherited from colmat-rs,
  recorded in `docs/worklogs/2026-11-20-colmat-core-views-port.md` and fixed in #14.
- `NOTICE`: the provenance and licence of the code ported from colmat-rs and faer.

- Repository scaffolding: agent and contribution rules, CI, and the design-record layout.
- `colmat-traits` (#6): `ComplexField`/`RealField` with scalar arithmetic only, `Precision`
  (`eps`, `safe_min`, `safe_max`), and SIMD providers split from the scalar trait (`SimdLayout`,
  `SimdField`, `SimdPairs`; `NoSimd`, `NativeSimd`, `ComplexSimd`). Ported from colmat-rs `faer-traits`.
- `colmat-traits` tests (#8): property tests (proptest) for every provider and for the scaled complex functions, an
  explicit AVX2/FMA pass on x86-64, and weekly mutation testing. They found two defects inherited from colmat-rs
  (`sqrt` cancellation, `recip` underflow near overflow), recorded in `docs/worklogs/2026-11-20-colmat-traits-test-strength.md`.
- Workspace setup: MSRV 1.89, pinned toolchain 1.96.0, rustfmt and lint configuration, Miri in the
  nightly heavy tests.
