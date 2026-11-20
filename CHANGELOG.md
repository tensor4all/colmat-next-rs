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

### Added

- Repository scaffolding: agent and contribution rules, CI, and the design-record layout.
- `colmat-traits` (#6): `ComplexField`/`RealField` with scalar arithmetic only, `Precision`
  (`eps`, `safe_min`, `safe_max`), and SIMD providers split from the scalar trait (`SimdLayout`,
  `SimdField`, `SimdPairs`; `NoSimd`, `NativeSimd`, `ComplexSimd`). Ported from colmat-rs `faer-traits`.
- `colmat-traits` tests (#8): property tests (proptest) for every provider and for the scaled complex functions, an
  explicit AVX2/FMA pass on x86-64, and weekly mutation testing. They found two defects inherited from colmat-rs
  (`sqrt` cancellation, `recip` underflow near overflow), recorded in `docs/worklogs/2026-11-20-colmat-traits-test-strength.md`.
- Workspace setup: MSRV 1.89, pinned toolchain 1.96.0, rustfmt and lint configuration, Miri in the
  nightly heavy tests.
