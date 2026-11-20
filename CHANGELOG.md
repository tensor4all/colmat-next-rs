# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the crates follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Releases are tagged
`<crate>-v<version>`.

## [Unreleased]

### Added

- Repository scaffolding: agent and contribution rules, CI, and the design-record layout.
- `colmat-traits` (#6): `ComplexField`/`RealField` with scalar arithmetic only, `Precision`
  (`eps`, `safe_min`, `safe_max`), and SIMD providers split from the scalar trait (`SimdLayout`,
  `SimdField`, `SimdPairs`; `NoSimd`, `NativeSimd`, `ComplexSimd`). Ported from colmat-rs `faer-traits`.
- Workspace setup: MSRV 1.89, pinned toolchain 1.96.0, rustfmt and lint configuration, Miri in the
  nightly heavy tests.
