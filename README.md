# colmat-next-rs

Next-generation colmat development.

**Status: trial.** This repository is a trial of building the next-generation colmat design as a
new codebase. Whether development continues here or inside
[colmat-rs](https://github.com/tensor4all/colmat-rs) has not been decided; that decision belongs to
the colmat maintainers. Until then, the design references are
[#2](https://github.com/tensor4all/colmat-next-rs/issues/2) and `tensor4all-meta`
[`docs/design/next-generation-dense-linear-algebra.md`](https://github.com/tensor4all/tensor4all-meta/blob/main/docs/design/next-generation-dense-linear-algebra.md),
and nothing in this repository overrides them.

No crate is published from this repository. The workspace has one crate so far:

- [`colmat-traits`](crates/colmat-traits): the scalar contract (arithmetic and precision, S0) and
  optional SIMD providers (S1), proposed in this trial for #2 (#6).

## Background

[colmat-rs](https://github.com/tensor4all/colmat-rs) is a pure-Rust dense and sparse linear algebra
library for scientific and HPC work, a fork of [faer](https://codeberg.org/sarah-quinones/faer). #2
(copied from the archived tensor4all/t4a-faer#415) discusses a next generation of its dense linear
algebra: coarse-grained primitives, a scalar-dependent cost model, a candidate `colmat-core` split,
and xprec (`Df64`) support. The `tensor4all-meta` design document above records that discussion,
with status "design under review".

## Working in this repository

- Contributors: read [`CONTRIBUTING.md`](CONTRIBUTING.md); agents start from [`AGENTS.md`](AGENTS.md).
- Design notes: [`docs/design/`](docs/design/README.md).

## License

MIT for now; see [`LICENSE`](LICENSE). This is a placeholder: the license choice belongs to the
maintainers and has not been made. Code ported from colmat-rs or faer keeps its upstream notices; see
[`REPOSITORY_RULES.md`](REPOSITORY_RULES.md).
