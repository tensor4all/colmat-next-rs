# Design records

One file per durable decision that future work must follow: the contract, the chosen approach
and why, the alternatives rejected, and the constraints that remain. A pull request that changes
durable design intent updates the relevant record in the same pull request
(rules/common/repository.md, "Work Logs And Design Records").

## Where the design lives today

- [#2](https://github.com/tensor4all/colmat-next-rs/issues/2): goals, a candidate crate layout, and
  the decisions recorded in its comments (AoS storage with SoA registers, primitive kinds P1–P10,
  the scalar layers S0–S3). The `Precision` contract is proposed there, and its values are still
  under discussion.
- `tensor4all-meta` `docs/design/next-generation-dense-linear-algebra.md`
  (tensor4all/tensor4all-meta#104; status: design under review), sections 5–7a.

A note in this directory records what this trial adopts for its code. It cites where each
decision was made (an issue comment, the design document); a choice made only here is marked as
proposed.

## Records

- [`kernel-selection.md`](kernel-selection.md): how a primitive chooses its kernel (a static
  provider per scalar, a host kernel table of function pointers resolved once, and a plan per
  call that serves the scratch query, the estimate and the run), thresholds generated from
  measurements, and the tuning harness (#44, #45, #46; proposed).
- [`trial-architecture.md`](trial-architecture.md): how this trial applies #2 and the design
  document, the proposed crate layout and boundaries, what is ported from colmat-rs, and the first
  slice (proposed).
