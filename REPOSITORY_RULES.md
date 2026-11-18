# Repository rules

Working rules for this trial repository. They are not decisions of the colmat maintainers. Rules
2 and 3 restate [#2](https://github.com/tensor4all/colmat-next-rs/issues/2); rule 1 applies the
shared provenance rule; rules 4 and 5 are this trial's own and can change with the maintainers'
review.

1. **Provenance of ported code.** Code taken or closely translated from colmat-rs or faer keeps
   faer's copyright notice and the MIT text, and the file says where it came from (repository, path,
   commit). Code from LAPACK, Eigen, SuiteSparse or other projects that colmat-rs carries under its
   own `COPYING.*` files keeps those terms: copy the matching `COPYING.*` file in the same change, and
   set the package `license` expression accordingly (`rules/common/provenance.md`).
2. **Only `threadteam` starts tasks.** No crate here depends on `rayon` directly; `deny.toml`
   rejects it. A primitive takes the `Par` it is given and never starts threads of its own (#2,
   goal 4).
3. **CPU only.** GPU backends belong in a separate library; the traits here do not grow stream,
   async or batch abstractions in anticipation of one (#2, scope).
4. **This is not a fork.** The shared Rust rules apply to every file, including test placement
   (`src/<module>/tests/`); the upstream-layout exception for forks does not apply here.
5. **Docs follow the code.** Until a crate exists, README, `llms.txt` and usage skills describe no
   API. Each is added in the change that adds the code it describes.
