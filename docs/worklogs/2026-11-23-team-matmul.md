# colmat-core matrix product on a team: the tile grid and the depth split (#28)

A record of #28: how a product on a `Par::Team` is split. #27 measured the product against
colmat-rs `71337cdc` on one thread only; the team measurements showed two regressions, and #28
(shinaoka, 2026-11-23) chose the fixes: the tile grid for products, and for the column-major
matrix-vector product the depth split above a measured size, with the partial sums from the
caller's scratch. Everything here describes this trial.

Setup: Apple M1 Pro (8 performance + 2 efficiency cores), Rust 1.96.0, `Accum::Replace`,
threadteam team widths 1, 2 and 4 (width 8 reaches the efficiency cores and varies by about ±30 %).
Median of 21 batches of at least 5 ms each; inputs built before timing. x86-64 was not measured.

## Products: the tile grid

`split_product` gave each participant a block of columns of `dst` (rows when `dst` had fewer
columns than participants), so for a tall product every participant packed the whole `lhs`. It
now cuts `dst` into the grid of `utils::partition::tile_grid`, as colmat-rs's `gemm_tiled` did:
the grid's columns go to sub-teams (`for_each_with_par`), each of which deals its rows. The depth
is not split, so the result stays bitwise that of `Par::Seq`.

`f64` `4096 x 64 x 64`, width 2: 1.26x colmat-rs before, 1.00x after; width 4: 1.21x, 1.00x.
Every product shape measured (`256^3` to `2048^3`, `4096 x 64 x 64`, `64 x 4096 x 64`,
`256 x 256 x 4096`) is at 0.97–1.03x colmat-rs on widths 1–4 for all four native types.

## The column-major matrix-vector product: rows or depth

The row split gives every participant a segment of every column; it keeps `Par::Seq`'s order and
needs no buffer. The depth split (colmat-rs's) gives every participant a block of whole columns,
summed into its own column of an `nrows x jobs` buffer, and then adds the columns in order, here
on the whole team by rows.

Both were measured on `m` in {64, 256, 1024, 4096, 16384}, `k` in {16, 64, 256, 1024, 4096},
widths 2 and 4, all four types (200 cases, twice). Depth over rows:

- Shallow products: the depth split loses below a depth of 256, by 1.2–4x at `m >= 4096` (its
  partial sums and their addition cost as much as the work they divide). At a depth of 256 the two
  are within about 10 %, either way.
- Depth 1024 and 4096, matrix in cache (8 MiB or less): the row split wins by up to 2x at width 4
  when the columns are long (`f64` `256 x 4096`: 49 against 85 µs; `c32` `1024 x 1024`: 153
  against 183 µs).
- Depth 1024 and 4096, matrix larger than 8 MiB: the depth split wins by 1.1–3x
  (`f64` `1024 x 4096` at width 4: 425 against 887 µs, the row split slower than one thread at
  657 µs; `f32` `4096 x 4096` at width 4: 619 against 2109 µs).
- Columns of a few cache lines: the depth split wins by 2–5x (`f32` `64 x 4096` at width 4:
  9 against 42 µs; `c32` `64 x 1024`: 9 against 16 µs), cache resident or not.

The rule (`generic::depth_split`): split the depth when `k >= 1024` and either the matrix is
larger than 8 MiB or a column is at most 512 bytes. It depends on the shape and the scalar only,
not the width, so `matmul_scratch` can tell from the shape whether the partials are needed, and
asks for `nrows x min(degree, k)` elements then. The boundary at 8 MiB is this host's: an `f64`
`1024 x 1024` product (exactly 8 MiB) stays on the row split, 1.15–1.45x colmat-rs. The cache
size of other hosts is not read; that would be the cost model's job once it has per-host rows.

After the change, against colmat-rs `71337cdc`, the 200 cases: `f64` and `f32` have a median of
1.00x (worst 1.45x, the 8 MiB boundary above); `4096 x 1 x 4096` went from 1.57x/1.53x
(`f64`, widths 2/4) and 1.73x/2.23x (`f32`) to 1.02x/1.01x and 1.01x/0.97x. `c64` and `c32` have a
median of 1.25x and 1.39x: their one-thread loads and stores are slower (#26), which the split
does not change.

## Rounding

The depth split rounds differently from `Par::Seq`. It is the same on every call of the same
width: the partition of the depth (`k.div_ceil(width)` columns per job) and the order of the
addition (job 0 first) depend on the shape and the width only. The error stays within the
reference bound of the tests (`(k + 2) eps |alpha| sum |a||b|`, any order of summation). The
bitwise `Par::Seq` test covers the products and the matrix-vector shapes below the split; the new
tests check the split shapes against the bound (both accumulations, the conjugations, widths 2–4,
an `m == 1` product with a row-major `rhs`), its determinism per width, and the rule itself. The
no-allocation test covers a depth-split shape on teams of 2 and 4.

## Not done

- A depth split of the row-major matrix-vector product (a dot product per row): it already reads
  whole rows, and was not measured to lose.
- Reading the cache size of the host for the 8 MiB bound.
