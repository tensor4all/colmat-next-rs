//! Static partitions of a parallel region's work (faer#168).
//!
//! A part computed once from the shape, before the region starts, so every
//! participant knows its share without negotiating it: the static-ownership
//! split OpenBLAS's level-3 drivers use. The triangle (`syrk_thread`) is
//! shared by the products into a triangle and, in turn, by the factorizations
//! whose trailing update is one (Cholesky, LU, the symmetric reductions); the
//! tile grid by the products into a rectangle.
//!
//! These are public so that a driver outside this crate can cut its regions
//! the same way (design record §5, "Drivers").
//!
//! # Provenance
//!
//! Ported from colmat-rs at `71337cdc`: `faer/src/utils/partition.rs`
//! (`lower_area_split`, `lower_area_parts`, lines 10-90, with their tests) and
//! `faer/src/utils/team.rs` (`ROW_GRAIN`, `COL_GRAIN`, `tile_grid`,
//! `grain_range`, lines 443-496, with `grid_uses_the_team_and_stays_in_bounds`).
//! Changed: public, and compiled on every target (colmat-rs gated the
//! equal-area split to x86-64 with its private kernel; debt K6). faer's MIT
//! notice: `LICENSE-faer`.

/// The first column of part `i` of `parts` when the lower part of an `m x n`
/// matrix (row `r`, column `c`, `r >= c`) is cut into column ranges of equal
/// area.
///
/// Column `c` holds `m - c` entries, so the area left of `c` is `c*m - c*(c -
/// 1)/2`. Boundaries are rounded to the nearest column, then up to multiples of
/// `align` (the kernel's column step), are nondecreasing, and part `parts - 1`
/// ends at `n`; a part may be empty unless `parts <= lower_area_parts(m, n,
/// align)`. An upper triangle is the lower one of the transpose, and a strict
/// triangle shifts the rows by one before calling this.
pub fn lower_area_split(
	m: usize,
	n: usize,
	i: usize,
	parts: usize,
	align: usize,
) -> usize {
	if i == 0 {
		return 0;
	}
	if i >= parts {
		return n;
	}
	// In integers, because the crate builds without `std` (no `f64::sqrt`). The
	// boundary is the smaller root `c` of `c*m - c*(c - 1)/2 = want`, `want =
	// total * i / parts`, rounded to the nearest column: with `B = 2m + 1`, `c
	// = (B - sqrt(B^2 - 8 want)) / 2`, so the nearest column is the largest
	// `k` with `parts * (B + 1 - 2k)^2 >= q`, where `q = parts * B^2 - 8 *
	// total * i`.
	let (m, w, p) = (m as u128, n.min(m) as u128, parts as u128);
	let total = w * m - w * w.saturating_sub(1) / 2;
	let big_b = 2 * m + 1;
	let q = p * big_b * big_b - 8 * total * i as u128;
	let fits =
		|k: u128| k <= big_b.div_ceil(2) && p * (big_b + 1 - 2 * k).pow(2) >= q;
	let mut k = (big_b + 1 - (q / p).isqrt()) / 2;
	while fits(k + 1) {
		k += 1;
	}
	while !fits(k) {
		k -= 1;
	}
	(k as usize).next_multiple_of(align.max(1)).min(n)
}

/// The most parts [`lower_area_split`] can cut the lower part of an `m x n`
/// matrix into without leaving a part empty.
///
/// Part `i` holds the area `total / parts`, and a part spanning `dc` columns
/// from column `c` holds at most `dc * (m - c) <= dc * m` entries. So
/// `total / parts >= align * m` makes every exact boundary advance by at least
/// `align` columns. Rounding to the nearest column then up to a multiple of
/// `align` keeps that step: rounding to the nearest column commutes with an
/// integer shift, and `align | align * k` makes the round-up of two columns
/// `align` apart differ by exactly `align`. Without the cap the boundaries
/// collapse: at 64 parts on the `128 x 128` trailing block of a 1024 Cholesky
/// the exact parts are one column wide, every boundary rounds up to 8, and 7 of
/// the 64 parts are empty while the section still wakes all 64 participants
/// (faer#168).
///
/// Callers pass this as the region width, so the participants a collapsed part
/// would have woken stay parked. It returns at least 1: an empty matrix is one
/// part that does nothing, not zero parts.
pub fn lower_area_parts(m: usize, n: usize, align: usize) -> usize {
	// the area of the lower part of the block, in integers (`total` above)
	let (m, w) = (m as u128, n.min(m) as u128);
	let total = w * m - w * w.saturating_sub(1) / 2;
	let strip = align.max(1) as u128 * m;
	if strip == 0 {
		return 1;
	}
	usize::try_from(total / strip).unwrap_or(usize::MAX).max(1)
}

/// Granularity of the row edges of a tile ([`tile_grid`]): a multiple of the register tile
/// of the `gemm` micro kernels (f64 on aarch64: 8 rows, 4 columns), so a tile does not end in
/// a partial micro panel unless the matrix does.
pub const ROW_GRAIN: usize = 8;
/// Granularity of the column edges of a tile, see [`ROW_GRAIN`].
pub const COL_GRAIN: usize = 4;

/// The grid `(pm, pn)` with `pm * pn <= nt` that minimises the packed volume
/// `k * (m / pm + n / pn)` per member, among grids that use as many members as the shape
/// allows (every tile at least one grain wide).
pub fn tile_grid(m: usize, n: usize, nt: usize) -> (usize, usize) {
	let max_pm = (m / ROW_GRAIN).max(1);
	let max_pn = (n / COL_GRAIN).max(1);
	let mut best = (1, 1);
	let mut best_used = 1;
	let mut best_cost = f64::INFINITY;
	for pm in 1..=nt.min(max_pm) {
		for pn in 1..=(nt / pm).min(max_pn) {
			let used = pm * pn;
			let cost = m as f64 / pm as f64 + n as f64 / pn as f64;
			// more members first, then the smaller packed volume
			if used > best_used || (used == best_used && cost < best_cost) {
				best = (pm, pn);
				best_used = used;
				best_cost = cost;
			}
		}
	}
	best
}

/// `[start, end)` of part `idx` of `0..len` cut into `parts` blocks whose edges are multiples
/// of `grain` (the last block takes the remainder).
pub fn grain_range(
	len: usize,
	parts: usize,
	idx: usize,
	grain: usize,
) -> (usize, usize) {
	let units = len.div_ceil(grain);
	let edge = |i: usize| -> usize {
		let base = units / parts;
		let rem = units % parts;
		let u = if i < rem {
			i * (base + 1)
		} else {
			rem + i * base
		};
		(u * grain).min(len)
	};
	(edge(idx), edge(idx + 1))
}

#[cfg(test)]
mod tests {
	use super::*;

	/// The boundaries are the nearest columns to the exact roots, decided in
	/// integers (no floating point, so every host cuts the same way).
	/// Reference values from the closed form.
	#[test]
	fn lower_boundaries_are_the_rounded_roots() {
		let cuts = |m, n, parts: usize, count: usize| -> Vec<usize> {
			(1..=count)
				.map(|i| lower_area_split(m, n, i, parts, 1))
				.collect()
		};
		assert_eq!(
			cuts(4096, 4096, 16, 8),
			[130, 265, 404, 549, 700, 858, 1024, 1200]
		);
		assert_eq!(cuts(1024, 1024, 64, 8), [8, 16, 24, 33, 41, 49, 58, 66]);
		assert_eq!(cuts(2047, 2048, 7, 6), [152, 317, 500, 707, 953, 1274]);
	}

	/// The parts cover `0..n` in order, start on multiples of `align`, and hold
	/// equal areas up to one aligned column strip.
	#[test]
	fn lower_parts_cover_the_triangle_with_equal_areas() {
		for &(m, n) in &[
			(4096usize, 4096usize),
			(2047, 2048),
			(301, 301),
			(40, 40),
			(9, 10),
			(900, 300),
		] {
			for align in [1usize, 8] {
				for parts in [1usize, 2, 3, 4, 7, 16, 64] {
					let cuts: Vec<usize> = (0..=parts)
						.map(|i| lower_area_split(m, n, i, parts, align))
						.collect();
					assert_eq!((cuts[0], cuts[parts]), (0, n));
					assert!(
						cuts.windows(2).all(|w| w[0] <= w[1]),
						"{m}x{n}/{parts}: {cuts:?}"
					);
					assert!(
						cuts[1..parts]
							.iter()
							.all(|c| c % align == 0 || *c == n)
					);
					let area = |c0: usize, c1: usize| {
						(c0..c1.min(m)).map(|c| m - c).sum::<usize>()
					};
					let total = area(0, n);
					for w in cuts.windows(2) {
						assert!(
							area(w[0], w[1]).abs_diff(total / parts)
								<= align * m,
							"{m}x{n}/{parts}/{align}: {cuts:?}"
						);
					}
				}
			}
		}
	}

	/// The cap is the request the collapse comes from: at 64 parts on the
	/// `128 x 128` trailing block of the 1024 Cholesky the exact parts are one
	/// column wide, so every boundary rounds up to 8 and seven parts are empty
	/// while the region still wakes all 64 participants.
	#[test]
	fn lower_area_parts_caps_the_trailing_block_of_a_1024_cholesky() {
		let cuts: Vec<usize> = (0..=64)
			.map(|i| lower_area_split(128, 128, i, 64, 8))
			.collect();
		assert_eq!(cuts[1..8], [8; 7]);
		assert_eq!(lower_area_parts(128, 128, 8), 8);
		// the whole 1024 trailing block still takes every part
		assert_eq!(lower_area_parts(1024, 1024, 8), 64);
		assert_eq!(lower_area_parts(2048, 2048, 8), 128);
		assert_eq!(lower_area_parts(4096, 4096, 8), 256);
	}

	/// A part holds at least one aligned column strip, so `parts` at or below
	/// the cap never leaves one empty. Boundaries are nondecreasing, the first
	/// is 0 and the last is `n`.
	///
	/// Every `parts` up to the cap is checked, so the calls grow as `cap^2 / 2`:
	/// 4096 x 4096 with `align = 1` alone (cap 2048) makes about 2.1M calls.
	/// Under Miri that did not finish within the nightly's 60 minutes (#21), so
	/// Miri skips 2048 and above, and 1024 with `align = 1`: about 28k calls,
	/// which keep the small sizes, both aligns and 1024 at `align = 8`.
	#[test]
	fn lower_area_parts_keeps_every_part_non_empty() {
		for &(m, n) in &[
			(4096usize, 4096usize),
			(2048, 2048),
			(1024, 1024),
			(300, 1024),
			(256, 256),
			(128, 128),
			(130, 131),
			(64, 64),
			(40, 40),
			(9, 10),
			(8, 8),
			(2, 2),
			(1, 1),
		] {
			for align in [1usize, 8] {
				if cfg!(miri) && (m >= 2048 || (m >= 1024 && align == 1)) {
					continue;
				}
				let cap = lower_area_parts(m, n, align);
				assert!(cap >= 1, "{m}x{n}/{align}");
				for parts in 1..=cap {
					let cuts: Vec<usize> = (0..=parts)
						.map(|i| lower_area_split(m, n, i, parts, align))
						.collect();
					assert_eq!((cuts[0], cuts[parts]), (0, n));
					assert!(
						cuts.windows(2).all(|w| w[0] < w[1]),
						"{m}x{n}/{parts}/{align}: {cuts:?}"
					);
				}
			}
		}
	}

	/// The cap's own boundaries: a small area, a block with more rows than
	/// columns (one aligned strip covers the whole triangle), a block with
	/// fewer rows than columns, and an empty matrix.
	#[test]
	fn lower_area_parts_boundaries() {
		// 8 x 8 holds 36 entries, less than one 8-column strip of 64
		assert_eq!(lower_area_parts(8, 8, 8), 1);
		// 8 columns is already one aligned strip, whatever the row count
		assert_eq!(lower_area_parts(100_000, 8, 8), 1);
		assert_eq!(lower_area_parts(1, 1000, 8), 1);
		assert_eq!(lower_area_parts(1, 1, 8), 1);
		assert_eq!(lower_area_parts(0, 0, 8), 1);
		assert_eq!(lower_area_parts(0, 5, 8), 1);
		assert_eq!(lower_area_parts(5, 0, 8), 1);
		// with a one-column step the cap is the half-area over one column,
		// so the square is cut in two parts per column pair
		assert_eq!(lower_area_parts(1024, 1024, 1), 512);
		assert_eq!(lower_area_parts(1000, 1000, 1), 500);
	}

	#[test]
	fn grid_uses_the_team_and_stays_in_bounds() {
		for nt in 1..=12 {
			for &(m, n) in &[
				(2048, 2048),
				(4096, 64),
				(64, 4096),
				(100, 7),
				(8, 4),
				(1, 1000),
			] {
				let (pm, pn) = tile_grid(m, n, nt);
				assert!(pm * pn <= nt && pm >= 1 && pn >= 1, "{m}x{n} nt={nt}");
				// the tiles partition the output
				let mut rows = 0;
				for i in 0..pm {
					let (a, b) = grain_range(m, pm, i, ROW_GRAIN);
					assert_eq!(a, rows);
					rows = b;
				}
				assert_eq!(rows, m);
				let mut cols = 0;
				for j in 0..pn {
					let (a, b) = grain_range(n, pn, j, COL_GRAIN);
					assert_eq!(a, cols);
					cols = b;
				}
				assert_eq!(cols, n);
			}
		}
		assert_eq!(tile_grid(2048, 2048, 6).0 * tile_grid(2048, 2048, 6).1, 6);
	}
}
