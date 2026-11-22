//! Kani proof harnesses (#17, a trial).
//!
//! Each harness proves, for every input within its stated bound, a claim
//! that a `// SAFETY:` comment of this crate relies on, or (#20) that the
//! split arithmetic of the execution layer cuts its range exactly. Run with
//! `cargo kani -p colmat-core`. Kani does not model unwinding or the
//! Stacked/Tree Borrows aliasing rules, so panic safety and aliasing stay
//! with Miri (`heavy-tests.yml`).

use crate::perm::PermRef;
use crate::utils::bound::{Dim, One, Zero};
use crate::zip::{
	MatLayoutTransform as L, transposed, triangular_lower_ncols,
	triangular_lower_rows,
};
use crate::{MatRef, Shape};

fn any_layout() -> L {
	match kani::any::<u8>() % 4 {
		0 => L::None,
		1 => L::ReverseRows,
		2 => L::Transpose,
		_ => L::TransposeReverseRows,
	}
}

fn swaps_dims(layout: L) -> bool {
	matches!(layout, L::Transpose | L::TransposeReverseRows)
}

/// The index of the original view for index `(i, j)` of the view that
/// `with_layout(_, layout)` produced, which has `m` rows. The same map as
/// the `from_dyn_idx` calls of the triangular iterations in `zip.rs`.
fn to_original(layout: L, m: usize, i: usize, j: usize) -> (usize, usize) {
	match layout {
		L::None => (i, j),
		L::ReverseRows => (m - 1 - i, j),
		L::Transpose => (j, i),
		L::TransposeReverseRows => (j, m - 1 - i),
	}
}

/// The inverse of [`to_original`].
fn to_view(layout: L, m: usize, p: usize, q: usize) -> (usize, usize) {
	match layout {
		L::None => (p, q),
		L::ReverseRows => (m - 1 - p, q),
		L::Transpose => (q, p),
		L::TransposeReverseRows => (m - 1 - q, p),
	}
}

/// The triangular iterations of `zip.rs` stay in bounds and visit exactly
/// the triangle, for every layout, both triangles, with and without the
/// diagonal, and every dimension (no bound: the claim has no loop).
///
/// The `// SAFETY:` comments of the triangular arms rely on `start <= end <=
/// m` for each column `j < ncols`; the iteration is correct only if, in
/// addition, the visited indices map back onto the triangle and every
/// element of the triangle is visited.
#[kani::proof]
fn triangular_ranges_are_in_bounds_and_cover_exactly_the_triangle() {
	let layout = any_layout();
	let upper: bool = kani::any();
	let strict: bool = kani::any();
	let (big_m, big_n): (usize, usize) = (kani::any(), kani::any());
	let (m, n) = if swaps_dims(layout) {
		(big_n, big_m)
	} else {
		(big_m, big_n)
	};
	// The upper triangle is iterated as the lower triangle of the transposed
	// layout (`for_each_mat_triangular_upper_with_index`).
	let rows_layout = if upper { transposed(layout) } else { layout };
	let s = strict as usize;
	let in_triangle =
		|p: usize, q: usize| if upper { p + s <= q } else { q + s <= p };
	let ncols = triangular_lower_ncols(rows_layout, m, n);
	assert!(ncols <= n);

	// Every visited index is in bounds and in the triangle.
	let j: usize = kani::any();
	kani::assume(j < ncols);
	let (start, end) = triangular_lower_rows(rows_layout, m, j, s);
	assert!(start <= end && end <= m);
	let i: usize = kani::any();
	kani::assume(start <= i && i < end);
	let (p, q) = to_original(layout, m, i, j);
	assert!(p < big_m && q < big_n);
	assert!(in_triangle(p, q));

	// Every element of the triangle is visited.
	let (p, q): (usize, usize) = (kani::any(), kani::any());
	kani::assume(p < big_m && q < big_n && in_triangle(p, q));
	let (i, j) = to_view(layout, m, p, q);
	assert!(j < ncols);
	let (start, end) = triangular_lower_rows(rows_layout, m, j, s);
	assert!(start <= i && i < end);
}

/// `Shape::cast_idx_slice` reads each entry back for the shapes whose index
/// is stored as `I` (`usize` and `Dim`).
#[kani::proof]
fn cast_idx_slice_reads_the_entries_back() {
	let a: [u32; 3] = kani::any();
	let s = <usize as Shape>::cast_idx_slice::<u32>(&a);
	assert!(s.len() == 3 && s[0] == a[0] && s[1] == a[1] && s[2] == a[2]);

	let k: [usize; 3] = kani::any();
	kani::assume(k[0] < 3 && k[1] < 3 && k[2] < 3);
	Dim::with(3, |n| {
		let idx = [n.idx(k[0]), n.idx(k[1]), n.idx(k[2])];
		let s = <Dim<'_> as Shape>::cast_idx_slice::<usize>(&idx);
		assert!(s == &k[..]);
	});
}

/// For `One`, whose index is the zero-sized `Zero`, the cast panics instead
/// of reading `I`s from a zero-sized slice (#14).
#[kani::proof]
#[kani::should_panic]
fn cast_idx_slice_panics_for_one() {
	let z = [Zero, Zero];
	let _ = <One as Shape>::cast_idx_slice::<u32>(&z);
}

/// `PermRef::new_checked` checks `forward[i] < n` and `inverse[forward[i]] ==
/// i`; the unchecked reads of a permutation rely on the whole permutation
/// invariant (every entry of both arrays `< n`, each the other's inverse).
/// Proves that the check implies the invariant, then reads every entry
/// through the bound arrays.
#[kani::proof]
#[kani::unwind(5)]
fn perm_new_checked_implies_the_permutation_invariant() {
	let fwd: [u32; 3] = kani::any();
	let inv: [u32; 3] = kani::any();
	for i in 0..3 {
		kani::assume((fwd[i] as usize) < 3);
		kani::assume(inv[fwd[i] as usize] as usize == i);
	}
	let p = PermRef::<u32>::new_checked(&fwd, &inv, 3);
	for i in 0..3 {
		assert!((inv[i] as usize) < 3);
		assert!(fwd[inv[i] as usize] as usize == i);
	}
	Dim::with(3, |n| {
		let (f, v) = p.as_shape(n).bound_arrays();
		let i = n.idx(kani::any_where(|i: &usize| *i < 3));
		let fi = f[i].zx();
		assert!(v[fi].zx() == i);
	});
}

/// `MatRef::split_at` gives four parts whose element `(i, j)` is the
/// parent's element at the shifted index, for every shape up to `3 x 3`,
/// both major orders, reversed or not, and every split point. Kani checks
/// that every read through the raw pointers stays in the buffer.
#[kani::proof]
fn split_at_parts_read_the_parent() {
	let buf: [u32; 9] = [0, 1, 2, 3, 4, 5, 6, 7, 8];
	let m: usize = kani::any_where(|m: &usize| *m <= 3);
	let n: usize = kani::any_where(|n: &usize| *n <= 3);
	let col_major: bool = kani::any();
	let (rs, cs) = if col_major {
		(1isize, m.max(1) as isize)
	} else {
		(n.max(1) as isize, 1isize)
	};
	// SAFETY: every `(i, j)` with `i < m <= 3`, `j < n <= 3` maps to offset
	// `i * rs + j * cs <= 8` in `buf`, which is borrowed for the view's
	// lifetime; the pointer is non-null.
	let a = unsafe { MatRef::from_raw_parts(buf.as_ptr(), m, n, rs, cs) };
	let a = if kani::any() { a.reverse_rows() } else { a };

	let i0: usize = kani::any_where(|i0: &usize| *i0 <= m);
	let j0: usize = kani::any_where(|j0: &usize| *j0 <= n);
	let (tl, tr, bl, br) = a.split_at(i0, j0);
	let i: usize = kani::any_where(|i: &usize| *i < m);
	let j: usize = kani::any_where(|j: &usize| *j < n);
	let got = match (i < i0, j < j0) {
		(true, true) => tl[(i, j)],
		(true, false) => tr[(i, j - j0)],
		(false, true) => bl[(i - i0, j)],
		(false, false) => br[(i - i0, j - j0)],
	};
	assert!(got == a[(i, j)]);
}

/// A dimension up to `max`: the split arithmetic divides by symbolic values, which CBMC cannot
/// decide over all of `usize` in reasonable time (none of the four harnesses below finished in
/// 240 s unbounded), so these harnesses bound their inputs. `partition::lower_area_split`
/// (`u128` products and `isqrt`) did not finish in 500 s even at `16 x 16`; its unit tests
/// cover it instead (#20 work log).
fn up_to(max: usize) -> usize {
	kani::any_where(|x: &usize| *x <= max)
}

/// `par_split_indices` cuts `0..n` into `chunk_count` consecutive blocks that cover it, whose
/// lengths differ by at most one. The item loops deal their blocks with it, and `deal` splits a
/// producer at these lengths. `n <= 1024`, counts up to 16.
#[kani::proof]
fn par_split_indices_cuts_consecutive_blocks() {
	use crate::utils::thread::par_split_indices;
	let n = up_to(1024);
	let count = up_to(16);
	kani::assume(count > 0);
	let idx = up_to(15);
	kani::assume(idx < count);
	let (start, len) = par_split_indices(n, idx, count);
	assert!(start <= n && len <= n - start);
	assert!(len == n / count || len == n / count + 1);
	if idx == 0 {
		assert!(start == 0);
	}
	if idx + 1 < count {
		let (next, _) = par_split_indices(n, idx + 1, count);
		assert!(next == start + len);
	} else {
		assert!(start + len == n);
	}
}

/// The recursive halving of `deal` hands each half at least one participant and at least one
/// item: for `2 <= jobs <= min(len, size)`, the left half has `left = jobs / 2` blocks, its
/// participants are `1..size`, its items `1..len`. Teams up to 256, items up to 4096.
#[kani::proof]
fn deal_halves_are_non_empty() {
	use crate::utils::thread::par_split_indices;
	let size = up_to(256);
	let len = up_to(4096);
	let jobs = up_to(256);
	kani::assume(jobs >= 2 && jobs <= size && jobs <= len);
	let left = jobs / 2;
	let (left_size, _) = par_split_indices(size, left, jobs);
	let (left_len, _) = par_split_indices(len, left, jobs);
	assert!(left_size >= 1 && left_size < size);
	assert!(left_len >= 1 && left_len < len);
}

/// The parallel layout copy shares `lines` lines among at most `degree` tasks in whole tiles,
/// every task non-empty: the producer of chunks it deals has exactly `n_tasks` items (the
/// `debug_assert!` in `copy_across_layouts`). Lines up to 4096, teams up to 64.
#[kani::proof]
fn copy_tasks_are_whole_tiles_and_non_empty() {
	use crate::layout::{COPY_TILE, copy_tasks};
	let lines = up_to(4096);
	let degree = up_to(64);
	kani::assume(degree >= 1);
	let (chunk, n_tasks) = copy_tasks(lines, degree);
	if lines == 0 {
		// nothing to share: no task, and the copy takes its serial path
		assert!(n_tasks == 0);
		return;
	}
	assert!(n_tasks >= 1 && n_tasks <= degree);
	assert!(chunk % COPY_TILE == 0 && chunk >= COPY_TILE);
	// the chunks of `chunk` lines number `n_tasks`, the last one non-empty
	assert!((n_tasks - 1) * chunk < lines && lines <= n_tasks * chunk);
}

/// `grain_range` cuts `0..len` into `parts` consecutive ranges whose inner edges are multiples
/// of `grain`, covering it. The grains in use (`ROW_GRAIN`, `COL_GRAIN`), lengths up to 1024,
/// up to 16 parts.
#[kani::proof]
fn grain_range_cuts_consecutive_ranges() {
	use crate::utils::partition::{COL_GRAIN, ROW_GRAIN, grain_range};
	let len = up_to(1024);
	let parts = up_to(16);
	kani::assume(parts > 0);
	let grain = if kani::any() { ROW_GRAIN } else { COL_GRAIN };
	let idx = up_to(15);
	kani::assume(idx < parts);
	let (start, end) = grain_range(len, parts, idx, grain);
	assert!(start <= end && end <= len);
	assert!(start % grain == 0 || start == len);
	if idx == 0 {
		assert!(start == 0);
	}
	if idx + 1 < parts {
		assert!(grain_range(len, parts, idx + 1, grain).0 == end);
	} else {
		assert!(end == len);
	}
}
