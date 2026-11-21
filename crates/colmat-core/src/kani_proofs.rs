//! Kani proof harnesses (#17, a trial).
//!
//! Each harness proves, for every input within its stated bound, a claim
//! that a `// SAFETY:` comment of this crate relies on. Run with
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
