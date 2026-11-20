//! Property tests of the views against a reference index model (#12).
//!
//! Every view is a pointer, a shape and two strides, so the element at
//! `(i, j)` is the one at offset `base + i * rs + j * cs` of the buffer. The
//! [`Model`] below is that formula and nothing else; each test builds a view
//! over a buffer whose element at offset `k` holds `k`, applies an operation
//! to the view and the same operation to the model, and compares every
//! element. Shapes include 0 and 1, strides are negative, zero (read-only
//! views) and padded.
//!
//! The ported unit tests of colmat-rs stay in the modules they came from.
//! These are new in this repository. `prop_` tests run 256 cases per PR and
//! 65 536 nightly (`PROPTEST_CASES`).

#![allow(clippy::needless_range_loop)]

use crate::internal_prelude::*;
use crate::layout::{copy_across_layouts, tile_copy_pays};
use crate::perm::permute_rows;
use proptest::prelude::*;
use std::assert;

/// Offset formula of a strided view.
#[derive(Copy, Clone, Debug)]
struct Model {
	base: isize,
	m: usize,
	n: usize,
	rs: isize,
	cs: isize,
}

impl Model {
	fn off(&self, i: usize, j: usize) -> usize {
		assert!(i < self.m && j < self.n);
		(self.base + i as isize * self.rs + j as isize * self.cs) as usize
	}

	/// Smallest buffer that holds every element, with `base` chosen so that
	/// the smallest offset is 0.
	fn with_base(m: usize, n: usize, rs: isize, cs: isize) -> (Self, usize) {
		let mut lo = 0isize;
		let mut hi = 0isize;
		if m > 0 && n > 0 {
			for (i, j) in [(0, 0), (m - 1, 0), (0, n - 1), (m - 1, n - 1)] {
				let o = i as isize * rs + j as isize * cs;
				lo = lo.min(o);
				hi = hi.max(o);
			}
		}
		let model = Model {
			base: -lo,
			m,
			n,
			rs,
			cs,
		};
		(model, (hi - lo + 1) as usize)
	}

	fn transpose(self) -> Self {
		Model {
			m: self.n,
			n: self.m,
			rs: self.cs,
			cs: self.rs,
			..self
		}
	}

	fn sub(self, i0: usize, j0: usize, m: usize, n: usize) -> Self {
		Model {
			base: self.base + i0 as isize * self.rs + j0 as isize * self.cs,
			m,
			n,
			..self
		}
	}

	fn reverse_rows(self) -> Self {
		let base = if self.m == 0 {
			self.base
		} else {
			self.base + (self.m - 1) as isize * self.rs
		};
		Model {
			base,
			rs: -self.rs,
			..self
		}
	}

	fn reverse_cols(self) -> Self {
		self.transpose().reverse_rows().transpose()
	}
}

/// A buffer holding its own offsets, and a read-only view over it.
fn buffer(len: usize) -> Vec<f64> {
	(0..len).map(|k| k as f64).collect()
}

fn view<'a>(buf: &'a [f64], md: &Model) -> MatRef<'a, f64> {
	// SAFETY: `Model::with_base` sized `buf` so that every in-bounds `(i, j)`
	// maps into it, and nothing writes to `buf` while the view lives.
	unsafe {
		MatRef::from_raw_parts(
			buf.as_ptr().wrapping_offset(md.base),
			md.m,
			md.n,
			md.rs,
			md.cs,
		)
	}
}

#[track_caller]
fn check(v: MatRef<'_, f64>, md: &Model) {
	assert_eq!((v.nrows(), v.ncols()), (md.m, md.n));
	for j in 0..md.n {
		for i in 0..md.m {
			assert_eq!(v[(i, j)], md.off(i, j) as f64, "({i}, {j}) of {md:?}");
		}
	}
}

/// Any strides, including zero and negative (read-only views may alias).
fn any_strided() -> impl Strategy<Value = (usize, usize, isize, isize)> {
	(0usize..9, 0usize..9, -12isize..=12, -12isize..=12)
}

/// Layouts in which no two elements share memory: column- or row-major with a
/// padded leading dimension, each direction possibly reversed.
fn disjoint_layout() -> impl Strategy<Value = (usize, usize, isize, isize)> {
	(0usize..9, 0usize..9, 0usize..3, any::<[bool; 3]>()).prop_map(
		|(m, n, pad, [row_major, flip_r, flip_c])| {
			let (rs, cs) = if row_major {
				((n + pad).max(1) as isize, 1)
			} else {
				(1, (m + pad).max(1) as isize)
			};
			let rs = if flip_r { -rs } else { rs };
			let cs = if flip_c { -cs } else { cs };
			(m, n, rs, cs)
		},
	)
}

proptest! {
	#[test]
	fn prop_view_reads_the_model((m, n, rs, cs) in any_strided()) {
		let (md, len) = Model::with_base(m, n, rs, cs);
		let buf = buffer(len);
		let v = view(&buf, &md);
		check(v, &md);
		check(v.transpose(), &md.transpose());
		check(v.reverse_rows(), &md.reverse_rows());
		check(v.reverse_cols(), &md.reverse_cols());
		check(v.reverse_rows_and_cols(), &md.reverse_rows().reverse_cols());
	}

	#[test]
	fn prop_subviews_and_splits(
		(m, n, rs, cs) in any_strided(),
		a in 0usize..=8, b in 0usize..=8, c in 0usize..=8, d in 0usize..=8,
	) {
		let (md, len) = Model::with_base(m, n, rs, cs);
		let buf = buffer(len);
		let v = view(&buf, &md);
		let (i0, j0) = (a.min(m), b.min(n));
		let (mm, nn) = (c.min(m - i0), d.min(n - j0));
		check(v.submatrix(i0, j0, mm, nn), &md.sub(i0, j0, mm, nn));
		check(v.subrows(i0, mm), &md.sub(i0, 0, mm, n));
		check(v.subcols(j0, nn), &md.sub(0, j0, m, nn));

		let (tl, tr, bl, br) = v.split_at(i0, j0);
		check(tl, &md.sub(0, 0, i0, j0));
		check(tr, &md.sub(0, j0, i0, n - j0));
		check(bl, &md.sub(i0, 0, m - i0, j0));
		check(br, &md.sub(i0, j0, m - i0, n - j0));
		let (top, bot) = v.split_at_row(i0);
		check(top, &md.sub(0, 0, i0, n));
		check(bot, &md.sub(i0, 0, m - i0, n));
		let (left, right) = v.split_at_col(j0);
		check(left, &md.sub(0, 0, m, j0));
		check(right, &md.sub(0, j0, m, n - j0));
	}

	#[test]
	fn prop_rows_cols_and_diagonal((m, n, rs, cs) in any_strided()) {
		let (md, len) = Model::with_base(m, n, rs, cs);
		let buf = buffer(len);
		let v = view(&buf, &md);
		for j in 0..n {
			let col = v.col(j);
			prop_assert_eq!(col.nrows(), m);
			for i in 0..m {
				prop_assert_eq!(col[i], md.off(i, j) as f64);
			}
		}
		for i in 0..m {
			let row = v.row(i);
			prop_assert_eq!(row.ncols(), n);
			for j in 0..n {
				prop_assert_eq!(row[j], md.off(i, j) as f64);
			}
		}
		let diag = v.diagonal().column_vector();
		prop_assert_eq!(diag.nrows(), m.min(n));
		for k in 0..m.min(n) {
			prop_assert_eq!(diag[k], md.off(k, k) as f64);
		}
		let iter: Vec<_> = v.col_iter().map(|c| c.nrows()).collect();
		prop_assert_eq!(iter, vec![m; n]);
		let iter: Vec<_> = v.row_iter().map(|r| r.ncols()).collect();
		prop_assert_eq!(iter, vec![n; m]);
	}

	/// The four quadrants of a mutable split, and the two halves of the row and
	/// column splits, cover every element of the parent exactly once.
	#[test]
	fn prop_mutable_splits_are_disjoint_and_cover(
		(m, n, rs, cs) in disjoint_layout(),
		a in 0usize..=8, b in 0usize..=8,
	) {
		let (md, len) = Model::with_base(m, n, rs, cs);
		let (i0, j0) = (a.min(m), b.min(n));
		let mut buf = vec![0.0f64; len];
		{
			// SAFETY: as in `view`, and `buf` is borrowed mutably for the view's
			// lifetime; `disjoint_layout` maps distinct `(i, j)` to distinct
			// offsets.
			let mut v = unsafe {
				MatMut::from_raw_parts_mut(
					buf.as_mut_ptr().wrapping_offset(md.base),
					m, n, rs, cs,
				)
			};
			let (tl, tr, bl, br) = v.rb_mut().split_at_mut(i0, j0);
			for (q, mut part) in [tl, tr, bl, br].into_iter().enumerate() {
				part.fill(0.0);
				zip!(part.rb_mut()).for_each(|unzip!(x)| *x += (q + 1) as f64);
			}
			let (mut top, mut bot) = v.rb_mut().split_at_row_mut(i0);
			zip!(top.rb_mut()).for_each(|unzip!(x)| *x += 10.0);
			zip!(bot.rb_mut()).for_each(|unzip!(x)| *x += 20.0);
			let (mut left, mut right) = v.rb_mut().split_at_col_mut(j0);
			zip!(left.rb_mut()).for_each(|unzip!(x)| *x += 100.0);
			zip!(right.rb_mut()).for_each(|unzip!(x)| *x += 200.0);
		}
		let mut touched = 0;
		for j in 0..n {
			for i in 0..m {
				let q = match (i < i0, j < j0) {
					(true, true) => 1.0,
					(true, false) => 2.0,
					(false, true) => 3.0,
					(false, false) => 4.0,
				};
				let h = if i < i0 { 10.0 } else { 20.0 };
				let w = if j < j0 { 100.0 } else { 200.0 };
				prop_assert_eq!(buf[md.off(i, j)], q + h + w);
				touched += 1;
			}
		}
		// Padding is never written.
		let written = buf.iter().filter(|&&x| x != 0.0).count();
		prop_assert_eq!(written, touched);
	}

	/// `zip!` over views of different layouts visits matching elements.
	#[test]
	fn prop_zip_matches_elements(
		(m, n, rs, cs) in disjoint_layout(),
		row_major_b in any::<bool>(),
	) {
		let (md, len) = Model::with_base(m, n, rs, cs);
		let buf = buffer(len);
		let a = view(&buf, &md);
		let b = Mat::<f64>::from_fn(m, n, |i, j| (i * 100 + j) as f64);
		let b_t = b.transpose().to_owned();
		let b_view = if row_major_b { b_t.transpose() } else { b.as_ref() };
		let mut out = Mat::<f64>::zeros(m, n);
		zip!(&mut out, a, b_view).for_each(|unzip!(o, x, y)| *o = *x * 1000.0 + *y);
		for j in 0..n {
			for i in 0..m {
				prop_assert_eq!(out[(i, j)], md.off(i, j) as f64 * 1000.0 + (i * 100 + j) as f64);
			}
		}
		let mut count = 0usize;
		zip!(a).for_each(|unzip!(_x)| count += 1);
		prop_assert_eq!(count, m * n);
	}

	/// Owned matrices: `from_fn`, growth through `resize_with` (which may
	/// reallocate), `truncate`, `push_col`, `clone` and `to_owned` keep every
	/// element where it belongs.
	#[test]
	fn prop_owned_matrix_resizes(
		m0 in 0usize..7, n0 in 0usize..7, m1 in 0usize..12, n1 in 0usize..12,
		reserve in any::<bool>(),
	) {
		let f = |i: usize, j: usize| (i * 1000 + j) as f64;
		let mut a = Mat::<f64>::from_fn(m0, n0, f);
		if reserve {
			a.reserve(m1, n1);
			prop_assert!(a.row_capacity() >= m1 && a.col_capacity() >= n1);
		}
		a.resize_with(m1, n1, f);
		prop_assert_eq!(a.shape(), (m1, n1));
		for j in 0..n1 {
			for i in 0..m1 {
				prop_assert_eq!(a[(i, j)], f(i, j));
			}
		}
		let (mt, nt) = (m1 / 2, n1 / 2);
		a.truncate(mt, nt);
		prop_assert_eq!(a.shape(), (mt, nt));
		let extra = Col::<f64>::from_fn(mt, |i| f(i, nt));
		a.push_col(extra.as_ref());
		prop_assert_eq!(a.shape(), (mt, nt + 1));
		let b = a.clone();
		let c = a.as_ref().reverse_rows().to_owned();
		for j in 0..=nt {
			for i in 0..mt {
				prop_assert_eq!(b[(i, j)], f(i, j));
				prop_assert_eq!(c[(mt - 1 - i, j)], f(i, j));
			}
		}
		prop_assert!(a == b);
	}

	/// The tiled layout copy writes what an elementwise copy writes, whatever
	/// the two layouts and their directions.
	#[test]
	fn prop_layout_copy_matches_elementwise(
		(m, n, rs, cs) in disjoint_layout(),
		big in any::<bool>(),
	) {
		// Shapes past one tile in each direction on some cases.
		let (m, n) = if big { (m * 5, n * 5) } else { (m, n) };
		let (rs, cs) = if rs.unsigned_abs() == 1 {
			(rs, cs.signum() * (m.max(1) as isize))
		} else {
			(rs.signum() * (n.max(1) as isize), cs)
		};
		let (md, len) = Model::with_base(m, n, rs, cs);
		let buf = buffer(len);
		let src = view(&buf, &md);
		let mut cm = Mat::<f64>::zeros(m, n);
		copy_across_layouts(cm.as_mut(), src);
		let mut rm_t = Mat::<f64>::zeros(n, m);
		copy_across_layouts(rm_t.as_mut().transpose_mut(), src);
		for j in 0..n {
			for i in 0..m {
				prop_assert_eq!(cm[(i, j)], md.off(i, j) as f64);
				prop_assert_eq!(rm_t[(j, i)], md.off(i, j) as f64);
			}
		}
	}

	/// `permute_rows` takes row `perm[i]` of the source to row `i`, and the
	/// inverse array undoes it.
	#[test]
	fn prop_permute_rows_follows_the_forward_array(
		perm in (0usize..9).prop_flat_map(|n| Just((0..n).collect::<Vec<_>>()).prop_shuffle()),
		ncols in 0usize..5,
		row_major in any::<bool>(),
	) {
		let n = perm.len();
		let mut inv = vec![0usize; n];
		for (i, &p) in perm.iter().enumerate() {
			inv[p] = i;
		}
		let p = Perm::<usize>::new_checked(perm.clone().into_boxed_slice(), inv.clone().into_boxed_slice(), n);
		prop_assert_eq!(p.as_ref().arrays().0, &perm[..]);
		prop_assert_eq!(p.as_ref().inverse().arrays().0, &inv[..]);
		let src = Mat::<f64>::from_fn(n, ncols, |i, j| (i * 10 + j) as f64);
		let mut dst_t = Mat::<f64>::zeros(ncols, n);
		let mut dst_c = Mat::<f64>::zeros(n, ncols);
		let dst = if row_major { dst_t.as_mut().transpose_mut() } else { dst_c.as_mut() };
		permute_rows(dst, src.as_ref(), p.as_ref());
		let dst = if row_major { dst_t.transpose() } else { dst_c.as_ref() };
		for i in 0..n {
			for j in 0..ncols {
				prop_assert_eq!(dst[(i, j)], src[(perm[i], j)]);
			}
		}
		let mut back = Mat::<f64>::zeros(n, ncols);
		permute_rows(back.as_mut(), dst, p.as_ref().inverse());
		prop_assert!(back == src);
	}
}

/// Matrices with the same row count and different column counts are unequal
/// (colmat-rs panicked here).
#[test]
fn matrices_of_different_widths_are_unequal() {
	let a = Mat::<f64>::zeros(2, 3);
	let b = Mat::<f64>::zeros(2, 4);
	assert!(a != b);
	assert!(a.as_ref() != b.as_ref());
	assert!(Mat::<f64>::zeros(0, 3) != Mat::<f64>::zeros(0, 4));
	assert!(a == Mat::<f64>::zeros(2, 3));
}

#[test]
#[should_panic]
fn new_checked_rejects_a_non_inverse_pair() {
	let _ = Perm::<usize>::new_checked(
		vec![1, 0, 2].into_boxed_slice(),
		vec![0, 1, 2].into_boxed_slice(),
		3,
	);
}

/// Conjugation is a view: `conjugate()` reads the same memory as the
/// conjugate, and `to_owned` materialises it in the canonical type.
#[test]
fn conjugate_view_reads_the_conjugate() {
	let a =
		Mat::<c64>::from_fn(3, 2, |i, j| c64::new(i as f64, j as f64 + 1.0));
	let c = a.as_ref().conjugate();
	let owned: Mat<c64> = c.to_owned();
	let adj: Mat<c64> = a.as_ref().adjoint().to_owned();
	for j in 0..2 {
		for i in 0..3 {
			assert_eq!(owned[(i, j)], a[(i, j)].conj());
			assert_eq!(adj[(j, i)], a[(i, j)].conj());
		}
	}
}

/// The thresholds of the tiled copy, at the boundaries colmat-rs measured.
#[test]
fn tile_copy_pays_at_the_measured_boundaries() {
	assert!(tile_copy_pays::<f64>(512, 512, 512, 512));
	assert!(!tile_copy_pays::<f64>(511, 511, 511, 511));
	assert!(!tile_copy_pays::<f32>(724, 724, 724, 725));
	assert!(tile_copy_pays::<f32>(725, 725, 725, 725));
	// power-of-two strides on both sides, below the working set
	assert!(tile_copy_pays::<f64>(256, 256, 256, 256));
	// only the destination's
	assert!(!tile_copy_pays::<f64>(256, 256, 256, 260));
	// destination below 2048 bytes
	assert!(!tile_copy_pays::<f64>(128, 128, 128, 128));
}

/// The triangular copy takes the tiles above the threshold and writes only
/// the lower triangle.
#[cfg(feature = "std")]
#[test]
fn triangular_lower_copy_tiles_and_leaves_the_upper_triangle() {
	// Under Miri only the small shapes (the working-set cases alone take over
	// half an hour there): `(256, 8)` takes the tiles through the power-of-two
	// strides (2048-byte destination column stride, 64-byte source row stride).
	let cases: &[(usize, usize, bool)] = if cfg!(miri) {
		&[(256, 8, true), (64, 64, false)]
	} else {
		&[
			(512, 512, true),
			(520, 511, true),
			(256, 8, true),
			(64, 64, false),
			(17, 300, false),
		]
	};
	for &(m, n, tiled) in cases {
		let src_t = Mat::<f64>::from_fn(n, m, |j, i| (i * 1000 + j) as f64);
		let src = src_t.transpose();
		let mut dst = Mat::<f64>::from_fn(m, n, |_, _| -1.0);
		let before = crate::layout::tiled_copies();
		dst.copy_from_triangular_lower(src);
		assert_eq!(
			crate::layout::tiled_copies() - before,
			tiled as usize,
			"{m}x{n}"
		);
		for j in 0..n {
			for i in 0..m {
				let want = if i >= j { src[(i, j)] } else { -1.0 };
				assert_eq!(dst[(i, j)], want, "({i}, {j}) of {m}x{n}");
			}
		}
	}
}
