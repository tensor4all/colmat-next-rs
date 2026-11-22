//! Layout-exhaustive tests of the view operations (#16).
//!
//! Each test runs every shape up to a small size against every layout from
//! [`layouts`] and compares with the exact expected result: the element read
//! or written at each position, and the padding around the view left as it
//! was. The randomized tests in `tests.rs` sample the same space; these
//! enumerate it, the form that found the column skipped by the upper-triangle
//! `ReverseRows` arm (#14). Under Miri the sizes and the layout set are
//! smaller.

use crate::internal_prelude::*;
use crate::test_support::{Strided, layouts};
use crate::zip::Diag;
use std::assert;

/// The value of element `(i, j)` in every source view below.
fn code(i: usize, j: usize) -> f64 {
	(i * 64 + j) as f64
}

const PAD: f64 = -1.0;

/// The layouts a test visits: all of them, or under Miri one per family
/// (column-major and row-major, forward and fully reversed).
fn test_layouts(m: usize, n: usize) -> Vec<(isize, isize)> {
	let all = layouts(m, n);
	if cfg!(miri) {
		all.into_iter()
			.filter(|&(rs, cs)| rs.abs() == 1 || cs.abs() == 1)
			.step_by(2)
			.collect()
	} else {
		all
	}
}

#[track_caller]
fn assert_reads(
	v: MatRef<'_, f64>,
	shape: (usize, usize),
	want: impl Fn(usize, usize) -> f64,
	context: &str,
) {
	assert_eq!(v.shape(), shape, "{context}");
	for j in 0..shape.1 {
		for i in 0..shape.0 {
			assert_eq!(v[(i, j)], want(i, j), "({i}, {j}), {context}");
		}
	}
}

#[test]
fn views_read_the_offset_model_in_every_layout() {
	let max = if cfg!(miri) { 3 } else { 5 };
	for m in 0..=max {
		for n in 0..=max {
			for l in test_layouts(m, n) {
				let s = Strided::new(m, n, l, PAD, code);
				let a = s.view();
				let cx = format!("{m}x{n} strides {l:?}");
				assert_reads(a, (m, n), code, &cx);
				assert_reads(
					a.transpose(),
					(n, m),
					|i, j| code(j, i),
					&format!("transpose, {cx}"),
				);
				assert_reads(
					a.reverse_rows(),
					(m, n),
					|i, j| code(m - 1 - i, j),
					&format!("reverse_rows, {cx}"),
				);
				assert_reads(
					a.reverse_cols(),
					(m, n),
					|i, j| code(i, n - 1 - j),
					&format!("reverse_cols, {cx}"),
				);

				let d = a.diagonal().column_vector();
				assert_eq!(d.nrows(), m.min(n), "diagonal, {cx}");
				for k in 0..m.min(n) {
					assert_eq!(d[k], code(k, k), "diagonal {k}, {cx}");
				}
				for i in 0..m {
					let r = a.row(i);
					assert_eq!(r.ncols(), n, "row {i}, {cx}");
					for j in 0..n {
						assert_eq!(r[j], code(i, j), "row {i} col {j}, {cx}");
					}
				}
				for j in 0..n {
					let c = a.col(j);
					assert_eq!(c.nrows(), m, "col {j}, {cx}");
					for i in 0..m {
						assert_eq!(c[i], code(i, j), "col {j} row {i}, {cx}");
					}
				}

				for i0 in 0..=m {
					for j0 in 0..=n {
						let cx = format!("at ({i0}, {j0}), {cx}");
						let (tl, tr, bl, br) = a.split_at(i0, j0);
						assert_reads(
							tl,
							(i0, j0),
							code,
							&format!("split top left {cx}"),
						);
						assert_reads(
							tr,
							(i0, n - j0),
							|i, j| code(i, j0 + j),
							&format!("split top right {cx}"),
						);
						assert_reads(
							bl,
							(m - i0, j0),
							|i, j| code(i0 + i, j),
							&format!("split bottom left {cx}"),
						);
						assert_reads(
							br,
							(m - i0, n - j0),
							|i, j| code(i0 + i, j0 + j),
							&format!("split bottom right {cx}"),
						);
						if j0 == 0 {
							let (top, bot) = a.split_at_row(i0);
							assert_reads(
								top,
								(i0, n),
								code,
								&format!("split_at_row top {cx}"),
							);
							assert_reads(
								bot,
								(m - i0, n),
								|i, j| code(i0 + i, j),
								&format!("split_at_row bottom {cx}"),
							);
						}
						if i0 == 0 {
							let (left, right) = a.split_at_col(j0);
							assert_reads(
								left,
								(m, j0),
								code,
								&format!("split_at_col left {cx}"),
							);
							assert_reads(
								right,
								(m, n - j0),
								|i, j| code(i, j0 + j),
								&format!("split_at_col right {cx}"),
							);
						}
						for r in 0..=m - i0 {
							for c in 0..=n - j0 {
								assert_reads(
									a.submatrix(i0, j0, r, c),
									(r, c),
									|i, j| code(i0 + i, j0 + j),
									&format!("submatrix {r}x{c} {cx}"),
								);
							}
						}
					}
				}
			}
		}
	}
}

/// Checks that element `(i, j)` of `s` is `want(i, j)` and that nothing
/// outside the view was written.
#[track_caller]
fn assert_buffer(
	s: &Strided<f64>,
	want: impl Fn(usize, usize) -> f64,
	context: &str,
) {
	for j in 0..s.n {
		for i in 0..s.m {
			assert_eq!(s.get(i, j), want(i, j), "({i}, {j}), {context}");
		}
	}
	s.assert_padding_untouched(context);
}

#[test]
fn mutable_splits_and_submatrices_write_each_element_once_in_every_layout() {
	let max = if cfg!(miri) { 3 } else { 5 };
	for m in 0..=max {
		for n in 0..=max {
			for l in test_layouts(m, n) {
				for i0 in 0..=m {
					for j0 in 0..=n {
						let cx =
							format!("{m}x{n} strides {l:?} at ({i0}, {j0})");

						// The four parts are alive together and written in
						// interleaved order, so an overlap or a stale pointer
						// shows up as a count other than 1 (or under Miri).
						let mut s = Strided::new(m, n, l, PAD, |_, _| 0.0);
						{
							let (mut tl, mut tr, mut bl, mut br) =
								s.view_mut().split_at_mut(i0, j0);
							for j in 0..n {
								for i in 0..m {
									match (i < i0, j < j0) {
										(true, true) => tl[(i, j)] += 1.0,
										(true, false) => tr[(i, j - j0)] += 1.0,
										(false, true) => bl[(i - i0, j)] += 1.0,
										(false, false) => {
											br[(i - i0, j - j0)] += 1.0
										},
									}
								}
							}
						}
						assert_buffer(
							&s,
							|_, _| 1.0,
							&format!("split_at_mut {cx}"),
						);

						if j0 == 0 {
							let mut s = Strided::new(m, n, l, PAD, |_, _| 0.0);
							{
								let (mut top, mut bot) =
									s.view_mut().split_at_row_mut(i0);
								for j in 0..n {
									for i in 0..m {
										if i < i0 {
											top[(i, j)] += 1.0;
										} else {
											bot[(i - i0, j)] += 1.0;
										}
									}
								}
							}
							assert_buffer(
								&s,
								|_, _| 1.0,
								&format!("split_at_row_mut {cx}"),
							);
						}
						if i0 == 0 {
							let mut s = Strided::new(m, n, l, PAD, |_, _| 0.0);
							{
								let (mut left, mut right) =
									s.view_mut().split_at_col_mut(j0);
								for j in 0..n {
									for i in 0..m {
										if j < j0 {
											left[(i, j)] += 1.0;
										} else {
											right[(i, j - j0)] += 1.0;
										}
									}
								}
							}
							assert_buffer(
								&s,
								|_, _| 1.0,
								&format!("split_at_col_mut {cx}"),
							);
						}

						for r in 0..=m - i0 {
							for c in 0..=n - j0 {
								let mut s =
									Strided::new(m, n, l, PAD, |_, _| 0.0);
								{
									let mut sub = s
										.view_mut()
										.submatrix_mut(i0, j0, r, c);
									for j in 0..c {
										for i in 0..r {
											sub[(i, j)] += 1.0;
										}
									}
								}
								let inside = |i: usize, j: usize| {
									(i0..i0 + r).contains(&i)
										&& (j0..j0 + c).contains(&j)
								};
								assert_buffer(
									&s,
									|i, j| if inside(i, j) { 1.0 } else { 0.0 },
									&format!("submatrix_mut {r}x{c} {cx}"),
								);
							}
						}
					}
				}
			}
		}
	}
}

#[test]
fn copies_match_elementwise_for_every_pair_of_layouts() {
	// Sizes on both sides of the 16-element tile.
	let sizes: &[usize] = if cfg!(miri) {
		&[0, 1, 17]
	} else {
		&[0, 1, 2, 15, 16, 17, 33]
	};
	let tiled_before = crate::layout::tiled_copies();
	for &m in sizes {
		for &n in sizes {
			let mut pairs = Vec::new();
			for dl in test_layouts(m, n) {
				for sl in test_layouts(m, n) {
					pairs.push((dl, sl));
				}
			}
			// A column-major destination and a row-major source whose strides
			// are 2048 bytes: `tile_copy_pays` holds at every size, so the
			// tiled triangular path runs.
			for s1 in [1isize, -1] {
				for s2 in [1isize, -1] {
					pairs.push(((1, s1 * 256), (s2 * 256, 1)));
				}
			}
			for (dl, sl) in pairs {
				let cx =
					format!("{m}x{n} dst strides {dl:?} src strides {sl:?}");
				let src = Strided::new(m, n, sl, PAD, code);

				let mut dst = Strided::new(m, n, dl, PAD, |_, _| -2.0);
				dst.view_mut().copy_from(src.view());
				assert_buffer(&dst, code, &format!("copy_from {cx}"));

				let mut dst = Strided::new(m, n, dl, PAD, |_, _| -2.0);
				dst.view_mut().copy_from_triangular_lower(src.view());
				assert_buffer(
					&dst,
					|i, j| if i >= j { code(i, j) } else { -2.0 },
					&format!("copy_from_triangular_lower {cx}"),
				);

				let mut dst = Strided::new(m, n, dl, PAD, |_, _| -2.0);
				crate::layout::copy_across_layouts(
					dst.view_mut(),
					src.view(),
					crate::Par::Seq,
				);
				assert_buffer(&dst, code, &format!("copy_across_layouts {cx}"));

				let owned = src.view().to_owned();
				assert_reads(
					owned.as_ref(),
					(m, n),
					code,
					&format!("to_owned {cx}"),
				);
			}
		}
	}
	assert!(
		crate::layout::tiled_copies() > tiled_before,
		"the tiled triangular path never ran"
	);
}

/// The parallel split of `copy_across_layouts` (#20): on teams of 2 to 4 the copy writes
/// exactly the serial result, in every pair of layouts, at sizes that give each participant one
/// or several tiles or none, and nothing outside the destination view. Not under Miri: the team
/// is threads of a pool.
#[cfg(all(feature = "threadteam", not(miri)))]
#[test]
fn parallel_copies_match_elementwise_for_every_pair_of_layouts() {
	let sizes: &[usize] = &[0, 1, 15, 16, 17, 33, 64, 70];
	for width in 2..=4 {
		crate::utils::team::with_team(width, |mut par| {
			assert!(par.degree() == width);
			for &m in sizes {
				for &n in sizes {
					for dl in layouts(m, n) {
						for sl in layouts(m, n) {
							let cx = format!(
								"width {width}, {m}x{n} dst strides {dl:?} src strides {sl:?}"
							);
							let src = Strided::new(m, n, sl, PAD, code);
							let mut dst =
								Strided::new(m, n, dl, PAD, |_, _| -2.0);
							crate::layout::copy_across_layouts(
								dst.view_mut(),
								src.view(),
								par.reborrow(),
							);
							assert_buffer(&dst, code, &cx);
						}
					}
				}
			}
		});
	}
}

#[test]
fn zip_of_two_views_pairs_the_same_positions_in_every_pair_of_layouts() {
	let max = if cfg!(miri) { 3 } else { 4 };
	for m in 0..=max {
		for n in 0..=max {
			for la in test_layouts(m, n) {
				for lb in test_layouts(m, n) {
					let cx = format!("{m}x{n} strides {la:?} and {lb:?}");
					let mut a = Strided::new(m, n, la, PAD, code);
					let mut b = Strided::new(m, n, lb, PAD, |_, _| 0.0);

					let mut count = 0;
					zip!(a.view_mut(), b.view_mut()).for_each_with_index(
						|i, j, unzip!(x, y)| {
							assert_eq!(*x, code(i, j), "({i}, {j}), {cx}");
							*y = *x + 0.5;
							count += 1;
						},
					);
					assert_eq!(count, m * n, "{cx}");
					assert_buffer(
						&b,
						|i, j| code(i, j) + 0.5,
						&format!("for_each_with_index {cx}"),
					);

					zip!(a.view(), b.view_mut())
						.for_each(|unzip!(x, y)| *y = 2.0 * *x);
					assert_buffer(
						&b,
						|i, j| 2.0 * code(i, j),
						&format!("for_each {cx}"),
					);
					a.assert_padding_untouched(&cx);

					for upper in [false, true] {
						for diag in [Diag::Include, Diag::Skip] {
							let strict = matches!(diag, Diag::Skip);
							let mut want = Vec::new();
							for j in 0..n {
								for i in 0..m {
									let inside = if upper {
										i < j || (!strict && i == j)
									} else {
										i > j || (!strict && i == j)
									};
									if inside {
										want.push((i, j));
									}
								}
							}
							let mut got = Vec::new();
							let f =
								|i: usize, j: usize, x: &f64, y: &mut f64| {
									assert_eq!(
										(*x, *y),
										(code(i, j), 2.0 * code(i, j)),
										"({i}, {j}), {cx}"
									);
									got.push((i, j));
								};
							let mut f = f;
							if upper {
								zip!(a.view(), b.view_mut())
									.for_each_triangular_upper_with_index(
										diag,
										|i, j, unzip!(x, y)| f(i, j, x, y),
									);
							} else {
								zip!(a.view(), b.view_mut())
									.for_each_triangular_lower_with_index(
										diag,
										|i, j, unzip!(x, y)| f(i, j, x, y),
									);
							}
							got.sort_unstable();
							want.sort_unstable();
							assert_eq!(
								got, want,
								"triangular upper {upper} strict {strict}, {cx}"
							);
						}
					}
				}
			}
		}
	}
}
