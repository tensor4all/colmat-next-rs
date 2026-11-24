//! Regression tests for the soundness defects inherited from colmat-rs (#14;
//! reported as tensor4all/colmat-rs#254, #255 and #256). Each one failed under
//! Miri before its fix; the scheduled Miri jobs run them.

use crate::internal_prelude::*;
use crate::perm::{Perm, PermRef};
use crate::test_support::{Strided, layouts};
use crate::utils::bound::{Array, Dim, One, Zero};
use crate::zip::Diag;
use std::assert;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

std::thread_local! {
	/// Ids of the `Tracked` values dropped on this thread, in order.
	static DROPPED: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
	/// Id whose next drop panics.
	static PANIC_ON_DROP: RefCell<Option<u32>> = const { RefCell::new(None) };
}

/// A value that logs its drop, and panics in it once if asked to.
#[derive(Debug)]
struct Tracked(u32);

impl Drop for Tracked {
	fn drop(&mut self) {
		DROPPED.with(|d| d.borrow_mut().push(self.0));
		let panic = PANIC_ON_DROP
			.with(|p| p.borrow_mut().take_if(|id| *id == self.0).is_some());
		if panic {
			panic!("drop of {}", self.0);
		}
	}
}

fn take_dropped() -> Vec<u32> {
	DROPPED.with(|d| core::mem::take(&mut *d.borrow_mut()))
}

/// Checks that every id in `created` was dropped exactly once and nothing
/// else was dropped.
#[track_caller]
fn assert_dropped_once(created: &[u32], context: &str) {
	let dropped = take_dropped();
	let mut count = BTreeMap::<u32, usize>::new();
	for id in &dropped {
		*count.entry(*id).or_default() += 1;
	}
	let mut want: Vec<u32> = created.to_vec();
	want.sort_unstable();
	let got: Vec<u32> = count.keys().copied().collect();
	assert!(
		count.values().all(|&c| c == 1),
		"{context}: drops {dropped:?}"
	);
	assert_eq!(got, want, "{context}: drops {dropped:?}");
}

/// Uses `m` again after a caught panic: every element it still owns is live
/// (created, not yet dropped, and distinct), and it can grow and shrink
/// again. The ids of the elements it creates are added to `created`.
#[track_caller]
fn use_again_after_panic(
	m: &mut Mat<Tracked>,
	created: &mut Vec<u32>,
	context: &str,
) {
	let dropped = DROPPED.with(|d| d.borrow().clone());
	let mut seen = Vec::new();
	for j in 0..m.ncols() {
		for i in 0..m.nrows() {
			let id = m[(i, j)].0;
			assert!(
				created.contains(&id),
				"{context}: ({i}, {j}) holds {id}, never created"
			);
			assert!(
				!dropped.contains(&id),
				"{context}: ({i}, {j}) holds {id}, already dropped"
			);
			seen.push(id);
		}
	}
	let n_seen = seen.len();
	seen.sort_unstable();
	seen.dedup();
	assert_eq!(seen.len(), n_seen, "{context}: an element is owned twice");

	let (m2, n2) = (m.nrows() + 1, m.ncols() + 2);
	m.resize_with(m2, n2, |i, j| {
		let id = 2000 + (i * 10 + j) as u32;
		created.push(id);
		Tracked(id)
	});
	assert_eq!((m.nrows(), m.ncols()), (m2, n2), "{context}");
	m.truncate(1, 1);
	assert_eq!((m.nrows(), m.ncols()), (1, 1), "{context}");
}

#[test]
fn resize_with_drops_each_element_once_when_f_panics() {
	// Growth in rows, in columns and in both, with `f` panicking at each call.
	let cases = [(2, 1, 4, 1), (2, 2, 2, 4), (2, 2, 4, 3), (0, 2, 3, 2)];
	for (m0, n0, m1, n1) in cases {
		let new_calls = m1 * n1 - m0 * n0;
		for panic_at in 0..new_calls {
			take_dropped();
			let mut created = Vec::new();
			let mut m = Mat::from_fn(m0, n0, |i, j| {
				let id = (i * 10 + j) as u32;
				created.push(id);
				Tracked(id)
			});
			let mut calls = 0;
			let result = catch_unwind(AssertUnwindSafe(|| {
				m.resize_with(m1, n1, |i, j| {
					if calls == panic_at {
						panic!("f at {panic_at}");
					}
					calls += 1;
					let id = 1000 + (i * 10 + j) as u32;
					created.push(id);
					Tracked(id)
				})
			}));
			assert!(result.is_err());
			let context =
				format!("{m0}x{n0} -> {m1}x{n1}, panic at call {panic_at}");
			use_again_after_panic(&mut m, &mut created, &context);
			drop(m);
			assert_dropped_once(&created, &context);
		}
	}
}

#[test]
fn truncate_drops_each_element_once_when_a_drop_panics() {
	// Columns removed, rows removed, and both.
	let cases = [(1, 3, 1, 1), (3, 1, 1, 1), (3, 3, 1, 2), (2, 2, 0, 0)];
	for (m0, n0, m1, n1) in cases {
		for victim_i in 0..m0 {
			for victim_j in 0..n0 {
				if victim_i < m1 && victim_j < n1 {
					continue;
				}
				take_dropped();
				let mut created = Vec::new();
				let mut m = Mat::from_fn(m0, n0, |i, j| {
					let id = (i * 10 + j) as u32;
					created.push(id);
					Tracked(id)
				});
				let victim = (victim_i * 10 + victim_j) as u32;
				PANIC_ON_DROP.with(|p| *p.borrow_mut() = Some(victim));
				let result =
					catch_unwind(AssertUnwindSafe(|| m.truncate(m1, n1)));
				assert!(result.is_err());
				assert!((m.nrows(), m.ncols()) <= (m0, n0));
				let context = format!(
					"{m0}x{n0} -> {m1}x{n1}, panic in drop of {victim}"
				);
				use_again_after_panic(&mut m, &mut created, &context);
				drop(m);
				assert_dropped_once(&created, &context);
			}
		}
	}
}

/// Runs `body` on an `m x n` view with strides `(rs, cs)` whose element
/// `(i, j)` holds `i * 16 + j`.
fn with_view(
	m: usize,
	n: usize,
	rs: isize,
	cs: isize,
	body: impl FnOnce(MatMut<'_, u32>),
) {
	let mut s =
		Strided::new(m, n, (rs, cs), u32::MAX, |i, j| (i * 16 + j) as u32);
	body(s.view_mut());
}

#[test]
fn triangular_iterations_visit_exactly_the_triangle_in_every_layout() {
	let max = if cfg!(miri) { 4 } else { 6 };
	for m in 0..=max {
		for n in 0..=max {
			for (rs, cs) in layouts(m, n) {
				for upper in [false, true] {
					for diag in [Diag::Include, Diag::Skip] {
						let strict = matches!(diag, Diag::Skip);
						let in_triangle =
							|i: usize, j: usize| match (upper, strict) {
								(false, false) => i >= j,
								(false, true) => i > j,
								(true, false) => i <= j,
								(true, true) => i < j,
							};
						let mut want: Vec<u32> = Vec::new();
						for i in 0..m {
							for j in 0..n {
								if in_triangle(i, j) {
									want.push((i * 16 + j) as u32);
								}
							}
						}
						want.sort_unstable();
						let context = format!(
							"{m}x{n} strides ({rs}, {cs}) upper {upper} strict {strict}"
						);

						let mut got = Vec::new();
						with_view(m, n, rs, cs, |a| {
							let mut f = |i: usize, j: usize, x: &mut u32| {
								assert_eq!(
									*x,
									(i * 16 + j) as u32,
									"{context}"
								);
								got.push(*x);
							};
							if upper {
								zip!(a).for_each_triangular_upper_with_index(
									diag,
									|i, j, unzip!(x)| f(i, j, x),
								);
							} else {
								zip!(a).for_each_triangular_lower_with_index(
									diag,
									|i, j, unzip!(x)| f(i, j, x),
								);
							}
						});
						got.sort_unstable();
						assert_eq!(got, want, "with index, {context}");

						let mut got = Vec::new();
						with_view(m, n, rs, cs, |a| {
							if upper {
								zip!(a).for_each_triangular_upper(
									diag,
									|unzip!(x)| got.push(*x),
								);
							} else {
								zip!(a).for_each_triangular_lower(
									diag,
									|unzip!(x)| got.push(*x),
								);
							}
						});
						got.sort_unstable();
						assert_eq!(got, want, "without index, {context}");
					}
				}
			}
		}
	}
}

#[test]
fn zip_eq_new_unchecked_with_matching_shapes_visits_every_pair() {
	let mut a = Mat::<u32>::from_fn(3, 2, |i, j| (i * 16 + j) as u32);
	let mut b = Mat::<u32>::from_fn(3, 2, |_, _| 0);
	// SAFETY: both operands are `3 x 2`.
	let z =
		unsafe { crate::zip::ZipEq::new_unchecked(zip!(&mut a), zip!(&mut b)) };
	z.for_each(
		|crate::zip::Zip(crate::zip::Last(x), crate::zip::Last(y))| *y = *x + 1,
	);
	assert!(b == Mat::from_fn(3, 2, |i, j| (i * 16 + j) as u32 + 1));
}

#[test]
fn permutations_of_shape_one_panic_instead_of_casting_zero_sized_indices() {
	let z = [Zero];
	let r = catch_unwind(|| {
		PermRef::<u32, One>::new_checked(&z, &z, One);
	});
	assert!(r.is_err());

	let r = catch_unwind(|| {
		let p = Perm::<u32>::new_checked(Box::new([0u32]), Box::new([0u32]), 1);
		drop(p.into_shape(One));
	});
	assert!(r.is_err());

	let r = catch_unwind(|| {
		let p = Perm::<u32>::new_checked(Box::new([0u32]), Box::new([0u32]), 1);
		let _ = p.as_ref().as_shape(One);
	});
	assert!(r.is_err());

	// Shapes whose index is stored as `I` still convert.
	let p =
		Perm::<u32>::new_checked(Box::new([1u32, 0]), Box::new([1u32, 0]), 2);
	Dim::with(2, |n| {
		let q = p.as_ref().as_shape(n);
		assert_eq!(q.as_shape(2usize).arrays().0, &[1u32, 0][..]);
	});
	assert_eq!(p.into_shape(2usize).as_ref().arrays().0, &[1u32, 0][..]);
}

#[test]
fn array_ranges_are_checked_and_as_mut_borrows_the_array() {
	let mut v = vec![1.0f64, 2.0, 3.0];
	Dim::with(v.len(), |n| {
		let a = Array::from_mut(&mut v, n);
		let r = catch_unwind(AssertUnwindSafe(|| {
			a[n.idx_inc(2)..n.idx_inc(1)].len()
		}));
		assert!(r.is_err());
		let r = catch_unwind(AssertUnwindSafe(|| {
			core::ops::IndexMut::index_mut(&mut *a, n.idx_inc(2)..n.idx_inc(1))
				.len()
		}));
		assert!(r.is_err());
		assert_eq!(&a[n.idx_inc(1)..n.idx_inc(3)], &[2.0, 3.0][..]);
		a.as_mut()[0] = 4.0;
		a.as_mut()[1] = 5.0;
		assert_eq!(a.as_ref(), &[4.0, 5.0, 3.0][..]);
	});
}
