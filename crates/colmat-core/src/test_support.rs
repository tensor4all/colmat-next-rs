//! Strided test buffers shared by the layout-exhaustive tests (#14, #16).
//!
//! A [`Strided`] owns a buffer and describes an `m x n` view of it with
//! strides `(rs, cs)`; the slots that the view does not cover hold a padding
//! value, so a test can check that an operation wrote nowhere else.

use crate::internal_prelude::*;

/// The strides of the 16 layouts the exhaustive tests visit: column-major
/// and row-major, with unit or doubled inner stride, each axis forward or
/// reversed. For every one, distinct `(i, j)` map to distinct offsets.
pub(crate) fn layouts(m: usize, n: usize) -> Vec<(isize, isize)> {
	let mut out = Vec::new();
	for inner in [1isize, 2] {
		for s1 in [1isize, -1] {
			for s2 in [1isize, -1] {
				out.push((s1 * inner, s2 * inner * m.max(1) as isize));
				out.push((s2 * inner * n.max(1) as isize, s1 * inner));
			}
		}
	}
	out
}

/// An `m x n` view over an owned buffer, with strides `(rs, cs)`.
pub(crate) struct Strided<T> {
	buf: Vec<T>,
	base: usize,
	pub(crate) m: usize,
	pub(crate) n: usize,
	rs: isize,
	cs: isize,
	pad: T,
}

impl<T: Copy + PartialEq + core::fmt::Debug> Strided<T> {
	/// A buffer just large enough for the view, every slot `pad` except the
	/// view's elements, which are `fill(i, j)`.
	pub(crate) fn new(
		m: usize,
		n: usize,
		(rs, cs): (isize, isize),
		pad: T,
		fill: impl Fn(usize, usize) -> T,
	) -> Self {
		let span = |len: usize, s: isize| (len.max(1) - 1) as isize * s;
		let lo = span(m, rs).min(0) + span(n, cs).min(0);
		let hi = span(m, rs).max(0) + span(n, cs).max(0);
		let mut this = Self {
			buf: vec![pad; (hi - lo + 1) as usize],
			base: (-lo) as usize,
			m,
			n,
			rs,
			cs,
			pad,
		};
		for j in 0..n {
			for i in 0..m {
				let k = this.offset(i, j);
				this.buf[k] = fill(i, j);
			}
		}
		this
	}

	fn offset(&self, i: usize, j: usize) -> usize {
		(self.base as isize + i as isize * self.rs + j as isize * self.cs)
			as usize
	}

	/// The element `(i, j)`, read from the buffer, not through a view.
	pub(crate) fn get(&self, i: usize, j: usize) -> T {
		self.buf[self.offset(i, j)]
	}

	pub(crate) fn view(&self) -> MatRef<'_, T> {
		// SAFETY: the buffer holds every offset `base + i * rs + j * cs` for
		// `i < m`, `j < n` (`new` sized it from their extremes), its pointer
		// is non-null, and the view borrows `self` for its lifetime.
		unsafe {
			MatRef::from_raw_parts(
				self.buf.as_ptr().add(self.base),
				self.m,
				self.n,
				self.rs,
				self.cs,
			)
		}
	}

	pub(crate) fn view_mut(&mut self) -> MatMut<'_, T> {
		// SAFETY: as in `view`; in addition, distinct `(i, j)` map to
		// distinct offsets (every stride pair of `layouts`, and every
		// padded pair the tests use, has this property), and the view
		// borrows `self` mutably.
		unsafe {
			MatMut::from_raw_parts_mut(
				self.buf.as_mut_ptr().add(self.base),
				self.m,
				self.n,
				self.rs,
				self.cs,
			)
		}
	}

	/// Panics if a slot outside the view no longer holds the padding value.
	#[track_caller]
	pub(crate) fn assert_padding_untouched(&self, context: &str) {
		let mut inside = vec![false; self.buf.len()];
		for j in 0..self.n {
			for i in 0..self.m {
				inside[self.offset(i, j)] = true;
			}
		}
		for (k, slot) in self.buf.iter().enumerate() {
			if !inside[k] {
				assert_eq!(
					*slot, self.pad,
					"{context}: slot {k} outside the view was written"
				);
			}
		}
	}
}
