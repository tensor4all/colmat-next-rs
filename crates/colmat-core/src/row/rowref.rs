use super::*;
use crate::utils::bound::{Array, Dim, Partition};
use crate::{ContiguousFwd, Idx, IdxInc};
use equator::{assert, debug_assert};
/// see [`super::RowRef`]
pub struct Ref<'a, T, Cols = usize, CStride = isize> {
	pub(crate) trans: ColRef<'a, T, Cols, CStride>,
}
impl<T, Rows: Copy, CStride: Copy> Copy for Ref<'_, T, Rows, CStride> {}
impl<T, Rows: Copy, CStride: Copy> Clone for Ref<'_, T, Rows, CStride> {
	#[inline]
	fn clone(&self) -> Self {
		*self
	}
}
impl<'short, T, Rows: Copy, CStride: Copy> Reborrow<'short>
	for Ref<'_, T, Rows, CStride>
{
	type Target = Ref<'short, T, Rows, CStride>;

	#[inline]
	fn rb(&'short self) -> Self::Target {
		*self
	}
}
impl<'short, T, Rows: Copy, CStride: Copy> ReborrowMut<'short>
	for Ref<'_, T, Rows, CStride>
{
	type Target = Ref<'short, T, Rows, CStride>;

	#[inline]
	fn rb_mut(&'short mut self) -> Self::Target {
		*self
	}
}
impl<'a, T, Rows: Copy, CStride: Copy> IntoConst for Ref<'a, T, Rows, CStride> {
	type Target = Ref<'a, T, Rows, CStride>;

	#[inline]
	fn into_const(self) -> Self::Target {
		self
	}
}
// SAFETY: a `Ref` is a shared view, like `&'a [T]`: through it, any thread
// can only obtain `&T` to the elements, which is sound to share across
// threads when `T: Sync`; the shape and stride are plain values.
unsafe impl<T: Sync, Rows: Sync, CStride: Sync> Sync
	for Ref<'_, T, Rows, CStride>
{
}
// SAFETY: as for `Sync` above: sending a shared view to another thread only
// gives that thread `&T` access, which is sound when `T: Sync` (as for
// `&T: Send`).
unsafe impl<T: Sync, Rows: Send, CStride: Send> Send
	for Ref<'_, T, Rows, CStride>
{
}
impl<'a, T> RowRef<'a, T> {
	/// creates a row view over the given element
	#[inline]
	pub fn from_ref(value: &'a T) -> Self {
		// SAFETY: `value` is valid for reads for `'a`; the only in-bounds index
		// is `0`, at offset `0`, so the view invariant holds.
		unsafe { RowRef::from_raw_parts(value as *const T, 1, 1) }
	}

	/// creates a `RowRef` from slice views over the row vector data, the result
	/// has the same number of columns as the length of the input slice
	#[inline]
	pub fn from_slice(slice: &'a [T]) -> Self {
		let len = slice.len();
		// SAFETY: with stride `1`, column `j < len` is `slice[j]`, which is
		// valid for reads for `'a`, so the view invariant holds.
		unsafe { Self::from_raw_parts(slice.as_ptr(), len, 1) }
	}
}
impl<'a, T, Cols: Shape, CStride: Stride> RowRef<'a, T, Cols, CStride> {
	/// creates a `RowRef` from pointers to the column vector data, number of
	/// rows, and row stride
	///
	/// # Safety
	/// this function has the same safety requirements as
	/// [`MatRef::from_raw_parts(ptr, 1, ncols, 0, col_stride)`]
	#[inline(always)]
	#[track_caller]
	pub const unsafe fn from_raw_parts(
		ptr: *const T,
		ncols: Cols,
		col_stride: CStride,
	) -> Self {
		// SAFETY: the caller upholds this function's contract (# Safety
		// above), which is that of `ColRef::from_raw_parts` with the same
		// pointer, length and stride: a row is stored as its transpose.
		unsafe {
			Self(Ref {
				trans: ColRef::from_raw_parts(ptr, ncols, col_stride),
			})
		}
	}

	/// returns a pointer to the row data
	#[inline(always)]
	pub fn as_ptr(&self) -> *const T {
		self.trans.as_ptr()
	}

	/// returns the number of rows of the row (always 1)
	#[inline(always)]
	pub fn nrows(&self) -> usize {
		1
	}

	/// returns the number of columns of the row
	#[inline(always)]
	pub fn ncols(&self) -> Cols {
		self.trans.nrows()
	}

	/// returns the number of rows and columns of the row
	#[inline(always)]
	pub fn shape(&self) -> (usize, Cols) {
		(self.nrows(), self.ncols())
	}

	/// returns the column stride of the row
	#[inline(always)]
	pub fn col_stride(&self) -> CStride {
		self.trans.row_stride()
	}

	/// returns a newly allocated row holding `f` applied to each element of
	/// `self`
	#[inline]
	pub fn map<U>(&self, f: impl FnMut(&T) -> U) -> Row<U, Cols> {
		let mut f = f;
		zip!(self).map(|unzip!(x)| f(x))
	}

	/// calls `f` on each element of `self`
	#[inline]
	pub fn for_each(&self, f: impl FnMut(&T)) {
		let mut f = f;
		zip!(self).for_each(|unzip!(x)| f(x))
	}

	/// returns a raw pointer to the element at the given index
	#[inline(always)]
	pub fn ptr_at(&self, col: IdxInc<Cols>) -> *const T {
		self.trans.ptr_at(col)
	}

	/// returns a raw pointer to the element at the given index, assuming the
	/// provided index is within the row bounds
	///
	/// # Safety
	/// the behavior is undefined if any of the following conditions are
	/// violated:
	/// * `col < self.ncols()`
	#[inline(always)]
	#[track_caller]
	pub unsafe fn ptr_inbounds_at(&self, col: Idx<Cols>) -> *const T {
		// SAFETY: the caller upholds this function's contract (# Safety
		// above): `col < self.ncols()`, which is `self.trans.nrows()`.
		unsafe {
			debug_assert!(all(col < self.ncols()));
			self.trans.ptr_inbounds_at(col)
		}
	}

	/// splits the row vertically at the given column into two parts and returns
	/// an array of each subrow, in the following order:
	/// * left
	/// * right
	///
	/// # panics
	/// the function panics if the following condition is violated:
	/// * `col <= self.ncols()`
	#[inline]
	#[track_caller]
	pub fn split_at_col(
		self,
		col: IdxInc<Cols>,
	) -> (RowRef<'a, T, usize, CStride>, RowRef<'a, T, usize, CStride>) {
		assert!(all(col <= self.ncols()));
		let rs = self.col_stride();
		let top = self.as_ptr();
		let bot = self.ptr_at(col);
		// SAFETY: `col <= self.ncols()` (checked above), so the left view
		// covers columns `0..col` and the right view, starting at `bot`,
		// columns `col..ncols` of `self`; both inherit the view invariant
		// from `self` for `'a`.
		unsafe {
			(
				RowRef::from_raw_parts(top, col.unbound(), rs),
				RowRef::from_raw_parts(
					bot,
					self.ncols().unbound() - col.unbound(),
					rs,
				),
			)
		}
	}

	/// returns a view over the transpose of `self`
	#[inline(always)]
	pub fn transpose(self) -> ColRef<'a, T, Cols, CStride> {
		self.trans
	}

	/// returns a view over the conjugate of `self`
	#[inline(always)]
	pub fn conjugate(self) -> RowRef<'a, T::Conj, Cols, CStride>
	where
		T: Conjugate,
	{
		RowRef {
			0: Ref {
				trans: self.trans.conjugate(),
			},
		}
	}

	/// returns an unconjugated view over `self`
	#[inline(always)]
	pub fn canonical(self) -> RowRef<'a, T::Canonical, Cols, CStride>
	where
		T: Conjugate,
	{
		RowRef {
			0: Ref {
				trans: self.trans.canonical(),
			},
		}
	}

	/// returns a view over the conjugate transpose of `self`
	#[inline(always)]
	pub fn adjoint(self) -> ColRef<'a, T::Conj, Cols, CStride>
	where
		T: Conjugate,
	{
		self.conjugate().transpose()
	}

	#[inline(always)]
	#[track_caller]
	pub(crate) fn at(self, col: Idx<Cols>) -> &'a T {
		assert!(all(col < self.ncols()));
		// SAFETY: `col < self.ncols()` (checked above).
		unsafe { self.at_unchecked(col) }
	}

	/// # Safety
	/// `col < self.ncols()`.
	#[inline(always)]
	#[track_caller]
	pub(crate) unsafe fn at_unchecked(self, col: Idx<Cols>) -> &'a T {
		// SAFETY: the caller guarantees `col < self.ncols()` (# Safety
		// above), so by the view invariant the pointer is valid for reads
		// for `'a`.
		unsafe { &*self.ptr_inbounds_at(col) }
	}

	/// returns a reference to the element at the given index, or a subrow if
	/// `col` is a range, with bound checks
	///
	/// # panics
	/// the function panics if any of the following conditions are violated:
	/// * `col` must be contained in `[0, self.ncols())`
	#[track_caller]
	#[inline(always)]
	pub fn get<ColRange>(
		self,
		col: ColRange,
	) -> <RowRef<'a, T, Cols, CStride> as RowIndex<ColRange>>::Target
	where
		RowRef<'a, T, Cols, CStride>: RowIndex<ColRange>,
	{
		<RowRef<'a, T, Cols, CStride> as RowIndex<ColRange>>::get(self, col)
	}

	/// returns a reference to the element at the given index, or a subrow if
	/// `col` is a range, without bound checks
	///
	/// # Safety
	/// the behavior is undefined if any of the following conditions are
	/// violated:
	/// * `col` must be contained in `[0, self.ncols())`
	#[track_caller]
	#[inline(always)]
	pub unsafe fn get_unchecked<ColRange>(
		self,
		col: ColRange,
	) -> <RowRef<'a, T, Cols, CStride> as RowIndex<ColRange>>::Target
	where
		RowRef<'a, T, Cols, CStride>: RowIndex<ColRange>,
	{
		// SAFETY: the caller upholds this function's contract (# Safety
		// above), which is that of `RowIndex::get_unchecked`.
		unsafe {
			<RowRef<'a, T, Cols, CStride> as RowIndex<ColRange>>::get_unchecked(
				self, col,
			)
		}
	}

	/// returns a view over the `self`, with the columns in reversed order
	#[inline]
	pub fn reverse_cols(self) -> RowRef<'a, T, Cols, CStride::Rev> {
		RowRef {
			0: Ref {
				trans: self.trans.reverse_rows(),
			},
		}
	}

	/// returns a view over the subrow starting at column `col_start`, and with
	/// number of columns `ncols`
	///
	/// # panics
	/// the function panics if any of the following conditions are violated:
	/// * `col_start <= self.ncols()`
	/// * `ncols <= self.ncols() - col_start`
	#[inline]
	pub fn subcols<V: Shape>(
		self,
		col_start: IdxInc<Cols>,
		ncols: V,
	) -> RowRef<'a, T, V, CStride> {
		assert!(all(col_start <= self.ncols()));
		{
			let ncols = ncols.unbound();
			let full_ncols = self.ncols().unbound();
			let col_start = col_start.unbound();
			assert!(all(ncols <= full_ncols - col_start));
		}
		let cs = self.col_stride();
		// SAFETY: `col_start + ncols <= self.ncols()` (checked above), so the
		// view covers columns `col_start..col_start + ncols` of `self` and
		// inherits the view invariant from it for `'a`.
		unsafe { RowRef::from_raw_parts(self.ptr_at(col_start), ncols, cs) }
	}

	/// returns the input row with the given column shape after checking that it
	/// matches the current column shape
	#[inline]
	#[track_caller]
	pub fn as_col_shape<V: Shape>(self, ncols: V) -> RowRef<'a, T, V, CStride> {
		assert!(all(self.ncols().unbound() == ncols.unbound()));
		// SAFETY: same pointer, stride and (checked above) number of columns
		// as `self`, so the view invariant carries over.
		unsafe {
			RowRef::from_raw_parts(self.as_ptr(), ncols, self.col_stride())
		}
	}

	/// returns the input row with dynamic column shape
	#[inline]
	pub fn as_dyn_cols(self) -> RowRef<'a, T, usize, CStride> {
		// SAFETY: same pointer, number of columns and stride as `self`, so
		// the view invariant carries over.
		unsafe {
			RowRef::from_raw_parts(
				self.as_ptr(),
				self.ncols().unbound(),
				self.col_stride(),
			)
		}
	}

	/// returns the input row with dynamic stride
	#[inline]
	pub fn as_dyn_stride(self) -> RowRef<'a, T, Cols, isize> {
		// SAFETY: same pointer, number of columns and stride value as `self`,
		// so the view invariant carries over.
		unsafe {
			RowRef::from_raw_parts(
				self.as_ptr(),
				self.ncols(),
				self.col_stride().element_stride(),
			)
		}
	}

	/// returns an iterator over the elements of the row
	#[inline]
	pub fn iter(
		self,
	) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = &'a T>
	where
		Cols: 'a,
	{
		self.trans.iter()
	}

	/// returns a view over the row with a static column stride equal to `+1`,
	/// or `None` otherwise
	#[inline]
	pub fn try_as_row_major(
		self,
	) -> Option<RowRef<'a, T, Cols, ContiguousFwd>> {
		if self.col_stride().element_stride() == 1 {
			// SAFETY: the stride is `1` (checked above), which is what
			// `ContiguousFwd` denotes, so the view covers the same elements as
			// `self` and the view invariant carries over.
			Some(unsafe {
				RowRef::from_raw_parts(
					self.as_ptr(),
					self.ncols(),
					ContiguousFwd,
				)
			})
		} else {
			None
		}
	}

	/// returns a mutable view over the same elements as `self`
	///
	/// # Safety
	/// the returned view must satisfy the view invariant for mutable views:
	/// for `'a`, its elements must be valid for writes, and no other
	/// reference to them (including `self` and its copies) may be used while
	/// it is live, typically because `self` was obtained from a `RowMut` that
	/// is not used again
	#[inline(always)]
	#[doc(hidden)]
	pub unsafe fn const_cast(self) -> RowMut<'a, T, Cols, CStride> {
		// SAFETY: the caller upholds this function's contract (# Safety
		// above), which is that of `ColRef::const_cast` for `self.trans`.
		unsafe {
			RowMut {
				0: Mut {
					trans: self.trans.const_cast(),
				},
			}
		}
	}

	/// returns a matrix view over `self`
	#[inline]
	pub fn as_mat(self) -> MatRef<'a, T, usize, Cols, isize, CStride> {
		self.transpose().as_mat().transpose()
	}

	/// interprets the row as a diagonal matrix
	#[inline]
	pub fn as_diagonal(self) -> DiagRef<'a, T, Cols, CStride> {
		DiagRef {
			0: crate::diag::Ref { inner: self.trans },
		}
	}

	#[inline]
	pub(crate) fn __at(self, i: Idx<Cols>) -> &'a T {
		self.at(i)
	}
}
impl<
	T,
	Cols: Shape,
	CStride: Stride,
	Inner: for<'short> Reborrow<'short, Target = Ref<'short, T, Cols, CStride>>,
> generic::Row<Inner>
{
	/// returns a view over `self`
	#[inline]
	pub fn as_ref(&self) -> RowRef<'_, T, Cols, CStride> {
		self.rb()
	}

	/// returns `true` if all of the elements of `self` are finite.
	/// otherwise returns `false`.
	#[inline]
	pub fn is_all_finite(&self) -> bool
	where
		T: Conjugate,
	{
		self.rb().transpose().is_all_finite()
	}

	/// returns `true` if any of the elements of `self` is `NaN`.
	/// otherwise returns `false`.
	#[inline]
	pub fn has_nan(&self) -> bool
	where
		T: Conjugate,
	{
		self.rb().transpose().has_nan()
	}

	/// returns a newly allocated row holding the cloned values of `self`
	#[inline]
	pub fn cloned(&self) -> Row<T, Cols>
	where
		T: Clone,
	{
		self.rb().transpose().cloned().into_transpose()
	}

	/// returns a newly allocated row holding the (possibly conjugated) values
	/// of `self`
	#[inline]
	pub fn to_owned(&self) -> Row<T::Canonical, Cols>
	where
		T: Conjugate,
	{
		self.rb().transpose().to_owned().into_transpose()
	}
}
impl<'a, T, Rows: Shape> RowRef<'a, T, Rows, ContiguousFwd> {
	/// returns a reference over the elements as a slice
	#[inline]
	pub fn as_slice(self) -> &'a [T] {
		self.transpose().as_slice()
	}
}
impl<'a, 'ROWS, T> RowRef<'a, T, Dim<'ROWS>, ContiguousFwd> {
	/// returns a reference over the elements as a lifetime-bound slice
	#[inline]
	pub fn as_array(self) -> &'a Array<'ROWS, T> {
		self.transpose().as_array()
	}
}
impl<'COLS, 'a, T, CStride: Stride> RowRef<'a, T, Dim<'COLS>, CStride> {
	#[doc(hidden)]
	#[inline]
	pub fn split_cols_with<'LEFT, 'RIGHT>(
		self,
		col: Partition<'LEFT, 'RIGHT, 'COLS>,
	) -> (
		RowRef<'a, T, Dim<'LEFT>, CStride>,
		RowRef<'a, T, Dim<'RIGHT>, CStride>,
	) {
		let (a, b) = self.split_at_col(col.midpoint());
		(a.as_col_shape(col.head), b.as_col_shape(col.tail))
	}
}
impl<T: core::fmt::Debug, Cols: Shape, CStride: Stride> core::fmt::Debug
	for Ref<'_, T, Cols, CStride>
{
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		fn imp<T: core::fmt::Debug>(
			f: &mut core::fmt::Formatter<'_>,
			this: RowRef<'_, T, Dim<'_>>,
		) -> core::fmt::Result {
			f.debug_list()
				.entries(this.ncols().indices().map(|j| this.at(j)))
				.finish()
		}
		let this = generic::Row::from_inner_ref(self);
		with_dim!(N, this.ncols().unbound());
		imp(f, this.as_col_shape(N).as_dyn_stride())
	}
}
impl<'a, T> RowRef<'a, T, usize, isize>
where
	T: RealField,
{
	/// Returns the maximum element in the row, or `None` if the row is empty
	pub(crate) fn internal_max(self) -> Option<T> {
		if self.nrows().unbound() == 0 || self.ncols() == 0 {
			return None;
		}
		let mut max_val = self.get(0);
		self.iter().for_each(|val| {
			if val > max_val {
				max_val = val;
			}
		});
		Some((*max_val).clone())
	}

	/// Returns the minimum element in the row, or `None` if the row is empty
	pub(crate) fn internal_min(self) -> Option<T> {
		if self.nrows().unbound() == 0 || self.ncols() == 0 {
			return None;
		}
		let mut min_val = self.get(0);
		self.iter().for_each(|val| {
			if val < min_val {
				min_val = val;
			}
		});
		Some((*min_val).clone())
	}
}
impl<'a, T, Cols: Shape, CStride: Stride> RowRef<'a, T, Cols, CStride>
where
	T: RealField,
{
	/// Returns the maximum element in the row, or `None` if the row is empty
	pub fn max(&self) -> Option<T> {
		self.as_dyn_cols().as_dyn_stride().internal_max()
	}

	/// Returns the minimum element in the row, or `None` if the row is empty
	pub fn min(&self) -> Option<T> {
		self.as_dyn_cols().as_dyn_stride().internal_min()
	}
}
#[cfg(test)]
mod tests {
	use crate::Row;
	#[test]
	fn test_row_min() {
		let row: Row<f64> = Row::from_fn(5, |x| (x + 1) as f64);
		let rowref = row.as_ref();
		assert_eq!(rowref.min(), Some(1.0));
		let empty: Row<f64> = Row::from_fn(0, |_| 0.0);
		let emptyref = empty.as_ref();
		assert_eq!(emptyref.min(), None);
	}
	#[test]
	fn test_row_max() {
		let row: Row<f64> = Row::from_fn(5, |x| (x + 1) as f64);
		let rowref = row.as_ref();
		assert_eq!(rowref.max(), Some(5.0));
		let empty: Row<f64> = Row::from_fn(0, |_| 0.0);
		let emptyref = empty.as_ref();
		assert_eq!(emptyref.max(), None);
	}
}
