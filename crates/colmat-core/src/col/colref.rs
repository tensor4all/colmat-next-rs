use super::*;
use crate::utils::bound::{Array, Dim, Partition};
use crate::{ContiguousFwd, Idx, IdxInc};
use core::marker::PhantomData;
use core::ptr::NonNull;
use equator::assert;
use generativity::Guard;
/// see [`super::ColRef`]
pub struct Ref<'a, T, Rows = usize, RStride = isize> {
	pub(super) imp: ColView<T, Rows, RStride>,
	pub(super) __marker: PhantomData<&'a T>,
}
impl<T, Rows: Copy, RStride: Copy> Copy for Ref<'_, T, Rows, RStride> {}
impl<T, Rows: Copy, RStride: Copy> Clone for Ref<'_, T, Rows, RStride> {
	#[inline]
	fn clone(&self) -> Self {
		*self
	}
}
impl<'short, T, Rows: Copy, RStride: Copy> Reborrow<'short>
	for Ref<'_, T, Rows, RStride>
{
	type Target = Ref<'short, T, Rows, RStride>;

	#[inline]
	fn rb(&'short self) -> Self::Target {
		*self
	}
}
impl<'short, T, Rows: Copy, RStride: Copy> ReborrowMut<'short>
	for Ref<'_, T, Rows, RStride>
{
	type Target = Ref<'short, T, Rows, RStride>;

	#[inline]
	fn rb_mut(&'short mut self) -> Self::Target {
		*self
	}
}
impl<'a, T, Rows: Copy, RStride: Copy> IntoConst for Ref<'a, T, Rows, RStride> {
	type Target = Ref<'a, T, Rows, RStride>;

	#[inline]
	fn into_const(self) -> Self::Target {
		self
	}
}
// SAFETY: a `Ref` is a shared borrow of `T` values (like `&'a T`) plus plain
// `Rows`/`RStride` values; sharing it across threads only gives shared access
// to the `T` values, which is sound for `T: Sync`.
unsafe impl<T: Sync, Rows: Sync, RStride: Sync> Sync
	for Ref<'_, T, Rows, RStride>
{
}
// SAFETY: sending a `Ref` to another thread lets that thread read the `T`
// values through a shared borrow, exactly like sending `&'a T`, which is sound
// for `T: Sync`; `Rows` and `RStride` are moved by value and are `Send`.
unsafe impl<T: Sync, Rows: Send, RStride: Send> Send
	for Ref<'_, T, Rows, RStride>
{
}
impl<'a, T> ColRef<'a, T> {
	/// creates a column view over the given element
	#[inline]
	pub fn from_ref(value: &'a T) -> Self {
		// SAFETY: `value` is a reference, so the pointer is non-null, aligned
		// and valid for reads of one element for `'a`, with no writes by other
		// means; with one row, index 0 is the only in-bounds index.
		unsafe { ColRef::from_raw_parts(value as *const T, 1, 1) }
	}

	/// creates a `ColRef` from slice views over the column vector data, the
	/// result has the same number of rows as the length of the input slice
	#[inline]
	pub fn from_slice(slice: &'a [T]) -> Self {
		let len = slice.len();
		// SAFETY: a slice pointer is non-null and aligned (also when empty),
		// and its `len` contiguous elements (row stride 1) are valid for reads
		// for `'a`, with no writes by other means.
		unsafe { Self::from_raw_parts(slice.as_ptr(), len, 1) }
	}
}
impl<'a, T, Rows: Shape, RStride: Stride> ColRef<'a, T, Rows, RStride> {
	/// creates a `ColRef` from pointers to the column vector data, number of
	/// rows, and row stride
	///
	/// # Safety
	/// this function has the same safety requirements as
	/// [`MatRef::from_raw_parts(ptr, nrows, 1, row_stride, 0)`]; in
	/// particular, `ptr` must be non-null and properly aligned, even for an
	/// empty column
	#[inline(always)]
	#[track_caller]
	pub const unsafe fn from_raw_parts(
		ptr: *const T,
		nrows: Rows,
		row_stride: RStride,
	) -> Self {
		// SAFETY: the caller upholds this function's contract (# Safety above),
		// which includes `ptr` being non-null.
		unsafe {
			Self(Ref {
				imp: ColView {
					ptr: NonNull::new_unchecked(ptr as *mut T),
					nrows,
					row_stride,
				},
				__marker: PhantomData,
			})
		}
	}

	/// returns a pointer to the column data
	#[inline(always)]
	pub fn as_ptr(&self) -> *const T {
		self.imp.ptr.as_ptr() as *const T
	}

	/// returns the number of rows of the column
	#[inline(always)]
	pub fn nrows(&self) -> Rows {
		self.imp.nrows
	}

	/// returns the number of columns of the column (always `1`)
	#[inline(always)]
	pub fn ncols(&self) -> usize {
		1
	}

	/// returns the number of rows and columns of the column
	#[inline(always)]
	pub fn shape(&self) -> (Rows, usize) {
		(self.nrows(), self.ncols())
	}

	/// returns the row stride of the column, specified in number of elements,
	/// not in bytes
	#[inline(always)]
	pub fn row_stride(&self) -> RStride {
		self.imp.row_stride
	}

	/// returns a newly allocated column holding `f` applied to each element
	/// of `self`
	#[inline]
	pub fn map<U>(&self, f: impl FnMut(&T) -> U) -> Col<U, Rows> {
		let mut f = f;
		zip!(self).map(|unzip!(x)| f(x))
	}

	/// calls `f` on each element of `self`, in order
	#[inline]
	pub fn for_each(&self, f: impl FnMut(&T)) {
		let mut f = f;
		zip!(self).for_each(|unzip!(x)| f(x))
	}

	/// returns a raw pointer to the element at the given index
	#[inline(always)]
	pub fn ptr_at(&self, row: IdxInc<Rows>) -> *const T {
		let ptr = self.as_ptr();
		if row >= self.nrows() {
			ptr
		} else {
			ptr.wrapping_offset(
				row.unbound() as isize * self.row_stride().element_stride(),
			)
		}
	}

	/// returns a raw pointer to the element at the given index, assuming the
	/// provided index is within the column bounds
	///
	/// # Safety
	/// the behavior is undefined if any of the following conditions are
	/// violated:
	/// * `row < self.nrows()`
	#[inline(always)]
	#[track_caller]
	pub unsafe fn ptr_inbounds_at(&self, row: Idx<Rows>) -> *const T {
		// SAFETY: the caller guarantees `row < self.nrows()` (# Safety above),
		// so by the view invariant the offset pointer lies in the same
		// allocation as `self.as_ptr()`, as `offset` requires.
		unsafe {
			self.as_ptr().offset(
				row.unbound() as isize * self.row_stride().element_stride(),
			)
		}
	}

	/// splits the column horizontally at the given row into two parts and
	/// returns an array of each submatrix, in the following order:
	/// * top
	/// * bottom
	///
	/// # panics
	/// the function panics if the following condition is violated:
	/// * `row <= self.nrows()`
	#[inline]
	#[track_caller]
	pub fn split_at_row(
		self,
		row: IdxInc<Rows>,
	) -> (ColRef<'a, T, usize, RStride>, ColRef<'a, T, usize, RStride>) {
		assert!(all(row <= self.nrows()));
		let rs = self.row_stride();
		let top = self.as_ptr();
		let bot = self.ptr_at(row);
		// SAFETY: `row <= self.nrows()` (asserted above). The top view is rows
		// `[0, row)` of `self`. The bottom view is rows `[row, nrows)`:
		// `ptr_at(row)` is the address of row `row` when `row < nrows`, and
		// when `row == nrows` it is `self.as_ptr()` used as the base of an
		// empty view. Both address only elements of `self` with its stride, so
		// the view invariant carries over for `'a`.
		unsafe {
			(
				ColRef::from_raw_parts(top, row.unbound(), rs),
				ColRef::from_raw_parts(
					bot,
					self.nrows().unbound() - row.unbound(),
					rs,
				),
			)
		}
	}

	/// returns a view over the transpose of `self`
	#[inline(always)]
	pub fn transpose(self) -> RowRef<'a, T, Rows, RStride> {
		RowRef {
			0: crate::row::Ref { trans: self },
		}
	}

	/// returns a view over the conjugate of `self`
	#[inline(always)]
	pub fn conjugate(self) -> ColRef<'a, T::Conj, Rows, RStride>
	where
		T: Conjugate,
	{
		// SAFETY: same pointer, row count and stride as `self`, so the view
		// invariant carries over; `T: Conjugate` guarantees that `T` and
		// `T::Conj` have the same size, alignment and bit validity, so the
		// elements may be read as `T::Conj`.
		unsafe {
			ColRef::from_raw_parts(
				self.as_ptr() as *const T::Conj,
				self.nrows(),
				self.row_stride(),
			)
		}
	}

	/// returns an unconjugated view over `self`
	#[inline(always)]
	pub fn canonical(self) -> ColRef<'a, T::Canonical, Rows, RStride>
	where
		T: Conjugate,
	{
		// SAFETY: same pointer, row count and stride as `self`, so the view
		// invariant carries over; `T: Conjugate` guarantees that `T` and
		// `T::Canonical` have the same size, alignment and bit validity, so the
		// elements may be read as `T::Canonical`.
		unsafe {
			ColRef::from_raw_parts(
				self.as_ptr() as *const T::Canonical,
				self.nrows(),
				self.row_stride(),
			)
		}
	}

	/// returns a view over the conjugate transpose of `self`
	#[inline(always)]
	pub fn adjoint(self) -> RowRef<'a, T::Conj, Rows, RStride>
	where
		T: Conjugate,
	{
		self.conjugate().transpose()
	}

	#[inline(always)]
	#[track_caller]
	pub(crate) fn at(self, row: Idx<Rows>) -> &'a T {
		assert!(all(row < self.nrows()));
		// SAFETY: `row < self.nrows()` was asserted above, which is
		// `at_unchecked`'s contract.
		unsafe { self.at_unchecked(row) }
	}

	/// returns a reference to the element at `row`, without bound checks
	///
	/// # Safety
	/// `row < self.nrows()`
	#[inline(always)]
	#[track_caller]
	pub(crate) unsafe fn at_unchecked(self, row: Idx<Rows>) -> &'a T {
		// SAFETY: the caller guarantees `row < self.nrows()` (# Safety above),
		// which is `ptr_inbounds_at`'s precondition; by the view invariant the
		// result points to an element valid for reads for `'a`, so it may be
		// borrowed for `'a`.
		unsafe { &*self.ptr_inbounds_at(row) }
	}

	/// returns a reference to the element at the given index, or a subcolumn if
	/// `row` is a range
	///
	/// # panics
	/// the function panics if any of the following conditions are violated:
	/// * `row` must be contained in `[0, self.nrows())`
	#[track_caller]
	#[inline(always)]
	pub fn get<RowRange>(
		self,
		row: RowRange,
	) -> <ColRef<'a, T, Rows, RStride> as ColIndex<RowRange>>::Target
	where
		ColRef<'a, T, Rows, RStride>: ColIndex<RowRange>,
	{
		<ColRef<'a, T, Rows, RStride> as ColIndex<RowRange>>::get(self, row)
	}

	/// returns a reference to the element at the given index, or a subcolumn if
	/// `row` is a range, without bound checks
	///
	/// # Safety
	/// the behavior is undefined if any of the following conditions are
	/// violated:
	/// * `row` must be contained in `[0, self.nrows())`
	#[track_caller]
	#[inline(always)]
	pub unsafe fn get_unchecked<RowRange>(
		self,
		row: RowRange,
	) -> <ColRef<'a, T, Rows, RStride> as ColIndex<RowRange>>::Target
	where
		ColRef<'a, T, Rows, RStride>: ColIndex<RowRange>,
	{
		// SAFETY: the caller upholds this function's contract (# Safety above),
		// which is that of `ColIndex::get_unchecked`: `row` is in bounds of
		// `self`.
		unsafe {
			<ColRef<'a, T, Rows, RStride> as ColIndex<RowRange>>::get_unchecked(
				self, row,
			)
		}
	}

	/// returns a view over the `self`, with the rows in reversed order
	#[inline]
	pub fn reverse_rows(self) -> ColRef<'a, T, Rows, RStride::Rev> {
		// SAFETY: `nrows - 1` (saturating) is at most `nrows`, so it is a valid
		// `IdxInc<Rows>` for this column (for a branded `Rows` it does not
		// exceed the branded length).
		let row = unsafe {
			IdxInc::<Rows>::new_unbound(
				self.nrows().unbound().saturating_sub(1),
			)
		};
		let ptr = self.ptr_at(row);
		// SAFETY: if `nrows > 0`, `ptr` is the address of row `nrows - 1` of
		// `self` and the reversed stride makes row `i` of the new view address
		// row `nrows - 1 - i` of `self`; if `nrows == 0`, `ptr` is
		// `self.as_ptr()` and the view is empty. So the view addresses only
		// elements of `self`, and the view invariant carries over for `'a`.
		unsafe {
			ColRef::from_raw_parts(ptr, self.nrows(), self.row_stride().rev())
		}
	}

	/// returns a view over the column starting at row `row_start`, and with
	/// number of rows `nrows`
	///
	/// # panics
	/// the function panics if any of the following conditions are violated:
	/// * `row_start <= self.nrows()`
	/// * `nrows <= self.nrows() - row_start`
	#[inline]
	#[track_caller]
	pub fn subrows<V: Shape>(
		self,
		row_start: IdxInc<Rows>,
		nrows: V,
	) -> ColRef<'a, T, V, RStride> {
		assert!(all(row_start <= self.nrows()));
		{
			let nrows = nrows.unbound();
			let full_nrows = self.nrows().unbound();
			let row_start = row_start.unbound();
			assert!(all(nrows <= full_nrows - row_start));
		}
		let rs = self.row_stride();
		// SAFETY: `row_start <= self.nrows()` and `nrows <= self.nrows() -
		// row_start` (both asserted above), so rows `[row_start, row_start +
		// nrows)` of `self` are in bounds; `ptr_at(row_start)` is the address
		// of row `row_start` (or `self.as_ptr()` when the new view is empty).
		// The new view addresses a subset of the elements of `self` with its
		// stride, so the view invariant carries over.
		unsafe { ColRef::from_raw_parts(self.ptr_at(row_start), nrows, rs) }
	}

	/// returns the input column with the given row shape after checking that it
	/// matches the current row shape
	#[inline]
	#[track_caller]
	pub fn as_row_shape<V: Shape>(self, nrows: V) -> ColRef<'a, T, V, RStride> {
		assert!(all(self.nrows().unbound() == nrows.unbound()));
		// SAFETY: the new row count equals `self.nrows()` (asserted above) and
		// pointer and stride are unchanged, so the view addresses exactly the
		// elements of `self`.
		unsafe {
			ColRef::from_raw_parts(self.as_ptr(), nrows, self.row_stride())
		}
	}

	/// returns the input column with dynamic row shape
	#[inline]
	pub fn as_dyn_rows(self) -> ColRef<'a, T, usize, RStride> {
		// SAFETY: same pointer, row count (unbranded) and stride as `self`, so
		// the view addresses exactly the elements of `self`.
		unsafe {
			ColRef::from_raw_parts(
				self.as_ptr(),
				self.nrows().unbound(),
				self.row_stride(),
			)
		}
	}

	/// returns the input column with dynamic stride
	#[inline]
	pub fn as_dyn_stride(self) -> ColRef<'a, T, Rows, isize> {
		// SAFETY: same pointer and row count as `self`, and the dynamic stride
		// equals the element stride of `self`, so the view addresses exactly
		// the elements of `self`.
		unsafe {
			ColRef::from_raw_parts(
				self.as_ptr(),
				self.nrows(),
				self.row_stride().element_stride(),
			)
		}
	}

	/// returns an iterator over the elements of the column
	#[inline]
	pub fn iter(
		self,
	) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = &'a T>
	where
		Rows: 'a,
	{
		Rows::indices(Rows::start(), self.nrows().end())
			// SAFETY: `Rows::indices(Rows::start(), self.nrows().end())` yields
			// only indices `j < self.nrows()`, which is `at_unchecked`'s
			// contract.
			.map(move |j| unsafe { self.at_unchecked(j) })
	}

	/// returns a view over the column with a static row stride equal to `+1`,
	/// or `None` otherwise
	#[inline]
	pub fn try_as_col_major(
		self,
	) -> Option<ColRef<'a, T, Rows, ContiguousFwd>> {
		if self.row_stride().element_stride() == 1 {
			// SAFETY: the element stride is 1 (checked just above), so
			// `ContiguousFwd` addresses the same elements as the stride of
			// `self`; pointer and row count are unchanged.
			Some(unsafe {
				ColRef::from_raw_parts(
					self.as_ptr(),
					self.nrows(),
					ContiguousFwd,
				)
			})
		} else {
			None
		}
	}

	/// converts the view into a mutable view over the same elements
	///
	/// # Safety
	/// `self` must satisfy the safety requirements of
	/// [`ColMut::from_raw_parts_mut`] for the lifetime `'a`: in particular, no
	/// other reference may access its elements during `'a`, and no two in-bounds
	/// rows may address the same element. this holds, for example, when `self`
	/// was derived from a `ColMut<'a, ...>` that is not used again during `'a`.
	#[inline(always)]
	#[doc(hidden)]
	pub unsafe fn const_cast(self) -> ColMut<'a, T, Rows, RStride> {
		// SAFETY: the caller upholds this function's contract (# Safety above),
		// which is the contract of `ColMut::from_raw_parts_mut` for the same
		// pointer, row count and stride.
		unsafe {
			ColMut::from_raw_parts_mut(
				self.as_ptr() as *mut T,
				self.nrows(),
				self.row_stride(),
			)
		}
	}

	/// returns a matrix view over `self`
	#[inline]
	pub fn as_mat(self) -> MatRef<'a, T, Rows, usize, RStride, isize> {
		// SAFETY: the matrix has a single column, so the column stride never
		// contributes to an in-bounds address; the matrix addresses exactly the
		// elements of `self`, with the same pointer and row stride.
		unsafe {
			MatRef::from_raw_parts(
				self.as_ptr(),
				self.nrows(),
				self.ncols(),
				self.row_stride(),
				0,
			)
		}
	}

	#[inline]
	#[doc(hidden)]
	pub fn bind_r<'N>(self, row: Guard<'N>) -> ColRef<'a, T, Dim<'N>, RStride> {
		// SAFETY: `self.nrows().bind(row)` has the same value as
		// `self.nrows()`, and pointer and stride are unchanged, so the view
		// addresses exactly the elements of `self`.
		unsafe {
			ColRef::from_raw_parts(
				self.as_ptr(),
				self.nrows().bind(row),
				self.row_stride(),
			)
		}
	}

	#[inline]
	#[track_caller]
	pub(crate) fn __at(self, i: Idx<Rows>) -> &'a T {
		self.at(i)
	}

	/// interprets the column as a diagonal matrix
	#[inline]
	pub fn as_diagonal(self) -> DiagRef<'a, T, Rows, RStride> {
		DiagRef {
			0: crate::diag::Ref { inner: self },
		}
	}
}
impl<
	T,
	Rows: Shape,
	RStride: Stride,
	Inner: for<'short> Reborrow<'short, Target = Ref<'short, T, Rows, RStride>>,
> generic::Col<Inner>
{
	/// returns a newly allocated column holding the cloned values of `self`
	#[inline]
	pub fn cloned(&self) -> Col<T, Rows>
	where
		T: Clone,
	{
		fn imp<'M, T: Clone, RStride: Stride>(
			this: ColRef<'_, T, Dim<'M>, RStride>,
		) -> Col<T, Dim<'M>> {
			Col::from_fn(this.nrows(), |i| this.at(i).clone())
		}
		let this = self.rb();
		with_dim!(M, this.nrows().unbound());
		imp(this.as_row_shape(M)).into_row_shape(this.nrows())
	}

	/// returns a newly allocated column holding the (possibly conjugated)
	/// values of `self`
	#[inline]
	pub fn to_owned(&self) -> Col<T::Canonical, Rows>
	where
		T: Conjugate,
	{
		fn imp<'M, T, RStride: Stride>(
			this: ColRef<'_, T, Dim<'M>, RStride>,
		) -> Col<T::Canonical, Dim<'M>>
		where
			T: Conjugate,
		{
			Col::from_fn(this.nrows(), |i| Conj::apply::<T>(this.at(i)))
		}
		let this = self.rb();
		with_dim!(M, this.nrows().unbound());
		imp(this.as_row_shape(M)).into_row_shape(this.nrows())
	}

	/// returns a view over `self`
	#[inline]
	pub fn as_ref(&self) -> ColRef<'_, T, Rows, RStride> {
		self.rb()
	}

	/// returns `true` if all of the elements of `self` are finite.
	/// otherwise returns `false`.
	#[inline]
	pub fn is_all_finite(&self) -> bool
	where
		T: Conjugate,
	{
		fn imp<T: ComplexField>(A: ColRef<'_, T>) -> bool {
			with_dim!({
				let M = A.nrows();
			});
			let A = A.as_row_shape(M);
			for i in M.indices() {
				if !A[i].is_finite() {
					return false;
				}
			}
			true
		}
		imp(self.rb().as_dyn_rows().as_dyn_stride().canonical())
	}

	/// returns `true` if any of the elements of `self` is `NaN`.
	/// otherwise returns `false`.
	#[inline]
	pub fn has_nan(&self) -> bool
	where
		T: Conjugate,
	{
		fn imp<T: ComplexField>(A: ColRef<'_, T>) -> bool {
			with_dim!({
				let M = A.nrows();
			});
			let A = A.as_row_shape(M);
			for i in M.indices() {
				if A[i].is_nan() {
					return true;
				}
			}
			false
		}
		imp(self.rb().as_dyn_rows().as_dyn_stride().canonical())
	}
}
impl<'a, T, Rows: Shape> ColRef<'a, T, Rows, ContiguousFwd> {
	/// returns a reference over the elements as a slice
	#[inline]
	pub fn as_slice(self) -> &'a [T] {
		// SAFETY: the row stride is `ContiguousFwd` (+1), so by the view
		// invariant the `nrows` elements starting at `as_ptr()` are contiguous
		// in one allocation and valid for reads for `'a`; the pointer is
		// non-null and aligned even for an empty column (`from_raw_parts`
		// contract).
		unsafe {
			core::slice::from_raw_parts(self.as_ptr(), self.nrows().unbound())
		}
	}
}
impl<'a, 'ROWS, T> ColRef<'a, T, Dim<'ROWS>, ContiguousFwd> {
	/// returns a reference over the elements as a lifetime-bound slice
	#[inline]
	pub fn as_array(self) -> &'a Array<'ROWS, T> {
		// SAFETY: `Array<'ROWS, T>` is `#[repr(transparent)]` over `[T]`, and
		// the slice has length `self.nrows()`, which is the length tied to
		// `'ROWS`, as `Array` requires.
		unsafe { &*(self.as_slice() as *const [_] as *const Array<'ROWS, T>) }
	}
}
impl<'ROWS, 'a, T, RStride: Stride> ColRef<'a, T, Dim<'ROWS>, RStride> {
	#[doc(hidden)]
	#[inline]
	pub fn split_rows_with<'TOP, 'BOT>(
		self,
		row: Partition<'TOP, 'BOT, 'ROWS>,
	) -> (
		ColRef<'a, T, Dim<'TOP>, RStride>,
		ColRef<'a, T, Dim<'BOT>, RStride>,
	) {
		let (a, b) = self.split_at_row(row.midpoint());
		(a.as_row_shape(row.head), b.as_row_shape(row.tail))
	}
}
impl<T: core::fmt::Debug, Rows: Shape, RStride: Stride> core::fmt::Debug
	for Ref<'_, T, Rows, RStride>
{
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		generic::Col::from_inner_ref(self).transpose().fmt(f)
	}
}
impl<'a, T> ColRef<'a, T, usize, isize>
where
	T: RealField,
{
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

	/// Returns the minimum element in the column, or `None` if the column is
	/// empty
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
impl<'a, T, Rows: Shape, RStride: Stride> ColRef<'a, T, Rows, RStride>
where
	T: RealField,
{
	/// Returns the maximum element in the column, or `None` if the column is
	/// empty
	pub fn max(&self) -> Option<T> {
		self.as_dyn_rows().as_dyn_stride().internal_max()
	}

	/// Returns the minimum element in the column, or `None` if the column is
	/// empty
	pub fn min(&self) -> Option<T> {
		self.as_dyn_rows().as_dyn_stride().internal_min()
	}
}
#[cfg(test)]
mod tests {
	use crate::Col;
	#[test]
	fn test_col_min() {
		let col: Col<f64> = Col::from_fn(5, |x| (x + 1) as f64);
		let colref = col.as_ref();
		assert_eq!(colref.min(), Some(1.0));
		let empty: Col<f64> = Col::from_fn(0, |_| 0.0);
		let emptyref = empty.as_ref();
		assert_eq!(emptyref.min(), None);
	}
	#[test]
	fn test_col_max() {
		let col: Col<f64> = Col::from_fn(5, |x| (x + 1) as f64);
		let colref = col.as_ref();
		assert_eq!(colref.max(), Some(5.0));
		let empty: Col<f64> = Col::from_fn(0, |_| 0.0);
		let emptyref = empty.as_ref();
		assert_eq!(emptyref.max(), None);
	}
}
