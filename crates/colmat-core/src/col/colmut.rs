use super::*;
use crate::utils::bound::{Array, Dim, Partition};
use crate::{ContiguousFwd, Idx, IdxInc};
use core::marker::PhantomData;
use core::ptr::NonNull;
use equator::assert;
use generativity::Guard;
/// see [`super::ColMut`]
pub struct Mut<'a, T, Rows = usize, RStride = isize> {
	pub(super) imp: ColView<T, Rows, RStride>,
	pub(super) __marker: PhantomData<&'a mut T>,
}
impl<'short, T, Rows: Copy, RStride: Copy> Reborrow<'short>
	for Mut<'_, T, Rows, RStride>
{
	type Target = Ref<'short, T, Rows, RStride>;

	#[inline]
	fn rb(&'short self) -> Self::Target {
		Ref {
			imp: self.imp,
			__marker: PhantomData,
		}
	}
}
impl<'short, T, Rows: Copy, RStride: Copy> ReborrowMut<'short>
	for Mut<'_, T, Rows, RStride>
{
	type Target = Mut<'short, T, Rows, RStride>;

	#[inline]
	fn rb_mut(&'short mut self) -> Self::Target {
		Mut {
			imp: self.imp,
			__marker: PhantomData,
		}
	}
}
impl<'a, T, Rows: Copy, RStride: Copy> IntoConst for Mut<'a, T, Rows, RStride> {
	type Target = Ref<'a, T, Rows, RStride>;

	#[inline]
	fn into_const(self) -> Self::Target {
		Ref {
			imp: self.imp,
			__marker: PhantomData,
		}
	}
}
// SAFETY: a `Mut` is an exclusive borrow of `T` values (like `&'a mut T`) plus
// plain `Rows`/`RStride` values; through `&Mut` only shared access to the `T`
// values is possible without `unsafe`, so sharing it across threads is sound
// for `T: Sync`.
unsafe impl<T: Sync, Rows: Sync, RStride: Sync> Sync
	for Mut<'_, T, Rows, RStride>
{
}
// SAFETY: sending a `Mut` moves its exclusive borrow to another thread, like
// sending `&'a mut T`, which is sound for `T: Send`; `Rows` and `RStride` are
// moved by value and are `Send`.
unsafe impl<T: Send, Rows: Send, RStride: Send> Send
	for Mut<'_, T, Rows, RStride>
{
}
impl<'a, T> ColMut<'a, T> {
	/// creates a column view over the given element
	#[inline]
	pub fn from_mut(value: &'a mut T) -> Self {
		// SAFETY: `value` is an exclusive reference, so the pointer is
		// non-null, aligned and valid for reads and writes of one element for
		// `'a`, with no other access; with one row, index 0 is the only
		// in-bounds index.
		unsafe { ColMut::from_raw_parts_mut(value as *mut T, 1, 1) }
	}

	/// creates a `ColMut` from slice views over the column vector data, the
	/// result has the same number of rows as the length of the input slice
	#[inline]
	pub fn from_slice_mut(slice: &'a mut [T]) -> Self {
		let len = slice.len();
		// SAFETY: a slice pointer is non-null and aligned (also when empty),
		// and the `len` contiguous, distinct elements (row stride 1) of an
		// exclusive slice are valid for reads and writes for `'a`, with no
		// other access.
		unsafe { Self::from_raw_parts_mut(slice.as_mut_ptr(), len, 1) }
	}
}
impl<'a, T, Rows: Shape, RStride: Stride> ColMut<'a, T, Rows, RStride> {
	/// creates a `ColMut` from pointers to the column vector data, number of
	/// rows, and row stride
	///
	/// # Safety
	/// this function has the same safety requirements as
	/// [`MatMut::from_raw_parts_mut(ptr, nrows, 1, row_stride, 0)`]
	#[inline(always)]
	#[track_caller]
	pub const unsafe fn from_raw_parts_mut(
		ptr: *mut T,
		nrows: Rows,
		row_stride: RStride,
	) -> Self {
		// SAFETY: the caller upholds this function's contract (# Safety above),
		// which includes `ptr` being non-null.
		unsafe {
			Self(Mut {
				imp: ColView {
					ptr: NonNull::new_unchecked(ptr),
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
		self.rb().as_ptr()
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
	pub fn map<U>(&self, f: impl Fn(&T) -> U) -> Col<U, Rows> {
		self.rb().map(f)
	}

	/// calls `f` on each element of `self`, in order
	#[inline]
	pub fn for_each(&self, f: impl Fn(&T)) {
		self.rb().for_each(f)
	}

	/// returns a newly allocated column holding `f` applied to a mutable
	/// reference to each element of `self`
	#[inline]
	pub fn map_mut<U>(&mut self, f: impl FnMut(&mut T) -> U) -> Col<U, Rows> {
		let mut f = f;
		zip!(self).map(|unzip!(x)| f(x))
	}

	/// calls `f` on a mutable reference to each element of `self`, in order
	#[inline]
	pub fn for_each_mut(&mut self, f: impl FnMut(&mut T)) {
		let mut f = f;
		zip!(self).for_each(|unzip!(x)| f(x))
	}

	/// returns a raw pointer to the element at the given index
	#[inline(always)]
	pub fn ptr_at(&self, row: IdxInc<Rows>) -> *const T {
		self.rb().ptr_at(row)
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
		// which is `ColRef::ptr_inbounds_at`'s precondition for the reborrowed
		// view (same pointer, row count and stride).
		unsafe { self.rb().ptr_inbounds_at(row) }
	}

	#[inline]
	#[track_caller]
	/// see [`ColRef::split_at_row`]
	pub fn split_at_row(
		self,
		row: IdxInc<Rows>,
	) -> (ColRef<'a, T, usize, RStride>, ColRef<'a, T, usize, RStride>) {
		self.into_const().split_at_row(row)
	}

	#[inline(always)]
	/// see [`ColRef::transpose`]
	pub fn transpose(self) -> RowRef<'a, T, Rows, RStride> {
		self.into_const().transpose()
	}

	#[inline(always)]
	/// see [`ColRef::conjugate`]
	pub fn conjugate(self) -> ColRef<'a, T::Conj, Rows, RStride>
	where
		T: Conjugate,
	{
		self.into_const().conjugate()
	}

	#[inline(always)]
	/// see [`ColRef::canonical`]
	pub fn canonical(self) -> ColRef<'a, T::Canonical, Rows, RStride>
	where
		T: Conjugate,
	{
		self.into_const().canonical()
	}

	#[inline(always)]
	/// see [`ColRef::adjoint`]
	pub fn adjoint(self) -> RowRef<'a, T::Conj, Rows, RStride>
	where
		T: Conjugate,
	{
		self.into_const().adjoint()
	}

	#[track_caller]
	#[inline(always)]
	/// see [`ColRef::get`]
	pub fn get<RowRange>(
		self,
		row: RowRange,
	) -> <ColRef<'a, T, Rows, RStride> as ColIndex<RowRange>>::Target
	where
		ColRef<'a, T, Rows, RStride>: ColIndex<RowRange>,
	{
		<ColRef<'a, T, Rows, RStride> as ColIndex<RowRange>>::get(
			self.into_const(),
			row,
		)
	}

	/// see [`ColRef::get_unchecked`]
	///
	/// # Safety
	/// the given ranges must be in bounds of the view.
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
		// which is that of `ColIndex::get_unchecked` for the const view of
		// `self` (same elements).
		unsafe {
			<ColRef<'a, T, Rows, RStride> as ColIndex<RowRange>>::get_unchecked(
				self.into_const(),
				row,
			)
		}
	}

	#[inline]
	/// see [`ColRef::reverse_rows`]
	pub fn reverse_rows(self) -> ColRef<'a, T, Rows, RStride::Rev> {
		self.into_const().reverse_rows()
	}

	#[inline]
	/// see [`ColRef::subrows`]
	pub fn subrows<V: Shape>(
		self,
		row_start: IdxInc<Rows>,
		nrows: V,
	) -> ColRef<'a, T, V, RStride> {
		self.into_const().subrows(row_start, nrows)
	}

	#[inline]
	#[track_caller]
	/// see [`ColRef::as_row_shape`]
	pub fn as_row_shape<V: Shape>(self, nrows: V) -> ColRef<'a, T, V, RStride> {
		self.into_const().as_row_shape(nrows)
	}

	#[inline]
	/// see [`ColRef::as_dyn_rows`]
	pub fn as_dyn_rows(self) -> ColRef<'a, T, usize, RStride> {
		self.into_const().as_dyn_rows()
	}

	#[inline]
	/// see [`ColRef::as_dyn_stride`]
	pub fn as_dyn_stride(self) -> ColRef<'a, T, Rows, isize> {
		self.into_const().as_dyn_stride()
	}

	#[inline]
	/// see [`ColRef::iter`]
	pub fn iter(
		self,
	) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = &'a T>
	where
		Rows: 'a,
	{
		self.into_const().iter()
	}

	#[inline]
	/// see [`ColRef::try_as_col_major`]
	pub fn try_as_col_major(
		self,
	) -> Option<ColRef<'a, T, Rows, ContiguousFwd>> {
		self.into_const().try_as_col_major()
	}

	#[inline]
	/// see [`ColRef::try_as_col_major`]
	pub fn try_as_col_major_mut(
		self,
	) -> Option<ColMut<'a, T, Rows, ContiguousFwd>> {
		self.into_const()
			.try_as_col_major()
			// SAFETY: `x` addresses exactly the elements of `self`, which was
			// consumed, so `x` inherits its exclusive borrow for `'a` and the
			// mutable view invariant (`const_cast`'s contract).
			.map(|x| unsafe { x.const_cast() })
	}

	/// returns `self` unchanged; mirrors [`ColRef::const_cast`]
	///
	/// # Safety
	/// none: this is the identity on a mutable view. it is `unsafe` only to
	/// mirror [`ColRef::const_cast`].
	#[inline(always)]
	#[doc(hidden)]
	pub unsafe fn const_cast(self) -> ColMut<'a, T, Rows, RStride> {
		self
	}

	#[inline]
	#[doc(hidden)]
	pub fn bind_r<'N>(self, row: Guard<'N>) -> ColMut<'a, T, Dim<'N>, RStride> {
		// SAFETY: same pointer, stride and row count (`bind` only brands it) as
		// `self`, which is consumed, so the result inherits its exclusive
		// borrow and the mutable view invariant.
		unsafe {
			ColMut::from_raw_parts_mut(
				self.as_ptr_mut(),
				self.nrows().bind(row),
				self.row_stride(),
			)
		}
	}

	#[inline]
	/// see [`ColRef::as_mat`]
	pub fn as_mat(self) -> MatRef<'a, T, Rows, usize, RStride, isize> {
		self.into_const().as_mat()
	}

	#[inline]
	/// see [`ColRef::as_mat`]
	pub fn as_mat_mut(self) -> MatMut<'a, T, Rows, usize, RStride, isize> {
		// SAFETY: `self` is consumed, so the result inherits its exclusive
		// borrow for `'a`; the single-column matrix from `as_mat` addresses
		// exactly the elements of `self`, so the mutable view invariant carries
		// over (`MatRef::const_cast`'s contract).
		unsafe { self.into_const().as_mat().const_cast() }
	}

	#[inline]
	/// see [`ColRef::as_diagonal`]
	pub fn as_diagonal(self) -> DiagRef<'a, T, Rows, RStride> {
		DiagRef {
			0: crate::diag::Ref {
				inner: self.into_const(),
			},
		}
	}
}
impl<
	T,
	Rows: Shape,
	RStride: Stride,
	Inner: for<'short> ReborrowMut<'short, Target = Mut<'short, T, Rows, RStride>>,
> generic::Col<Inner>
{
	/// copies `other` into `self`
	#[inline]
	pub fn copy_from<RhsT: Conjugate<Canonical = T>>(
		&mut self,
		other: impl AsColRef<T = RhsT, Rows = Rows>,
	) where
		T: ComplexField,
	{
		let other = other.as_col_ref();
		let this = self.rb_mut();
		assert!(all(
			this.nrows() == other.nrows(),
			this.ncols() == other.ncols(),
		));
		let m = this.nrows();
		with_dim!(M, m.unbound());
		imp(
			self.rb_mut().as_row_shape_mut(M).as_dyn_stride_mut(),
			other.as_row_shape(M).canonical(),
			Conj::get::<RhsT>(),
		);
		#[allow(clippy::extra_unused_lifetimes)] // the lifetime appears inside `Dim<'M>` argument types
		pub fn imp<'M, 'N, T: ComplexField>(
			this: ColMut<'_, T, Dim<'M>>,
			other: ColRef<'_, T, Dim<'M>>,
			conj_: Conj,
		) {
			match conj_ {
				Conj::No => {
					zip!(this, other)
						.for_each(|unzip!(dst, src)| *dst = src.copy());
				},
				Conj::Yes => {
					zip!(this, other)
						.for_each(|unzip!(dst, src)| *dst = src.conj());
				},
			}
		}
	}

	/// fills all the elements of `self` with `value`
	#[inline]
	pub fn fill(&mut self, value: T)
	where
		T: Clone,
	{
		fn cloner<T: Clone>(
			value: T,
		) -> impl for<'a> FnMut(crate::zip::Last<&'a mut T>) {
			#[inline(always)]
			move |x| *x.0 = value.clone()
		}
		z!(self.rb_mut().as_dyn_rows_mut()).for_each(cloner::<T>(value));
	}

	#[inline]
	/// returns a view over `self`
	pub fn as_mut(&mut self) -> ColMut<'_, T, Rows, RStride> {
		self.rb_mut()
	}
}
impl<'a, T, Rows: Shape, RStride: Stride> ColMut<'a, T, Rows, RStride> {
	#[inline(always)]
	/// see [`ColRef::as_ptr`]
	pub fn as_ptr_mut(&self) -> *mut T {
		self.rb().as_ptr() as *mut T
	}

	#[inline(always)]
	/// see [`ColRef::ptr_at`]
	pub fn ptr_at_mut(&self, row: IdxInc<Rows>) -> *mut T {
		self.rb().ptr_at(row) as *mut T
	}

	/// see [`ColRef::ptr_inbounds_at`]
	///
	/// # Safety
	/// `row` must be in bounds of the view, and no other borrow of the element may be live while the returned pointer is used.
	#[inline(always)]
	#[track_caller]
	pub unsafe fn ptr_inbounds_at_mut(&self, row: Idx<Rows>) -> *mut T {
		// SAFETY: the caller guarantees `row < self.nrows()` (# Safety above),
		// which is `ColRef::ptr_inbounds_at`'s precondition.
		unsafe { self.rb().ptr_inbounds_at(row) as *mut T }
	}

	#[inline]
	#[track_caller]
	/// see [`ColRef::split_at_row`]
	pub fn split_at_row_mut(
		self,
		row: IdxInc<Rows>,
	) -> (ColMut<'a, T, usize, RStride>, ColMut<'a, T, usize, RStride>) {
		let (a, b) = self.into_const().split_at_row(row);
		// SAFETY: `self` is consumed, so the two views inherit its exclusive
		// borrow for `'a`. They are rows `[0, row)` and `[row, nrows)` of
		// `self`, and distinct rows of a mutable view address distinct elements
		// (mutable view invariant), so the two views do not alias
		// (`const_cast`'s contract).
		unsafe { (a.const_cast(), b.const_cast()) }
	}

	#[inline(always)]
	/// see [`ColRef::transpose`]
	pub fn transpose_mut(self) -> RowMut<'a, T, Rows, RStride> {
		// SAFETY: `self` is consumed, so the result inherits its exclusive
		// borrow for `'a`, and it addresses exactly the elements of `self`, so
		// the mutable view invariant carries over (`const_cast`'s contract).
		unsafe { self.into_const().transpose().const_cast() }
	}

	#[inline(always)]
	/// see [`ColRef::conjugate`]
	pub fn conjugate_mut(self) -> ColMut<'a, T::Conj, Rows, RStride>
	where
		T: Conjugate,
	{
		// SAFETY: `self` is consumed, so the result inherits its exclusive
		// borrow for `'a` and addresses exactly its elements (`const_cast`'s
		// contract); `T: Conjugate` guarantees that `T` and `T::Conj` have the
		// same size, alignment and bit validity, so values written as `T::Conj`
		// are valid `T`.
		unsafe { self.into_const().conjugate().const_cast() }
	}

	#[inline(always)]
	/// see [`ColRef::canonical`]
	pub fn canonical_mut(self) -> ColMut<'a, T::Canonical, Rows, RStride>
	where
		T: Conjugate,
	{
		// SAFETY: `self` is consumed, so the result inherits its exclusive
		// borrow for `'a` and addresses exactly its elements (`const_cast`'s
		// contract); `T: Conjugate` guarantees that `T` and `T::Canonical` have
		// the same size, alignment and bit validity, so values written as
		// `T::Canonical` are valid `T`.
		unsafe { self.into_const().canonical().const_cast() }
	}

	#[inline(always)]
	/// see [`ColRef::adjoint`]
	pub fn adjoint_mut(self) -> RowMut<'a, T::Conj, Rows, RStride>
	where
		T: Conjugate,
	{
		// SAFETY: `self` is consumed, so the result inherits its exclusive
		// borrow for `'a` and addresses exactly its elements (`const_cast`'s
		// contract); `T: Conjugate` guarantees that `T` and `T::Conj` have the
		// same size, alignment and bit validity, so values written as `T::Conj`
		// are valid `T`.
		unsafe { self.into_const().adjoint().const_cast() }
	}

	#[inline(always)]
	#[track_caller]
	pub(crate) fn at_mut(self, row: Idx<Rows>) -> &'a mut T {
		assert!(all(row < self.nrows()));
		// SAFETY: `row < self.nrows()` was asserted above, which is
		// `at_mut_unchecked`'s contract.
		unsafe { self.at_mut_unchecked(row) }
	}

	/// returns a mutable reference to the element at `row`, without bound
	/// checks
	///
	/// # Safety
	/// `row < self.nrows()`
	#[inline(always)]
	#[track_caller]
	pub(crate) unsafe fn at_mut_unchecked(self, row: Idx<Rows>) -> &'a mut T {
		// SAFETY: the caller guarantees `row < self.nrows()` (# Safety above),
		// which is `ptr_inbounds_at_mut`'s precondition; by the mutable view
		// invariant the element is valid for reads and writes for `'a` with no
		// other access, and `self` is consumed, so it may be borrowed mutably
		// for `'a`.
		unsafe { &mut *self.ptr_inbounds_at_mut(row) }
	}

	#[track_caller]
	#[inline(always)]
	/// see [`ColRef::get`]
	pub fn get_mut<RowRange>(
		self,
		row: RowRange,
	) -> <ColMut<'a, T, Rows, RStride> as ColIndex<RowRange>>::Target
	where
		ColMut<'a, T, Rows, RStride>: ColIndex<RowRange>,
	{
		<ColMut<'a, T, Rows, RStride> as ColIndex<RowRange>>::get(self, row)
	}

	/// see [`ColRef::get_unchecked`]
	///
	/// # Safety
	/// the given ranges must be in bounds of the view and must not overlap any other live borrow of the same elements.
	#[track_caller]
	#[inline(always)]
	pub unsafe fn get_mut_unchecked<RowRange>(
		self,
		row: RowRange,
	) -> <ColMut<'a, T, Rows, RStride> as ColIndex<RowRange>>::Target
	where
		ColMut<'a, T, Rows, RStride>: ColIndex<RowRange>,
	{
		// SAFETY: the caller upholds this function's contract (# Safety above),
		// which is that of `ColIndex::get_unchecked`: `row` is in bounds of
		// `self`.
		unsafe {
			<ColMut<'a, T, Rows, RStride> as ColIndex<RowRange>>::get_unchecked(
				self, row,
			)
		}
	}

	#[inline]
	/// see [`ColRef::reverse_rows`]
	pub fn reverse_rows_mut(self) -> ColMut<'a, T, Rows, RStride::Rev> {
		// SAFETY: `self` is consumed, so the result inherits its exclusive
		// borrow for `'a`, and it addresses exactly the elements of `self`, so
		// the mutable view invariant carries over (`const_cast`'s contract).
		unsafe { self.into_const().reverse_rows().const_cast() }
	}

	#[inline]
	#[track_caller]
	/// see [`ColRef::subrows`]
	pub fn subrows_mut<V: Shape>(
		self,
		row_start: IdxInc<Rows>,
		nrows: V,
	) -> ColMut<'a, T, V, RStride> {
		// SAFETY: `subrows` checks the bounds and addresses a subset of the
		// elements of `self`; `self` is consumed, so the result inherits its
		// exclusive borrow for `'a` and the mutable view invariant
		// (`const_cast`'s contract).
		unsafe { self.into_const().subrows(row_start, nrows).const_cast() }
	}

	#[inline]
	#[track_caller]
	/// see [`ColRef::as_row_shape`]
	pub fn as_row_shape_mut<V: Shape>(
		self,
		nrows: V,
	) -> ColMut<'a, T, V, RStride> {
		// SAFETY: `self` is consumed, so the result inherits its exclusive
		// borrow for `'a`, and it addresses exactly the elements of `self`, so
		// the mutable view invariant carries over (`const_cast`'s contract).
		unsafe { self.into_const().as_row_shape(nrows).const_cast() }
	}

	#[inline]
	/// see [`ColRef::as_dyn_rows`]
	pub fn as_dyn_rows_mut(self) -> ColMut<'a, T, usize, RStride> {
		// SAFETY: `self` is consumed, so the result inherits its exclusive
		// borrow for `'a`, and it addresses exactly the elements of `self`, so
		// the mutable view invariant carries over (`const_cast`'s contract).
		unsafe { self.into_const().as_dyn_rows().const_cast() }
	}

	#[inline]
	/// see [`ColRef::as_dyn_stride`]
	pub fn as_dyn_stride_mut(self) -> ColMut<'a, T, Rows, isize> {
		// SAFETY: `self` is consumed, so the result inherits its exclusive
		// borrow for `'a`, and it addresses exactly the elements of `self`, so
		// the mutable view invariant carries over (`const_cast`'s contract).
		unsafe { self.into_const().as_dyn_stride().const_cast() }
	}

	#[inline]
	/// see [`ColRef::iter`]
	pub fn iter_mut(
		self,
	) -> impl 'a + ExactSizeIterator + DoubleEndedIterator<Item = &'a mut T>
	where
		Rows: 'a,
	{
		let this = self.into_const();
		Rows::indices(Rows::start(), this.nrows().end())
			// SAFETY: `this` addresses exactly the elements of `self`, which
			// was consumed, so `const_cast`'s contract holds. The indices are
			// each `j < nrows`, yielded once, so `at_mut_unchecked`'s
			// precondition holds and the `&'a mut T` it returns are to distinct
			// elements (distinct rows of a mutable view address distinct
			// elements).
			.map(move |j| unsafe { this.const_cast().at_mut_unchecked(j) })
	}

	#[inline]
	/// see [`ColRef::as_diagonal`]
	pub fn as_diagonal_mut(self) -> DiagMut<'a, T, Rows, RStride> {
		DiagMut {
			0: crate::diag::Mut { inner: self },
		}
	}

	#[inline]
	#[track_caller]
	pub(crate) fn __at_mut(self, i: Idx<Rows>) -> &'a mut T {
		self.at_mut(i)
	}
}
impl<'a, T, Rows: Shape> ColMut<'a, T, Rows, ContiguousFwd> {
	/// returns a reference over the elements as a slice
	#[inline]
	pub fn as_slice_mut(self) -> &'a mut [T] {
		// SAFETY: the row stride is `ContiguousFwd` (+1), so by the mutable
		// view invariant the `nrows` elements starting at `as_ptr_mut()` are
		// contiguous in one allocation and valid for reads and writes for `'a`
		// with no other access (`self` is consumed); the pointer is non-null
		// and aligned even for an empty column (`from_raw_parts_mut` contract).
		unsafe {
			core::slice::from_raw_parts_mut(
				self.as_ptr_mut(),
				self.nrows().unbound(),
			)
		}
	}
}
impl<'a, 'ROWS, T> ColMut<'a, T, Dim<'ROWS>, ContiguousFwd> {
	/// returns a reference over the elements as a lifetime-bound slice
	#[inline]
	pub fn as_array_mut(self) -> &'a mut Array<'ROWS, T> {
		// SAFETY: `Array<'ROWS, T>` is `#[repr(transparent)]` over `[T]`, and
		// the slice has length `self.nrows()`, which is the length tied to
		// `'ROWS`, as `Array` requires.
		unsafe {
			&mut *(self.as_slice_mut() as *mut [_] as *mut Array<'ROWS, T>)
		}
	}
}
impl<'ROWS, 'a, T, RStride: Stride> ColMut<'a, T, Dim<'ROWS>, RStride> {
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
impl<'ROWS, 'a, T, RStride: Stride> ColMut<'a, T, Dim<'ROWS>, RStride> {
	#[doc(hidden)]
	#[inline]
	pub fn split_rows_with_mut<'TOP, 'BOT>(
		self,
		row: Partition<'TOP, 'BOT, 'ROWS>,
	) -> (
		ColMut<'a, T, Dim<'TOP>, RStride>,
		ColMut<'a, T, Dim<'BOT>, RStride>,
	) {
		let (a, b) = self.split_at_row_mut(row.midpoint());
		(a.as_row_shape_mut(row.head), b.as_row_shape_mut(row.tail))
	}
}
impl<T: core::fmt::Debug, Rows: Shape, RStride: Stride> core::fmt::Debug
	for Mut<'_, T, Rows, RStride>
{
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		self.rb().fmt(f)
	}
}
impl<'a, T, Rows: Shape> ColMut<'a, T, Rows>
where
	T: RealField,
{
	/// Returns the maximum element in the column, or `None` if the column is
	/// empty
	pub fn max(&self) -> Option<T> {
		self.rb().as_dyn_rows().as_dyn_stride().internal_max()
	}

	/// Returns the minimum element in the column, or `None` if the column is
	/// empty
	pub fn min(&self) -> Option<T> {
		self.rb().as_dyn_rows().as_dyn_stride().internal_min()
	}
}
#[cfg(test)]
mod tests {
	use crate::Col;
	#[test]
	fn test_col_min() {
		let mut col: Col<f64> = Col::from_fn(5, |x| (x + 1) as f64);
		let colmut = col.as_mut();
		assert_eq!(colmut.min(), Some(1.0));
		let mut empty: Col<f64> = Col::from_fn(0, |_| 0.0);
		let emptymut = empty.as_mut();
		assert_eq!(emptymut.min(), None);
	}
	#[test]
	fn test_col_max() {
		let mut col: Col<f64> = Col::from_fn(5, |x| (x + 1) as f64);
		let colmut = col.as_mut();
		assert_eq!(colmut.max(), Some(5.0));
		let mut empty: Col<f64> = Col::from_fn(0, |_| 0.0);
		let emptymut = empty.as_mut();
		assert_eq!(emptymut.max(), None);
	}
}
