//! Ported from colmat-rs `faer/src/linalg/zip.rs` at `71337cdc`.
//! Changed: moved from `linalg::zip` to `zip`; nothing else beyond the `SAFETY` comments; every unsafe block carries a `// SAFETY:` comment.
//! Known defects inherited from colmat-rs are marked `TODO(review)` (see
//! `docs/worklogs/2026-11-20-colmat-core-views-port.md`). faer's MIT notice:
//! `LICENSE-faer`.
#![allow(missing_docs)]
use crate::col::{Col, ColMut, ColRef};
use crate::mat::{Mat, MatMut, MatRef};
use crate::row::{Row, RowMut, RowRef};
use crate::{ContiguousFwd, Idx, Shape, Stride, Unbind, diag};
use core::marker::PhantomData;
use equator::{assert, debug_assert};
use reborrow::*;
pub trait IntoView {
	type Target;
	fn into_view(self) -> Self::Target;
}
impl<'a, T, Rows: Shape, Cols: Shape> IntoView for &'a mut Mat<T, Rows, Cols> {
	type Target = MatMut<'a, T, Rows, Cols, ContiguousFwd>;

	#[inline]
	fn into_view(self) -> Self::Target {
		self.as_mut().try_as_col_major_mut().unwrap()
	}
}
impl<'a, T, Rows: Shape, Cols: Shape> IntoView for &'a Mat<T, Rows, Cols> {
	type Target = MatRef<'a, T, Rows, Cols, ContiguousFwd>;

	#[inline]
	fn into_view(self) -> Self::Target {
		self.as_ref().try_as_col_major().unwrap()
	}
}
impl<'a, T, Len: Shape> IntoView for &'a mut Col<T, Len> {
	type Target = ColMut<'a, T, Len, ContiguousFwd>;

	#[inline]
	fn into_view(self) -> Self::Target {
		self.as_mut().try_as_col_major_mut().unwrap()
	}
}
impl<'a, T, Len: Shape> IntoView for &'a Col<T, Len> {
	type Target = ColRef<'a, T, Len, ContiguousFwd>;

	#[inline]
	fn into_view(self) -> Self::Target {
		self.as_ref().try_as_col_major().unwrap()
	}
}
impl<'a, T, Len: Shape> IntoView for &'a mut Row<T, Len> {
	type Target = RowMut<'a, T, Len, ContiguousFwd>;

	#[inline]
	fn into_view(self) -> Self::Target {
		self.as_mut().try_as_row_major_mut().unwrap()
	}
}
impl<'a, T, Len: Shape> IntoView for &'a Row<T, Len> {
	type Target = RowRef<'a, T, Len, ContiguousFwd>;

	#[inline]
	fn into_view(self) -> Self::Target {
		self.as_ref().try_as_row_major().unwrap()
	}
}
impl<'a, T, Len: Shape> IntoView for &'a mut diag::Diag<T, Len> {
	type Target = diag::DiagMut<'a, T, Len, ContiguousFwd>;

	#[inline]
	fn into_view(self) -> Self::Target {
		self.as_mut()
			.column_vector_mut()
			.try_as_col_major_mut()
			.unwrap()
			.as_diagonal_mut()
	}
}
impl<'a, T, Len: Shape> IntoView for &'a diag::Diag<T, Len> {
	type Target = diag::DiagRef<'a, T, Len, ContiguousFwd>;

	#[inline]
	fn into_view(self) -> Self::Target {
		self.as_ref()
			.column_vector()
			.try_as_col_major()
			.unwrap()
			.as_diagonal()
	}
}
impl<'a, T, Rows: Shape, Cols: Shape, RStride: Stride, CStride: Stride> IntoView
	for MatMut<'a, T, Rows, Cols, RStride, CStride>
{
	type Target = Self;

	#[inline]
	fn into_view(self) -> Self::Target {
		self
	}
}
impl<'a, T, Rows: Shape, Cols: Shape, RStride: Stride, CStride: Stride> IntoView
	for MatRef<'a, T, Rows, Cols, RStride, CStride>
{
	type Target = Self;

	#[inline]
	fn into_view(self) -> Self::Target {
		self
	}
}
impl<'a, T, Rows: Shape, Cols: Shape, RStride: Stride, CStride: Stride> IntoView
	for &'a MatMut<'_, T, Rows, Cols, RStride, CStride>
{
	type Target = MatRef<'a, T, Rows, Cols, RStride, CStride>;

	#[inline]
	fn into_view(self) -> Self::Target {
		self.rb()
	}
}
impl<'a, T, Rows: Shape, Cols: Shape, RStride: Stride, CStride: Stride> IntoView
	for &'a MatRef<'_, T, Rows, Cols, RStride, CStride>
{
	type Target = MatRef<'a, T, Rows, Cols, RStride, CStride>;

	#[inline]
	fn into_view(self) -> Self::Target {
		*self
	}
}
impl<'a, T, Rows: Shape, Cols: Shape, RStride: Stride, CStride: Stride> IntoView
	for &'a mut MatMut<'_, T, Rows, Cols, RStride, CStride>
{
	type Target = MatMut<'a, T, Rows, Cols, RStride, CStride>;

	#[inline]
	fn into_view(self) -> Self::Target {
		self.rb_mut()
	}
}
impl<'a, T, Rows: Shape, Cols: Shape, RStride: Stride, CStride: Stride> IntoView
	for &'a mut MatRef<'_, T, Rows, Cols, RStride, CStride>
{
	type Target = MatRef<'a, T, Rows, Cols, RStride, CStride>;

	#[inline]
	fn into_view(self) -> Self::Target {
		*self
	}
}
impl<'a, T, Rows: Shape, RStride: Stride> IntoView
	for ColMut<'a, T, Rows, RStride>
{
	type Target = Self;

	#[inline]
	fn into_view(self) -> Self::Target {
		self
	}
}
impl<'a, T, Rows: Shape, RStride: Stride> IntoView
	for ColRef<'a, T, Rows, RStride>
{
	type Target = Self;

	#[inline]
	fn into_view(self) -> Self::Target {
		self
	}
}
impl<'a, T, Rows: Shape, RStride: Stride> IntoView
	for &'a ColMut<'_, T, Rows, RStride>
{
	type Target = ColRef<'a, T, Rows, RStride>;

	#[inline]
	fn into_view(self) -> Self::Target {
		self.rb()
	}
}
impl<'a, T, Rows: Shape, RStride: Stride> IntoView
	for &'a ColRef<'_, T, Rows, RStride>
{
	type Target = ColRef<'a, T, Rows, RStride>;

	#[inline]
	fn into_view(self) -> Self::Target {
		*self
	}
}
impl<'a, T, Rows: Shape, RStride: Stride> IntoView
	for &'a mut ColMut<'_, T, Rows, RStride>
{
	type Target = ColMut<'a, T, Rows, RStride>;

	#[inline]
	fn into_view(self) -> Self::Target {
		self.rb_mut()
	}
}
impl<'a, T, Rows: Shape, RStride: Stride> IntoView
	for &'a mut ColRef<'_, T, Rows, RStride>
{
	type Target = ColRef<'a, T, Rows, RStride>;

	#[inline]
	fn into_view(self) -> Self::Target {
		*self
	}
}
impl<'a, T, Rows: Shape, DStride: Stride> IntoView
	for diag::DiagMut<'a, T, Rows, DStride>
{
	type Target = Self;

	#[inline]
	fn into_view(self) -> Self::Target {
		self
	}
}
impl<'a, T, Rows: Shape, DStride: Stride> IntoView
	for diag::DiagRef<'a, T, Rows, DStride>
{
	type Target = Self;

	#[inline]
	fn into_view(self) -> Self::Target {
		self
	}
}
impl<'a, T, Rows: Shape, DStride: Stride> IntoView
	for &'a diag::DiagMut<'_, T, Rows, DStride>
{
	type Target = diag::DiagRef<'a, T, Rows, DStride>;

	#[inline]
	fn into_view(self) -> Self::Target {
		self.rb()
	}
}
impl<'a, T, Rows: Shape, DStride: Stride> IntoView
	for &'a diag::DiagRef<'_, T, Rows, DStride>
{
	type Target = diag::DiagRef<'a, T, Rows, DStride>;

	#[inline]
	fn into_view(self) -> Self::Target {
		*self
	}
}
impl<'a, T, Rows: Shape, DStride: Stride> IntoView
	for &'a mut diag::DiagMut<'_, T, Rows, DStride>
{
	type Target = diag::DiagMut<'a, T, Rows, DStride>;

	#[inline]
	fn into_view(self) -> Self::Target {
		self.rb_mut()
	}
}
impl<'a, T, Rows: Shape, DStride: Stride> IntoView
	for &'a mut diag::DiagRef<'_, T, Rows, DStride>
{
	type Target = diag::DiagRef<'a, T, Rows, DStride>;

	#[inline]
	fn into_view(self) -> Self::Target {
		*self
	}
}
impl<'a, T, Cols: Shape, CStride: Stride> IntoView
	for RowMut<'a, T, Cols, CStride>
{
	type Target = Self;

	#[inline]
	fn into_view(self) -> Self::Target {
		self
	}
}
impl<'a, T, Cols: Shape, CStride: Stride> IntoView
	for RowRef<'a, T, Cols, CStride>
{
	type Target = Self;

	#[inline]
	fn into_view(self) -> Self::Target {
		self
	}
}
impl<'a, T, Cols: Shape, CStride: Stride> IntoView
	for &'a RowMut<'_, T, Cols, CStride>
{
	type Target = RowRef<'a, T, Cols, CStride>;

	#[inline]
	fn into_view(self) -> Self::Target {
		self.rb()
	}
}
impl<'a, T, Cols: Shape, CStride: Stride> IntoView
	for &'a RowRef<'_, T, Cols, CStride>
{
	type Target = RowRef<'a, T, Cols, CStride>;

	#[inline]
	fn into_view(self) -> Self::Target {
		*self
	}
}
impl<'a, T, Cols: Shape, CStride: Stride> IntoView
	for &'a mut RowMut<'_, T, Cols, CStride>
{
	type Target = RowMut<'a, T, Cols, CStride>;

	#[inline]
	fn into_view(self) -> Self::Target {
		self.rb_mut()
	}
}
impl<'a, T, Cols: Shape, CStride: Stride> IntoView
	for &'a mut RowRef<'_, T, Cols, CStride>
{
	type Target = RowRef<'a, T, Cols, CStride>;

	#[inline]
	fn into_view(self) -> Self::Target {
		*self
	}
}
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Diag {
	Skip,
	Include,
}
/// matrix layout transformation. used for zipping optimizations
#[derive(Copy, Clone)]
pub enum MatLayoutTransform {
	/// matrix is used as-is
	None,
	/// matrix rows are reversed
	ReverseRows,
	/// matrix is transposed
	Transpose,
	/// matrix is transposed, then rows are reversed
	TransposeReverseRows,
}
/// vector layout transformation. used for zipping optimizations
#[derive(Copy, Clone)]
pub enum VecLayoutTransform {
	/// vector is used as-is
	None,
	/// vector is reversed
	Reverse,
}
/// type with a given matrix shape
pub trait MatIndex {
	type Kind;
	/// type of rows
	type Rows: Copy + Eq + core::fmt::Debug;
	/// type of columns
	type Cols: Copy + Eq + core::fmt::Debug;
	/// returns the number of rows
	fn nrows(this: &Self) -> Self::Rows;
	/// returns the number of columns
	fn ncols(this: &Self) -> Self::Cols;
	/// indexing type
	type Index: Copy;
	/// layout transformation type
	type LayoutTransform: Copy;
	/// item produced by the zip views
	type Item;
	/// matrix type with type erased dimensions
	type Dyn: MatIndex<
			Kind = Self::Kind,
			Dyn = Self::Dyn,
			LayoutTransform = Self::LayoutTransform,
			Item = Self::Item,
			Slice = Self::Slice,
		>;
	type Slice: for<'a> SliceFamily<'a, Self::Item>;
	///
	/// # Safety
	/// the index must lie in bounds of every zipped operand, every operand must
	/// be contiguous along the iteration direction (`is_contiguous`), and
	/// `n_elems` elements must remain from the index in that direction. The
	/// returned slice is not tied to the borrow of `this`, so no element may be
	/// handed out (by this function or `get_unchecked`) more than once.
	/// returns slice at index of length `n_elems`
	unsafe fn get_slice_unchecked<'a>(
		this: &'a mut Self,
		idx: Self::Index,
		n_elems: usize,
	) -> <Self::Slice as SliceFamily<'a, Self::Item>>::Slice;
	///
	/// # Safety
	/// the erased index must denote a valid index of this operand.
	/// converts a type erased index back to its original representation
	unsafe fn from_dyn_idx(idx: <Self::Dyn as MatIndex>::Index) -> Self::Index;
	///
	/// # Safety
	/// `index` must lie in bounds of every zipped operand. The returned item is
	/// not tied to the borrow of `this`, so no element may be handed out (by
	/// this function or `get_slice_unchecked`) more than once.
	/// get the item at the given index, skipping bound checks
	unsafe fn get_unchecked(this: &mut Self, index: Self::Index) -> Self::Item;
	///
	/// # Safety
	/// the slice position must be in bounds and not yet consumed: at least one
	/// element of every operand's slice must remain.
	/// get the item at the given slice position, skipping bound checks
	unsafe fn next_unchecked<'a>(
		slice: &mut <Self::Slice as SliceFamily<'a, Self::Item>>::Slice,
	) -> Self::Item;
	/// checks if the zip matrices are contiguous
	fn is_contiguous(this: &Self) -> bool;
	/// computes the preferred iteration layout of the matrices
	fn preferred_layout(this: &Self) -> Self::LayoutTransform;
	/// applies the layout transformation to the matrices
	fn with_layout(this: Self, layout: Self::LayoutTransform) -> Self::Dyn;
}
pub trait SliceFamily<'a, T, Outlives = &'a Self> {
	type Slice;
}
pub struct Slice<T>(pub T);
pub struct SliceRef<'b, T>(pub &'b T);
pub struct SliceMut<'b, T>(pub &'b mut T);
impl<'a, T> SliceFamily<'a, T> for Slice<T> {
	type Slice = &'a [T];
}
impl<'a, 'b, T> SliceFamily<'a, &'b T> for SliceRef<'b, T> {
	type Slice = &'b [T];
}
impl<'a, 'b, T> SliceFamily<'a, &'b mut T> for SliceMut<'b, T> {
	type Slice = &'b mut [T];
}
impl<'a, T, F: SliceFamily<'a, T>> SliceFamily<'a, Last<T>> for Last<F> {
	type Slice = Last<F::Slice>;
}
impl<'a, T, U, F: SliceFamily<'a, T>, G: SliceFamily<'a, U>>
	SliceFamily<'a, Zip<T, U>> for Zip<F, G>
{
	type Slice = Zip<F::Slice, G::Slice>;
}
/// single matrix
#[derive(Copy, Clone, Debug)]
pub struct LastEq<Kind, Mat>(pub Mat, pub PhantomData<Kind>);
/// single element
#[derive(Copy, Clone, Debug)]
pub struct Last<Mat>(pub Mat);
/// zipped matrices
#[derive(Copy, Clone, Debug)]
pub struct ZipEq<Kind, Head, Tail>(pub Head, pub Tail, PhantomData<Kind>);
/// zipped elements
#[derive(Copy, Clone, Debug)]
pub struct Zip<Head, Tail>(pub Head, pub Tail);
/// single matrix view
impl<
	Kind,
	Rows: Copy + Eq + core::fmt::Debug,
	Cols: Copy + Eq + core::fmt::Debug,
	Head: MatIndex<Kind = Kind, Rows = Rows, Cols = Cols>,
	Tail: MatIndex<Kind = Kind, Rows = Rows, Cols = Cols>,
> ZipEq<Kind, Head, Tail>
{
	/// creates a zip matrix, after asserting that the dimensions match
	#[inline(always)]
	#[track_caller]
	pub fn new(head: Head, tail: Tail) -> Self {
		assert!(all(
			Head::nrows(&head) == Tail::nrows(&tail),
			Head::ncols(&head) == Tail::ncols(&tail),
		));
		Self(head, tail, PhantomData)
	}

	/// creates a zip matrix, assuming that the dimensions match
	// TODO(review): this is a safe function, but the unsafe iteration code
	// (`for_each*`, `map*`) relies on the operands having equal dimensions,
	// which is only `debug_assert!`ed here. In builds without debug
	// assertions, `ZipEq::new_unchecked(zip!(&mut a), zip!(&mut b))
	// .for_each(..)` with `a` 64x64 and `b` 1x1 reads `b` out of bounds
	// (reproduced in release). It probably should be an `unsafe fn`.
	#[inline(always)]
	#[track_caller]
	pub fn new_unchecked(head: Head, tail: Tail) -> Self {
		debug_assert!(all(
			Head::nrows(&head) == Tail::nrows(&tail),
			Head::ncols(&head) == Tail::ncols(&tail),
		));
		Self(head, tail, PhantomData)
	}
}
impl<
	Kind,
	Rows: Copy + Eq + core::fmt::Debug,
	Cols: Copy + Eq + core::fmt::Debug,
	Mat: MatIndex<Rows = Rows, Cols = Cols, Kind = Kind>,
> MatIndex for LastEq<Kind, Mat>
{
	type Cols = Mat::Cols;
	type Dyn = LastEq<<Mat::Dyn as MatIndex>::Kind, Mat::Dyn>;
	type Index = Mat::Index;
	type Item = Last<Mat::Item>;
	type Kind = Mat::Kind;
	type LayoutTransform = Mat::LayoutTransform;
	type Rows = Mat::Rows;
	type Slice = Last<Mat::Slice>;

	#[inline(always)]
	fn nrows(this: &Self) -> Self::Rows {
		Mat::nrows(&this.0)
	}

	#[inline(always)]
	fn ncols(this: &Self) -> Self::Cols {
		Mat::ncols(&this.0)
	}

	#[inline]
	unsafe fn get_slice_unchecked<'a>(
		this: &'a mut Self,
		idx: Self::Index,
		n_elems: usize,
	) -> <Self::Slice as SliceFamily<'a, Self::Item>>::Slice {
		// SAFETY: the caller upholds this function's contract (# Safety on
		// `MatIndex::get_slice_unchecked`), which `LastEq` forwards unchanged
		// to its single operand.
		unsafe { Last(Mat::get_slice_unchecked(&mut this.0, idx, n_elems)) }
	}

	#[inline]
	unsafe fn from_dyn_idx(idx: <Self::Dyn as MatIndex>::Index) -> Self::Index {
		// SAFETY: the caller upholds this function's contract (# Safety on
		// `MatIndex::from_dyn_idx`), which `LastEq` forwards unchanged to its
		// single operand.
		unsafe { Mat::from_dyn_idx(idx) }
	}

	#[inline]
	unsafe fn get_unchecked(this: &mut Self, index: Self::Index) -> Self::Item {
		// SAFETY: the caller upholds this function's contract (# Safety on
		// `MatIndex::get_unchecked`), which `LastEq` forwards unchanged to its
		// single operand.
		unsafe { Last(Mat::get_unchecked(&mut this.0, index)) }
	}

	#[inline]
	unsafe fn next_unchecked<'a>(
		slice: &mut <Self::Slice as SliceFamily<'a, Self::Item>>::Slice,
	) -> Self::Item {
		// SAFETY: the caller upholds this function's contract (# Safety on
		// `MatIndex::next_unchecked`), which `LastEq` forwards unchanged to its
		// single operand.
		unsafe { Last(Mat::next_unchecked(&mut slice.0)) }
	}

	#[inline]
	fn is_contiguous(this: &Self) -> bool {
		Mat::is_contiguous(&this.0)
	}

	#[inline]
	fn preferred_layout(this: &Self) -> Self::LayoutTransform {
		Mat::preferred_layout(&this.0)
	}

	#[inline]
	fn with_layout(this: Self, layout: Self::LayoutTransform) -> Self::Dyn {
		LastEq(Mat::with_layout(this.0, layout), PhantomData)
	}
}
impl<
	Kind,
	Rows: Copy + Eq + core::fmt::Debug,
	Cols: Copy + Eq + core::fmt::Debug,
	L: MatIndex<Kind = Kind, Rows = Rows, Cols = Cols>,
	R: MatIndex<
			Kind = Kind,
			Rows = Rows,
			Cols = Cols,
			Index = L::Index,
			LayoutTransform = L::LayoutTransform,
		>,
> MatIndex for ZipEq<Kind, L, R>
where
	R::Dyn: MatIndex<
			Rows = <L::Dyn as MatIndex>::Rows,
			Cols = <L::Dyn as MatIndex>::Cols,
			Index = <L::Dyn as MatIndex>::Index,
		>,
{
	type Cols = L::Cols;
	type Dyn = ZipEq<<L::Dyn as MatIndex>::Kind, L::Dyn, R::Dyn>;
	type Index = L::Index;
	type Item = Zip<L::Item, R::Item>;
	type Kind = L::Kind;
	type LayoutTransform = L::LayoutTransform;
	type Rows = L::Rows;
	type Slice = Zip<L::Slice, R::Slice>;

	#[inline(always)]
	fn nrows(this: &Self) -> Self::Rows {
		L::nrows(&this.0)
	}

	#[inline(always)]
	fn ncols(this: &Self) -> Self::Cols {
		L::ncols(&this.0)
	}

	#[inline]
	unsafe fn get_slice_unchecked<'a>(
		this: &'a mut Self,
		idx: Self::Index,
		n_elems: usize,
	) -> <Self::Slice as SliceFamily<'a, Self::Item>>::Slice {
		// SAFETY: the caller upholds `get_slice_unchecked`'s contract for the
		// zip. Both operands have the same dimensions (asserted by
		// `ZipEq::new`; see the TODO(review) on `ZipEq::new_unchecked`) and the
		// same index type, and the zip is contiguous only if both are, so the
		// same `idx` and `n_elems` satisfy the contract for each operand; the
		// two calls borrow disjoint fields of `this`.
		unsafe {
			Zip(
				L::get_slice_unchecked(&mut this.0, idx, n_elems),
				R::get_slice_unchecked(&mut this.1, idx, n_elems),
			)
		}
	}

	#[inline]
	unsafe fn from_dyn_idx(idx: <Self::Dyn as MatIndex>::Index) -> Self::Index {
		// SAFETY: `Self::Index` is `L::Index` and the zip's erased index is
		// `L::Dyn`'s; the caller guarantees it denotes a valid index of the
		// zip, which is a valid index of `L`.
		unsafe { L::from_dyn_idx(idx) }
	}

	#[inline]
	unsafe fn get_unchecked(this: &mut Self, index: Self::Index) -> Self::Item {
		// SAFETY: the caller guarantees `index` is in bounds of every zipped
		// operand and is not handed out twice (# Safety on
		// `MatIndex::get_unchecked`); both operands have the same dimensions
		// (see `ZipEq::new`), so this holds for each of them.
		unsafe {
			Zip(
				L::get_unchecked(&mut this.0, index),
				R::get_unchecked(&mut this.1, index),
			)
		}
	}

	#[inline]
	unsafe fn next_unchecked<'a>(
		slice: &mut <Self::Slice as SliceFamily<'a, Self::Item>>::Slice,
	) -> Self::Item {
		// SAFETY: the zipped slice holds one slice per operand, all created by
		// `get_slice_unchecked` with the same `n_elems` and advanced in
		// lockstep here, so the caller's guarantee that an element remains
		// holds for each of them.
		unsafe {
			Zip(
				L::next_unchecked(&mut slice.0),
				R::next_unchecked(&mut slice.1),
			)
		}
	}

	#[inline]
	fn is_contiguous(this: &Self) -> bool {
		L::is_contiguous(&this.0) && R::is_contiguous(&this.1)
	}

	#[inline]
	fn preferred_layout(this: &Self) -> Self::LayoutTransform {
		L::preferred_layout(&this.0)
	}

	#[inline]
	fn with_layout(this: Self, layout: Self::LayoutTransform) -> Self::Dyn {
		ZipEq(
			L::with_layout(this.0, layout),
			R::with_layout(this.1, layout),
			PhantomData,
		)
	}
}
impl<'b, T, Rows: Shape, Cols: Shape, RStride: Stride, CStride: Stride> MatIndex
	for MatMut<'b, T, Rows, Cols, RStride, CStride>
{
	type Cols = Cols;
	type Dyn = MatMut<'b, T, usize, usize, isize, isize>;
	type Index = (Idx<Rows>, Idx<Cols>);
	type Item = &'b mut T;
	type Kind = kind::Mat;
	type LayoutTransform = MatLayoutTransform;
	type Rows = Rows;
	type Slice = SliceMut<'b, T>;

	#[inline]
	fn nrows(this: &Self) -> Self::Rows {
		this.nrows()
	}

	#[inline]
	fn ncols(this: &Self) -> Self::Cols {
		this.ncols()
	}

	#[inline]
	unsafe fn get_slice_unchecked<'a>(
		this: &'a mut Self,
		idx: Self::Index,
		n_elems: usize,
	) -> <Self::Slice as SliceFamily<'a, Self::Item>>::Slice {
		// SAFETY: the caller guarantees `idx` is in bounds (the precondition of
		// `ptr_inbounds_at_mut`), that the row stride is 1 (`is_contiguous`)
		// and that `n_elems` elements remain from `idx` (# Safety on
		// `MatIndex::get_slice_unchecked`); by the view invariant these
		// consecutive elements lie in one allocation valid for reads and writes
		// for `'b`, and since the caller hands out each element at most once,
		// the `&'b mut [T]` is unique.
		unsafe {
			let ptr = this.ptr_inbounds_at_mut(idx.0, idx.1);
			core::slice::from_raw_parts_mut(ptr, n_elems)
		}
	}

	#[inline]
	unsafe fn from_dyn_idx(idx: <Self::Dyn as MatIndex>::Index) -> Self::Index {
		// SAFETY: the caller guarantees the erased index denotes a valid index
		// of this operand, which is the invariant of `Idx<Rows>` and
		// `Idx<Cols>` (for a `Dim<'n>`, `idx < n`).
		unsafe {
			(
				Idx::<Rows>::new_unbound(idx.0),
				Idx::<Cols>::new_unbound(idx.1),
			)
		}
	}

	#[inline]
	unsafe fn get_unchecked(
		this: &mut Self,
		(i, j): Self::Index,
	) -> Self::Item {
		// SAFETY: the caller guarantees `(i, j)` is in bounds (the precondition
		// of `ptr_inbounds_at_mut`); by the view invariant the element is valid
		// for reads and writes for `'b`, and since the caller hands out each
		// element at most once, the `&'b mut T` is unique.
		unsafe {
			let ptr = this.rb().ptr_inbounds_at_mut(i, j);
			&mut *ptr
		}
	}

	#[inline(always)]
	unsafe fn next_unchecked<'a>(
		slice: &mut <Self::Slice as SliceFamily<'a, Self::Item>>::Slice,
	) -> Self::Item {
		// SAFETY: the caller guarantees at least one element of `slice` remains
		// (# Safety on `MatIndex::next_unchecked`), so `split_first_mut`
		// returns `Some`.
		unsafe {
			let (head, tail) =
				core::mem::take(slice).split_first_mut().unwrap_unchecked();
			*slice = tail;
			head
		}
	}

	#[inline(always)]
	fn is_contiguous(this: &Self) -> bool {
		this.row_stride().element_stride() == 1
	}

	#[inline(always)]
	fn preferred_layout(this: &Self) -> Self::LayoutTransform {
		let rs = this.row_stride().element_stride();
		let cs = this.col_stride().element_stride();
		let nrows = this.nrows().unbound();
		let ncols = this.ncols().unbound();
		if nrows > 1 && rs == 1 {
			MatLayoutTransform::None
		} else if nrows > 1 && rs == -1 {
			MatLayoutTransform::ReverseRows
		} else if ncols > 1 && cs == 1 {
			MatLayoutTransform::Transpose
		} else if ncols > 1 && cs == -1 {
			MatLayoutTransform::TransposeReverseRows
		} else {
			MatLayoutTransform::None
		}
	}

	#[inline(always)]
	fn with_layout(this: Self, layout: Self::LayoutTransform) -> Self::Dyn {
		use MatLayoutTransform::*;
		let this = this.as_dyn_mut().as_dyn_stride_mut();
		match layout {
			None => this,
			ReverseRows => this.reverse_rows_mut(),
			Transpose => this.transpose_mut(),
			TransposeReverseRows => this.transpose_mut().reverse_rows_mut(),
		}
	}
}
impl<'b, T, Rows: Shape, Cols: Shape, RStride: Stride, CStride: Stride> MatIndex
	for MatRef<'b, T, Rows, Cols, RStride, CStride>
{
	type Cols = Cols;
	type Dyn = MatRef<'b, T, usize, usize, isize, isize>;
	type Index = (Idx<Rows>, Idx<Cols>);
	type Item = &'b T;
	type Kind = kind::Mat;
	type LayoutTransform = MatLayoutTransform;
	type Rows = Rows;
	type Slice = SliceRef<'b, T>;

	#[inline]
	fn nrows(this: &Self) -> Self::Rows {
		this.nrows()
	}

	#[inline]
	fn ncols(this: &Self) -> Self::Cols {
		this.ncols()
	}

	#[inline]
	unsafe fn get_slice_unchecked<'a>(
		this: &'a mut Self,
		idx: Self::Index,
		n_elems: usize,
	) -> <Self::Slice as SliceFamily<'a, Self::Item>>::Slice {
		// SAFETY: the caller guarantees `idx` is in bounds (the precondition of
		// `ptr_inbounds_at`), that the row stride is 1 (`is_contiguous`) and
		// that `n_elems` elements remain from `idx` (# Safety on
		// `MatIndex::get_slice_unchecked`); by the view invariant these
		// consecutive elements lie in one allocation valid for reads for `'b`.
		unsafe {
			let ptr = this.ptr_inbounds_at(idx.0, idx.1);
			core::slice::from_raw_parts(ptr, n_elems)
		}
	}

	#[inline]
	unsafe fn from_dyn_idx(idx: <Self::Dyn as MatIndex>::Index) -> Self::Index {
		// SAFETY: the caller guarantees the erased index denotes a valid index
		// of this operand, which is the invariant of `Idx<Rows>` and
		// `Idx<Cols>` (for a `Dim<'n>`, `idx < n`).
		unsafe {
			(
				Idx::<Rows>::new_unbound(idx.0),
				Idx::<Cols>::new_unbound(idx.1),
			)
		}
	}

	#[inline]
	unsafe fn get_unchecked(
		this: &mut Self,
		(i, j): Self::Index,
	) -> Self::Item {
		// SAFETY: the caller guarantees `(i, j)` is in bounds (the precondition
		// of `ptr_inbounds_at`); by the view invariant the element is valid for
		// reads for `'b`.
		unsafe {
			let ptr = this.rb().ptr_inbounds_at(i, j);
			&*ptr
		}
	}

	#[inline(always)]
	unsafe fn next_unchecked<'a>(
		slice: &mut <Self::Slice as SliceFamily<'a, Self::Item>>::Slice,
	) -> Self::Item {
		// SAFETY: the caller guarantees at least one element of `slice` remains
		// (# Safety on `MatIndex::next_unchecked`), so `split_first` returns
		// `Some`.
		unsafe {
			let (head, tail) =
				core::mem::take(slice).split_first().unwrap_unchecked();
			*slice = tail;
			head
		}
	}

	#[inline(always)]
	fn is_contiguous(this: &Self) -> bool {
		this.row_stride().element_stride() == 1
	}

	#[inline(always)]
	fn preferred_layout(this: &Self) -> Self::LayoutTransform {
		let rs = this.row_stride().element_stride();
		let cs = this.col_stride().element_stride();
		let nrows = this.nrows().unbound();
		let ncols = this.ncols().unbound();
		if nrows > 1 && rs == 1 {
			MatLayoutTransform::None
		} else if nrows > 1 && rs == -1 {
			MatLayoutTransform::ReverseRows
		} else if ncols > 1 && cs == 1 {
			MatLayoutTransform::Transpose
		} else if ncols > 1 && cs == -1 {
			MatLayoutTransform::TransposeReverseRows
		} else {
			MatLayoutTransform::None
		}
	}

	#[inline(always)]
	fn with_layout(this: Self, layout: Self::LayoutTransform) -> Self::Dyn {
		use MatLayoutTransform::*;
		let this = this.as_dyn().as_dyn_stride();
		match layout {
			None => this,
			ReverseRows => this.reverse_rows(),
			Transpose => this.transpose(),
			TransposeReverseRows => this.transpose().reverse_rows(),
		}
	}
}
impl<'b, T, Len: Shape, Strd: Stride> MatIndex for ColMut<'b, T, Len, Strd> {
	type Cols = ();
	type Dyn = ColMut<'b, T, usize, isize>;
	type Index = Idx<Len>;
	type Item = &'b mut T;
	type Kind = kind::Col;
	type LayoutTransform = VecLayoutTransform;
	type Rows = Len;
	type Slice = SliceMut<'b, T>;

	#[inline]
	fn nrows(this: &Self) -> Self::Rows {
		this.nrows()
	}

	#[inline]
	fn ncols(_: &Self) -> Self::Cols {}

	#[inline]
	unsafe fn get_slice_unchecked<'a>(
		this: &'a mut Self,
		idx: Self::Index,
		n_elems: usize,
	) -> <Self::Slice as SliceFamily<'a, Self::Item>>::Slice {
		// SAFETY: the caller guarantees `idx` is in bounds (the precondition of
		// `ptr_inbounds_at_mut`), that the row stride is 1 (`is_contiguous`)
		// and that `n_elems` elements remain from `idx` (# Safety on
		// `MatIndex::get_slice_unchecked`); by the view invariant these
		// consecutive elements lie in one allocation valid for reads and writes
		// for `'b`, and since the caller hands out each element at most once,
		// the `&'b mut [T]` is unique.
		unsafe {
			let ptr = this.ptr_inbounds_at_mut(idx);
			core::slice::from_raw_parts_mut(ptr, n_elems)
		}
	}

	#[inline]
	unsafe fn from_dyn_idx(idx: <Self::Dyn as MatIndex>::Index) -> Self::Index {
		// SAFETY: the caller guarantees the erased index denotes a valid index
		// of this operand, which is the invariant of `Idx<Len>` (for a
		// `Dim<'n>`, `idx < n`).
		unsafe { Idx::<Len>::new_unbound(idx) }
	}

	#[inline]
	unsafe fn get_unchecked(this: &mut Self, i: Self::Index) -> Self::Item {
		// SAFETY: the caller guarantees `i` is in bounds (the precondition of
		// `ptr_inbounds_at_mut`); by the view invariant the element is valid
		// for reads and writes for `'b`, and since the caller hands out each
		// element at most once, the `&'b mut T` is unique.
		unsafe {
			let ptr = this.ptr_inbounds_at_mut(i);
			&mut *ptr
		}
	}

	#[inline(always)]
	unsafe fn next_unchecked<'a>(
		slice: &mut <Self::Slice as SliceFamily<'a, Self::Item>>::Slice,
	) -> Self::Item {
		// SAFETY: the caller guarantees at least one element of `slice` remains
		// (# Safety on `MatIndex::next_unchecked`), so `split_first_mut`
		// returns `Some`.
		unsafe {
			let (head, tail) =
				core::mem::take(slice).split_first_mut().unwrap_unchecked();
			*slice = tail;
			head
		}
	}

	#[inline(always)]
	fn is_contiguous(this: &Self) -> bool {
		this.row_stride().element_stride() == 1
	}

	#[inline(always)]
	fn preferred_layout(this: &Self) -> Self::LayoutTransform {
		let strd = this.row_stride().element_stride();
		let len = this.nrows().unbound();
		if len > 1 && strd == 1 {
			VecLayoutTransform::None
		} else if len > 1 && strd == -1 {
			VecLayoutTransform::Reverse
		} else {
			VecLayoutTransform::None
		}
	}

	#[inline(always)]
	fn with_layout(this: Self, layout: Self::LayoutTransform) -> Self::Dyn {
		use VecLayoutTransform::*;
		let this = this.as_dyn_rows_mut().as_dyn_stride_mut();
		match layout {
			None => this,
			Reverse => this.reverse_rows_mut(),
		}
	}
}
impl<'b, T, Len: Shape, Strd: Stride> MatIndex
	for diag::DiagMut<'b, T, Len, Strd>
{
	type Cols = Len;
	type Dyn = diag::DiagMut<'b, T, usize, isize>;
	type Index = Idx<Len>;
	type Item = &'b mut T;
	type Kind = kind::Diag;
	type LayoutTransform = VecLayoutTransform;
	type Rows = Len;
	type Slice = SliceMut<'b, T>;

	#[inline]
	fn nrows(this: &Self) -> Self::Rows {
		this.dim()
	}

	#[inline]
	fn ncols(this: &Self) -> Self::Cols {
		this.dim()
	}

	#[inline]
	unsafe fn get_slice_unchecked<'a>(
		this: &'a mut Self,
		idx: Self::Index,
		n_elems: usize,
	) -> <Self::Slice as SliceFamily<'a, Self::Item>>::Slice {
		// SAFETY: the caller guarantees `idx` is in bounds (the precondition of
		// `ptr_inbounds_at_mut`), that the diagonal stride is 1
		// (`is_contiguous`) and that `n_elems` elements remain from `idx` (#
		// Safety on `MatIndex::get_slice_unchecked`); by the view invariant
		// these consecutive elements lie in one allocation valid for reads and
		// writes for `'b`, and since the caller hands out each element at most
		// once, the `&'b mut [T]` is unique. The diagonal's column vector has
		// the same length and stride as the diagonal.
		unsafe {
			let ptr =
				this.rb_mut().column_vector_mut().ptr_inbounds_at_mut(idx);
			core::slice::from_raw_parts_mut(ptr, n_elems)
		}
	}

	#[inline]
	unsafe fn from_dyn_idx(idx: <Self::Dyn as MatIndex>::Index) -> Self::Index {
		// SAFETY: the caller guarantees the erased index denotes a valid index
		// of this operand, which is the invariant of `Idx<Len>` (for a
		// `Dim<'n>`, `idx < n`).
		unsafe { Idx::<Len>::new_unbound(idx) }
	}

	#[inline]
	unsafe fn get_unchecked(this: &mut Self, i: Self::Index) -> Self::Item {
		// SAFETY: the caller guarantees `i` is in bounds (the precondition of
		// `ptr_inbounds_at_mut`); by the view invariant the element is valid
		// for reads and writes for `'b`, and since the caller hands out each
		// element at most once, the `&'b mut T` is unique. The diagonal's
		// column vector has the same length as the diagonal.
		unsafe {
			let ptr = this.rb_mut().column_vector_mut().ptr_inbounds_at_mut(i);
			&mut *ptr
		}
	}

	#[inline(always)]
	unsafe fn next_unchecked<'a>(
		slice: &mut <Self::Slice as SliceFamily<'a, Self::Item>>::Slice,
	) -> Self::Item {
		// SAFETY: the caller guarantees at least one element of `slice` remains
		// (# Safety on `MatIndex::next_unchecked`), so `split_first_mut`
		// returns `Some`.
		unsafe {
			let (head, tail) =
				core::mem::take(slice).split_first_mut().unwrap_unchecked();
			*slice = tail;
			head
		}
	}

	#[inline(always)]
	fn is_contiguous(this: &Self) -> bool {
		this.stride().element_stride() == 1
	}

	#[inline(always)]
	fn preferred_layout(this: &Self) -> Self::LayoutTransform {
		let strd = this.stride().element_stride();
		let len = this.dim().unbound();
		if len > 1 && strd == 1 {
			VecLayoutTransform::None
		} else if len > 1 && strd == -1 {
			VecLayoutTransform::Reverse
		} else {
			VecLayoutTransform::None
		}
	}

	#[inline(always)]
	fn with_layout(this: Self, layout: Self::LayoutTransform) -> Self::Dyn {
		use VecLayoutTransform::*;
		let this = this.as_dyn_mut().as_dyn_stride_mut();
		match layout {
			None => this,
			Reverse => this
				.column_vector_mut()
				.reverse_rows_mut()
				.as_diagonal_mut(),
		}
	}
}
impl<'b, T, Len: Shape, Strd: Stride> MatIndex for RowMut<'b, T, Len, Strd> {
	type Cols = Len;
	type Dyn = RowMut<'b, T, usize, isize>;
	type Index = Idx<Len>;
	type Item = &'b mut T;
	type Kind = kind::Row;
	type LayoutTransform = VecLayoutTransform;
	type Rows = ();
	type Slice = SliceMut<'b, T>;

	#[inline]
	fn nrows(_: &Self) -> Self::Rows {}

	#[inline]
	fn ncols(this: &Self) -> Self::Cols {
		this.ncols()
	}

	#[inline]
	unsafe fn get_slice_unchecked<'a>(
		this: &'a mut Self,
		idx: Self::Index,
		n_elems: usize,
	) -> <Self::Slice as SliceFamily<'a, Self::Item>>::Slice {
		// SAFETY: the caller guarantees `idx` is in bounds (the precondition of
		// `ptr_inbounds_at_mut`), that the column stride is 1 (`is_contiguous`)
		// and that `n_elems` elements remain from `idx` (# Safety on
		// `MatIndex::get_slice_unchecked`); by the view invariant these
		// consecutive elements lie in one allocation valid for reads and writes
		// for `'b`, and since the caller hands out each element at most once,
		// the `&'b mut [T]` is unique.
		unsafe {
			let ptr = this.ptr_inbounds_at_mut(idx);
			core::slice::from_raw_parts_mut(ptr, n_elems)
		}
	}

	#[inline]
	unsafe fn from_dyn_idx(idx: <Self::Dyn as MatIndex>::Index) -> Self::Index {
		// SAFETY: the caller guarantees the erased index denotes a valid index
		// of this operand, which is the invariant of `Idx<Len>` (for a
		// `Dim<'n>`, `idx < n`).
		unsafe { Idx::<Len>::new_unbound(idx) }
	}

	#[inline]
	unsafe fn get_unchecked(this: &mut Self, i: Self::Index) -> Self::Item {
		// SAFETY: the caller guarantees `i` is in bounds (the precondition of
		// `ptr_inbounds_at_mut`); by the view invariant the element is valid
		// for reads and writes for `'b`, and since the caller hands out each
		// element at most once, the `&'b mut T` is unique.
		unsafe {
			let ptr = this.rb().ptr_inbounds_at_mut(i);
			&mut *ptr
		}
	}

	#[inline(always)]
	unsafe fn next_unchecked<'a>(
		slice: &mut <Self::Slice as SliceFamily<'a, Self::Item>>::Slice,
	) -> Self::Item {
		// SAFETY: the caller guarantees at least one element of `slice` remains
		// (# Safety on `MatIndex::next_unchecked`), so `split_first_mut`
		// returns `Some`.
		unsafe {
			let (head, tail) =
				core::mem::take(slice).split_first_mut().unwrap_unchecked();
			*slice = tail;
			head
		}
	}

	#[inline(always)]
	fn is_contiguous(this: &Self) -> bool {
		this.col_stride().element_stride() == 1
	}

	#[inline(always)]
	fn preferred_layout(this: &Self) -> Self::LayoutTransform {
		let strd = this.col_stride().element_stride();
		let len = this.ncols().unbound();
		if len > 1 && strd == 1 {
			VecLayoutTransform::None
		} else if len > 1 && strd == -1 {
			VecLayoutTransform::Reverse
		} else {
			VecLayoutTransform::None
		}
	}

	#[inline(always)]
	fn with_layout(this: Self, layout: Self::LayoutTransform) -> Self::Dyn {
		use VecLayoutTransform::*;
		let this = this.as_dyn_cols_mut().as_dyn_stride_mut();
		match layout {
			None => this,
			Reverse => this.reverse_cols_mut(),
		}
	}
}
impl<'b, T, Len: Shape, Strd: Stride> MatIndex for ColRef<'b, T, Len, Strd> {
	type Cols = ();
	type Dyn = ColRef<'b, T, usize, isize>;
	type Index = Idx<Len>;
	type Item = &'b T;
	type Kind = kind::Col;
	type LayoutTransform = VecLayoutTransform;
	type Rows = Len;
	type Slice = SliceRef<'b, T>;

	#[inline]
	fn nrows(this: &Self) -> Self::Rows {
		this.nrows()
	}

	#[inline]
	fn ncols(_: &Self) -> Self::Cols {}

	#[inline]
	unsafe fn get_slice_unchecked<'a>(
		this: &'a mut Self,
		idx: Self::Index,
		n_elems: usize,
	) -> <Self::Slice as SliceFamily<'a, Self::Item>>::Slice {
		// SAFETY: the caller guarantees `idx` is in bounds (the precondition of
		// `ptr_inbounds_at`), that the row stride is 1 (`is_contiguous`) and
		// that `n_elems` elements remain from `idx` (# Safety on
		// `MatIndex::get_slice_unchecked`); by the view invariant these
		// consecutive elements lie in one allocation valid for reads for `'b`.
		unsafe {
			let ptr = this.ptr_inbounds_at(idx);
			core::slice::from_raw_parts(ptr, n_elems)
		}
	}

	#[inline]
	unsafe fn from_dyn_idx(idx: <Self::Dyn as MatIndex>::Index) -> Self::Index {
		// SAFETY: the caller guarantees the erased index denotes a valid index
		// of this operand, which is the invariant of `Idx<Len>` (for a
		// `Dim<'n>`, `idx < n`).
		unsafe { Idx::<Len>::new_unbound(idx) }
	}

	#[inline]
	unsafe fn get_unchecked(this: &mut Self, i: Self::Index) -> Self::Item {
		// SAFETY: the caller guarantees `i` is in bounds (the precondition of
		// `ptr_inbounds_at`); by the view invariant the element is valid for
		// reads for `'b`.
		unsafe {
			let ptr = this.rb().ptr_inbounds_at(i);
			&*ptr
		}
	}

	#[inline(always)]
	unsafe fn next_unchecked<'a>(
		slice: &mut <Self::Slice as SliceFamily<'a, Self::Item>>::Slice,
	) -> Self::Item {
		// SAFETY: the caller guarantees at least one element of `slice` remains
		// (# Safety on `MatIndex::next_unchecked`), so `split_first` returns
		// `Some`.
		unsafe {
			let (head, tail) =
				core::mem::take(slice).split_first().unwrap_unchecked();
			*slice = tail;
			head
		}
	}

	#[inline(always)]
	fn is_contiguous(this: &Self) -> bool {
		this.row_stride().element_stride() == 1
	}

	#[inline(always)]
	fn preferred_layout(this: &Self) -> Self::LayoutTransform {
		let strd = this.row_stride().element_stride();
		let len = this.nrows().unbound();
		if len > 1 && strd == 1 {
			VecLayoutTransform::None
		} else if len > 1 && strd == -1 {
			VecLayoutTransform::Reverse
		} else {
			VecLayoutTransform::None
		}
	}

	#[inline(always)]
	fn with_layout(this: Self, layout: Self::LayoutTransform) -> Self::Dyn {
		use VecLayoutTransform::*;
		let this = this.as_dyn_rows().as_dyn_stride();
		match layout {
			None => this,
			Reverse => this.reverse_rows(),
		}
	}
}
impl<'b, T, Len: Shape, Strd: Stride> MatIndex
	for diag::DiagRef<'b, T, Len, Strd>
{
	type Cols = Len;
	type Dyn = diag::DiagRef<'b, T, usize, isize>;
	type Index = Idx<Len>;
	type Item = &'b T;
	type Kind = kind::Diag;
	type LayoutTransform = VecLayoutTransform;
	type Rows = Len;
	type Slice = SliceRef<'b, T>;

	#[inline]
	fn nrows(this: &Self) -> Self::Rows {
		this.dim()
	}

	#[inline]
	fn ncols(this: &Self) -> Self::Cols {
		this.dim()
	}

	#[inline]
	unsafe fn get_slice_unchecked<'a>(
		this: &'a mut Self,
		idx: Self::Index,
		n_elems: usize,
	) -> <Self::Slice as SliceFamily<'a, Self::Item>>::Slice {
		// SAFETY: the caller guarantees `idx` is in bounds (the precondition of
		// `ptr_inbounds_at`), that the diagonal stride is 1 (`is_contiguous`)
		// and that `n_elems` elements remain from `idx` (# Safety on
		// `MatIndex::get_slice_unchecked`); by the view invariant these
		// consecutive elements lie in one allocation valid for reads for `'b`.
		// The diagonal's column vector has the same length and stride as the
		// diagonal.
		unsafe {
			let ptr = this.column_vector().ptr_inbounds_at(idx);
			core::slice::from_raw_parts(ptr, n_elems)
		}
	}

	#[inline]
	unsafe fn from_dyn_idx(idx: <Self::Dyn as MatIndex>::Index) -> Self::Index {
		// SAFETY: the caller guarantees the erased index denotes a valid index
		// of this operand, which is the invariant of `Idx<Len>` (for a
		// `Dim<'n>`, `idx < n`).
		unsafe { Idx::<Len>::new_unbound(idx) }
	}

	#[inline]
	unsafe fn get_unchecked(this: &mut Self, i: Self::Index) -> Self::Item {
		// SAFETY: the caller guarantees `i` is in bounds (the precondition of
		// `ptr_inbounds_at`); by the view invariant the element is valid for
		// reads for `'b`. The diagonal's column vector has the same length as
		// the diagonal.
		unsafe {
			let ptr = this.column_vector().ptr_inbounds_at(i);
			&*ptr
		}
	}

	#[inline(always)]
	unsafe fn next_unchecked<'a>(
		slice: &mut <Self::Slice as SliceFamily<'a, Self::Item>>::Slice,
	) -> Self::Item {
		// SAFETY: the caller guarantees at least one element of `slice` remains
		// (# Safety on `MatIndex::next_unchecked`), so `split_first` returns
		// `Some`.
		unsafe {
			let (head, tail) =
				core::mem::take(slice).split_first().unwrap_unchecked();
			*slice = tail;
			head
		}
	}

	#[inline(always)]
	fn is_contiguous(this: &Self) -> bool {
		this.stride().element_stride() == 1
	}

	#[inline(always)]
	fn preferred_layout(this: &Self) -> Self::LayoutTransform {
		let strd = this.stride().element_stride();
		let len = this.dim().unbound();
		if len > 1 && strd == 1 {
			VecLayoutTransform::None
		} else if len > 1 && strd == -1 {
			VecLayoutTransform::Reverse
		} else {
			VecLayoutTransform::None
		}
	}

	#[inline(always)]
	fn with_layout(this: Self, layout: Self::LayoutTransform) -> Self::Dyn {
		use VecLayoutTransform::*;
		let this = this.as_dyn().as_dyn_stride();
		match layout {
			None => this,
			Reverse => this.column_vector().reverse_rows().as_diagonal(),
		}
	}
}
impl<'b, T, Len: Shape, Strd: Stride> MatIndex for RowRef<'b, T, Len, Strd> {
	type Cols = Len;
	type Dyn = RowRef<'b, T, usize, isize>;
	type Index = Idx<Len>;
	type Item = &'b T;
	type Kind = kind::Row;
	type LayoutTransform = VecLayoutTransform;
	type Rows = ();
	type Slice = SliceRef<'b, T>;

	#[inline]
	fn nrows(_: &Self) -> Self::Rows {}

	#[inline]
	fn ncols(this: &Self) -> Self::Cols {
		this.ncols()
	}

	#[inline]
	unsafe fn get_slice_unchecked<'a>(
		this: &'a mut Self,
		idx: Self::Index,
		n_elems: usize,
	) -> <Self::Slice as SliceFamily<'a, Self::Item>>::Slice {
		// SAFETY: the caller guarantees `idx` is in bounds (the precondition of
		// `ptr_inbounds_at`), that the column stride is 1 (`is_contiguous`) and
		// that `n_elems` elements remain from `idx` (# Safety on
		// `MatIndex::get_slice_unchecked`); by the view invariant these
		// consecutive elements lie in one allocation valid for reads for `'b`.
		unsafe {
			let ptr = this.ptr_inbounds_at(idx);
			core::slice::from_raw_parts(ptr, n_elems)
		}
	}

	#[inline]
	unsafe fn from_dyn_idx(idx: <Self::Dyn as MatIndex>::Index) -> Self::Index {
		// SAFETY: the caller guarantees the erased index denotes a valid index
		// of this operand, which is the invariant of `Idx<Len>` (for a
		// `Dim<'n>`, `idx < n`).
		unsafe { Idx::<Len>::new_unbound(idx) }
	}

	#[inline]
	unsafe fn get_unchecked(this: &mut Self, i: Self::Index) -> Self::Item {
		// SAFETY: the caller guarantees `i` is in bounds (the precondition of
		// `ptr_inbounds_at`); by the view invariant the element is valid for
		// reads for `'b`.
		unsafe {
			let ptr = this.rb().ptr_inbounds_at(i);
			&*ptr
		}
	}

	#[inline(always)]
	unsafe fn next_unchecked<'a>(
		slice: &mut <Self::Slice as SliceFamily<'a, Self::Item>>::Slice,
	) -> Self::Item {
		// SAFETY: the caller guarantees at least one element of `slice` remains
		// (# Safety on `MatIndex::next_unchecked`), so `split_first` returns
		// `Some`.
		unsafe {
			let (head, tail) =
				core::mem::take(slice).split_first().unwrap_unchecked();
			*slice = tail;
			head
		}
	}

	#[inline(always)]
	fn is_contiguous(this: &Self) -> bool {
		this.col_stride().element_stride() == 1
	}

	#[inline(always)]
	fn preferred_layout(this: &Self) -> Self::LayoutTransform {
		let strd = this.col_stride().element_stride();
		let len = this.ncols().unbound();
		if len > 1 && strd == 1 {
			VecLayoutTransform::None
		} else if len > 1 && strd == -1 {
			VecLayoutTransform::Reverse
		} else {
			VecLayoutTransform::None
		}
	}

	#[inline(always)]
	fn with_layout(this: Self, layout: Self::LayoutTransform) -> Self::Dyn {
		use VecLayoutTransform::*;
		let this = this.as_dyn_cols().as_dyn_stride();
		match layout {
			None => this,
			Reverse => this.reverse_cols(),
		}
	}
}
#[inline(always)]
fn annotate_noalias_mat<Z: MatIndex>(
	f: &mut impl FnMut(<Z as MatIndex>::Item),
	mut slice: <Z::Slice as SliceFamily<'_, Z::Item>>::Slice,
	i_begin: usize,
	i_end: usize,
	_j: usize,
) {
	for _ in i_begin..i_end {
		// SAFETY: callers pass a slice of `i_end - i_begin` elements obtained
		// from `get_slice_unchecked`, so an element remains in each of the
		// `i_end - i_begin` iterations.
		unsafe { f(Z::next_unchecked(&mut slice)) };
	}
}
#[inline(always)]
fn annotate_noalias_mat_with_index<
	const TRANSPOSE: bool,
	const REVERSE_ROWS: bool,
	Z: MatIndex<Index = (RowIdx, ColIdx)>,
	RowIdx,
	ColIdx,
>(
	f: &mut impl FnMut(RowIdx, ColIdx, <Z as MatIndex>::Item),
	mut slice: <Z::Slice as SliceFamily<'_, Z::Item>>::Slice,
	i_begin: usize,
	i_end: usize,
	j: usize,
) where
	Z::Dyn: MatIndex<Index = (usize, usize)>,
{
	if const { !TRANSPOSE } {
		if const { !REVERSE_ROWS } {
			for i in i_begin..i_end {
				// SAFETY: callers pass a slice of `i_end - i_begin` elements
				// obtained from `get_slice_unchecked`, so an element remains in
				// each iteration; `[i_begin, i_end)` is the range of rows of
				// `Z` that the slice covers in column `j`, so the index passed
				// to `from_dyn_idx` is a valid index of `Z`.
				unsafe {
					let (ii, jj) = Z::from_dyn_idx((i, j));
					f(ii, jj, Z::next_unchecked(&mut slice))
				};
			}
		} else {
			for i in i_begin..i_end {
				// SAFETY: callers pass a slice of `i_end - i_begin` elements
				// obtained from `get_slice_unchecked`, so an element remains in
				// each iteration; `[i_begin, i_end)` is the range of rows of
				// `Z` that the slice covers in column `j`, so the index,
				// mirrored inside that range, passed to `from_dyn_idx` is a
				// valid index of `Z`.
				unsafe {
					let (ii, jj) =
						Z::from_dyn_idx((i_begin + (i_end - i - 1), j));
					f(ii, jj, Z::next_unchecked(&mut slice))
				};
			}
		}
	} else {
		if const { !REVERSE_ROWS } {
			for i in i_begin..i_end {
				// SAFETY: callers pass a slice of `i_end - i_begin` elements
				// obtained from `get_slice_unchecked`, so an element remains in
				// each iteration; `[i_begin, i_end)` is the range of rows of
				// `Z`'s transpose that the slice covers in column `j` (row `j`
				// of `Z`), so the index passed to `from_dyn_idx` is a valid
				// index of `Z`.
				unsafe {
					let (ii, jj) = Z::from_dyn_idx((j, i));
					f(ii, jj, Z::next_unchecked(&mut slice))
				};
			}
		} else {
			for i in i_begin..i_end {
				// SAFETY: callers pass a slice of `i_end - i_begin` elements
				// obtained from `get_slice_unchecked`, so an element remains in
				// each iteration; `[i_begin, i_end)` is the range of rows of
				// `Z`'s transpose that the slice covers in column `j` (row `j`
				// of `Z`), so the index, mirrored inside that range, passed to
				// `from_dyn_idx` is a valid index of `Z`.
				unsafe {
					let (ii, jj) =
						Z::from_dyn_idx((j, i_begin + (i_end - i - 1)));
					f(ii, jj, Z::next_unchecked(&mut slice))
				};
			}
		}
	}
}
#[inline(always)]
fn annotate_noalias_col<Z: MatIndex>(
	f: &mut impl FnMut(<Z as MatIndex>::Item),
	mut slice: <Z::Slice as SliceFamily<'_, Z::Item>>::Slice,
	i_begin: usize,
	i_end: usize,
) {
	for _ in i_begin..i_end {
		// SAFETY: callers pass a slice of `i_end - i_begin` elements obtained
		// from `get_slice_unchecked`, so an element remains in each of the
		// `i_end - i_begin` iterations.
		unsafe { f(Z::next_unchecked(&mut slice)) };
	}
}
#[inline(always)]
fn annotate_noalias_col_with_index<Z: MatIndex<Index = Idx>, Idx>(
	f: &mut impl FnMut(Idx, <Z as MatIndex>::Item),
	mut slice: <Z::Slice as SliceFamily<'_, Z::Item>>::Slice,
	i_begin: usize,
	i_end: usize,
	reverse: bool,
) where
	Z::Dyn: MatIndex<Item = Z::Item, Index = usize>,
{
	if !reverse {
		for i in i_begin..i_end {
			// SAFETY: callers pass the slice of all `i_end - i_begin` elements
			// from `get_slice_unchecked`, so an element remains in each
			// iteration, and the index passed to `from_dyn_idx` lies in
			// `[i_begin, i_end)`, the range of `Z` (which has the length of
			// `Z::Dyn`).
			unsafe {
				let ii = Z::from_dyn_idx(i);
				f(ii, Z::next_unchecked(&mut slice))
			};
		}
	} else {
		for i in i_begin..i_end {
			// SAFETY: callers pass the slice of all `i_end - i_begin` elements
			// from `get_slice_unchecked`, so an element remains in each
			// iteration, and the index, mirrored inside that range, passed to
			// `from_dyn_idx` lies in `[i_begin, i_end)`, the range of `Z`
			// (which has the length of `Z::Dyn`).
			unsafe {
				let ii = Z::from_dyn_idx(i_begin + (i_end - i - 1));
				f(ii, Z::next_unchecked(&mut slice))
			};
		}
	}
}
#[inline(always)]
fn for_each_mat<Z: MatIndex>(z: Z, mut f: impl FnMut(<Z as MatIndex>::Item))
where
	Z::Dyn: MatIndex<
			Item = Z::Item,
			Slice = Z::Slice,
			Rows = usize,
			Cols = usize,
			Index = (usize, usize),
		>,
{
	let layout = Z::preferred_layout(&z);
	let mut z = Z::with_layout(z, layout);
	let m = Z::Dyn::nrows(&z);
	let n = Z::Dyn::ncols(&z);
	if m == 0 || n == 0 {
		return;
	}
	// SAFETY: `(m, n)` are the dimensions of `z`, which are those of every
	// zipped operand (`ZipEq::new` asserts they match; see the TODO(review) on
	// `ZipEq::new_unchecked`), so every `(i, j)` with `i < m`, `j < n` is in
	// bounds. The contiguous branch takes, for each column, the slice of all
	// `m` rows from row 0 (row stride 1, checked by `is_contiguous`); each
	// element is handed out exactly once.
	unsafe {
		if Z::Dyn::is_contiguous(&z) {
			for j in 0..n {
				annotate_noalias_mat::<Z::Dyn>(
					&mut f,
					Z::Dyn::get_slice_unchecked(&mut z, (0, j), m),
					0,
					m,
					j,
				);
			}
		} else {
			for j in 0..n {
				for i in 0..m {
					f(Z::Dyn::get_unchecked(&mut z, (i, j)))
				}
			}
		}
	}
}
#[inline(always)]
fn for_each_mat_with_index<
	RowIdx,
	ColIdx,
	Z: MatIndex<Index = (RowIdx, ColIdx), LayoutTransform = MatLayoutTransform>,
>(
	z: Z,
	mut f: impl FnMut(RowIdx, ColIdx, <Z as MatIndex>::Item),
) where
	Z::Dyn: MatIndex<
			Rows = usize,
			Cols = usize,
			Index = (usize, usize),
			Slice = Z::Slice,
			Item = Z::Item,
		>,
{
	let layout = Z::preferred_layout(&z);
	let mut z = Z::with_layout(z, layout);
	let m = Z::Dyn::nrows(&z);
	let n = Z::Dyn::ncols(&z);
	if m == 0 || n == 0 {
		return;
	}
	match layout {
		// SAFETY: `(m, n)` are the dimensions of `z`, which are those of every
		// zipped operand (`ZipEq::new` asserts they match; see the TODO(review)
		// on `ZipEq::new_unchecked`), so every `(i, j)` with `i < m`, `j < n`
		// is in bounds. The contiguous branch takes, for each column, the slice
		// of all `m` rows from row 0 (row stride 1, checked by
		// `is_contiguous`); each element is handed out exactly once.
		// `from_dyn_idx` receives `(i, j)`, which undoes the layout transform
		// `with_layout` applied, so it is a valid index of `Z`.
		MatLayoutTransform::None => unsafe {
			if Z::Dyn::is_contiguous(&z) {
				for j in 0..n {
					annotate_noalias_mat_with_index::<false, false, Z, _, _>(
						&mut f,
						Z::Dyn::get_slice_unchecked(&mut z, (0, j), m),
						0,
						m,
						j,
					);
				}
			} else {
				for j in 0..n {
					for i in 0..m {
						let (ii, jj) = Z::from_dyn_idx((i, j));
						f(ii, jj, Z::Dyn::get_unchecked(&mut z, (i, j)))
					}
				}
			}
		},
		// SAFETY: `(m, n)` are the dimensions of `z`, which are those of every
		// zipped operand (`ZipEq::new` asserts they match; see the TODO(review)
		// on `ZipEq::new_unchecked`), so every `(i, j)` with `i < m`, `j < n`
		// is in bounds. The contiguous branch takes, for each column, the slice
		// of all `m` rows from row 0 (row stride 1, checked by
		// `is_contiguous`); each element is handed out exactly once.
		// `from_dyn_idx` receives `(m - i - 1, j)`, which undoes the layout
		// transform `with_layout` applied, so it is a valid index of `Z`.
		MatLayoutTransform::ReverseRows => unsafe {
			if Z::Dyn::is_contiguous(&z) {
				for j in 0..n {
					annotate_noalias_mat_with_index::<false, true, Z, _, _>(
						&mut f,
						Z::Dyn::get_slice_unchecked(&mut z, (0, j), m),
						0,
						m,
						j,
					);
				}
			} else {
				for j in 0..n {
					for i in 0..m {
						let (ii, jj) = Z::from_dyn_idx((m - i - 1, j));
						f(ii, jj, Z::Dyn::get_unchecked(&mut z, (i, j)))
					}
				}
			}
		},
		// SAFETY: `(m, n)` are the dimensions of `z`, which are those of every
		// zipped operand (`ZipEq::new` asserts they match; see the TODO(review)
		// on `ZipEq::new_unchecked`), so every `(i, j)` with `i < m`, `j < n`
		// is in bounds. The contiguous branch takes, for each column, the slice
		// of all `m` rows from row 0 (row stride 1, checked by
		// `is_contiguous`); each element is handed out exactly once.
		// `from_dyn_idx` receives `(j, i)`, which undoes the layout transform
		// `with_layout` applied, so it is a valid index of `Z`.
		MatLayoutTransform::Transpose => unsafe {
			if Z::Dyn::is_contiguous(&z) {
				for j in 0..n {
					annotate_noalias_mat_with_index::<true, false, Z, _, _>(
						&mut f,
						Z::Dyn::get_slice_unchecked(&mut z, (0, j), m),
						0,
						m,
						j,
					);
				}
			} else {
				for j in 0..n {
					for i in 0..m {
						let (ii, jj) = Z::from_dyn_idx((j, i));
						f(ii, jj, Z::Dyn::get_unchecked(&mut z, (i, j)))
					}
				}
			}
		},
		// SAFETY: `(m, n)` are the dimensions of `z`, which are those of every
		// zipped operand (`ZipEq::new` asserts they match; see the TODO(review)
		// on `ZipEq::new_unchecked`), so every `(i, j)` with `i < m`, `j < n`
		// is in bounds. The contiguous branch takes, for each column, the slice
		// of all `m` rows from row 0 (row stride 1, checked by
		// `is_contiguous`); each element is handed out exactly once.
		// `from_dyn_idx` receives `(j, m - i - 1)`, which undoes the layout
		// transform `with_layout` applied, so it is a valid index of `Z`.
		MatLayoutTransform::TransposeReverseRows => unsafe {
			if Z::Dyn::is_contiguous(&z) {
				for j in 0..n {
					annotate_noalias_mat_with_index::<true, true, Z, _, _>(
						&mut f,
						Z::Dyn::get_slice_unchecked(&mut z, (0, j), m),
						0,
						m,
						j,
					);
				}
			} else {
				for j in 0..n {
					for i in 0..m {
						let (ii, jj) = Z::from_dyn_idx((j, m - i - 1));
						f(ii, jj, Z::Dyn::get_unchecked(&mut z, (i, j)))
					}
				}
			}
		},
	}
}
#[inline(always)]
fn for_each_mat_triangular_lower_with_index<
	RowIdx,
	ColIdx,
	Z: MatIndex<Index = (RowIdx, ColIdx), LayoutTransform = MatLayoutTransform>,
>(
	z: Z,
	diag: Diag,
	mut f: impl FnMut(RowIdx, ColIdx, <Z as MatIndex>::Item),
) where
	Z::Dyn: MatIndex<
			Rows = usize,
			Cols = usize,
			Index = (usize, usize),
			Item = Z::Item,
			Slice = Z::Slice,
		>,
{
	let layout = Z::preferred_layout(&z);
	let mut z = Z::with_layout(z, layout);
	let m = Z::Dyn::nrows(&z);
	let n = Z::Dyn::ncols(&z);
	let strict = match diag {
		Diag::Skip => true,
		Diag::Include => false,
	};
	let strict = strict as usize;
	if m == 0 || n == 0 {
		return;
	}
	match layout {
		// SAFETY: TODO(review): for `n > m + 1 - strict` (a wide matrix; for
		// the upper triangle under `Transpose`, a tall one) some columns have
		// `start = j + strict > end = m`. `start == end` does not catch it: the
		// non-contiguous loop is empty, but the contiguous branch calls
		// `get_slice_unchecked` at row `start >= m` with length `end - start`,
		// which wraps to nearly `usize::MAX` in builds without overflow checks,
		// violating its contract (no element is read, as the index range is
		// empty). Otherwise `(m, n)` are the dimensions of `z`, which are those
		// of every zipped operand (see `ZipEq::new`). In column `j`, `start = j
		// + strict <= end = m` when the TODO above does not apply, so each `(i,
		// j)` with `start <= i < end` is in bounds and the contiguous branch
		// (row stride 1, checked by `is_contiguous`) slices `end - start` rows
		// from `(start, j)`; `from_dyn_idx` receives the index with the layout
		// transform undone, which is a valid index of `Z`; each element is
		// handed out once.
		MatLayoutTransform::None => unsafe {
			if Z::Dyn::is_contiguous(&z) {
				for j in 0..n {
					let start = j + strict;
					let end = m;
					if start == end {
						continue;
					}
					annotate_noalias_mat_with_index::<false, false, Z, _, _>(
						&mut f,
						Z::Dyn::get_slice_unchecked(
							&mut z,
							(start, j),
							end - start,
						),
						start,
						end,
						j,
					);
				}
			} else {
				for j in 0..n {
					let start = j + strict;
					let end = m;
					if start == end {
						continue;
					}
					for i in start..end {
						let (ii, jj) = Z::from_dyn_idx((i, j));
						f(ii, jj, Z::Dyn::get_unchecked(&mut z, (i, j)))
					}
				}
			}
		},
		// SAFETY: `(m, n)` are the dimensions of `z`, which are those of every
		// zipped operand (see `ZipEq::new`). In column `j`, `j < min(m, n)`, so
		// `end = m - j - strict` does not underflow and `0 <= end <= m`, so
		// each `(i, j)` with `start <= i < end` is in bounds and the contiguous
		// branch (row stride 1, checked by `is_contiguous`) slices `end -
		// start` rows from `(start, j)`; `from_dyn_idx` receives the index with
		// the layout transform undone, which is a valid index of `Z`; each
		// element is handed out once.
		MatLayoutTransform::ReverseRows => unsafe {
			if Z::Dyn::is_contiguous(&z) {
				for j in 0..Ord::min(m, n) {
					let start = 0;
					let end = m - j - strict;
					if start == end {
						continue;
					}
					annotate_noalias_mat_with_index::<false, true, Z, _, _>(
						&mut f,
						Z::Dyn::get_slice_unchecked(
							&mut z,
							(start, j),
							end - start,
						),
						j + strict + start,
						j + strict + end,
						j,
					);
				}
			} else {
				for j in 0..Ord::min(m, n) {
					let start = 0;
					let end = m - j - strict;
					if start == end {
						continue;
					}
					for i in start..end {
						let (ii, jj) = Z::from_dyn_idx((m - i - 1, j));
						f(ii, jj, Z::Dyn::get_unchecked(&mut z, (i, j)))
					}
				}
			}
		},
		// SAFETY: `(m, n)` are the dimensions of `z`, which are those of every
		// zipped operand (see `ZipEq::new`). In column `j`, `end = min(m, j + 1
		// - strict) <= m`, so each `(i, j)` with `start <= i < end` is in
		// bounds and the contiguous branch (row stride 1, checked by
		// `is_contiguous`) slices `end - start` rows from `(start, j)`;
		// `from_dyn_idx` receives the index with the layout transform undone,
		// which is a valid index of `Z`; each element is handed out once.
		MatLayoutTransform::Transpose => unsafe {
			if Z::Dyn::is_contiguous(&z) {
				for j in 0..n {
					let start = 0;
					let end = Ord::min(m, j + (1 - strict));
					if start == end {
						continue;
					}
					annotate_noalias_mat_with_index::<true, false, Z, _, _>(
						&mut f,
						Z::Dyn::get_slice_unchecked(
							&mut z,
							(0, j),
							end - start,
						),
						start,
						end,
						j,
					);
				}
			} else {
				for j in 0..n {
					let start = 0;
					let end = Ord::min(m, j + (1 - strict));
					if start == end {
						continue;
					}
					for i in start..end {
						let (ii, jj) = Z::from_dyn_idx((j, i));
						f(ii, jj, Z::Dyn::get_unchecked(&mut z, (i, j)))
					}
				}
			}
		},
		// SAFETY: `(m, n)` are the dimensions of `z`, which are those of every
		// zipped operand (see `ZipEq::new`). In column `j`, `start = m - min(j
		// + 1 - strict, m)` lies in `[0, m]`, so each `(i, j)` with `start <= i
		// < end` is in bounds and the contiguous branch (row stride 1, checked
		// by `is_contiguous`) slices `end - start` rows from `(start, j)`;
		// `from_dyn_idx` receives the index with the layout transform undone,
		// which is a valid index of `Z`; each element is handed out once.
		MatLayoutTransform::TransposeReverseRows => unsafe {
			if Z::Dyn::is_contiguous(&z) {
				for j in 0..n {
					let start = m - Ord::min(j + (1 - strict), m);
					let end = m;
					if start == end {
						continue;
					}
					annotate_noalias_mat_with_index::<true, true, Z, _, _>(
						&mut f,
						Z::Dyn::get_slice_unchecked(
							&mut z,
							(start, j),
							end - start,
						),
						0,
						end - start,
						j,
					);
				}
			} else {
				for j in 0..n {
					let start = m - Ord::min(j + (1 - strict), m);
					let end = m;
					if start == end {
						continue;
					}
					for i in start..end {
						let (ii, jj) = Z::from_dyn_idx((j, m - i - 1));
						f(ii, jj, Z::Dyn::get_unchecked(&mut z, (i, j)))
					}
				}
			}
		},
	}
}
#[inline(always)]
fn for_each_mat_triangular_upper_with_index<
	RowIdx,
	ColIdx,
	Z: MatIndex<Index = (RowIdx, ColIdx), LayoutTransform = MatLayoutTransform>,
>(
	z: Z,
	diag: Diag,
	mut f: impl FnMut(RowIdx, ColIdx, <Z as MatIndex>::Item),
) where
	Z::Dyn: MatIndex<
			Rows = usize,
			Cols = usize,
			Index = (usize, usize),
			Item = Z::Item,
			Slice = Z::Slice,
		>,
{
	let layout = Z::preferred_layout(&z);
	let mut z = Z::with_layout(z, layout);
	let m = Z::Dyn::nrows(&z);
	let n = Z::Dyn::ncols(&z);
	let strict = match diag {
		Diag::Skip => true,
		Diag::Include => false,
	};
	let strict = strict as usize;
	if m == 0 || n == 0 {
		return;
	}
	match layout {
		// SAFETY: `(m, n)` are the dimensions of `z`, which are those of every
		// zipped operand (see `ZipEq::new`). In column `j`, `end = min(m, j + 1
		// - strict) <= m`, so each `(i, j)` with `start <= i < end` is in
		// bounds and the contiguous branch (row stride 1, checked by
		// `is_contiguous`) slices `end - start` rows from `(start, j)`;
		// `from_dyn_idx` receives the index with the layout transform undone,
		// which is a valid index of `Z`; each element is handed out once.
		MatLayoutTransform::None => unsafe {
			if Z::Dyn::is_contiguous(&z) {
				for j in 0..n {
					let start = 0;
					let end = Ord::min(m, j + (1 - strict));
					if start == end {
						continue;
					}
					annotate_noalias_mat_with_index::<false, false, Z, _, _>(
						&mut f,
						Z::Dyn::get_slice_unchecked(
							&mut z,
							(start, j),
							end - start,
						),
						start,
						end,
						j,
					);
				}
			} else {
				for j in 0..n {
					let start = 0;
					let end = Ord::min(m, j + (1 - strict));
					if start == end {
						continue;
					}
					for i in start..end {
						let (ii, jj) = Z::from_dyn_idx((i, j));
						f(ii, jj, Z::Dyn::get_unchecked(&mut z, (i, j)))
					}
				}
			}
		},
		// SAFETY: `(m, n)` are the dimensions of `z`, which are those of every
		// zipped operand (see `ZipEq::new`). In column `j`, `j < min(m, n)` and
		// `start = m - min(j + 1 - strict, m)` lies in `[0, m]`, so each `(i,
		// j)` with `start <= i < end` is in bounds and the contiguous branch
		// (row stride 1, checked by `is_contiguous`) slices `end - start` rows
		// from `(start, j)`; `from_dyn_idx` receives the index with the layout
		// transform undone, which is a valid index of `Z`; each element is
		// handed out once.
		MatLayoutTransform::ReverseRows => unsafe {
			if Z::Dyn::is_contiguous(&z) {
				for j in 0..Ord::min(m, n) {
					let start = m - Ord::min(j + (1 - strict), m);
					let end = m;
					if start == end {
						continue;
					}
					annotate_noalias_mat_with_index::<false, true, Z, _, _>(
						&mut f,
						Z::Dyn::get_slice_unchecked(
							&mut z,
							(start, j),
							end - start,
						),
						0,
						end - start,
						j,
					);
				}
			} else {
				for j in 0..Ord::min(m, n) {
					let start = m - Ord::min(j + (1 - strict), m);
					let end = m;
					if start == end {
						continue;
					}
					for i in start..end {
						let (ii, jj) = Z::from_dyn_idx((m - i - 1, j));
						f(ii, jj, Z::Dyn::get_unchecked(&mut z, (i, j)))
					}
				}
			}
		},
		// SAFETY: TODO(review): for `n > m + 1 - strict` (a wide matrix; for
		// the upper triangle under `Transpose`, a tall one) some columns have
		// `start = j + strict > end = m`. `start == end` does not catch it: the
		// non-contiguous loop is empty, but the contiguous branch calls
		// `get_slice_unchecked` at row `start >= m` with length `end - start`,
		// which wraps to nearly `usize::MAX` in builds without overflow checks,
		// violating its contract (no element is read, as the index range is
		// empty). Otherwise `(m, n)` are the dimensions of `z`, which are those
		// of every zipped operand (see `ZipEq::new`). In column `j`, `start = j
		// + strict <= end = m` when the TODO above does not apply, so each `(i,
		// j)` with `start <= i < end` is in bounds and the contiguous branch
		// (row stride 1, checked by `is_contiguous`) slices `end - start` rows
		// from `(start, j)`; `from_dyn_idx` receives the index with the layout
		// transform undone, which is a valid index of `Z`; each element is
		// handed out once.
		MatLayoutTransform::Transpose => unsafe {
			if Z::Dyn::is_contiguous(&z) {
				for j in 0..n {
					let start = j + strict;
					let end = m;
					if start == end {
						continue;
					}
					annotate_noalias_mat_with_index::<true, false, Z, _, _>(
						&mut f,
						Z::Dyn::get_slice_unchecked(
							&mut z,
							(start, j),
							end - start,
						),
						start,
						end,
						j,
					);
				}
			} else {
				for j in 0..n {
					let start = j + strict;
					let end = m;
					if start == end {
						continue;
					}
					for i in start..end {
						let (ii, jj) = Z::from_dyn_idx((j, i));
						f(ii, jj, Z::Dyn::get_unchecked(&mut z, (i, j)))
					}
				}
			}
		},
		// SAFETY: TODO(review): `end = m - j - strict` underflows when `j +
		// strict > m`, which happens when `Z` has more rows than columns (`n >
		// m` here). In builds without overflow checks it wraps to nearly
		// `usize::MAX`, `start == end` does not catch it, and the
		// non-contiguous loop then calls `get_unchecked` with rows `>= m`: an
		// out-of-bounds access from safe code (e.g. a 4x2
		// `zip!(a.transpose_mut().reverse_cols_mut(), &mut
		// b).for_each_triangular_upper_with_index(..)`, reproduced in release).
		// When `j + strict <= m`, `(m, n)` are the dimensions of `z`, which are
		// those of every zipped operand (see `ZipEq::new`). In column `j`, `0
		// <= end <= m`, so each `(i, j)` with `start <= i < end` is in bounds
		// and the contiguous branch (row stride 1, checked by `is_contiguous`)
		// slices `end - start` rows from `(start, j)`; `from_dyn_idx` receives
		// the index with the layout transform undone, which is a valid index of
		// `Z`; each element is handed out once.
		MatLayoutTransform::TransposeReverseRows => unsafe {
			if Z::Dyn::is_contiguous(&z) {
				for j in 0..n {
					let start = 0;
					let end = m - j - strict;
					if start == end {
						continue;
					}
					annotate_noalias_mat_with_index::<true, true, Z, _, _>(
						&mut f,
						Z::Dyn::get_slice_unchecked(
							&mut z,
							(start, j),
							end - start,
						),
						j + strict,
						j + strict + end - start,
						j,
					);
				}
			} else {
				for j in 0..n {
					let start = 0;
					let end = m - j - strict;
					if start == end {
						continue;
					}
					for i in start..end {
						let (ii, jj) = Z::from_dyn_idx((j, m - i - 1));
						f(ii, jj, Z::Dyn::get_unchecked(&mut z, (i, j)))
					}
				}
			}
		},
	}
}
#[inline(always)]
fn for_each_mat_triangular_lower<
	Z: MatIndex<LayoutTransform = MatLayoutTransform>,
>(
	z: Z,
	diag: Diag,
	transpose: bool,
	mut f: impl FnMut(<Z as MatIndex>::Item),
) where
	Z::Dyn: MatIndex<
			LayoutTransform = MatLayoutTransform,
			Item = Z::Item,
			Slice = Z::Slice,
			Rows = usize,
			Cols = usize,
			Index = (usize, usize),
			Dyn = Z::Dyn,
		>,
{
	use MatLayoutTransform::*;
	let z = if transpose {
		Z::with_layout(z, MatLayoutTransform::Transpose)
	} else {
		Z::with_layout(z, MatLayoutTransform::None)
	};
	let layout = Z::Dyn::preferred_layout(&z);
	let mut z = Z::Dyn::with_layout(z, layout);
	let m = Z::Dyn::nrows(&z);
	let n = Z::Dyn::ncols(&z);
	let n = match layout {
		None | ReverseRows => Ord::min(m, n),
		Transpose | TransposeReverseRows => n,
	};
	if m == 0 || n == 0 {
		return;
	}
	let strict = match diag {
		Diag::Skip => true,
		Diag::Include => false,
	};
	// SAFETY: `(m, n)` are the dimensions of `z`, which are those of every
	// zipped operand (see `ZipEq::new`), and `n` was clamped to `min(m, n)` for
	// the `None`/`ReverseRows` layouts. For every layout, `j < n` gives `0 <=
	// start <= end <= m` (`j + strict <= m` for `None`, `m - (j + strict) >= 0`
	// for `ReverseRows`, `min(.., m)` for the transposed layouts), so each `(i,
	// j)` with `start <= i < end` is in bounds and the contiguous branch (row
	// stride 1, checked by `is_contiguous`) slices `end - start` rows from
	// `(start, j)`; each element is handed out once.
	unsafe {
		if Z::Dyn::is_contiguous(&z) {
			for j in 0..n {
				let (start, end) = match layout {
					None => (j + strict as usize, m),
					ReverseRows => (0, (m - (j + strict as usize))),
					Transpose => (0, (j + !strict as usize).min(m)),
					TransposeReverseRows => {
						(m - (j + !strict as usize).min(m), m)
					},
				};
				let len = end - start;
				if start == end {
					continue;
				}
				annotate_noalias_mat::<Z::Dyn>(
					&mut f,
					Z::Dyn::get_slice_unchecked(&mut z, (start, j), len),
					start,
					end,
					j,
				);
			}
		} else {
			for j in 0..n {
				let (start, end) = match layout {
					None => (j + strict as usize, m),
					ReverseRows => (0, (m - (j + strict as usize))),
					Transpose => (0, (j + !strict as usize).min(m)),
					TransposeReverseRows => {
						(m - (j + !strict as usize).min(m), m)
					},
				};
				if start == end {
					continue;
				}
				for i in start..end {
					f(Z::Dyn::get_unchecked(&mut z, (i, j)))
				}
			}
		}
	}
}
#[inline(always)]
fn for_each_diag<Z: MatIndex>(z: Z, mut f: impl FnMut(<Z as MatIndex>::Item))
where
	Z::Dyn: MatIndex<
			Rows = usize,
			Cols = usize,
			Index = usize,
			Item = Z::Item,
			Slice = Z::Slice,
		>,
{
	let layout = Z::preferred_layout(&z);
	let mut z = Z::with_layout(z, layout);
	let m = Z::Dyn::nrows(&z);
	if m == 0 {
		return;
	}
	// SAFETY: `m` is the length of `z`, which is that of every zipped operand
	// (see `ZipEq::new`), so every index `< m` is in bounds; the contiguous
	// branch takes the slice of all `m` elements from index 0 (stride 1,
	// checked by `is_contiguous`); each element is handed out exactly once.
	unsafe {
		if Z::Dyn::is_contiguous(&z) {
			annotate_noalias_col::<Z::Dyn>(
				&mut f,
				Z::Dyn::get_slice_unchecked(&mut z, 0, m),
				0,
				m,
			);
		} else {
			for i in 0..m {
				f(Z::Dyn::get_unchecked(&mut z, i))
			}
		}
	}
}
#[inline(always)]
fn for_each_diag_with_index<
	Idx,
	Z: MatIndex<LayoutTransform = VecLayoutTransform, Index = Idx>,
>(
	z: Z,
	mut f: impl FnMut(Idx, <Z as MatIndex>::Item),
) where
	Z::Dyn: MatIndex<
			Rows = usize,
			Cols = usize,
			Index = usize,
			Item = Z::Item,
			Slice = Z::Slice,
		>,
{
	let layout = Z::preferred_layout(&z);
	let mut z = Z::with_layout(z, layout);
	let m = Z::Dyn::nrows(&z);
	if m == 0 {
		return;
	}
	// SAFETY: `m` is the length of `z`, which is that of every zipped operand
	// (see `ZipEq::new`), so every index `< m` is in bounds; the contiguous
	// branch takes the slice of all `m` elements from index 0 (stride 1,
	// checked by `is_contiguous`); each element is handed out exactly once.
	// `from_dyn_idx` receives `i` or `m - i - 1` (the layout transform undone),
	// a valid index of `Z`.
	unsafe {
		match layout {
			VecLayoutTransform::None => {
				if Z::Dyn::is_contiguous(&z) {
					annotate_noalias_col_with_index::<Z, _>(
						&mut f,
						Z::Dyn::get_slice_unchecked(&mut z, 0, m),
						0,
						m,
						false,
					);
				} else {
					for i in 0..m {
						f(Z::from_dyn_idx(i), Z::Dyn::get_unchecked(&mut z, i))
					}
				}
			},
			VecLayoutTransform::Reverse => {
				if Z::Dyn::is_contiguous(&z) {
					annotate_noalias_col_with_index::<Z, _>(
						&mut f,
						Z::Dyn::get_slice_unchecked(&mut z, 0, m),
						0,
						m,
						true,
					);
				} else {
					for i in 0..m {
						f(
							Z::from_dyn_idx(m - i - 1),
							Z::Dyn::get_unchecked(&mut z, i),
						)
					}
				}
			},
		}
	}
}
#[inline(always)]
fn for_each_col<Z: MatIndex<LayoutTransform = VecLayoutTransform>>(
	z: Z,
	mut f: impl FnMut(<Z as MatIndex>::Item),
) where
	Z::Dyn: MatIndex<
			Rows = usize,
			Cols = (),
			Index = usize,
			Item = Z::Item,
			Slice = Z::Slice,
		>,
{
	let layout = Z::preferred_layout(&z);
	let mut z = Z::with_layout(z, layout);
	let m = Z::Dyn::nrows(&z);
	if m == 0 {
		return;
	}
	// SAFETY: `m` is the length of `z`, which is that of every zipped operand
	// (see `ZipEq::new`), so every index `< m` is in bounds; the contiguous
	// branch takes the slice of all `m` elements from index 0 (stride 1,
	// checked by `is_contiguous`); each element is handed out exactly once.
	unsafe {
		if Z::Dyn::is_contiguous(&z) {
			annotate_noalias_col::<Z::Dyn>(
				&mut f,
				Z::Dyn::get_slice_unchecked(&mut z, 0, m),
				0,
				m,
			);
		} else {
			for i in 0..m {
				f(Z::Dyn::get_unchecked(&mut z, i))
			}
		}
	}
}
#[inline(always)]
fn for_each_col_with_index<
	Idx,
	Z: MatIndex<LayoutTransform = VecLayoutTransform, Index = Idx>,
>(
	z: Z,
	mut f: impl FnMut(Idx, <Z as MatIndex>::Item),
) where
	Z::Dyn: MatIndex<
			Rows = usize,
			Cols = (),
			Index = usize,
			Item = Z::Item,
			Slice = Z::Slice,
		>,
{
	let layout = Z::preferred_layout(&z);
	let mut z = Z::with_layout(z, layout);
	let m = Z::Dyn::nrows(&z);
	if m == 0 {
		return;
	}
	// SAFETY: `m` is the length of `z`, which is that of every zipped operand
	// (see `ZipEq::new`), so every index `< m` is in bounds; the contiguous
	// branch takes the slice of all `m` elements from index 0 (stride 1,
	// checked by `is_contiguous`); each element is handed out exactly once.
	// `from_dyn_idx` receives `i` or `m - i - 1` (the layout transform undone),
	// a valid index of `Z`.
	unsafe {
		match layout {
			VecLayoutTransform::None => {
				if Z::Dyn::is_contiguous(&z) {
					annotate_noalias_col_with_index::<Z, _>(
						&mut f,
						Z::Dyn::get_slice_unchecked(&mut z, 0, m),
						0,
						m,
						false,
					);
				} else {
					for i in 0..m {
						f(Z::from_dyn_idx(i), Z::Dyn::get_unchecked(&mut z, i))
					}
				}
			},
			VecLayoutTransform::Reverse => {
				if Z::Dyn::is_contiguous(&z) {
					annotate_noalias_col_with_index::<Z, _>(
						&mut f,
						Z::Dyn::get_slice_unchecked(&mut z, 0, m),
						0,
						m,
						true,
					);
				} else {
					for i in 0..m {
						f(
							Z::from_dyn_idx(m - i - 1),
							Z::Dyn::get_unchecked(&mut z, i),
						)
					}
				}
			},
		}
	}
}
#[inline(always)]
fn for_each_row_with_index<
	Idx,
	Z: MatIndex<LayoutTransform = VecLayoutTransform, Index = Idx>,
>(
	z: Z,
	mut f: impl FnMut(Idx, <Z as MatIndex>::Item),
) where
	Z::Dyn: MatIndex<
			Rows = (),
			Cols = usize,
			Index = usize,
			Item = Z::Item,
			Slice = Z::Slice,
		>,
{
	let layout = Z::preferred_layout(&z);
	let mut z = Z::with_layout(z, layout);
	let n = Z::Dyn::ncols(&z);
	if n == 0 {
		return;
	}
	// SAFETY: `n` is the length of `z`, which is that of every zipped operand
	// (see `ZipEq::new`), so every index `< n` is in bounds; the contiguous
	// branch takes the slice of all `n` elements from index 0 (stride 1,
	// checked by `is_contiguous`); each element is handed out exactly once.
	// `from_dyn_idx` receives `i` or `n - i - 1` (the layout transform undone),
	// a valid index of `Z`.
	unsafe {
		match layout {
			VecLayoutTransform::None => {
				if Z::Dyn::is_contiguous(&z) {
					annotate_noalias_col_with_index::<Z, _>(
						&mut f,
						Z::Dyn::get_slice_unchecked(&mut z, 0, n),
						0,
						n,
						false,
					);
				} else {
					for i in 0..n {
						f(Z::from_dyn_idx(i), Z::Dyn::get_unchecked(&mut z, i))
					}
				}
			},
			VecLayoutTransform::Reverse => {
				if Z::Dyn::is_contiguous(&z) {
					annotate_noalias_col_with_index::<Z, _>(
						&mut f,
						Z::Dyn::get_slice_unchecked(&mut z, 0, n),
						0,
						n,
						true,
					);
				} else {
					for i in 0..n {
						f(
							Z::from_dyn_idx(n - i - 1),
							Z::Dyn::get_unchecked(&mut z, i),
						)
					}
				}
			},
		}
	}
}
#[inline(always)]
fn for_each_row<Z: MatIndex>(z: Z, mut f: impl FnMut(<Z as MatIndex>::Item))
where
	Z::Dyn: MatIndex<
			Rows = (),
			Cols = usize,
			Index = usize,
			Item = Z::Item,
			Slice = Z::Slice,
		>,
{
	let layout = Z::preferred_layout(&z);
	let mut z = Z::with_layout(z, layout);
	let n = Z::Dyn::ncols(&z);
	if n == 0 {
		return;
	}
	// SAFETY: `n` is the length of `z`, which is that of every zipped operand
	// (see `ZipEq::new`), so every index `< n` is in bounds; the contiguous
	// branch takes the slice of all `n` elements from index 0 (stride 1,
	// checked by `is_contiguous`); each element is handed out exactly once.
	unsafe {
		if Z::Dyn::is_contiguous(&z) {
			annotate_noalias_col::<Z::Dyn>(
				&mut f,
				Z::Dyn::get_slice_unchecked(&mut z, 0, n),
				0,
				n,
			);
		} else {
			for j in 0..n {
				f(Z::Dyn::get_unchecked(&mut z, j))
			}
		}
	}
}
impl<
	Rows: Shape,
	Cols: Shape,
	M: MatIndex<
			Kind = kind::Mat,
			LayoutTransform = MatLayoutTransform,
			Rows = Rows,
			Cols = Cols,
			Index = (Idx<Rows>, Idx<Cols>),
		>,
> LastEq<kind::Mat, M>
where
	M::Dyn: MatIndex<Rows = usize, Cols = usize, Index = (usize, usize)>,
{
	/// applies `f` to each element of `self`
	#[inline(always)]
	pub fn for_each(self, f: impl FnMut(<Self as MatIndex>::Item)) {
		for_each_mat(self, f);
	}

	/// applies `f` to each element of `self`, while passing the indices of the
	/// position of the current element
	#[inline(always)]
	pub fn for_each_with_index(
		self,
		f: impl FnMut(Idx<Rows>, Idx<Cols>, <Self as MatIndex>::Item),
	) {
		for_each_mat_with_index(self, f);
	}

	/// applies `f` to each element of the lower triangular half of `self`,
	/// while passing the indices of the position of the current element
	///
	/// `diag` specifies whether the diagonal should be included or excluded
	#[inline(always)]
	pub fn for_each_triangular_lower_with_index(
		self,
		diag: Diag,
		f: impl FnMut(Idx<Rows>, Idx<Cols>, <Self as MatIndex>::Item),
	) {
		for_each_mat_triangular_lower_with_index(self, diag, f);
	}

	/// applies `f` to each element of the upper triangular half of `self`,
	/// while passing the indices of the position of the current element
	///
	/// `diag` specifies whether the diagonal should be included or excluded
	#[inline(always)]
	pub fn for_each_triangular_upper_with_index(
		self,
		diag: Diag,
		f: impl FnMut(Idx<Rows>, Idx<Cols>, <Self as MatIndex>::Item),
	) {
		for_each_mat_triangular_upper_with_index(self, diag, f);
	}

	/// applies `f` to each element of the lower triangular half of `self`
	///
	/// `diag` specifies whether the diagonal should be included or excluded
	#[inline(always)]
	pub fn for_each_triangular_lower(
		self,
		diag: Diag,
		f: impl FnMut(<Self as MatIndex>::Item),
	) {
		for_each_mat_triangular_lower(self, diag, false, f);
	}

	/// applies `f` to each element of the upper triangular half of `self`
	///
	/// `diag` specifies whether the diagonal should be included or excluded
	#[inline(always)]
	pub fn for_each_triangular_upper(
		self,
		diag: Diag,
		f: impl FnMut(<Self as MatIndex>::Item),
	) {
		for_each_mat_triangular_lower(self, diag, true, f);
	}

	/// applies `f` to each element of `self` and collect its result into a new
	/// matrix
	#[inline(always)]
	pub fn map<T>(
		self,
		f: impl FnMut(<Self as MatIndex>::Item) -> T,
	) -> Mat<T, Rows, Cols> {
		let (m, n) = (Self::nrows(&self), Self::ncols(&self));
		let mut f = f;
		let mut this = self;
		Mat::from_fn(
			m,
			n,
			#[inline(always)]
			// SAFETY: `Mat::from_fn` calls the closure once for each in-bounds
			// `(i, j)` of an `(m, n)` matrix, and `(m, n)` are the dimensions
			// of `this` (of every zipped operand, see `ZipEq::new`), so each
			// index is in bounds and handed out once.
			|i, j| f(unsafe { Self::get_unchecked(&mut this, (i, j)) }),
		)
	}

	/// applies `f` to each element of `self` and collect its result into a new
	/// matrix
	#[inline(always)]
	pub fn map_with_index<T>(
		self,
		f: impl FnMut(Idx<Rows>, Idx<Cols>, <Self as MatIndex>::Item) -> T,
	) -> Mat<T, Rows, Cols> {
		let (m, n) = (Self::nrows(&self), Self::ncols(&self));
		let mut f = f;
		let mut this = self;
		Mat::from_fn(
			m,
			n,
			#[inline(always)]
			// SAFETY: `Mat::from_fn` calls the closure once for each in-bounds
			// `(i, j)` of an `(m, n)` matrix, and `(m, n)` are the dimensions
			// of `this` (of every zipped operand, see `ZipEq::new`), so each
			// index is in bounds and handed out once.
			|i, j| f(i, j, unsafe { Self::get_unchecked(&mut this, (i, j)) }),
		)
	}
}
impl<
	Rows: Shape,
	Cols: Shape,
	L: MatIndex<
			Kind = kind::Mat,
			LayoutTransform = MatLayoutTransform,
			Rows = Rows,
			Cols = Cols,
			Index = (Idx<Rows>, Idx<Cols>),
		>,
	R: MatIndex<
			Kind = kind::Mat,
			LayoutTransform = MatLayoutTransform,
			Rows = Rows,
			Cols = Cols,
			Index = (Idx<Rows>, Idx<Cols>),
		>,
> ZipEq<kind::Mat, L, R>
where
	L::Dyn: MatIndex<Rows = usize, Cols = usize, Index = (usize, usize)>,
	R::Dyn: MatIndex<Rows = usize, Cols = usize, Index = (usize, usize)>,
{
	/// applies `f` to each element of `self`
	#[inline(always)]
	pub fn for_each(self, f: impl FnMut(<Self as MatIndex>::Item)) {
		for_each_mat(self, f);
	}

	/// applies `f` to each element of `self`, while passing the indices of the
	/// position of the current element
	#[inline(always)]
	pub fn for_each_with_index(
		self,
		f: impl FnMut(Idx<Rows>, Idx<Cols>, <Self as MatIndex>::Item),
	) {
		for_each_mat_with_index(self, f);
	}

	/// applies `f` to each element of the lower triangular half of `self`,
	/// while passing the indices of the position of the current element
	///
	/// `diag` specifies whether the diagonal should be included or excluded
	#[inline(always)]
	pub fn for_each_triangular_lower_with_index(
		self,
		diag: Diag,
		f: impl FnMut(Idx<Rows>, Idx<Cols>, <Self as MatIndex>::Item),
	) {
		for_each_mat_triangular_lower_with_index(self, diag, f);
	}

	/// applies `f` to each element of the upper triangular half of `self`,
	/// while passing the indices of the position of the current element
	///
	/// `diag` specifies whether the diagonal should be included or excluded
	#[inline(always)]
	pub fn for_each_triangular_upper_with_index(
		self,
		diag: Diag,
		f: impl FnMut(Idx<Rows>, Idx<Cols>, <Self as MatIndex>::Item),
	) {
		for_each_mat_triangular_upper_with_index(self, diag, f);
	}

	/// applies `f` to each element of the lower triangular half of `self`
	///
	/// `diag` specifies whether the diagonal should be included or excluded
	#[inline(always)]
	pub fn for_each_triangular_lower(
		self,
		diag: Diag,
		f: impl FnMut(<Self as MatIndex>::Item),
	) {
		for_each_mat_triangular_lower(self, diag, false, f);
	}

	/// applies `f` to each element of the upper triangular half of `self`
	///
	/// `diag` specifies whether the diagonal should be included or excluded
	#[inline(always)]
	pub fn for_each_triangular_upper(
		self,
		diag: Diag,
		f: impl FnMut(<Self as MatIndex>::Item),
	) {
		for_each_mat_triangular_lower(self, diag, true, f);
	}

	/// applies `f` to each element of `self` and collect its result into a new
	/// matrix
	#[inline(always)]
	pub fn map<T>(
		self,
		f: impl FnMut(<Self as MatIndex>::Item) -> T,
	) -> Mat<T, Rows, Cols> {
		let (m, n) = (Self::nrows(&self), Self::ncols(&self));
		let mut f = f;
		let mut this = self;
		Mat::from_fn(
			m,
			n,
			#[inline(always)]
			// SAFETY: `Mat::from_fn` calls the closure once for each in-bounds
			// `(i, j)` of an `(m, n)` matrix, and `(m, n)` are the dimensions
			// of `this` (of every zipped operand, see `ZipEq::new`), so each
			// index is in bounds and handed out once.
			|i, j| f(unsafe { Self::get_unchecked(&mut this, (i, j)) }),
		)
	}

	/// applies `f` to each element of `self` and collect its result into a new
	/// matrix
	#[inline(always)]
	pub fn map_with_index<T>(
		self,
		f: impl FnMut(Idx<Rows>, Idx<Cols>, <Self as MatIndex>::Item) -> T,
	) -> Mat<T, Rows, Cols> {
		let (m, n) = (Self::nrows(&self), Self::ncols(&self));
		let mut f = f;
		let mut this = self;
		Mat::from_fn(
			m,
			n,
			#[inline(always)]
			// SAFETY: `Mat::from_fn` calls the closure once for each in-bounds
			// `(i, j)` of an `(m, n)` matrix, and `(m, n)` are the dimensions
			// of `this` (of every zipped operand, see `ZipEq::new`), so each
			// index is in bounds and handed out once.
			|i, j| f(i, j, unsafe { Self::get_unchecked(&mut this, (i, j)) }),
		)
	}
}
impl<
	Rows: Shape,
	M: MatIndex<
			Kind = kind::Col,
			LayoutTransform = VecLayoutTransform,
			Rows = Rows,
			Cols = (),
			Index = Idx<Rows>,
		>,
> LastEq<kind::Col, M>
where
	M::Dyn: MatIndex<Rows = usize, Cols = (), Index = usize>,
{
	/// applies `f` to each element of `self`
	#[inline(always)]
	pub fn for_each(self, f: impl FnMut(<Self as MatIndex>::Item)) {
		for_each_col(self, f);
	}

	/// applies `f` to each element of `self`, while passing the indices of the
	/// position of the current element
	#[inline(always)]
	pub fn for_each_with_index(
		self,
		f: impl FnMut(Idx<Rows>, <Self as MatIndex>::Item),
	) {
		for_each_col_with_index(self, f);
	}

	/// applies `f` to each element of `self` and collect its result into a new
	/// matrix
	#[inline(always)]
	pub fn map<T>(
		self,
		f: impl FnMut(<Self as MatIndex>::Item) -> T,
	) -> Col<T, Rows> {
		let (m, _) = (Self::nrows(&self), Self::ncols(&self));
		let mut f = f;
		let mut this = self;
		Col::from_fn(
			m,
			#[inline(always)]
			// SAFETY: `Col::from_fn` calls the closure once for each index `<
			// m` (`n` for a row), the length of `this` (of every zipped
			// operand, see `ZipEq::new`), so each index is in bounds and handed
			// out once.
			|i| f(unsafe { Self::get_unchecked(&mut this, i) }),
		)
	}

	/// applies `f` to each element of `self` and collect its result into a new
	/// matrix
	#[inline(always)]
	pub fn map_with_index<T>(
		self,
		f: impl FnMut(Idx<Rows>, <Self as MatIndex>::Item) -> T,
	) -> Col<T, Rows> {
		let (m, _) = (Self::nrows(&self), Self::ncols(&self));
		let mut f = f;
		let mut this = self;
		Col::from_fn(
			m,
			#[inline(always)]
			// SAFETY: `Col::from_fn` calls the closure once for each index `<
			// m` (`n` for a row), the length of `this` (of every zipped
			// operand, see `ZipEq::new`), so each index is in bounds and handed
			// out once.
			|i| f(i, unsafe { Self::get_unchecked(&mut this, i) }),
		)
	}
}
impl<
	Rows: Shape,
	L: MatIndex<
			Kind = kind::Col,
			LayoutTransform = VecLayoutTransform,
			Rows = Rows,
			Cols = (),
			Index = Idx<Rows>,
		>,
	R: MatIndex<
			Kind = kind::Col,
			LayoutTransform = VecLayoutTransform,
			Rows = Rows,
			Cols = (),
			Index = Idx<Rows>,
		>,
> ZipEq<kind::Col, L, R>
where
	L::Dyn: MatIndex<Rows = usize, Cols = (), Index = usize>,
	R::Dyn: MatIndex<Rows = usize, Cols = (), Index = usize>,
{
	/// applies `f` to each element of `self`
	#[inline(always)]
	pub fn for_each(self, f: impl FnMut(<Self as MatIndex>::Item)) {
		for_each_col(self, f);
	}

	/// applies `f` to each element of `self`, while passing the indices of the
	/// position of the current element
	#[inline(always)]
	pub fn for_each_with_index(
		self,
		f: impl FnMut(Idx<Rows>, <Self as MatIndex>::Item),
	) {
		for_each_col_with_index(self, f);
	}

	/// applies `f` to each element of `self` and collect its result into a new
	/// matrix
	#[inline(always)]
	pub fn map<T>(
		self,
		f: impl FnMut(<Self as MatIndex>::Item) -> T,
	) -> Col<T, Rows> {
		let (m, _) = (Self::nrows(&self), Self::ncols(&self));
		let mut f = f;
		let mut this = self;
		Col::from_fn(
			m,
			#[inline(always)]
			// SAFETY: `Col::from_fn` calls the closure once for each index `<
			// m` (`n` for a row), the length of `this` (of every zipped
			// operand, see `ZipEq::new`), so each index is in bounds and handed
			// out once.
			|i| f(unsafe { Self::get_unchecked(&mut this, i) }),
		)
	}

	/// applies `f` to each element of `self` and collect its result into a new
	/// matrix
	#[inline(always)]
	pub fn map_with_index<T>(
		self,
		f: impl FnMut(Idx<Rows>, <Self as MatIndex>::Item) -> T,
	) -> Col<T, Rows> {
		let (m, _) = (Self::nrows(&self), Self::ncols(&self));
		let mut f = f;
		let mut this = self;
		Col::from_fn(
			m,
			#[inline(always)]
			// SAFETY: `Col::from_fn` calls the closure once for each index `<
			// m` (`n` for a row), the length of `this` (of every zipped
			// operand, see `ZipEq::new`), so each index is in bounds and handed
			// out once.
			|i| f(i, unsafe { Self::get_unchecked(&mut this, i) }),
		)
	}
}
impl<
	Dim: Shape,
	M: MatIndex<
			Kind = kind::Diag,
			LayoutTransform = VecLayoutTransform,
			Rows = Dim,
			Cols = Dim,
			Index = Idx<Dim>,
		>,
> LastEq<kind::Diag, M>
where
	M::Dyn: MatIndex<Rows = usize, Cols = usize, Index = usize>,
{
	/// applies `f` to each element of `self`
	#[inline(always)]
	pub fn for_each(self, f: impl FnMut(<Self as MatIndex>::Item)) {
		for_each_diag(self, f);
	}

	/// applies `f` to each element of `self`, while passing the indices of the
	/// position of the current element
	#[inline(always)]
	pub fn for_each_with_index(
		self,
		f: impl FnMut(Idx<Dim>, <Self as MatIndex>::Item),
	) {
		for_each_diag_with_index(self, f);
	}

	/// applies `f` to each element of `self` and collect its result into a new
	/// matrix
	#[inline(always)]
	pub fn map<T>(
		self,
		f: impl FnMut(<Self as MatIndex>::Item) -> T,
	) -> diag::Diag<T, Dim> {
		let (m, _) = (Self::nrows(&self), Self::ncols(&self));
		let mut f = f;
		let mut this = self;
		Col::from_fn(
			m,
			#[inline(always)]
			// SAFETY: `Col::from_fn` calls the closure once for each index `<
			// m` (`n` for a row), the length of `this` (of every zipped
			// operand, see `ZipEq::new`), so each index is in bounds and handed
			// out once.
			|i| f(unsafe { Self::get_unchecked(&mut this, i) }),
		)
		.into_diagonal()
	}

	/// applies `f` to each element of `self` and collect its result into a new
	/// matrix
	#[inline(always)]
	pub fn map_with_index<T>(
		self,
		f: impl FnMut(Idx<Dim>, <Self as MatIndex>::Item) -> T,
	) -> diag::Diag<T, Dim> {
		let (m, _) = (Self::nrows(&self), Self::ncols(&self));
		let mut f = f;
		let mut this = self;
		Col::from_fn(
			m,
			#[inline(always)]
			// SAFETY: `Col::from_fn` calls the closure once for each index `<
			// m` (`n` for a row), the length of `this` (of every zipped
			// operand, see `ZipEq::new`), so each index is in bounds and handed
			// out once.
			|i| f(i, unsafe { Self::get_unchecked(&mut this, i) }),
		)
		.into_diagonal()
	}
}
impl<
	Dim: Shape,
	L: MatIndex<
			Kind = kind::Diag,
			LayoutTransform = VecLayoutTransform,
			Rows = Dim,
			Cols = Dim,
			Index = Idx<Dim>,
		>,
	R: MatIndex<
			Kind = kind::Diag,
			LayoutTransform = VecLayoutTransform,
			Rows = Dim,
			Cols = Dim,
			Index = Idx<Dim>,
		>,
> ZipEq<kind::Diag, L, R>
where
	L::Dyn: MatIndex<Rows = usize, Cols = usize, Index = usize>,
	R::Dyn: MatIndex<Rows = usize, Cols = usize, Index = usize>,
{
	/// applies `f` to each element of `self`
	#[inline(always)]
	pub fn for_each(self, f: impl FnMut(<Self as MatIndex>::Item)) {
		for_each_diag(self, f);
	}

	/// applies `f` to each element of `self`, while passing the indices of the
	/// position of the current element
	#[inline(always)]
	pub fn for_each_with_index(
		self,
		f: impl FnMut(Idx<Dim>, <Self as MatIndex>::Item),
	) {
		for_each_diag_with_index(self, f);
	}

	/// applies `f` to each element of `self` and collect its result into a new
	/// matrix
	#[inline(always)]
	pub fn map<T>(
		self,
		f: impl FnMut(<Self as MatIndex>::Item) -> T,
	) -> diag::Diag<T, Dim> {
		let (m, _) = (Self::nrows(&self), Self::ncols(&self));
		let mut f = f;
		let mut this = self;
		Col::from_fn(
			m,
			#[inline(always)]
			// SAFETY: `Col::from_fn` calls the closure once for each index `<
			// m` (`n` for a row), the length of `this` (of every zipped
			// operand, see `ZipEq::new`), so each index is in bounds and handed
			// out once.
			|i| f(unsafe { Self::get_unchecked(&mut this, i) }),
		)
		.into_diagonal()
	}

	/// applies `f` to each element of `self` and collect its result into a new
	/// matrix
	#[inline(always)]
	pub fn map_with_index<T>(
		self,
		f: impl FnMut(Idx<Dim>, <Self as MatIndex>::Item) -> T,
	) -> diag::Diag<T, Dim> {
		let (m, _) = (Self::nrows(&self), Self::ncols(&self));
		let mut f = f;
		let mut this = self;
		Col::from_fn(
			m,
			#[inline(always)]
			// SAFETY: `Col::from_fn` calls the closure once for each index `<
			// m` (`n` for a row), the length of `this` (of every zipped
			// operand, see `ZipEq::new`), so each index is in bounds and handed
			// out once.
			|i| f(i, unsafe { Self::get_unchecked(&mut this, i) }),
		)
		.into_diagonal()
	}
}
impl<
	Cols: Shape,
	M: MatIndex<
			Kind = kind::Row,
			LayoutTransform = VecLayoutTransform,
			Rows = (),
			Cols = Cols,
			Index = Idx<Cols>,
		>,
> LastEq<kind::Row, M>
where
	M::Dyn: MatIndex<Rows = (), Cols = usize, Index = usize>,
{
	/// applies `f` to each element of `self`
	#[inline(always)]
	pub fn for_each(self, f: impl FnMut(<Self as MatIndex>::Item)) {
		for_each_row(self, f);
	}

	/// applies `f` to each element of `self`, while passing the indices of the
	/// position of the current element
	#[inline(always)]
	pub fn for_each_with_index(
		self,
		f: impl FnMut(Idx<Cols>, <Self as MatIndex>::Item),
	) {
		for_each_row_with_index(self, f);
	}

	/// applies `f` to each element of `self` and collect its result into a new
	/// matrix
	#[inline(always)]
	pub fn map<T>(
		self,
		f: impl FnMut(<Self as MatIndex>::Item) -> T,
	) -> Row<T, Cols> {
		let (_, n) = (Self::nrows(&self), Self::ncols(&self));
		let mut f = f;
		let mut this = self;
		Row::from_fn(
			n,
			#[inline(always)]
			// SAFETY: `Row::from_fn` calls the closure once for each index `<
			// m` (`n` for a row), the length of `this` (of every zipped
			// operand, see `ZipEq::new`), so each index is in bounds and handed
			// out once.
			|i| f(unsafe { Self::get_unchecked(&mut this, i) }),
		)
	}

	/// applies `f` to each element of `self` and collect its result into a new
	/// matrix
	#[inline(always)]
	pub fn map_with_index<T>(
		self,
		f: impl FnMut(Idx<Cols>, <Self as MatIndex>::Item) -> T,
	) -> Row<T, Cols> {
		let (_, n) = (Self::nrows(&self), Self::ncols(&self));
		let mut f = f;
		let mut this = self;
		Row::from_fn(
			n,
			#[inline(always)]
			// SAFETY: `Row::from_fn` calls the closure once for each index `<
			// m` (`n` for a row), the length of `this` (of every zipped
			// operand, see `ZipEq::new`), so each index is in bounds and handed
			// out once.
			|i| f(i, unsafe { Self::get_unchecked(&mut this, i) }),
		)
	}
}
impl<
	Cols: Shape,
	L: MatIndex<
			Kind = kind::Row,
			LayoutTransform = VecLayoutTransform,
			Rows = (),
			Cols = Cols,
			Index = Idx<Cols>,
		>,
	R: MatIndex<
			Kind = kind::Row,
			LayoutTransform = VecLayoutTransform,
			Rows = (),
			Cols = Cols,
			Index = Idx<Cols>,
		>,
> ZipEq<kind::Row, L, R>
where
	L::Dyn: MatIndex<Rows = (), Cols = usize, Index = usize>,
	R::Dyn: MatIndex<Rows = (), Cols = usize, Index = usize>,
{
	/// applies `f` to each element of `self`
	#[inline(always)]
	pub fn for_each(self, f: impl FnMut(<Self as MatIndex>::Item)) {
		for_each_row(self, f);
	}

	/// applies `f` to each element of `self`, while passing the indices of the
	/// position of the current element
	#[inline(always)]
	pub fn for_each_with_index(
		self,
		f: impl FnMut(Idx<Cols>, <Self as MatIndex>::Item),
	) {
		for_each_row_with_index(self, f);
	}

	/// applies `f` to each element of `self` and collect its result into a new
	/// matrix
	#[inline(always)]
	pub fn map<T>(
		self,
		f: impl FnMut(<Self as MatIndex>::Item) -> T,
	) -> Row<T, Cols> {
		let (_, n) = (Self::nrows(&self), Self::ncols(&self));
		let mut f = f;
		let mut this = self;
		Row::from_fn(
			n,
			#[inline(always)]
			// SAFETY: `Row::from_fn` calls the closure once for each index `<
			// m` (`n` for a row), the length of `this` (of every zipped
			// operand, see `ZipEq::new`), so each index is in bounds and handed
			// out once.
			|i| f(unsafe { Self::get_unchecked(&mut this, i) }),
		)
	}

	/// applies `f` to each element of `self` and collect its result into a new
	/// matrix
	#[inline(always)]
	pub fn map_with_index<T>(
		self,
		f: impl FnMut(Idx<Cols>, <Self as MatIndex>::Item) -> T,
	) -> Row<T, Cols> {
		let (_, n) = (Self::nrows(&self), Self::ncols(&self));
		let mut f = f;
		let mut this = self;
		Row::from_fn(
			n,
			#[inline(always)]
			// SAFETY: `Row::from_fn` calls the closure once for each index `<
			// m` (`n` for a row), the length of `this` (of every zipped
			// operand, see `ZipEq::new`), so each index is in bounds and handed
			// out once.
			|i| f(i, unsafe { Self::get_unchecked(&mut this, i) }),
		)
	}
}

#[doc(hidden)]
pub mod kind {
	pub struct Col;
	pub struct Row;
	pub struct Mat;
	pub struct Diag;
}
