//! The caller-owned workspace: temporary matrices carved out of a [`MemStack`].
//!
//! A function that needs temporary storage takes a `stack: &mut MemStack` and has a companion
//! `*_scratch` function that returns its [`StackReq`]. The caller allocates once (for example
//! with [`MemBuffer::new`]), may combine requirements ([`StackReq::all_of`] for storage used at
//! the same time, [`StackReq::any_of`] for storage used in turn), and reuses the buffer across
//! calls, so the call itself does not allocate.
//!
//! The contract proposed in this trial (design record §5, from the design document §6
//! "Caller-owned workspace"):
//!
//! * the `*_scratch` query and the call agree on the path, so a buffer of exactly the queried
//!   size suffices;
//! * the call checks that the stack can hold its requirement **before it writes any output**,
//!   and panics if it cannot. Which error a caller gets back instead of a panic is open (design
//!   document §11, question 3);
//! * the call allocates nothing on the heap and keeps no thread-local buffer.
//!
//! ```
//! use colmat_core::workspace::{MemBuffer, MemStack, temp_mat_scratch, temp_mat_zeroed};
//! let mut buf = MemBuffer::new(temp_mat_scratch::<f64>(4, 3));
//! let stack = MemStack::new(&mut buf);
//! let (mut tmp, _rest) = temp_mat_zeroed::<f64, _, _>(4, 3, stack);
//! use colmat_core::mat::AsMatMut;
//! let mut tmp = tmp.as_mat_mut();
//! tmp[(3, 2)] = 1.0;
//! assert_eq!(tmp[(0, 0)], 0.0);
//! ```
//!
//! # Provenance
//!
//! Ported from colmat-rs `faer/src/linalg/mod.rs` at `71337cdc` (lines 29-73 and 602-770:
//! `padded_col_stride`, `temp_mat_scratch`, `DynMat`, `DropGuard`, `temp_mat_uninit`,
//! `temp_mat_zeroed`). Changed: a module of its own (colmat-rs: `colmat::linalg`); the
//! capacity is checked at entry with a message naming the requirement; `temp_mat_zeroed` no
//! longer makes a first, discarded allocation. faer's MIT notice: `LICENSE-faer`.

use crate::Shape;
use crate::mat::matown::align_for;
use crate::mat::{AsMatMut, AsMatRef, Mat, MatMut, MatRef};
use colmat_traits::ComplexField;
use core::marker::PhantomData;
pub use dyn_stack::{MemBuffer, MemStack, StackReq};

/// The column stride an owned matrix with `nrows` rows gets: the row capacity is rounded up so
/// the byte stride is a multiple of the element alignment (`matown::align_for`), so a
/// `505 x 505` f64 matrix has a 512-row stride. The padding only exists for a `T` without drop
/// glue.
pub(crate) fn padded_col_stride<T>(nrows: usize) -> usize {
	let size = core::mem::size_of::<T>();
	let align = align_for(
		size,
		core::mem::align_of::<T>(),
		core::mem::needs_drop::<T>(),
	);
	// a zero-sized element has a zero stride, so there is nothing to pad and `align / size` is
	// undefined
	if size > 0 && align > size {
		nrows.next_multiple_of(align / size)
	} else {
		nrows
	}
}

/// The alignment and length of the storage of a temporary `nrows x ncols` matrix.
fn temp_mat_layout<T>(nrows: usize, ncols: usize) -> (usize, usize, usize) {
	let align = align_for(
		core::mem::size_of::<T>(),
		core::mem::align_of::<T>(),
		core::mem::needs_drop::<T>(),
	);
	let col_stride = padded_col_stride::<T>(nrows);
	let len = col_stride
		.checked_mul(ncols)
		.expect("the size of the temporary matrix overflows");
	(align, col_stride, len)
}

/// returns the stack requirements for creating a temporary matrix with the
/// given dimensions.
pub fn temp_mat_scratch<T: ComplexField>(
	nrows: usize,
	ncols: usize,
) -> StackReq {
	let (align, _, len) = temp_mat_layout::<T>(nrows, ncols);
	StackReq::new_aligned::<T>(len, align)
}

/// A temporary column-major matrix in a [`MemStack`]: its elements are dropped with it, and the
/// stack space is released when the borrow of the stack ends.
struct DynMat<'a, T: ComplexField, Rows: Shape, Cols: Shape> {
	ptr: *mut T,
	nrows: Rows,
	ncols: Cols,
	col_stride: usize,
	__marker: PhantomData<(&'a T, T)>,
}

impl<T: ComplexField, Rows: Shape, Cols: Shape> Drop
	for DynMat<'_, T, Rows, Cols>
{
	#[inline]
	fn drop(&mut self) {
		// SAFETY: `ptr` holds `col_stride * ncols` elements of the stack, all initialised when
		// `T` needs drop (`temp_mat_uninit` and `temp_mat_zeroed` fill every one, and a type
		// with drop glue has no padding rows); for a type without drop glue this drops nothing.
		unsafe {
			core::ptr::drop_in_place(core::ptr::slice_from_raw_parts_mut(
				self.ptr,
				self.col_stride * self.ncols.unbound(),
			))
		};
	}
}

impl<T: ComplexField, Rows: Shape, Cols: Shape> AsMatRef
	for DynMat<'_, T, Rows, Cols>
{
	type Cols = Cols;
	type Owned = Mat<T, Rows, Cols>;
	type Rows = Rows;
	type T = T;

	fn as_mat_ref(&self) -> MatRef<'_, T, Rows, Cols> {
		// SAFETY: `ptr` holds a column-major `nrows x ncols` matrix with column stride
		// `col_stride` in stack memory borrowed for `'a`, which outlives `&self`; the view
		// invariant holds for the elements the caller has initialised, which the safety
		// contract of `temp_mat_uninit` requires before any read.
		unsafe {
			MatRef::from_raw_parts(
				self.ptr as *const T,
				self.nrows,
				self.ncols,
				1,
				self.col_stride as isize,
			)
		}
	}
}

impl<T: ComplexField, Rows: Shape, Cols: Shape> AsMatMut
	for DynMat<'_, T, Rows, Cols>
{
	fn as_mat_mut(&mut self) -> MatMut<'_, T, Rows, Cols> {
		// SAFETY: as in `as_mat_ref`; `&mut self` makes the view the only access to the
		// storage for its lifetime.
		unsafe {
			MatMut::from_raw_parts_mut(
				self.ptr,
				self.nrows,
				self.ncols,
				1,
				self.col_stride as isize,
			)
		}
	}
}

/// Drops the first `len` elements at `ptr` if a fill panics part way.
struct DropGuard<T> {
	ptr: *mut T,
	len: usize,
}

impl<T> Drop for DropGuard<T> {
	#[inline]
	fn drop(&mut self) {
		// SAFETY: the first `len` elements at `ptr` were written by the fill and are owned by
		// nobody else until the guard is forgotten.
		unsafe {
			core::ptr::drop_in_place(core::ptr::slice_from_raw_parts_mut(
				self.ptr, self.len,
			))
		};
	}
}

/// The storage of a temporary matrix, filled with `fill()` when `always` or when `T` needs
/// drop, otherwise left uninitialised.
#[track_caller]
fn temp_mat_storage<
	'a,
	T: ComplexField + 'a,
	Rows: Shape + 'a,
	Cols: Shape + 'a,
>(
	nrows: Rows,
	ncols: Cols,
	stack: &'a mut MemStack,
	always: bool,
	fill: fn() -> T,
) -> (DynMat<'a, T, Rows, Cols>, &'a mut MemStack) {
	let (align, col_stride, len) =
		temp_mat_layout::<T>(nrows.unbound(), ncols.unbound());
	let req = StackReq::new_aligned::<T>(len, align);
	equator::assert!(
		stack.can_hold(req),
		"the stack holds {} bytes; a {} x {} temporary matrix needs {} bytes at alignment {}",
		stack.len_bytes(),
		nrows.unbound(),
		ncols.unbound(),
		req.size_bytes(),
		req.align_bytes(),
	);
	let (uninit, stack) = stack.make_aligned_uninit::<T>(len, align);
	let ptr = uninit.as_mut_ptr() as *mut T;
	if always || core::mem::needs_drop::<T>() {
		let mut guard = DropGuard { ptr, len: 0 };
		for k in 0..len {
			// SAFETY: `k < len`, the length of `uninit`; the slot is uninitialised and owned
			// by this function, and `guard` counts it once written.
			unsafe { ptr.add(k).write(fill()) };
			guard.len += 1;
		}
		core::mem::forget(guard);
	}
	(
		DynMat {
			ptr,
			nrows,
			ncols,
			col_stride,
			__marker: PhantomData,
		},
		stack,
	)
}

/// creates a temporary matrix of uninit values, from the given memory stack.
///
/// # Safety
/// The returned matrix may contain uninitialized elements (types that need drop are filled with
/// `T::nan_impl` instead). The caller must initialize every element before reading it or forming a
/// reference to it, including through any view derived from the returned matrix, and no other
/// reference may alias its elements while it is alive.
///
/// # Panics
/// Panics if `stack` cannot hold [`temp_mat_scratch::<T>(nrows, ncols)`](temp_mat_scratch).
#[track_caller]
pub unsafe fn temp_mat_uninit<
	'a,
	T: ComplexField + 'a,
	Rows: Shape + 'a,
	Cols: Shape + 'a,
>(
	nrows: Rows,
	ncols: Cols,
	stack: &'a mut MemStack,
) -> (
	impl 'a + AsMatMut<T = T, Rows = Rows, Cols = Cols>,
	&'a mut MemStack,
) {
	temp_mat_storage(nrows, ncols, stack, false, T::nan_impl)
}

/// creates a temporary matrix of zero values, from the given memory stack.
///
/// # Panics
/// Panics if `stack` cannot hold [`temp_mat_scratch::<T>(nrows, ncols)`](temp_mat_scratch).
#[track_caller]
pub fn temp_mat_zeroed<
	'a,
	T: ComplexField + 'a,
	Rows: Shape + 'a,
	Cols: Shape + 'a,
>(
	nrows: Rows,
	ncols: Cols,
	stack: &'a mut MemStack,
) -> (
	impl 'a + AsMatMut<T = T, Rows = Rows, Cols = Cols>,
	&'a mut MemStack,
) {
	temp_mat_storage(nrows, ncols, stack, true, T::zero_impl)
}
