//! Matrix views, iteration and layout copies for colmat-next (trial).
//!
//! This crate holds the view layer of the trial (design record §4): owned
//! matrices and vectors ([`Mat`], [`Col`], [`Row`]), their borrowed views
//! ([`MatRef`], [`MatMut`], …), diagonal and permutation matrices
//! ([`diag`], [`perm`]), element-wise iteration over several views of the
//! same shape ([`zip!`]), and serial copies between layouts ([`layout`]).
//!
//! It has no parallel route and no kernels: parallel traversal, the
//! workspace contract and the matrix product come in later steps of the
//! design record (§7), on top of these types.
//!
//! # The view invariant
//!
//! Every view carries a pointer, its dimensions and its strides. The unsafe
//! code of this crate relies on one invariant, stated on the raw constructors
//! ([`MatRef::from_raw_parts`], [`MatMut::from_raw_parts_mut`] and their
//! column, row and diagonal counterparts): for every in-bounds index
//! `(i, j)`, `ptr + i * row_stride + j * col_stride` points to an element of
//! one allocation that is valid for reads (and, for mutable views, writes,
//! with no other live reference to it) for the lifetime of the view. Safe
//! methods only build views that satisfy it; the `// SAFETY:` comments in
//! the view modules refer to it as "the view invariant".
//!
//! # Provenance
//!
//! Ported from colmat-rs at `71337cdc` (`faer/src`: `mat/`, `col/`, `row/`,
//! `diag/`, `perm/`, `linalg/zip.rs`, `utils/bound.rs`, parts of `lib.rs`
//! and `linalg/mod.rs`), a fork of faer
//! (<https://codeberg.org/sarah-quinones/faer>). faer is Copyright (c) 2026
//! sarah quiñones el kazdadi; the modifications in colmat-rs are Copyright
//! (c) tensor4all contributors; both are distributed under the MIT licence,
//! whose text is in this crate's `LICENSE-faer` (see also `NOTICE`). Each
//! module states what it ports and what was changed.
#![cfg_attr(not(feature = "std"), no_std)]
#![allow(non_snake_case)]
#![warn(rustdoc::broken_intra_doc_links)]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

/// see: [`generativity::make_guard`]
#[macro_export]
macro_rules! make_guard {
    ($($name:ident),* $(,)?) => {
        $(
        // SAFETY: the three steps below are `generativity::make_guard!`
        // expanded: the brand lives on this stack frame and the guard is
        // never moved out of it.
        #[allow(unused_unsafe)]
        let $name = unsafe { $crate::generativity::Id::new() };
        // SAFETY: as above.
        #[allow(unused, unused_unsafe)]
        let lifetime_brand = unsafe {
            $crate::generativity::LifetimeBrand::new(&$name)
        };
        // SAFETY: as above.
        #[allow(unused_unsafe)]
        let $name = unsafe { $crate::generativity::Guard::new($name) };
        )*
    };
}

use colmat_traits::*;
#[allow(unused_imports)]
use equator::{assert, debug_assert};

pub use colmat_traits::{c32, c64};

#[doc(hidden)]
#[macro_export]
macro_rules! with_dim {
    ($name:ident, $value:expr $(,)?) => {
        let __val__ = $value; $crate::make_guard!($name); let $name =
        $crate::utils::bound::Dim::new(__val__, $name);
    };
    ({ $(let $name:ident = $value:expr;)* }) => {
        $(let __val__ = $value; $crate::make_guard!($name); let $name =
        $crate::utils::bound::Dim::new(__val__, $name);)*
    };
}

/// zips together matrix of the same size, so that coefficient-wise operations
/// can be performed on their elements.
///
/// # note
/// the order in which the matrix elements are traversed is unspecified.
///
/// # example
/// ```
/// use colmat_core::{Mat, mat, unzip, zip};
/// let nrows = 2;
/// let ncols = 3;
/// let a = mat![[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]];
/// let b = mat![[7.0, 9.0, 11.0], [8.0, 10.0, 12.0]];
/// let mut sum = Mat::<f64>::zeros(nrows, ncols);
/// zip!(&mut sum, &a, &b).for_each(|unzip!(sum, a, b)| {
///     *sum = a + b;
/// });
/// for i in 0..nrows {
///     for j in 0..ncols {
///         assert_eq!(sum[(i, j)], a[(i, j)] + b[(i, j)]);
///     }
/// }
/// ```
#[macro_export]
macro_rules! zip {
    ($head:expr $(,)?) => {
        $crate::zip::LastEq($crate::zip::IntoView::into_view($head),
        ::core::marker::PhantomData)
    };
    ($head:expr, $($tail:expr),* $(,)?) => {
        $crate::zip::ZipEq::new($crate::zip::IntoView::into_view($head),
        $crate::zip!($($tail,)*))
    };
}

/// expands to the type of zipped items.
///
/// # example
/// ```
/// use colmat_core::{Mat, Zip, mat, unzip, zip};
/// let nrows = 2;
/// let ncols = 3;
/// let a = mat![[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]];
/// let b = mat![[7.0, 9.0, 11.0], [8.0, 10.0, 12.0]];
/// let mut sum = Mat::<f64>::zeros(nrows, ncols);
/// zip!(&mut sum, &a, &b).for_each(
///     |unzip!(sum, a, b): Zip!(&mut f64, &f64, &f64)| {
///         *sum = a + b;
///     },
/// );
/// for i in 0..nrows {
///     for j in 0..ncols {
///         assert_eq!(sum[(i, j)], a[(i, j)] + b[(i, j)]);
///     }
/// }
/// ```
#[macro_export]
macro_rules! Zip {
    (..$(,)?) => {
        _
    };
    ($head:ty $(,)?) => {
        $crate::zip::Last::<$head >
    };
    ($head:ty, $($tail:tt)*) => {
        $crate::zip::Zip::<$head, $crate::Zip!($($tail)*) >
    };
}

/// used to undo the zipping by the [`zip!`] macro.
///
/// # example
/// ```
/// use colmat_core::{Mat, mat, unzip, zip};
/// let nrows = 2;
/// let ncols = 3;
/// let a = mat![[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]];
/// let b = mat![[7.0, 9.0, 11.0], [8.0, 10.0, 12.0]];
/// let mut sum = Mat::<f64>::zeros(nrows, ncols);
/// zip!(&mut sum, &a, &b).for_each(|unzip!(sum, a, b)| {
///     *sum = a + b;
/// });
/// for i in 0..nrows {
///     for j in 0..ncols {
///         assert_eq!(sum[(i, j)], a[(i, j)] + b[(i, j)]);
///     }
/// }
/// ```
#[macro_export]
macro_rules! unzip {
    (..$(,)?) => {
        _
    };
    ($head:pat $(,)?) => {
        $crate::zip::Last($head)
    };
    ($head:pat, $($tail:tt)*) => {
        $crate::zip::Zip($head, $crate::unzip!($($tail)*))
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! __transpose_impl {
    ([$([$($col:expr),*])*] $($v:expr;)*) => {
        [$([$($col,)*],)* [$($v,)*]]
    };
    ([$([$($col:expr),*])*] $($v0:expr, $($v:expr),*;)*) => {
        $crate::__transpose_impl!([$([$($col),*])* [$($v0),*]] $($($v),*;)*)
    };
}

/// creates a [`Mat`] containing the arguments.
///
/// ```
/// use colmat_core::mat;
/// let matrix = mat![
///     [1.0, 5.0, 9.0], //
///     [2.0, 6.0, 10.0],
///     [3.0, 7.0, 11.0],
///     [4.0, 8.0, 12.0f64],
/// ];
/// assert_eq!(matrix[(0, 0)], 1.0);
/// assert_eq!(matrix[(1, 0)], 2.0);
/// assert_eq!(matrix[(2, 0)], 3.0);
/// assert_eq!(matrix[(3, 0)], 4.0);
/// assert_eq!(matrix[(0, 1)], 5.0);
/// assert_eq!(matrix[(1, 1)], 6.0);
/// assert_eq!(matrix[(2, 1)], 7.0);
/// assert_eq!(matrix[(3, 1)], 8.0);
/// assert_eq!(matrix[(0, 2)], 9.0);
/// assert_eq!(matrix[(1, 2)], 10.0);
/// assert_eq!(matrix[(2, 2)], 11.0);
/// assert_eq!(matrix[(3, 2)], 12.0);
/// ```
#[macro_export]
macro_rules! mat {
    () => {
        { compile_error!("number of columns in the matrix is ambiguous"); }
    };
    ($([$($v:expr),* $(,)?]),* $(,)?) => {
        {
            let __data = ::core::mem::ManuallyDrop::new(
                $crate::__transpose_impl!([] $($($v),*;)*)
            );
            let __data = &*__data;
            let __ncols = __data.len();
            let __nrows = (*__data.get(0).unwrap()).len();
            $crate::mat::Mat::from_fn(__nrows, __ncols, |i, j| {
                // SAFETY: `from_fn` visits each `(i, j)` once, so each
                // element of the `ManuallyDrop` array is moved out once and
                // the array itself is never dropped.
                #[allow(unused_unsafe)]
                unsafe { ::core::ptr::from_ref(&__data[j][i]).read() }
            })
        }
    };
}

/// creates a [`col::Col`] containing the arguments
///
/// ```
/// use colmat_core::col;
/// let col_vec = col![3.0, 5.0, 7.0, 9.0];
/// assert_eq!(col_vec[0], 3.0);
/// assert_eq!(col_vec[1], 5.0);
/// assert_eq!(col_vec[2], 7.0);
/// assert_eq!(col_vec[3], 9.0);
/// ```
#[macro_export]
macro_rules! col {
    ($($v:expr),* $(,)?) => {
        {
            let __data = ::core::mem::ManuallyDrop::new([$($v,)*]);
            let __data = &*__data;
            let __len = __data.len();
            $crate::col::Col::from_fn(__len, |i| {
                // SAFETY: `from_fn` visits each `i` once, so each element of
                // the `ManuallyDrop` array is moved out once and the array
                // itself is never dropped.
                #[allow(unused_unsafe)]
                unsafe { ::core::ptr::from_ref(&__data[i]).read() }
            })
        }
    };
}

/// creates a [`row::Row`] containing the arguments
///
/// ```
/// use colmat_core::row;
/// let row_vec = row![3.0, 5.0, 7.0, 9.0];
/// assert_eq!(row_vec[0], 3.0);
/// assert_eq!(row_vec[1], 5.0);
/// assert_eq!(row_vec[2], 7.0);
/// assert_eq!(row_vec[3], 9.0);
/// ```
#[macro_export]
macro_rules! row {
    ($($v:expr),* $(,)?) => {
        {
            let __data = ::core::mem::ManuallyDrop::new([$($v,)*]);
            let __data = &*__data;
            let __len = __data.len();
            $crate::row::Row::from_fn(__len, |i| {
                // SAFETY: `from_fn` visits each `i` once, so each element of
                // the `ManuallyDrop` array is moved out once and the array
                // itself is never dropped.
                #[allow(unused_unsafe)]
                unsafe { ::core::ptr::from_ref(&__data[i]).read() }
            })
        }
    };
}

/// convenience function to concatenate a nested list of matrices into a single
/// big ['Mat']. concatonation pattern follows the numpy.block convention that
/// each sub-list must have an equal number of columns (net) but the boundaries
/// do not need to align. in other words, this sort of thing:
/// ```notcode
///   AAAbb
///   AAAbb
///   cDDDD
/// ```
/// is perfectly acceptable
#[doc(hidden)]
#[track_caller]
pub fn concat_impl<T: ComplexField>(
	blocks: &[&[(mat::MatRef<'_, T>, Conj)]],
) -> mat::Mat<T> {
	#[inline(always)]
	fn count_total_columns<T: ComplexField>(
		block_row: &[(mat::MatRef<'_, T>, Conj)],
	) -> usize {
		let mut out: usize = 0;
		for (elem, _) in block_row.iter() {
			out += elem.ncols();
		}
		out
	}
	#[inline(always)]
	#[track_caller]
	fn count_rows<T: ComplexField>(
		block_row: &[(mat::MatRef<'_, T>, Conj)],
	) -> usize {
		let mut out: usize = 0;
		for (i, (e, _)) in block_row.iter().enumerate() {
			if i == 0 {
				out = e.nrows();
			} else {
				assert!(e.nrows() == out);
			}
		}
		out
	}
	let mut n: usize = 0;
	let mut m: usize = 0;
	for row in blocks.iter() {
		n += count_rows(row);
	}
	for (i, row) in blocks.iter().enumerate() {
		let cols = count_total_columns(row);
		if i == 0 {
			m = cols;
		} else {
			assert!(cols == m);
		}
	}
	let mut mat = mat::Mat::<T>::zeros(n, m);
	let mut ni: usize = 0;
	let mut mj: usize;
	for row in blocks.iter() {
		mj = 0;
		for (elem, conj) in row.iter() {
			let mut dst =
				mat.as_mut()
					.submatrix_mut(ni, mj, elem.nrows(), elem.ncols());
			if *conj == Conj::No {
				dst.copy_from(elem);
			} else {
				dst.copy_from(elem.conjugate());
			}
			mj += elem.ncols();
		}
		ni += row[0].0.nrows();
	}
	mat
}

/// column vector
pub mod col;
/// diagonal matrix
pub mod diag;
/// element-wise equality
mod eq;
/// copies between matrices of different layouts
pub mod layout;
/// rectangular matrix
pub mod mat;
/// permutation matrix
pub mod perm;
/// row vector
pub mod row;
/// helper utilities
pub mod utils;
/// element-wise iteration over several views of the same shape
pub mod zip;

#[cfg(all(test, feature = "std"))]
mod execution_tests;
#[cfg(all(test, feature = "std"))]
mod exhaustive_tests;
#[cfg(kani)]
mod kani_proofs;
#[cfg(all(test, feature = "std"))]
mod soundness_tests;
#[cfg(all(test, feature = "std"))]
mod test_support;
#[cfg(test)]
mod tests;

/// native unsigned integer type
pub trait Index: traits::IndexCore + traits::Index + seal::Seal {}
impl<T: colmat_traits::Index<Signed: seal::Seal> + seal::Seal> Index for T {}
mod seal {
	pub trait Seal {}
	impl<T: colmat_traits::Seal> Seal for T {}
	impl Seal for crate::utils::bound::Dim<'_> {}
	impl<I: crate::Index> Seal for crate::utils::bound::Idx<'_, I> {}
	impl<I: crate::Index> Seal for crate::utils::bound::IdxInc<'_, I> {}
	impl<I: crate::Index> Seal for crate::utils::bound::MaybeIdx<'_, I> {}
	impl<I: crate::Index> Seal for crate::utils::bound::IdxIncOne<I> {}
	impl<I: crate::Index> Seal for crate::utils::bound::MaybeIdxOne<I> {}
	impl Seal for crate::utils::bound::One {}
	impl Seal for crate::utils::bound::Zero {}
	impl Seal for crate::ContiguousFwd {}
	impl Seal for crate::ContiguousBwd {}
}

/// sealed trait for types that can be created from "unbound" values, as long as
/// their struct preconditions are upheld
pub trait Unbind<I = usize>:
	Send + Sync + Copy + core::fmt::Debug + seal::Seal
{
	/// creates new value
	/// # Safety
	/// safety invariants must be upheld
	unsafe fn new_unbound(idx: I) -> Self;
	/// returns the unbound value, unconstrained by safety invariants
	fn unbound(self) -> I;
}

/// type that can be used to index into a range
pub type Idx<Dim, I = usize> = <Dim as ShapeIdx>::Idx<I>;
/// type that can be used to partition a range
pub type IdxInc<Dim, I = usize> = <Dim as ShapeIdx>::IdxInc<I>;
/// either an index or a negative value
pub type MaybeIdx<Dim, I = usize> = <Dim as ShapeIdx>::MaybeIdx<I>;

/// Whether `Idx<N, I>` has the size and alignment of `I`. For the sealed
/// shapes this holds exactly when it is `I` or a `#[repr(transparent)]`
/// wrapper of it, which excludes [`One`](utils::bound::One) (its index is the
/// zero-sized [`Zero`](utils::bound::Zero)). The slice casts between index
/// types of different shapes check it.
#[inline(always)]
pub(crate) const fn idx_has_index_layout<N: ShapeIdx + ?Sized, I: Index>()
-> bool {
	core::mem::size_of::<Idx<N, I>>() == core::mem::size_of::<I>()
		&& core::mem::align_of::<Idx<N, I>>() == core::mem::align_of::<I>()
}

/// base trait for [`Shape`]
pub trait ShapeIdx {
	/// type that can be used to index into a range
	type Idx<I: Index>: Unbind<I> + Ord + Eq;
	/// type that can be used to partition a range
	type IdxInc<I: Index>: Unbind<I> + Ord + Eq + From<Idx<Self, I>>;
	/// either an index or a negative value
	type MaybeIdx<I: Index>: Unbind<I::Signed> + Ord + Eq;
}

/// matrix dimension
///
/// The provided methods create indices with [`Unbind::new_unbound`]; each
/// one either checks the bound first or is an `unsafe fn` that requires it.
pub trait Shape:
	Unbind
	+ Ord
	+ ShapeIdx<
		Idx<usize>: Ord + Eq + PartialOrd<Self>,
		IdxInc<usize>: Ord + Eq + PartialOrd<Self>,
	>
{
	/// whether the types involved have any safety invariants
	const IS_BOUND: bool = true;
	/// bind the current value using a invariant lifetime guard
	#[inline]
	fn bind<'n>(self, guard: generativity::Guard<'n>) -> utils::bound::Dim<'n> {
		utils::bound::Dim::new(self.unbound(), guard)
	}
	/// cast a slice of bound values to unbound values
	///
	/// # Panics
	///
	/// Panics if `Idx<Self, I>` does not have the layout of `I`: for
	/// [`One`](utils::bound::One), whose index is the zero-sized
	/// [`Zero`](utils::bound::Zero).
	#[inline]
	#[track_caller]
	fn cast_idx_slice<I: Index>(slice: &[Idx<Self, I>]) -> &[I] {
		core::assert!(
			idx_has_index_layout::<Self, I>(),
			"this shape's index is not stored as `I`"
		);
		// SAFETY: `Shape` is sealed, and every `Idx<Self, I>` is `I`, a
		// `#[repr(transparent)]` wrapper of `I` (`utils::bound`), or the
		// zero-sized `Zero`, which the assertion above excludes; so the slice
		// has the layout of `[I]` with the same length.
		unsafe { core::slice::from_raw_parts(slice.as_ptr() as _, slice.len()) }
	}
	/// cast a slice of bound values to unbound values
	#[inline]
	fn cast_idx_inc_slice<I: Index>(slice: &[IdxInc<Self, I>]) -> &[I] {
		// SAFETY: `Shape` is sealed, and every `IdxInc<Self, I>` is `I` or a
		// `#[repr(transparent)]` wrapper of `I` (`utils::bound`, including
		// `IdxIncOne`), so the slice has the layout of `[I]`.
		unsafe { core::slice::from_raw_parts(slice.as_ptr() as _, slice.len()) }
	}
	/// returns the index `0`, which is always valid
	#[inline(always)]
	fn start() -> IdxInc<Self> {
		// SAFETY: `0 <= n` for every dimension `n`.
		unsafe { IdxInc::<Self>::new_unbound(0) }
	}
	/// returns the incremented value, as an inclusive index
	#[inline(always)]
	fn next(idx: Idx<Self>) -> IdxInc<Self> {
		// SAFETY: `idx < n`, so `idx + 1 <= n`.
		unsafe { IdxInc::<Self>::new_unbound(idx.unbound() + 1) }
	}
	/// returns the last value, equal to the dimension
	#[inline(always)]
	fn end(self) -> IdxInc<Self> {
		// SAFETY: `n <= n`.
		unsafe { IdxInc::<Self>::new_unbound(self.unbound()) }
	}
	/// checks if the index is valid, returning `Some(_)` in that case
	#[inline(always)]
	fn idx(self, idx: usize) -> Option<Idx<Self>> {
		if idx < self.unbound() {
			// SAFETY: checked just above.
			Some(unsafe { Idx::<Self>::new_unbound(idx) })
		} else {
			None
		}
	}
	/// checks if the index is valid, returning `Some(_)` in that case
	#[inline(always)]
	fn idx_inc(self, idx: usize) -> Option<IdxInc<Self>> {
		if idx <= self.unbound() {
			// SAFETY: checked just above.
			Some(unsafe { IdxInc::<Self>::new_unbound(idx) })
		} else {
			None
		}
	}
	/// checks if the index is valid, and panics otherwise
	#[inline(always)]
	fn checked_idx(self, idx: usize) -> Idx<Self> {
		equator::assert!(idx < self.unbound());
		// SAFETY: asserted just above.
		unsafe { Idx::<Self>::new_unbound(idx) }
	}
	/// checks if the index is valid, and panics otherwise
	#[inline(always)]
	fn checked_idx_inc(self, idx: usize) -> IdxInc<Self> {
		equator::assert!(idx <= self.unbound());
		// SAFETY: asserted just above.
		unsafe { IdxInc::<Self>::new_unbound(idx) }
	}
	/// assumes the index is valid
	/// # Safety
	/// the index must be valid
	#[inline(always)]
	unsafe fn unchecked_idx(self, idx: usize) -> Idx<Self> {
		equator::debug_assert!(idx < self.unbound());
		// SAFETY: the caller guarantees `idx < n`.
		unsafe { Idx::<Self>::new_unbound(idx) }
	}
	/// assumes the index is valid
	/// # Safety
	/// the index must be valid
	#[inline(always)]
	unsafe fn unchecked_idx_inc(self, idx: usize) -> IdxInc<Self> {
		equator::debug_assert!(idx <= self.unbound());
		// SAFETY: the caller guarantees `idx <= n`.
		unsafe { IdxInc::<Self>::new_unbound(idx) }
	}
	/// returns an iterator over the indices between `from` and `to`
	#[inline(always)]
	fn indices(
		from: IdxInc<Self>,
		to: IdxInc<Self>,
	) -> impl Clone + ExactSizeIterator + DoubleEndedIterator<Item = Idx<Self>>
	{
		(from.unbound()..to.unbound()).map(
			#[inline(always)]
			// SAFETY: `i < to <= n`.
			|i| unsafe { Idx::<Self>::new_unbound(i) },
		)
	}
}

impl<T: Send + Sync + Copy + core::fmt::Debug + colmat_traits::Seal> Unbind<T>
	for T
{
	#[inline(always)]
	unsafe fn new_unbound(idx: T) -> Self {
		idx
	}

	#[inline(always)]
	fn unbound(self) -> T {
		self
	}
}
impl ShapeIdx for usize {
	type Idx<I: Index> = I;
	type IdxInc<I: Index> = I;
	type MaybeIdx<I: Index> = I::Signed;
}
impl Shape for usize {
	const IS_BOUND: bool = false;
}

/// stride distance between two consecutive elements along a given dimension
pub trait Stride:
	seal::Seal + core::fmt::Debug + Copy + Send + Sync + 'static
{
	/// the reversed stride type
	type Rev: Stride<Rev = Self>;
	/// returns the reversed stride
	fn rev(self) -> Self::Rev;
	/// returns the stride in elements
	fn element_stride(self) -> isize;
}
impl Stride for isize {
	type Rev = Self;

	#[inline(always)]
	fn rev(self) -> Self::Rev {
		-self
	}

	#[inline(always)]
	fn element_stride(self) -> isize {
		self
	}
}
/// contiguous stride equal to `+1`
#[derive(Copy, Clone, Debug)]
pub struct ContiguousFwd;
/// contiguous stride equal to `-1`
#[derive(Copy, Clone, Debug)]
pub struct ContiguousBwd;
impl Stride for ContiguousFwd {
	type Rev = ContiguousBwd;

	#[inline(always)]
	fn rev(self) -> Self::Rev {
		ContiguousBwd
	}

	#[inline(always)]
	fn element_stride(self) -> isize {
		1
	}
}
impl Stride for ContiguousBwd {
	type Rev = ContiguousFwd;

	#[inline(always)]
	fn rev(self) -> Self::Rev {
		ContiguousFwd
	}

	#[inline(always)]
	fn element_stride(self) -> isize {
		-1
	}
}

/// memory allocation error
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TryReserveError {
	///rRequired allocation does not fit within `isize` bytes
	CapacityOverflow,
	/// allocator could not provide an allocation with the requested layout
	AllocError {
		/// requested layout
		layout: core::alloc::Layout,
	},
}

/// determines whether the input should be implicitly conjugated or not
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Conj {
	/// no implicit conjugation
	No,
	/// implicit conjugation
	Yes,
}

/// determines which side of a self-adjoint matrix should be accessed
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Side {
	/// lower triangular half
	Lower,
	/// upper triangular half
	Upper,
}

impl Conj {
	/// returns `self == Conj::Yes`
	#[inline]
	pub const fn is_conj(self) -> bool {
		matches!(self, Conj::Yes)
	}

	/// returns the composition of `self` and `other`
	#[inline]
	pub const fn compose(self, other: Self) -> Self {
		match (self, other) {
			(Conj::No, Conj::No) => Conj::No,
			(Conj::Yes, Conj::Yes) => Conj::No,
			(Conj::No, Conj::Yes) => Conj::Yes,
			(Conj::Yes, Conj::No) => Conj::Yes,
		}
	}

	/// returns `Conj::No` if `T` is the canonical representation, otherwise
	/// `Conj::Yes`
	#[inline]
	pub const fn get<T: Conjugate>() -> Self {
		if T::IS_CANONICAL { Self::No } else { Self::Yes }
	}

	#[inline]
	pub(crate) fn apply<T: Conjugate>(value: &T) -> T::Canonical {
		// SAFETY: `Conjugate`'s contract: `T` and `T::Canonical` have the
		// same size, alignment and bit validity.
		let value = unsafe { &*(value as *const T as *const T::Canonical) };
		if const { matches!(Self::get::<T>(), Conj::Yes) } {
			T::Canonical::conj_impl(value)
		} else {
			T::Canonical::copy_impl(value)
		}
	}
}

pub use col::{Col, ColMut, ColRef};
pub use mat::{Mat, MatMut, MatRef, MatUninitMut};
pub use row::{Row, RowMut, RowRef};

#[allow(unused_imports, dead_code)]
mod internal_prelude {
	pub trait DivCeil: Sized {
		fn msrv_div_ceil(self, rhs: Self) -> Self;
		fn msrv_checked_next_multiple_of(self, rhs: Self) -> Option<Self>;
	}
	impl DivCeil for usize {
		#[inline]
		fn msrv_div_ceil(self, rhs: Self) -> Self {
			let d = self / rhs;
			let r = self % rhs;
			if r > 0 { d + 1 } else { d }
		}

		#[inline]
		fn msrv_checked_next_multiple_of(self, rhs: Self) -> Option<Self> {
			match self.checked_rem(rhs)? {
				0 => Some(self),
				r => self.checked_add(rhs - r),
			}
		}
	}
	pub(crate) use self::{unzip as uz, zip as z};
	pub(crate) use crate::col::{Col, ColMut, ColRef};
	pub(crate) use crate::diag::{Diag, DiagMut, DiagRef};
	pub(crate) use crate::mat::{
		AsMat, AsMatMut, AsMatRef, Mat, MatMut, MatRef,
	};
	pub(crate) use crate::perm::{Perm, PermRef};
	pub(crate) use crate::prelude::*;
	pub(crate) use crate::row::{AsRowMut, AsRowRef, Row, RowMut, RowRef};
	pub(crate) use crate::utils::bound::{Array, Dim, Idx, IdxInc, MaybeIdx};
	pub(crate) use crate::{
		Conj, ContiguousBwd, ContiguousFwd, Shape, Side, Stride, Unbind,
		make_guard, unzip, zip,
	};
	pub use colmat_traits::ext::*;
	pub use colmat_traits::math_utils::*;
	pub use colmat_traits::{
		ComplexConj, ComplexField, Conjugate, Index, IndexCore, Real,
		RealField, SignedIndex,
	};
	pub use equator::{
		assert, assert as Assert, debug_assert, debug_assert as DebugAssert,
	};
	pub use num_complex::Complex;
	pub use reborrow::*;
	#[cfg(test)]
	pub(crate) use {alloc::boxed::Box, alloc::vec, alloc::vec::Vec};
}

/// useful imports for general usage of the library
pub mod prelude {
	pub use super::{c32, c64, col, mat, row, unzip, zip};
	pub use col::{Col, ColMut, ColRef};
	pub use mat::{Mat, MatMut, MatRef, MatUninitMut};
	pub use reborrow::{IntoConst, Reborrow, ReborrowMut};
	pub use row::{Row, RowMut, RowRef};
	/// see [`Default`]
	#[inline]
	pub fn default<T: Default>() -> T {
		Default::default()
	}
}

mod into_range {
	use super::*;
	pub trait IntoRange<I> {
		type Len<N: Shape>: Shape;
		fn into_range(self, min: I, max: I) -> core::ops::Range<I>;
	}
	impl<I> IntoRange<I> for core::ops::Range<I> {
		type Len<N: Shape> = usize;

		#[inline]
		fn into_range(self, _: I, _: I) -> core::ops::Range<I> {
			self
		}
	}
	impl<I> IntoRange<I> for core::ops::RangeFrom<I> {
		type Len<N: Shape> = usize;

		#[inline]
		fn into_range(self, _: I, max: I) -> core::ops::Range<I> {
			self.start..max
		}
	}
	impl<I> IntoRange<I> for core::ops::RangeTo<I> {
		type Len<N: Shape> = usize;

		#[inline]
		fn into_range(self, min: I, _: I) -> core::ops::Range<I> {
			min..self.end
		}
	}
	impl<I> IntoRange<I> for core::ops::RangeFull {
		type Len<N: Shape> = N;

		#[inline]
		fn into_range(self, min: I, max: I) -> core::ops::Range<I> {
			min..max
		}
	}
}

pub extern crate colmat_traits as traits;
#[doc(hidden)]
pub extern crate generativity;
pub extern crate num_complex as complex;
pub extern crate reborrow;

extern crate self as colmat_core;

// Defined last on purpose: a `macro_rules!` macro is in scope by name for
// the code after it, and a crate-local `concat!` there would shadow
// `core::concat!`, which the standard and Kani (#17) assertion macros expand
// to. `#[macro_export]` still exports it as `colmat_core::concat!`.
/// concatenates the matrices in each row horizontally,
/// then concatenates the results vertically
///
/// `concat![[a0, a1, a2], [b1, b2]]` results in the matrix
/// ```notcode
/// [a0 | a1 | a2][b0 | b1]
/// ```
#[macro_export]
macro_rules! concat {
    () => {
        { compile_error!("number of columns in the matrix is ambiguous"); }
    };
    ($([$($v:expr),* $(,)?]),* $(,)?) => {
        { $crate::concat_impl(& [$(& [$(($v).as_ref().__canonicalize(),)*],)*]) }
    };
}
