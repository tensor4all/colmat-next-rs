//! SIMD providers (S1), split from the scalar trait.
//!
//! A scalar names its provider in [`ComplexField::Simd`]. A provider is a
//! zero-sized type that implements, for that scalar:
//!
//! - [`SimdLayout`]: data movement. Element loads and stores (memory holds
//!   whole elements, AoS; a register may hold them split, SoA), masks over
//!   logical lanes, splats, selects and lane indices;
//! - [`SimdField`]: arithmetic, comparisons and reductions on registers;
//! - [`SimdPairs`], for real scalars: loads and stores of elements made of two
//!   values, split into two registers. [`ComplexSimd`] builds the provider of
//!   `Complex<T>` from it, and a future double-word type can use it the same
//!   way.
//!
//! # Logical lanes
//!
//! A register holds [`SimdLayout::lanes`] elements. Lane positions are
//! logical: [`SimdLayout::iota`] gives the element index each lane holds, so
//! a provider whose load permutes physical lanes (as an AVX2 deinterleave
//! does) stays correct for masks, index selects and pivot searches, as long
//! as its load, store, `mask_between` and `iota` agree.
//!
//! # Real registers
//!
//! Some operations return or take *real registers*: registers of `T` whose
//! lanes carry a real value in the real part. Other components are
//! unspecified. Comparisons and `mul_real` read only the real part.
//!
//! Ported from the SIMD items of colmat-rs `ComplexField` and their `f32`,
//! `f64` and `Complex<T>` implementations (`faer-traits/src/lib.rs` at
//! `71337cdc`, lines 1179-1192, 1308-1623, 1648-2579, 2763-3246). faer's MIT
//! notice: `LICENSE-faer`.

mod complex;
mod native;
mod no_simd;

pub use complex::ComplexSimd;
pub use native::NativeSimd;
pub use no_simd::NoSimd;

use crate::index::IndexCore;
use crate::scalar::{ComplexField, RealField};
use core::fmt::Debug;
use pulp::Simd;

/// An instruction-set dispatcher: runs a [`pulp::WithSimd`] closure on the
/// best instruction set available.
pub trait SimdArch: Copy + Default + Send + Sync + Debug {
	/// Runs `f` with the selected instruction set.
	fn dispatch<R>(self, f: impl pulp::WithSimd<Output = R>) -> R;
}
impl SimdArch for pulp::Arch {
	#[inline]
	fn dispatch<R>(self, f: impl pulp::WithSimd<Output = R>) -> R {
		pulp::Arch::dispatch(self, f)
	}
}
impl SimdArch for pulp::Scalar {
	#[inline]
	fn dispatch<R>(self, f: impl pulp::WithSimd<Output = R>) -> R {
		f.with_simd(self)
	}
}

/// Data movement for registers of `T`.
///
/// # Safety
///
/// Callers rely on these properties when they hand memory to the unsafe
/// functions, so an implementation must guarantee them:
///
/// - [`mask_load`](Self::mask_load) reads, and
///   [`mask_store`](Self::mask_store) writes, exactly the elements
///   `ptr.add(i)` for the logical indices `i` enabled in the mask, and no
///   other memory. A load reads as [`core::ptr::read`] would but without
///   taking ownership (it clones); a store writes as [`core::ptr::write`].
/// - [`mem_mask_between`](Self::mem_mask_between)`(start, end)` enables
///   exactly the indices in `start..end` that are below
///   [`lanes`](Self::lanes).
/// - [`mask_between`](Self::mask_between)`(start, end)` enables exactly the
///   lanes whose [`iota`](Self::iota) value lies in `start..end`.
/// - A register loaded from `src` and stored to `dst` writes `src[i]` to
///   `dst[i]` for every enabled `i`.
pub unsafe trait SimdLayout<T: ComplexField>: 'static + Sized {
	/// `false` for [`NoSimd`], whose registers hold one element.
	const IS_SIMD: bool;
	/// The dispatcher to use for this provider.
	type Arch: SimdArch;
	/// Per-instruction-set context.
	type Ctx<S: Simd>: Copy + Debug;
	/// A register of `T`.
	type Vec<S: Simd>: Clone + Debug;
	/// A mask over logical lanes, for selects and comparisons.
	type Mask<S: Simd>: Copy + Debug;
	/// A mask over elements in memory, for partial loads and stores.
	type MemMask<S: Simd>: Copy + Debug;
	/// The scalar type of lane indices.
	type Index: IndexCore;
	/// A register of lane indices.
	type IndexVec<S: Simd>: Copy + Debug;

	/// The context for `simd`.
	fn ctx<S: Simd>(simd: S) -> Self::Ctx<S>;
	/// The instruction set of `ctx`.
	fn simd<S: Simd>(ctx: &Self::Ctx<S>) -> S;
	/// Elements per register.
	fn lanes<S: Simd>() -> usize;

	/// Lanes whose logical index lies in `start..end`.
	fn mask_between<S: Simd>(
		ctx: &Self::Ctx<S>,
		start: usize,
		end: usize,
	) -> Self::Mask<S>;
	/// Elements `start..end` (clamped to [`lanes`](Self::lanes)) of a
	/// register's worth of memory.
	fn mem_mask_between<S: Simd>(
		ctx: &Self::Ctx<S>,
		start: usize,
		end: usize,
	) -> Self::MemMask<S>;
	/// Loads the enabled elements of `ptr[0..lanes]`; other lanes are
	/// unspecified.
	///
	/// # Safety
	///
	/// For every enabled index `i`, `ptr.add(i)` must be valid for reads,
	/// aligned for `T` and point to an initialised `T`.
	unsafe fn mask_load<S: Simd>(
		ctx: &Self::Ctx<S>,
		mask: Self::MemMask<S>,
		ptr: *const T,
	) -> Self::Vec<S>;
	/// Stores the enabled lanes of `value` to `ptr[0..lanes]`.
	///
	/// # Safety
	///
	/// For every enabled index `i`, `ptr.add(i)` must be valid for writes and
	/// aligned for `T`. The previous value there is overwritten without being
	/// dropped.
	unsafe fn mask_store<S: Simd>(
		ctx: &Self::Ctx<S>,
		mask: Self::MemMask<S>,
		ptr: *mut T,
		value: Self::Vec<S>,
	);
	/// Loads `src`, which must hold exactly [`lanes`](Self::lanes) elements.
	///
	/// # Panics
	///
	/// When `src.len() != lanes`.
	fn load<S: Simd>(ctx: &Self::Ctx<S>, src: &[T]) -> Self::Vec<S>;
	/// Stores to `dst`, which must hold exactly [`lanes`](Self::lanes)
	/// elements.
	///
	/// # Panics
	///
	/// When `dst.len() != lanes`.
	fn store<S: Simd>(ctx: &Self::Ctx<S>, dst: &mut [T], value: Self::Vec<S>);

	/// Every lane set to `value`.
	fn splat<S: Simd>(ctx: &Self::Ctx<S>, value: &T) -> Self::Vec<S>;
	/// A real register with every lane set to `value`.
	fn splat_real<S: Simd>(ctx: &Self::Ctx<S>, value: &T::Real)
	-> Self::Vec<S>;
	/// `mask ? lhs : rhs` per lane.
	fn select<S: Simd>(
		ctx: &Self::Ctx<S>,
		mask: Self::Mask<S>,
		lhs: Self::Vec<S>,
		rhs: Self::Vec<S>,
	) -> Self::Vec<S>;

	/// The logical element index of each lane.
	fn iota<S: Simd>(ctx: &Self::Ctx<S>) -> Self::IndexVec<S>;
	/// `mask ? lhs : rhs` per lane, on index registers.
	fn index_select<S: Simd>(
		ctx: &Self::Ctx<S>,
		mask: Self::Mask<S>,
		lhs: Self::IndexVec<S>,
		rhs: Self::IndexVec<S>,
	) -> Self::IndexVec<S>;
	/// Every lane set to `value`.
	fn index_splat<S: Simd>(
		ctx: &Self::Ctx<S>,
		value: Self::Index,
	) -> Self::IndexVec<S>;
	/// Lane-wise wrapping `lhs + rhs`.
	fn index_add<S: Simd>(
		ctx: &Self::Ctx<S>,
		lhs: Self::IndexVec<S>,
		rhs: Self::IndexVec<S>,
	) -> Self::IndexVec<S>;
	/// Lane-wise `lhs < rhs`.
	fn index_less_than<S: Simd>(
		ctx: &Self::Ctx<S>,
		lhs: Self::IndexVec<S>,
		rhs: Self::IndexVec<S>,
	) -> Self::Mask<S>;

	/// Lane-wise `lhs && rhs`.
	fn and_mask<S: Simd>(
		ctx: &Self::Ctx<S>,
		lhs: Self::Mask<S>,
		rhs: Self::Mask<S>,
	) -> Self::Mask<S>;
	/// Lane-wise `lhs || rhs`.
	fn or_mask<S: Simd>(
		ctx: &Self::Ctx<S>,
		lhs: Self::Mask<S>,
		rhs: Self::Mask<S>,
	) -> Self::Mask<S>;
	/// Lane-wise `!mask`.
	fn not_mask<S: Simd>(
		ctx: &Self::Ctx<S>,
		mask: Self::Mask<S>,
	) -> Self::Mask<S>;
	/// The first physical lane that is set, or [`lanes`](Self::lanes) when
	/// none is. Use [`iota`](Self::iota) to translate it to an element index.
	fn first_true_mask<S: Simd>(
		ctx: &Self::Ctx<S>,
		mask: Self::Mask<S>,
	) -> usize;
}

/// Arithmetic, comparisons and reductions on registers of `T`.
///
/// Each operation is the lane-wise counterpart of the scalar operation of
/// [`ComplexField`]; reductions combine lanes in an unspecified order.
pub trait SimdField<T: ComplexField>: SimdLayout<T> {
	/// `lhs + rhs`.
	fn add<S: Simd>(
		ctx: &Self::Ctx<S>,
		lhs: Self::Vec<S>,
		rhs: Self::Vec<S>,
	) -> Self::Vec<S>;
	/// `lhs - rhs`.
	fn sub<S: Simd>(
		ctx: &Self::Ctx<S>,
		lhs: Self::Vec<S>,
		rhs: Self::Vec<S>,
	) -> Self::Vec<S>;
	/// `-value`.
	fn neg<S: Simd>(ctx: &Self::Ctx<S>, value: Self::Vec<S>) -> Self::Vec<S>;
	/// `conj(value)`.
	fn conj<S: Simd>(ctx: &Self::Ctx<S>, value: Self::Vec<S>) -> Self::Vec<S>;
	/// Real register of `|re| + |im|`.
	fn abs1<S: Simd>(ctx: &Self::Ctx<S>, value: Self::Vec<S>) -> Self::Vec<S>;
	/// Real register of `max(|re|, |im|)`; NaN when any component is NaN.
	fn abs_max<S: Simd>(
		ctx: &Self::Ctx<S>,
		value: Self::Vec<S>,
	) -> Self::Vec<S>;
	/// `lhs * rhs` for a real register `rhs`.
	fn mul_real<S: Simd>(
		ctx: &Self::Ctx<S>,
		lhs: Self::Vec<S>,
		real_rhs: Self::Vec<S>,
	) -> Self::Vec<S>;
	/// `lhs * rhs` for a real register `rhs` of powers of two.
	fn mul_pow2<S: Simd>(
		ctx: &Self::Ctx<S>,
		lhs: Self::Vec<S>,
		real_rhs: Self::Vec<S>,
	) -> Self::Vec<S>;
	/// `lhs * rhs`.
	fn mul<S: Simd>(
		ctx: &Self::Ctx<S>,
		lhs: Self::Vec<S>,
		rhs: Self::Vec<S>,
	) -> Self::Vec<S>;
	/// `conj(lhs) * rhs`.
	fn conj_mul<S: Simd>(
		ctx: &Self::Ctx<S>,
		lhs: Self::Vec<S>,
		rhs: Self::Vec<S>,
	) -> Self::Vec<S>;
	/// `lhs * rhs + acc`, fused where the provider can.
	fn mul_add<S: Simd>(
		ctx: &Self::Ctx<S>,
		lhs: Self::Vec<S>,
		rhs: Self::Vec<S>,
		acc: Self::Vec<S>,
	) -> Self::Vec<S>;
	/// `conj(lhs) * rhs + acc`, fused where the provider can.
	fn conj_mul_add<S: Simd>(
		ctx: &Self::Ctx<S>,
		lhs: Self::Vec<S>,
		rhs: Self::Vec<S>,
		acc: Self::Vec<S>,
	) -> Self::Vec<S>;
	/// Real register of `re² + im²`.
	fn abs2<S: Simd>(ctx: &Self::Ctx<S>, value: Self::Vec<S>) -> Self::Vec<S>;
	/// Real register of `re² + im² + acc` for a real register `acc`.
	fn abs2_add<S: Simd>(
		ctx: &Self::Ctx<S>,
		value: Self::Vec<S>,
		acc: Self::Vec<S>,
	) -> Self::Vec<S>;
	/// Sum of all lanes.
	fn reduce_sum<S: Simd>(ctx: &Self::Ctx<S>, value: Self::Vec<S>) -> T;
	/// Maximum over all lanes of a real register, in the real part of the
	/// result.
	fn reduce_max<S: Simd>(ctx: &Self::Ctx<S>, value: Self::Vec<S>) -> T;
	/// Lane-wise `lhs == rhs` on whole values.
	fn equal<S: Simd>(
		ctx: &Self::Ctx<S>,
		lhs: Self::Vec<S>,
		rhs: Self::Vec<S>,
	) -> Self::Mask<S>;
	/// Lane-wise `lhs < rhs` on real registers.
	fn less_than<S: Simd>(
		ctx: &Self::Ctx<S>,
		real_lhs: Self::Vec<S>,
		real_rhs: Self::Vec<S>,
	) -> Self::Mask<S>;
	/// Lane-wise `lhs <= rhs` on real registers.
	fn less_than_or_equal<S: Simd>(
		ctx: &Self::Ctx<S>,
		real_lhs: Self::Vec<S>,
		real_rhs: Self::Vec<S>,
	) -> Self::Mask<S>;
	/// Lane-wise `lhs > rhs` on real registers.
	fn greater_than<S: Simd>(
		ctx: &Self::Ctx<S>,
		real_lhs: Self::Vec<S>,
		real_rhs: Self::Vec<S>,
	) -> Self::Mask<S>;
	/// Lane-wise `lhs >= rhs` on real registers.
	fn greater_than_or_equal<S: Simd>(
		ctx: &Self::Ctx<S>,
		real_lhs: Self::Vec<S>,
		real_rhs: Self::Vec<S>,
	) -> Self::Mask<S>;
}

/// Loads and stores of elements made of two `T` (`[T; 2]`, or any
/// `#[repr(C)]` pair such as `Complex<T>`), split into one register per
/// component.
///
/// # Safety
///
/// As for [`SimdLayout`], applied to pairs: the pair functions read or write
/// exactly the pairs enabled in the mask, and the second register of a load
/// holds the second components of the same logical elements, in the same
/// lanes, as the first. [`iota2`](Self::iota2) gives the element index each
/// lane holds after [`load2`](Self::load2).
pub unsafe trait SimdPairs<T: RealField>: SimdField<T> {
	/// A mask over pairs in memory.
	type MemMask2<S: Simd>: Copy + Debug;

	/// Pairs `start..end` (clamped to [`lanes`](SimdLayout::lanes)) of two
	/// registers' worth of memory.
	fn mem_mask2_between<S: Simd>(
		ctx: &Self::Ctx<S>,
		start: usize,
		end: usize,
	) -> Self::MemMask2<S>;
	/// Loads the enabled pairs of `ptr[0..lanes]` as (first, second)
	/// components.
	///
	/// # Safety
	///
	/// For every enabled index `i`, `ptr.add(i)` must be valid for reads,
	/// aligned and point to an initialised pair.
	unsafe fn mask_load2<S: Simd>(
		ctx: &Self::Ctx<S>,
		mask: Self::MemMask2<S>,
		ptr: *const [T; 2],
	) -> (Self::Vec<S>, Self::Vec<S>);
	/// Stores the enabled lanes of `first` and `second` as pairs.
	///
	/// # Safety
	///
	/// For every enabled index `i`, `ptr.add(i)` must be valid for writes and
	/// aligned. Previous values are overwritten without being dropped.
	unsafe fn mask_store2<S: Simd>(
		ctx: &Self::Ctx<S>,
		mask: Self::MemMask2<S>,
		ptr: *mut [T; 2],
		first: Self::Vec<S>,
		second: Self::Vec<S>,
	);
	/// Loads `src`, which must hold exactly `lanes` pairs.
	///
	/// # Panics
	///
	/// When `src.len() != lanes`.
	fn load2<S: Simd>(
		ctx: &Self::Ctx<S>,
		src: &[[T; 2]],
	) -> (Self::Vec<S>, Self::Vec<S>);
	/// Stores to `dst`, which must hold exactly `lanes` pairs.
	///
	/// # Panics
	///
	/// When `dst.len() != lanes`.
	fn store2<S: Simd>(
		ctx: &Self::Ctx<S>,
		dst: &mut [[T; 2]],
		first: Self::Vec<S>,
		second: Self::Vec<S>,
	);
	/// The logical pair index of each lane after a pair load.
	fn iota2<S: Simd>(ctx: &Self::Ctx<S>) -> Self::IndexVec<S>;
}

#[cfg(test)]
mod tests;
