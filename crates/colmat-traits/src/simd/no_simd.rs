//! The one-lane provider, built on S0 alone.

use super::{SimdField, SimdLayout, SimdPairs};
use crate::math_utils::{abs1, abs2, conj, from_real, imag, is_nan, real};
use crate::scalar::{ComplexField, RealField};
use pulp::Simd;

/// The provider for a scalar without SIMD: a register is one element, a mask
/// is one `bool`, and every operation is the scalar one.
///
/// Any [`ComplexField`] can name it, so a scalar-only type reaches every path
/// written against [`SimdField`], one element per step.
#[derive(Copy, Clone, Debug, Default)]
pub struct NoSimd;

// SAFETY: a register is the single element at `ptr.add(0)`; it is read and
// written only when the one-lane mask is set, and `mask_between`,
// `mem_mask_between` and `iota` all describe lane 0 as element 0.
unsafe impl<T: ComplexField> SimdLayout<T> for NoSimd {
	const IS_SIMD: bool = false;
	type Arch = pulp::Scalar;
	type Ctx<S: Simd> = S;
	type Vec<S: Simd> = T;
	type Mask<S: Simd> = bool;
	type MemMask<S: Simd> = bool;
	type Index = usize;
	type IndexVec<S: Simd> = usize;

	#[inline(always)]
	fn ctx<S: Simd>(simd: S) -> S {
		simd
	}

	#[inline(always)]
	fn simd<S: Simd>(ctx: &S) -> S {
		*ctx
	}

	#[inline(always)]
	fn lanes<S: Simd>() -> usize {
		1
	}

	#[inline(always)]
	fn mask_between<S: Simd>(_: &S, start: usize, end: usize) -> bool {
		start == 0 && end > 0
	}

	#[inline(always)]
	fn mem_mask_between<S: Simd>(_: &S, start: usize, end: usize) -> bool {
		start == 0 && end > 0
	}

	#[inline(always)]
	unsafe fn mask_load<S: Simd>(_: &S, mask: bool, ptr: *const T) -> T {
		if mask {
			// SAFETY: the caller guarantees that `ptr` points to an
			// initialised, aligned `T` when the lane is enabled.
			unsafe { (*ptr).clone() }
		} else {
			T::zero_impl()
		}
	}

	#[inline(always)]
	unsafe fn mask_store<S: Simd>(_: &S, mask: bool, ptr: *mut T, value: T) {
		if mask {
			// SAFETY: the caller guarantees that `ptr` is valid for writes and
			// aligned when the lane is enabled.
			unsafe { ptr.write(value) }
		}
	}

	#[inline(always)]
	fn load<S: Simd>(_: &S, src: &[T]) -> T {
		assert_eq!(src.len(), 1, "a NoSimd register holds one element");
		src[0].clone()
	}

	#[inline(always)]
	fn store<S: Simd>(_: &S, dst: &mut [T], value: T) {
		assert_eq!(dst.len(), 1, "a NoSimd register holds one element");
		dst[0] = value;
	}

	#[inline(always)]
	fn splat<S: Simd>(_: &S, value: &T) -> T {
		value.clone()
	}

	#[inline(always)]
	fn splat_real<S: Simd>(_: &S, value: &T::Real) -> T {
		from_real(value)
	}

	#[inline(always)]
	fn select<S: Simd>(_: &S, mask: bool, lhs: T, rhs: T) -> T {
		if mask { lhs } else { rhs }
	}

	#[inline(always)]
	fn iota<S: Simd>(_: &S) -> usize {
		0
	}

	#[inline(always)]
	fn index_select<S: Simd>(
		_: &S,
		mask: bool,
		lhs: usize,
		rhs: usize,
	) -> usize {
		if mask { lhs } else { rhs }
	}

	#[inline(always)]
	fn index_splat<S: Simd>(_: &S, value: usize) -> usize {
		value
	}

	#[inline(always)]
	fn index_add<S: Simd>(_: &S, lhs: usize, rhs: usize) -> usize {
		lhs.wrapping_add(rhs)
	}

	#[inline(always)]
	fn index_less_than<S: Simd>(_: &S, lhs: usize, rhs: usize) -> bool {
		lhs < rhs
	}

	#[inline(always)]
	fn and_mask<S: Simd>(_: &S, lhs: bool, rhs: bool) -> bool {
		lhs && rhs
	}

	#[inline(always)]
	fn or_mask<S: Simd>(_: &S, lhs: bool, rhs: bool) -> bool {
		lhs || rhs
	}

	#[inline(always)]
	fn not_mask<S: Simd>(_: &S, mask: bool) -> bool {
		!mask
	}

	#[inline(always)]
	fn first_true_mask<S: Simd>(_: &S, mask: bool) -> usize {
		if mask { 0 } else { 1 }
	}
}

impl<T: ComplexField> SimdField<T> for NoSimd {
	#[inline(always)]
	fn add<S: Simd>(_: &S, lhs: T, rhs: T) -> T {
		lhs + rhs
	}

	#[inline(always)]
	fn sub<S: Simd>(_: &S, lhs: T, rhs: T) -> T {
		lhs - rhs
	}

	#[inline(always)]
	fn neg<S: Simd>(_: &S, value: T) -> T {
		-value
	}

	#[inline(always)]
	fn conj<S: Simd>(_: &S, value: T) -> T {
		conj(&value)
	}

	#[inline(always)]
	fn abs1<S: Simd>(_: &S, value: T) -> T {
		from_real(&abs1(&value))
	}

	#[inline(always)]
	fn abs_max<S: Simd>(_: &S, value: T) -> T {
		let re = abs1(&real(&value));
		let im = abs1(&imag(&value));
		// `re > im` is false when either is NaN; keep a NaN real part.
		let max = if re > im || is_nan(&re) { re } else { im };
		from_real(&max)
	}

	#[inline(always)]
	fn mul_real<S: Simd>(_: &S, lhs: T, real_rhs: T) -> T {
		T::mul_real_impl(&lhs, &real(&real_rhs))
	}

	#[inline(always)]
	fn mul_pow2<S: Simd>(_: &S, lhs: T, real_rhs: T) -> T {
		T::mul_pow2_impl(&lhs, &real(&real_rhs))
	}

	#[inline(always)]
	fn mul<S: Simd>(_: &S, lhs: T, rhs: T) -> T {
		lhs * rhs
	}

	#[inline(always)]
	fn conj_mul<S: Simd>(_: &S, lhs: T, rhs: T) -> T {
		conj(&lhs) * rhs
	}

	#[inline(always)]
	fn mul_add<S: Simd>(_: &S, lhs: T, rhs: T, acc: T) -> T {
		lhs * rhs + acc
	}

	#[inline(always)]
	fn conj_mul_add<S: Simd>(_: &S, lhs: T, rhs: T, acc: T) -> T {
		conj(&lhs) * rhs + acc
	}

	#[inline(always)]
	fn abs2<S: Simd>(_: &S, value: T) -> T {
		from_real(&abs2(&value))
	}

	#[inline(always)]
	fn abs2_add<S: Simd>(_: &S, value: T, acc: T) -> T {
		from_real(&(abs2(&value) + real(&acc)))
	}

	#[inline(always)]
	fn reduce_sum<S: Simd>(_: &S, value: T) -> T {
		value
	}

	#[inline(always)]
	fn reduce_max<S: Simd>(_: &S, value: T) -> T {
		value
	}

	#[inline(always)]
	fn equal<S: Simd>(_: &S, lhs: T, rhs: T) -> bool {
		lhs == rhs
	}

	#[inline(always)]
	fn less_than<S: Simd>(_: &S, real_lhs: T, real_rhs: T) -> bool {
		real(&real_lhs) < real(&real_rhs)
	}

	#[inline(always)]
	fn less_than_or_equal<S: Simd>(_: &S, real_lhs: T, real_rhs: T) -> bool {
		real(&real_lhs) <= real(&real_rhs)
	}

	#[inline(always)]
	fn greater_than<S: Simd>(_: &S, real_lhs: T, real_rhs: T) -> bool {
		real(&real_lhs) > real(&real_rhs)
	}

	#[inline(always)]
	fn greater_than_or_equal<S: Simd>(_: &S, real_lhs: T, real_rhs: T) -> bool {
		real(&real_lhs) >= real(&real_rhs)
	}
}

// SAFETY: a pair register is the single pair at `ptr.add(0)`, read and
// written only when the one-lane mask is set; `iota2` is 0 like `iota`.
unsafe impl<T: RealField> SimdPairs<T> for NoSimd {
	type MemMask2<S: Simd> = bool;

	#[inline(always)]
	fn mem_mask2_between<S: Simd>(_: &S, start: usize, end: usize) -> bool {
		start == 0 && end > 0
	}

	#[inline(always)]
	unsafe fn mask_load2<S: Simd>(
		_: &S,
		mask: bool,
		ptr: *const [T; 2],
	) -> (T, T) {
		if mask {
			// SAFETY: the caller guarantees that `ptr` points to an
			// initialised, aligned pair when the lane is enabled.
			let pair = unsafe { &*ptr };
			(pair[0].clone(), pair[1].clone())
		} else {
			(T::zero_impl(), T::zero_impl())
		}
	}

	#[inline(always)]
	unsafe fn mask_store2<S: Simd>(
		_: &S,
		mask: bool,
		ptr: *mut [T; 2],
		first: T,
		second: T,
	) {
		if mask {
			// SAFETY: the caller guarantees that `ptr` is valid for writes and
			// aligned when the lane is enabled.
			unsafe { ptr.write([first, second]) }
		}
	}

	#[inline(always)]
	fn load2<S: Simd>(_: &S, src: &[[T; 2]]) -> (T, T) {
		assert_eq!(src.len(), 1, "a NoSimd register holds one element");
		(src[0][0].clone(), src[0][1].clone())
	}

	#[inline(always)]
	fn store2<S: Simd>(_: &S, dst: &mut [[T; 2]], first: T, second: T) {
		assert_eq!(dst.len(), 1, "a NoSimd register holds one element");
		dst[0] = [first, second];
	}

	#[inline(always)]
	fn iota2<S: Simd>(_: &S) -> usize {
		0
	}
}
