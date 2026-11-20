//! The provider of `Complex<T>`, derived from the provider of `T`.
//!
//! Ported from colmat-rs's `impl ComplexField for Complex<T>` SIMD items
//! (`faer-traits/src/lib.rs` at `71337cdc`, lines 2763-3246). Registers are
//! split (one register of real parts, one of imaginary parts) and memory holds
//! whole `Complex<T>` values; the conversion happens in the pair loads and
//! stores of [`SimdPairs`]. The lane mask is built from `iota2` instead of a
//! plain lane range, so it stays correct when the pair shuffle permutes lanes.
//! faer's MIT notice: `LICENSE-faer`.

use super::{SimdField, SimdLayout, SimdPairs};
use crate::index::IndexCore;
use crate::scalar::RealField;
use core::marker::PhantomData;
use num_complex::Complex;
use pulp::Simd;

/// The provider of `Complex<T>` built on `P`, the provider of `T`.
///
/// A register is a `Complex` of two registers of `P` (real parts, imaginary
/// parts). A real register holds the same value in both.
pub struct ComplexSimd<P>(PhantomData<P>);

impl<P> core::fmt::Debug for ComplexSimd<P> {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		f.write_str("ComplexSimd")
	}
}

/// `Complex<T>` is `#[repr(C)] { re: T, im: T }` (documented by num-complex),
/// so it has the layout of `[T; 2]`.
#[inline(always)]
fn as_pairs<T>(src: &[Complex<T>]) -> &[[T; 2]] {
	// SAFETY: same size, alignment and element count; `[T; 2]` and
	// `Complex<T>` are both two consecutive `T` with no padding.
	unsafe {
		core::slice::from_raw_parts(src.as_ptr() as *const [T; 2], src.len())
	}
}
#[inline(always)]
fn as_pairs_mut<T>(dst: &mut [Complex<T>]) -> &mut [[T; 2]] {
	// SAFETY: as in `as_pairs`; the borrow is unique for its lifetime.
	unsafe {
		core::slice::from_raw_parts_mut(
			dst.as_mut_ptr() as *mut [T; 2],
			dst.len(),
		)
	}
}

// SAFETY: every memory access goes through `P`'s pair functions, which access
// exactly the enabled pairs (the `SimdPairs` contract) at `ptr.add(i)` with
// the layout of `Complex<T>` (see `as_pairs`). `mask_between` compares
// `iota2`, the logical index of each lane, against `start..end`, and `iota`
// returns the same `iota2`.
unsafe impl<T, P> SimdLayout<Complex<T>> for ComplexSimd<P>
where
	T: RealField,
	P: SimdPairs<T>,
{
	const IS_SIMD: bool = P::IS_SIMD;
	type Arch = P::Arch;
	type Ctx<S: Simd> = P::Ctx<S>;
	type Vec<S: Simd> = Complex<P::Vec<S>>;
	type Mask<S: Simd> = P::Mask<S>;
	type MemMask<S: Simd> = P::MemMask2<S>;
	type Index = P::Index;
	type IndexVec<S: Simd> = P::IndexVec<S>;

	#[inline(always)]
	fn ctx<S: Simd>(simd: S) -> Self::Ctx<S> {
		P::ctx(simd)
	}

	#[inline(always)]
	fn simd<S: Simd>(ctx: &Self::Ctx<S>) -> S {
		P::simd(ctx)
	}

	#[inline(always)]
	fn lanes<S: Simd>() -> usize {
		P::lanes::<S>()
	}

	#[inline(always)]
	fn mask_between<S: Simd>(
		ctx: &Self::Ctx<S>,
		start: usize,
		end: usize,
	) -> Self::Mask<S> {
		let lanes = P::lanes::<S>();
		let start = P::index_splat(ctx, P::Index::truncate(start.min(lanes)));
		let end = P::index_splat(ctx, P::Index::truncate(end.min(lanes)));
		let iota = P::iota2(ctx);
		P::and_mask(
			ctx,
			P::not_mask(ctx, P::index_less_than(ctx, iota, start)),
			P::index_less_than(ctx, iota, end),
		)
	}

	#[inline(always)]
	fn mem_mask_between<S: Simd>(
		ctx: &Self::Ctx<S>,
		start: usize,
		end: usize,
	) -> Self::MemMask<S> {
		P::mem_mask2_between(ctx, start, end)
	}

	#[inline(always)]
	unsafe fn mask_load<S: Simd>(
		ctx: &Self::Ctx<S>,
		mask: Self::MemMask<S>,
		ptr: *const Complex<T>,
	) -> Self::Vec<S> {
		// SAFETY: forwarded from the caller; `Complex<T>` has the layout of
		// `[T; 2]`.
		let (re, im) =
			unsafe { P::mask_load2(ctx, mask, ptr as *const [T; 2]) };
		Complex { re, im }
	}

	#[inline(always)]
	unsafe fn mask_store<S: Simd>(
		ctx: &Self::Ctx<S>,
		mask: Self::MemMask<S>,
		ptr: *mut Complex<T>,
		value: Self::Vec<S>,
	) {
		// SAFETY: forwarded from the caller; `Complex<T>` has the layout of
		// `[T; 2]`.
		unsafe {
			P::mask_store2(ctx, mask, ptr as *mut [T; 2], value.re, value.im)
		}
	}

	#[inline(always)]
	fn load<S: Simd>(ctx: &Self::Ctx<S>, src: &[Complex<T>]) -> Self::Vec<S> {
		let (re, im) = P::load2(ctx, as_pairs(src));
		Complex { re, im }
	}

	#[inline(always)]
	fn store<S: Simd>(
		ctx: &Self::Ctx<S>,
		dst: &mut [Complex<T>],
		value: Self::Vec<S>,
	) {
		P::store2(ctx, as_pairs_mut(dst), value.re, value.im)
	}

	#[inline(always)]
	fn splat<S: Simd>(ctx: &Self::Ctx<S>, value: &Complex<T>) -> Self::Vec<S> {
		Complex {
			re: P::splat(ctx, &value.re),
			im: P::splat(ctx, &value.im),
		}
	}

	#[inline(always)]
	fn splat_real<S: Simd>(ctx: &Self::Ctx<S>, value: &T) -> Self::Vec<S> {
		Complex {
			re: P::splat(ctx, value),
			im: P::splat(ctx, value),
		}
	}

	#[inline(always)]
	fn select<S: Simd>(
		ctx: &Self::Ctx<S>,
		mask: Self::Mask<S>,
		lhs: Self::Vec<S>,
		rhs: Self::Vec<S>,
	) -> Self::Vec<S> {
		Complex {
			re: P::select(ctx, mask, lhs.re, rhs.re),
			im: P::select(ctx, mask, lhs.im, rhs.im),
		}
	}

	#[inline(always)]
	fn iota<S: Simd>(ctx: &Self::Ctx<S>) -> Self::IndexVec<S> {
		P::iota2(ctx)
	}

	#[inline(always)]
	fn index_select<S: Simd>(
		ctx: &Self::Ctx<S>,
		mask: Self::Mask<S>,
		lhs: Self::IndexVec<S>,
		rhs: Self::IndexVec<S>,
	) -> Self::IndexVec<S> {
		P::index_select(ctx, mask, lhs, rhs)
	}

	#[inline(always)]
	fn index_splat<S: Simd>(
		ctx: &Self::Ctx<S>,
		value: Self::Index,
	) -> Self::IndexVec<S> {
		P::index_splat(ctx, value)
	}

	#[inline(always)]
	fn index_add<S: Simd>(
		ctx: &Self::Ctx<S>,
		lhs: Self::IndexVec<S>,
		rhs: Self::IndexVec<S>,
	) -> Self::IndexVec<S> {
		P::index_add(ctx, lhs, rhs)
	}

	#[inline(always)]
	fn index_less_than<S: Simd>(
		ctx: &Self::Ctx<S>,
		lhs: Self::IndexVec<S>,
		rhs: Self::IndexVec<S>,
	) -> Self::Mask<S> {
		P::index_less_than(ctx, lhs, rhs)
	}

	#[inline(always)]
	fn and_mask<S: Simd>(
		ctx: &Self::Ctx<S>,
		lhs: Self::Mask<S>,
		rhs: Self::Mask<S>,
	) -> Self::Mask<S> {
		P::and_mask(ctx, lhs, rhs)
	}

	#[inline(always)]
	fn or_mask<S: Simd>(
		ctx: &Self::Ctx<S>,
		lhs: Self::Mask<S>,
		rhs: Self::Mask<S>,
	) -> Self::Mask<S> {
		P::or_mask(ctx, lhs, rhs)
	}

	#[inline(always)]
	fn not_mask<S: Simd>(
		ctx: &Self::Ctx<S>,
		mask: Self::Mask<S>,
	) -> Self::Mask<S> {
		P::not_mask(ctx, mask)
	}

	#[inline(always)]
	fn first_true_mask<S: Simd>(
		ctx: &Self::Ctx<S>,
		mask: Self::Mask<S>,
	) -> usize {
		P::first_true_mask(ctx, mask)
	}
}

impl<T, P> SimdField<Complex<T>> for ComplexSimd<P>
where
	T: RealField,
	P: SimdPairs<T>,
{
	#[inline(always)]
	fn add<S: Simd>(
		ctx: &Self::Ctx<S>,
		lhs: Self::Vec<S>,
		rhs: Self::Vec<S>,
	) -> Self::Vec<S> {
		Complex {
			re: P::add(ctx, lhs.re, rhs.re),
			im: P::add(ctx, lhs.im, rhs.im),
		}
	}

	#[inline(always)]
	fn sub<S: Simd>(
		ctx: &Self::Ctx<S>,
		lhs: Self::Vec<S>,
		rhs: Self::Vec<S>,
	) -> Self::Vec<S> {
		Complex {
			re: P::sub(ctx, lhs.re, rhs.re),
			im: P::sub(ctx, lhs.im, rhs.im),
		}
	}

	#[inline(always)]
	fn neg<S: Simd>(ctx: &Self::Ctx<S>, value: Self::Vec<S>) -> Self::Vec<S> {
		Complex {
			re: P::neg(ctx, value.re),
			im: P::neg(ctx, value.im),
		}
	}

	#[inline(always)]
	fn conj<S: Simd>(ctx: &Self::Ctx<S>, value: Self::Vec<S>) -> Self::Vec<S> {
		Complex {
			re: value.re,
			im: P::neg(ctx, value.im),
		}
	}

	#[inline(always)]
	fn abs1<S: Simd>(ctx: &Self::Ctx<S>, value: Self::Vec<S>) -> Self::Vec<S> {
		let v = P::add(ctx, P::abs1(ctx, value.re), P::abs1(ctx, value.im));
		Complex {
			re: v.clone(),
			im: v,
		}
	}

	#[inline(always)]
	fn abs_max<S: Simd>(
		ctx: &Self::Ctx<S>,
		value: Self::Vec<S>,
	) -> Self::Vec<S> {
		let re = P::abs_max(ctx, value.re);
		let im = P::abs_max(ctx, value.im);
		let gt = P::greater_than(ctx, re.clone(), im.clone());
		// `re > im` is false when either side is NaN, and the select then
		// returns `im`: a NaN real part would otherwise read as the finite
		// imaginary magnitude. A NaN imaginary part already lands on `im`.
		let re_nan = P::not_mask(ctx, P::equal(ctx, re.clone(), re.clone()));
		let v = P::select(ctx, P::or_mask(ctx, gt, re_nan), re, im);
		Complex {
			re: v.clone(),
			im: v,
		}
	}

	#[inline(always)]
	fn mul_real<S: Simd>(
		ctx: &Self::Ctx<S>,
		lhs: Self::Vec<S>,
		real_rhs: Self::Vec<S>,
	) -> Self::Vec<S> {
		Complex {
			re: P::mul_real(ctx, lhs.re, real_rhs.re.clone()),
			im: P::mul_real(ctx, lhs.im, real_rhs.re),
		}
	}

	#[inline(always)]
	fn mul_pow2<S: Simd>(
		ctx: &Self::Ctx<S>,
		lhs: Self::Vec<S>,
		real_rhs: Self::Vec<S>,
	) -> Self::Vec<S> {
		Complex {
			re: P::mul_pow2(ctx, lhs.re, real_rhs.re.clone()),
			im: P::mul_pow2(ctx, lhs.im, real_rhs.re),
		}
	}

	#[inline(always)]
	fn mul<S: Simd>(
		ctx: &Self::Ctx<S>,
		lhs: Self::Vec<S>,
		rhs: Self::Vec<S>,
	) -> Self::Vec<S> {
		// re = l.re r.re - l.im r.im, im = l.re r.im + l.im r.re
		let im_im = P::mul(ctx, lhs.im.clone(), rhs.im.clone());
		let im_re = P::mul(ctx, lhs.im, rhs.re.clone());
		Complex {
			re: P::mul_add(ctx, lhs.re.clone(), rhs.re, P::neg(ctx, im_im)),
			im: P::mul_add(ctx, lhs.re, rhs.im, im_re),
		}
	}

	#[inline(always)]
	fn conj_mul<S: Simd>(
		ctx: &Self::Ctx<S>,
		lhs: Self::Vec<S>,
		rhs: Self::Vec<S>,
	) -> Self::Vec<S> {
		// re = l.re r.re + l.im r.im, im = l.re r.im - l.im r.re
		let im_im = P::mul(ctx, lhs.im.clone(), rhs.im.clone());
		let im_re = P::mul(ctx, lhs.im, rhs.re.clone());
		Complex {
			re: P::mul_add(ctx, lhs.re.clone(), rhs.re, im_im),
			im: P::mul_add(ctx, lhs.re, rhs.im, P::neg(ctx, im_re)),
		}
	}

	#[inline(always)]
	fn mul_add<S: Simd>(
		ctx: &Self::Ctx<S>,
		lhs: Self::Vec<S>,
		rhs: Self::Vec<S>,
		acc: Self::Vec<S>,
	) -> Self::Vec<S> {
		let re = P::mul_add(ctx, lhs.re.clone(), rhs.re.clone(), acc.re);
		let re =
			P::mul_add(ctx, P::neg(ctx, lhs.im.clone()), rhs.im.clone(), re);
		let im = P::mul_add(ctx, lhs.im, rhs.re, acc.im);
		let im = P::mul_add(ctx, lhs.re, rhs.im, im);
		Complex { re, im }
	}

	#[inline(always)]
	fn conj_mul_add<S: Simd>(
		ctx: &Self::Ctx<S>,
		lhs: Self::Vec<S>,
		rhs: Self::Vec<S>,
		acc: Self::Vec<S>,
	) -> Self::Vec<S> {
		let re = P::mul_add(ctx, lhs.re.clone(), rhs.re.clone(), acc.re);
		let re = P::mul_add(ctx, lhs.im.clone(), rhs.im.clone(), re);
		let im = P::mul_add(ctx, P::neg(ctx, lhs.im), rhs.re, acc.im);
		let im = P::mul_add(ctx, lhs.re, rhs.im, im);
		Complex { re, im }
	}

	#[inline(always)]
	fn abs2<S: Simd>(ctx: &Self::Ctx<S>, value: Self::Vec<S>) -> Self::Vec<S> {
		let v = P::abs2_add(ctx, value.re, P::abs2(ctx, value.im));
		Complex {
			re: v.clone(),
			im: v,
		}
	}

	#[inline(always)]
	fn abs2_add<S: Simd>(
		ctx: &Self::Ctx<S>,
		value: Self::Vec<S>,
		acc: Self::Vec<S>,
	) -> Self::Vec<S> {
		let v = P::abs2_add(ctx, value.re, P::abs2_add(ctx, value.im, acc.re));
		Complex {
			re: v.clone(),
			im: v,
		}
	}

	#[inline(always)]
	fn reduce_sum<S: Simd>(
		ctx: &Self::Ctx<S>,
		value: Self::Vec<S>,
	) -> Complex<T> {
		Complex {
			re: P::reduce_sum(ctx, value.re),
			im: P::reduce_sum(ctx, value.im),
		}
	}

	#[inline(always)]
	fn reduce_max<S: Simd>(
		ctx: &Self::Ctx<S>,
		value: Self::Vec<S>,
	) -> Complex<T> {
		Complex {
			re: P::reduce_max(ctx, value.re),
			im: P::reduce_max(ctx, value.im),
		}
	}

	#[inline(always)]
	fn equal<S: Simd>(
		ctx: &Self::Ctx<S>,
		lhs: Self::Vec<S>,
		rhs: Self::Vec<S>,
	) -> Self::Mask<S> {
		P::and_mask(
			ctx,
			P::equal(ctx, lhs.re, rhs.re),
			P::equal(ctx, lhs.im, rhs.im),
		)
	}

	#[inline(always)]
	fn less_than<S: Simd>(
		ctx: &Self::Ctx<S>,
		real_lhs: Self::Vec<S>,
		real_rhs: Self::Vec<S>,
	) -> Self::Mask<S> {
		P::less_than(ctx, real_lhs.re, real_rhs.re)
	}

	#[inline(always)]
	fn less_than_or_equal<S: Simd>(
		ctx: &Self::Ctx<S>,
		real_lhs: Self::Vec<S>,
		real_rhs: Self::Vec<S>,
	) -> Self::Mask<S> {
		P::less_than_or_equal(ctx, real_lhs.re, real_rhs.re)
	}

	#[inline(always)]
	fn greater_than<S: Simd>(
		ctx: &Self::Ctx<S>,
		real_lhs: Self::Vec<S>,
		real_rhs: Self::Vec<S>,
	) -> Self::Mask<S> {
		P::greater_than(ctx, real_lhs.re, real_rhs.re)
	}

	#[inline(always)]
	fn greater_than_or_equal<S: Simd>(
		ctx: &Self::Ctx<S>,
		real_lhs: Self::Vec<S>,
		real_rhs: Self::Vec<S>,
	) -> Self::Mask<S> {
		P::greater_than_or_equal(ctx, real_lhs.re, real_rhs.re)
	}
}
