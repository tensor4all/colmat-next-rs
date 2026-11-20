//! The scalar traits (S0): arithmetic and precision, without SIMD.
//!
//! Ported from colmat-rs `faer-traits/src/lib.rs` at `71337cdc`: the scalar
//! part of `ComplexField` and `RealField` (lines 1257-1307, 1625-1647), their
//! `f32`/`f64`/`Complex<T>` implementations (lines 1648-2763), and the scaled
//! complex `abs`, `recip` and `sqrt` (lines 337-419). The SIMD items of those
//! traits moved to [`crate::simd`]; the precision constants moved to
//! [`Precision`]. faer's MIT notice: `LICENSE-faer`.

use crate::conjugate::Conjugate;
use crate::math_utils::*;
use crate::ops::{NegByRef, RefOps, imply::Imply};
use crate::precision::Precision;
use crate::simd::{ComplexSimd, NativeSimd, SimdField, SimdPairs};
use core::fmt::Debug;
use num_complex::Complex;

/// A real or complex scalar: arithmetic, comparisons and the choice of SIMD
/// provider.
///
/// A new type implements the required `*_impl` functions and names a provider
/// in [`ComplexField::Simd`]; [`crate::NoSimd`] works for any type and runs
/// the vector paths one element at a time. Generic code calls the free
/// functions in [`crate::math_utils`] or the methods of [`crate::ext`] rather
/// than the `*_impl` functions.
pub trait ComplexField:
	Debug
	+ Clone
	+ Conjugate<Canonical = Self>
	+ PartialEq
	+ RefOps
	+ Imply<Self::Real, Is: RefOps>
{
	/// `true` for real types, `false` for complex ones.
	const IS_REAL: bool;
	/// The real type: `Self` for real types, the component type for complex
	/// ones.
	type Real: RealField;
	/// The SIMD provider for this type (S1). [`crate::NoSimd`] is always
	/// valid.
	type Simd: SimdField<Self>;

	/// Additive identity.
	fn zero_impl() -> Self;
	/// Multiplicative identity.
	fn one_impl() -> Self;
	/// A NaN value.
	fn nan_impl() -> Self;
	/// An infinite value.
	fn infinity_impl() -> Self;
	/// The value with real part `real` and zero imaginary part.
	fn from_real_impl(real: &Self::Real) -> Self;
	/// `real` rounded to `Self`.
	fn from_f64_impl(real: f64) -> Self;
	/// Real part.
	fn real_part_impl(value: &Self) -> Self::Real;
	/// Imaginary part (zero for real types).
	fn imag_part_impl(value: &Self) -> Self::Real;
	/// A copy of `value`.
	fn copy_impl(value: &Self) -> Self;
	/// Complex conjugate.
	fn conj_impl(value: &Self) -> Self;
	/// `1 / value`.
	fn recip_impl(value: &Self) -> Self;
	/// Principal square root.
	fn sqrt_impl(value: &Self) -> Self;
	/// Modulus, without overflow or underflow in intermediate squares.
	fn abs_impl(value: &Self) -> Self::Real;
	/// `|re| + |im|`.
	fn abs1_impl(value: &Self) -> Self::Real;
	/// `re² + im²`.
	fn abs2_impl(value: &Self) -> Self::Real;
	/// `lhs * rhs` for a real `rhs`.
	fn mul_real_impl(lhs: &Self, rhs: &Self::Real) -> Self;
	/// `lhs * rhs` for a real power of two `rhs` (exact unless it over- or
	/// underflows).
	fn mul_pow2_impl(lhs: &Self, rhs: &Self::Real) -> Self;
	/// Whether every component is finite.
	fn is_finite_impl(value: &Self) -> bool;
	/// Whether any component is NaN.
	// reason: `value != value` is the generic NaN test; `Self` has no is_nan.
	#[allow(clippy::eq_op)]
	fn is_nan_impl(value: &Self) -> bool {
		value != value
	}
}

/// A real scalar: ordered, with a precision contract.
///
/// The `num_traits` bounds are what `num_complex::Complex<Self>` needs for its
/// division operators (as in colmat-rs). Its SIMD provider must also load and store pairs of values, so that
/// `Complex<Self>` gets a provider derived from it ([`crate::ComplexSimd`]).
pub trait RealField:
	ComplexField<Real = Self, Conj = Self, Simd: SimdPairs<Self>>
	+ PartialOrd
	+ Precision
	+ num_traits::Num
	+ num_traits::NumAssign
{
	/// `self * lhs + rhs`, fused when the type has a fused multiply-add.
	///
	/// The default is the unfused expression; `f32` and `f64` override it
	/// with the hardware operation.
	#[inline(always)]
	fn mul_add_impl(&self, lhs: Self, rhs: Self) -> Self {
		self.clone() * lhs + rhs
	}
}

macro_rules! impl_native_real {
	($ty:ident, $sqrt:path, $fma:path) => {
		impl ComplexField for $ty {
			const IS_REAL: bool = true;
			type Real = Self;
			type Simd = NativeSimd;

			#[inline(always)]
			fn zero_impl() -> Self {
				0.0
			}

			#[inline(always)]
			fn one_impl() -> Self {
				1.0
			}

			#[inline(always)]
			fn nan_impl() -> Self {
				Self::NAN
			}

			#[inline(always)]
			fn infinity_impl() -> Self {
				Self::INFINITY
			}

			#[inline(always)]
			fn from_real_impl(value: &Self) -> Self {
				*value
			}

			#[inline(always)]
			fn from_f64_impl(value: f64) -> Self {
				value as _
			}

			#[inline(always)]
			fn real_part_impl(value: &Self) -> Self {
				*value
			}

			#[inline(always)]
			fn imag_part_impl(_: &Self) -> Self {
				0.0
			}

			#[inline(always)]
			fn copy_impl(value: &Self) -> Self {
				*value
			}

			#[inline(always)]
			fn conj_impl(value: &Self) -> Self {
				*value
			}

			#[inline(always)]
			fn recip_impl(value: &Self) -> Self {
				1.0 / *value
			}

			#[inline(always)]
			fn sqrt_impl(value: &Self) -> Self {
				$sqrt(*value)
			}

			#[inline(always)]
			fn abs_impl(value: &Self) -> Self {
				(*value).abs()
			}

			#[inline(always)]
			fn abs1_impl(value: &Self) -> Self {
				(*value).abs()
			}

			#[inline(always)]
			fn abs2_impl(value: &Self) -> Self {
				(*value) * (*value)
			}

			#[inline(always)]
			fn mul_real_impl(lhs: &Self, rhs: &Self) -> Self {
				(*lhs) * (*rhs)
			}

			#[inline(always)]
			fn mul_pow2_impl(lhs: &Self, rhs: &Self) -> Self {
				(*lhs) * (*rhs)
			}

			#[inline(always)]
			fn is_finite_impl(value: &Self) -> bool {
				(*value).is_finite()
			}

			#[inline(always)]
			fn is_nan_impl(value: &Self) -> bool {
				(*value).is_nan()
			}
		}

		impl RealField for $ty {
			#[inline(always)]
			fn mul_add_impl(&self, lhs: Self, rhs: Self) -> Self {
				$fma(*self, lhs, rhs)
			}
		}
	};
}

#[cfg(feature = "std")]
impl_native_real!(f64, f64::sqrt, f64::mul_add);
#[cfg(not(feature = "std"))]
impl_native_real!(f64, libm::sqrt, libm::fma);
#[cfg(feature = "std")]
impl_native_real!(f32, f32::sqrt, f32::mul_add);
#[cfg(not(feature = "std"))]
impl_native_real!(f32, libm::sqrtf, libm::fmaf);

impl<T: RealField> ComplexField for Complex<T> {
	const IS_REAL: bool = false;
	type Real = T;
	type Simd = ComplexSimd<T::Simd>;

	#[inline]
	fn zero_impl() -> Self {
		Complex {
			re: T::zero_impl(),
			im: T::zero_impl(),
		}
	}

	#[inline]
	fn one_impl() -> Self {
		Complex {
			re: T::one_impl(),
			im: T::zero_impl(),
		}
	}

	#[inline]
	fn nan_impl() -> Self {
		Complex {
			re: T::nan_impl(),
			im: T::nan_impl(),
		}
	}

	#[inline]
	fn infinity_impl() -> Self {
		Complex {
			re: T::infinity_impl(),
			im: T::infinity_impl(),
		}
	}

	#[inline]
	fn from_real_impl(real: &T) -> Self {
		Complex {
			re: real.clone(),
			im: T::zero_impl(),
		}
	}

	#[inline]
	fn from_f64_impl(real: f64) -> Self {
		Complex {
			re: T::from_f64_impl(real),
			im: T::zero_impl(),
		}
	}

	#[inline]
	fn real_part_impl(value: &Self) -> T {
		value.re.clone()
	}

	#[inline]
	fn imag_part_impl(value: &Self) -> T {
		value.im.clone()
	}

	#[inline]
	fn copy_impl(value: &Self) -> Self {
		value.clone()
	}

	#[inline]
	fn conj_impl(value: &Self) -> Self {
		Complex {
			re: value.re.clone(),
			im: value.im.neg_by_ref(),
		}
	}

	#[inline]
	fn recip_impl(value: &Self) -> Self {
		let (re, im) = complex_recip(value.re.clone(), value.im.clone());
		Complex { re, im }
	}

	#[inline]
	fn sqrt_impl(value: &Self) -> Self {
		let (re, im) = complex_sqrt(value.re.clone(), value.im.clone());
		Complex { re, im }
	}

	#[inline]
	fn abs_impl(value: &Self) -> T {
		complex_abs(value.re.clone(), value.im.clone())
	}

	#[inline]
	fn abs1_impl(value: &Self) -> T {
		T::abs1_impl(&value.re) + T::abs1_impl(&value.im)
	}

	#[inline]
	fn abs2_impl(value: &Self) -> T {
		T::abs2_impl(&value.re) + T::abs2_impl(&value.im)
	}

	#[inline]
	fn mul_real_impl(lhs: &Self, rhs: &T) -> Self {
		Complex {
			re: &lhs.re * rhs,
			im: &lhs.im * rhs,
		}
	}

	#[inline]
	fn mul_pow2_impl(lhs: &Self, rhs: &T) -> Self {
		Complex {
			re: T::mul_pow2_impl(&lhs.re, rhs),
			im: T::mul_pow2_impl(&lhs.im, rhs),
		}
	}

	#[inline]
	fn is_finite_impl(value: &Self) -> bool {
		T::is_finite_impl(&value.re) && T::is_finite_impl(&value.im)
	}

	#[inline]
	fn is_nan_impl(value: &Self) -> bool {
		T::is_nan_impl(&value.re) || T::is_nan_impl(&value.im)
	}
}

#[cfg(test)]
mod tests;

/// `|re + i im|` with scaling by exact powers of two, so that intermediate
/// squares neither overflow nor flush to zero.
pub(crate) fn complex_abs<T: RealField>(re: T, im: T) -> T {
	let (re, im) = &(re, im);
	let small = &sqrt_safe_min::<T>();
	let big = &sqrt_safe_max::<T>();
	let one = &one::<T>();
	let eps = &eps::<T>();
	let re_abs = &abs(re);
	let im_abs = &abs(im);
	// Exact powers of two that keep the square of the larger component, and
	// of any component within `sqrt(eps)` of it, normal. A component above
	// `big` scaled by `small` alone can square to `inf`, and one below `small`
	// scaled by `big` alone can square to a subnormal or zero, so those two
	// ranges scale further by `eps`.
	let (scale, unscale) = if re_abs > big || im_abs > big {
		(small * eps, big / eps)
	} else if re_abs > one || im_abs > one {
		(one.clone(), one.clone())
	} else if re_abs < small && im_abs < small {
		(big / eps, small * eps)
	} else {
		(big.clone(), small.clone())
	};
	sqrt(&(abs2(&(re * &scale)) + abs2(&(im * &scale)))) * unscale
}

/// `1 / (re + i im)` with the same scaling as [`complex_abs`].
pub(crate) fn complex_recip<T: RealField>(re: T, im: T) -> (T, T) {
	let (re, im) = &(re, im);
	if is_nan(re) || is_nan(im) {
		return (nan(), nan());
	}
	if *re == zero() && *im == zero() {
		return (infinity(), infinity());
	}
	if !is_finite(re) || !is_finite(im) {
		return (zero(), zero());
	}
	let small = &sqrt_safe_min::<T>();
	let big = &sqrt_safe_max::<T>();
	let one = &one::<T>();
	let re_abs = &abs(re);
	let im_abs = &abs(im);
	if re_abs > big || im_abs > big {
		let re = re * small;
		let im = im * small;
		let inv = &recip(&(abs2(&re) + abs2(&im)));
		((re * inv) * small, (-im * inv) * small)
	} else if re_abs > one || im_abs > one {
		let inv = &recip(&(abs2(re) + abs2(im)));
		(re * inv, -im * inv)
	} else {
		let re = re * big;
		let im = im * big;
		if &abs(&re) < small && &abs(&im) < small {
			// A subnormal input is still below `small` after one scaling,
			// where the squared magnitude's reciprocal would overflow.
			let re = re * big;
			let im = im * big;
			let inv = &recip(&(abs2(&re) + abs2(&im)));
			(((re * inv) * big) * big, ((-im * inv) * big) * big)
		} else {
			let inv = &recip(&(abs2(&re) + abs2(&im)));
			((re * inv) * big, (-im * inv) * big)
		}
	}
}

/// Principal square root of `re + i im`.
pub(crate) fn complex_sqrt<T: RealField>(re: T, im: T) -> (T, T) {
	let im_negative = im < zero();
	let half = &from_f64::<T>(0.5);
	let abs = &complex_abs(re.clone(), im);
	let zero = &zero::<T>();
	let re = &re;
	let mut sum = re + abs;
	if &sum < zero {
		sum = zero.clone();
	}
	let out_re = sqrt(&mul_pow2(&sum, half));
	let mut out_im = sqrt(&max(&mul_pow2(&(abs - re), half), zero));
	if im_negative {
		out_im = -out_im;
	}
	(out_re, out_im)
}
