//! Free functions and extension methods over [`ComplexField`] and
//! [`RealField`], the way generic code calls scalar arithmetic.
//!
//! Ported from colmat-rs `faer-traits/src/lib.rs` at `71337cdc` (`math_utils`,
//! lines 24-212, and `ext`, lines 5710-5879). The precision functions follow
//! [`crate::Precision`]; `digits` replaces `nbits` and is derived from `eps`. faer's
//! MIT notice: `LICENSE-faer`.

use crate::ops::ByRef;
pub use crate::ops::{AddByRef, DivByRef, MulByRef, NegByRef, SubByRef};
use crate::scalar::{ComplexField, RealField, complex_abs};

/// The precision parameter of `T` ([`crate::Precision::eps`]).
#[inline(always)]
#[must_use]
pub fn eps<T: RealField>() -> T {
	T::eps()
}
/// [`crate::Precision::safe_min`].
#[inline(always)]
#[must_use]
pub fn safe_min<T: RealField>() -> T {
	T::safe_min()
}
/// [`crate::Precision::safe_max`].
#[inline(always)]
#[must_use]
pub fn safe_max<T: RealField>() -> T {
	T::safe_max()
}
/// `sqrt(safe_min)`.
#[inline(always)]
#[must_use]
pub fn sqrt_safe_min<T: RealField>() -> T {
	sqrt(&T::safe_min())
}
/// `sqrt(safe_max)`.
#[inline(always)]
#[must_use]
pub fn sqrt_safe_max<T: RealField>() -> T {
	sqrt(&T::safe_max())
}
/// The number of base-2 digits that `eps` corresponds to: the smallest `d`
/// with `eps * 2^(d - 1) >= 1`. For `f64` this is 53, for `f32` 24.
///
/// Returns 0 when `eps` is not positive and finite, which a valid
/// [`crate::Precision`] implementation never declares.
#[must_use]
pub fn digits<T: RealField>() -> u32 {
	let eps = T::eps();
	let one = one::<T>();
	let positive = matches!(
		eps.partial_cmp(&zero::<T>()),
		Some(core::cmp::Ordering::Greater)
	);
	if !positive || !is_finite(&eps) {
		return 0;
	}
	let two = from_f64::<T>(2.0);
	let mut x = eps;
	let mut d = 1u32;
	while x < one {
		x *= &two;
		d += 1;
	}
	d
}
/// Additive identity.
#[inline(always)]
#[must_use]
pub fn zero<T: ComplexField>() -> T {
	T::zero_impl()
}
/// Multiplicative identity.
#[inline(always)]
#[must_use]
pub fn one<T: ComplexField>() -> T {
	T::one_impl()
}
/// A NaN value.
#[inline(always)]
#[must_use]
pub fn nan<T: ComplexField>() -> T {
	T::nan_impl()
}
/// An infinite value.
#[inline(always)]
#[must_use]
pub fn infinity<T: ComplexField>() -> T {
	T::infinity_impl()
}
/// The real part of `value`, as a `T`.
#[inline(always)]
#[must_use]
pub fn as_real<T: ComplexField>(value: &T) -> T {
	T::from_real_impl(&T::real_part_impl(value))
}
/// Real part.
#[inline(always)]
#[must_use]
pub fn real<T: ComplexField>(value: &T) -> T::Real {
	T::real_part_impl(value)
}
/// Imaginary part.
#[inline(always)]
#[must_use]
pub fn imag<T: ComplexField>(value: &T) -> T::Real {
	T::imag_part_impl(value)
}
/// `-value`.
#[inline(always)]
#[track_caller]
#[must_use]
pub fn neg<T: NegByRef>(value: &T) -> T::Output {
	value.neg_by_ref()
}
/// A copy of `value`.
#[inline(always)]
#[must_use]
pub fn copy<T: ComplexField>(value: &T) -> T {
	T::copy_impl(value)
}
/// Complex conjugate.
#[inline(always)]
#[must_use]
pub fn conj<T: ComplexField>(value: &T) -> T {
	T::conj_impl(value)
}
/// `lhs + rhs`.
#[inline(always)]
#[track_caller]
#[must_use]
pub fn add<T: AddByRef<U>, U>(lhs: &T, rhs: &U) -> T::Output {
	lhs.add_by_ref(rhs)
}
/// `lhs - rhs`.
#[inline(always)]
#[track_caller]
#[must_use]
pub fn sub<T: SubByRef<U>, U>(lhs: &T, rhs: &U) -> T::Output {
	lhs.sub_by_ref(rhs)
}
/// `lhs * rhs`.
#[inline(always)]
#[track_caller]
#[must_use]
pub fn mul<T: MulByRef<U>, U>(lhs: &T, rhs: &U) -> T::Output {
	lhs.mul_by_ref(rhs)
}
/// `lhs / rhs`.
#[inline(always)]
#[track_caller]
#[must_use]
pub fn div<T: DivByRef<U>, U>(lhs: &T, rhs: &U) -> T::Output {
	lhs.div_by_ref(rhs)
}
/// `lhs * rhs` for a real `rhs`.
#[inline(always)]
#[must_use]
pub fn mul_real<T: ComplexField>(lhs: &T, rhs: &T::Real) -> T {
	T::mul_real_impl(lhs, rhs)
}
/// `lhs * rhs` for a real power of two `rhs`.
#[inline(always)]
#[must_use]
pub fn mul_pow2<T: ComplexField>(lhs: &T, rhs: &T::Real) -> T {
	T::mul_pow2_impl(lhs, rhs)
}
/// `|re| + |im|`.
#[inline(always)]
#[must_use]
pub fn abs1<T: ComplexField>(value: &T) -> T::Real {
	T::abs1_impl(value)
}
/// `|value|` for a real type and `|re| + |im|` for a complex one (colmat-rs's
/// definition, kept as is).
#[inline(always)]
#[must_use]
pub fn absmax<T: ComplexField>(value: &T) -> T::Real {
	if const { T::IS_REAL } {
		T::abs1_impl(value)
	} else {
		T::Real::abs1_impl(&real(value)) + T::Real::abs1_impl(&imag(value))
	}
}
/// Modulus.
#[inline(always)]
#[must_use]
pub fn abs<T: ComplexField>(value: &T) -> T::Real {
	T::abs_impl(value)
}
/// `sqrt(lhs² + rhs²)` without overflow or underflow in the squares.
#[inline(always)]
#[must_use]
pub fn hypot<T: RealField>(lhs: &T, rhs: &T) -> T {
	complex_abs(lhs.clone(), rhs.clone())
}
/// `re² + im²`.
#[inline(always)]
#[must_use]
pub fn abs2<T: ComplexField>(value: &T) -> T::Real {
	T::abs2_impl(value)
}
/// The larger of the two; `rhs` when they are unordered.
#[inline(always)]
#[must_use]
pub fn max<T: RealField>(lhs: &T, rhs: &T) -> T {
	if lhs > rhs { copy(lhs) } else { copy(rhs) }
}
/// The smaller of the two; `rhs` when they are unordered.
#[inline(always)]
#[must_use]
pub fn min<T: RealField>(lhs: &T, rhs: &T) -> T {
	if lhs < rhs { copy(lhs) } else { copy(rhs) }
}
/// Whether any component is NaN.
#[inline(always)]
#[must_use]
pub fn is_nan<T: ComplexField>(value: &T) -> bool {
	T::is_nan_impl(value)
}
/// Whether every component is finite.
#[inline(always)]
#[must_use]
pub fn is_finite<T: ComplexField>(value: &T) -> bool {
	T::is_finite_impl(value)
}
/// Principal square root.
#[inline(always)]
#[must_use]
pub fn sqrt<T: ComplexField>(value: &T) -> T {
	T::sqrt_impl(value)
}
/// `1 / value`.
#[inline(always)]
#[must_use]
pub fn recip<T: ComplexField>(value: &T) -> T {
	T::recip_impl(value)
}
/// The value with real part `value` and zero imaginary part.
#[inline(always)]
#[must_use]
pub fn from_real<T: ComplexField>(value: &T::Real) -> T {
	T::from_real_impl(value)
}
/// `value` rounded to `T`.
#[inline(always)]
#[must_use]
pub fn from_f64<T: ComplexField>(value: f64) -> T {
	T::from_f64_impl(value)
}

/// Method syntax for the functions above, on any [`ComplexField`].
pub trait ComplexFieldExt: ComplexField {
	/// Additive identity.
	#[inline(always)]
	#[must_use]
	fn zero() -> Self {
		Self::zero_impl()
	}
	/// Multiplicative identity.
	#[inline(always)]
	#[must_use]
	fn one() -> Self {
		Self::one_impl()
	}
	/// A NaN value.
	#[inline(always)]
	#[must_use]
	fn nan() -> Self {
		Self::nan_impl()
	}
	/// An infinite value.
	#[inline(always)]
	#[must_use]
	fn infinity() -> Self {
		Self::infinity_impl()
	}
	/// `value` rounded to `Self`.
	#[inline(always)]
	#[must_use]
	fn from_f64(value: f64) -> Self {
		Self::from_f64_impl(value)
	}
	/// The real part, as `Self`.
	#[inline(always)]
	#[must_use]
	fn as_real(&self) -> Self {
		as_real(self)
	}
	/// Real part.
	#[inline(always)]
	#[must_use]
	fn real(&self) -> Self::Real {
		Self::real_part_impl(self)
	}
	/// Imaginary part.
	#[inline(always)]
	#[must_use]
	fn imag(&self) -> Self::Real {
		Self::imag_part_impl(self)
	}
	/// A copy.
	#[inline(always)]
	#[must_use]
	fn copy(&self) -> Self {
		Self::copy_impl(self)
	}
	/// Complex conjugate.
	#[inline(always)]
	#[must_use]
	fn conj(&self) -> Self {
		Self::conj_impl(self)
	}
	/// `self * rhs` for a real `rhs`.
	#[inline(always)]
	#[must_use]
	fn mul_real(&self, rhs: impl ByRef<Self::Real>) -> Self {
		Self::mul_real_impl(self, rhs.by_ref())
	}
	/// `self * rhs` for a real power of two `rhs`.
	#[inline(always)]
	#[must_use]
	fn mul_pow2(&self, rhs: impl ByRef<Self::Real>) -> Self {
		Self::mul_pow2_impl(self, rhs.by_ref())
	}
	/// `|re| + |im|`.
	#[inline(always)]
	#[must_use]
	fn abs1(&self) -> Self::Real {
		Self::abs1_impl(self)
	}
	/// See [`absmax`].
	#[inline(always)]
	#[must_use]
	fn absmax(&self) -> Self::Real {
		absmax(self)
	}
	/// Modulus.
	#[inline(always)]
	#[must_use]
	fn abs(&self) -> Self::Real {
		Self::abs_impl(self)
	}
	/// `re² + im²`.
	#[inline(always)]
	#[must_use]
	fn abs2(&self) -> Self::Real {
		Self::abs2_impl(self)
	}
	/// Whether any component is NaN.
	#[inline(always)]
	#[must_use]
	fn is_nan(&self) -> bool {
		Self::is_nan_impl(self)
	}
	/// Whether every component is finite.
	#[inline(always)]
	#[must_use]
	fn is_finite(&self) -> bool {
		Self::is_finite_impl(self)
	}
	/// Principal square root.
	#[inline(always)]
	#[must_use]
	fn sqrt(&self) -> Self {
		Self::sqrt_impl(self)
	}
	/// `1 / self`.
	#[inline(always)]
	#[must_use]
	fn recip(&self) -> Self {
		Self::recip_impl(self)
	}
}

/// Method syntax for the real-only functions above.
pub trait RealFieldExt: RealField {
	/// The larger of the two; `rhs` when they are unordered.
	#[inline(always)]
	#[must_use]
	fn fmax(&self, rhs: impl ByRef<Self>) -> Self {
		max(self, rhs.by_ref())
	}
	/// The smaller of the two; `rhs` when they are unordered.
	#[inline(always)]
	#[must_use]
	fn fmin(&self, rhs: impl ByRef<Self>) -> Self {
		min(self, rhs.by_ref())
	}
	/// `sqrt(self² + rhs²)` without overflow or underflow in the squares.
	#[inline(always)]
	#[must_use]
	fn hypot(&self, rhs: impl ByRef<Self>) -> Self {
		hypot(self, rhs.by_ref())
	}
	/// `self` as the real part of a `T`.
	#[inline(always)]
	#[must_use]
	fn to_cplx<T: ComplexField<Real = Self>>(&self) -> T {
		T::from_real_impl(self)
	}
}

impl<T: ComplexField> ComplexFieldExt for T {}
impl<T: RealField> RealFieldExt for T {}

#[cfg(test)]
mod tests;
