//! Test-only scalars and value generators shared by the unit tests.

use crate::{Complex, ComplexField, NoSimd, Precision, RealField};
use core::ops::*;

/// A scalar-only real type: implements S0 and [`Precision`] and names
/// [`NoSimd`]. It is deliberately not `Copy`, so that the generic paths are
/// checked against types that must be cloned (an MPFR-like type).
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Sf(pub f64);

macro_rules! sf_binop {
	($tr:ident, $f:ident, $atr:ident, $af:ident, $op:tt) => {
		impl $tr for Sf {
			type Output = Sf;
			fn $f(self, rhs: Sf) -> Sf {
				Sf(self.0 $op rhs.0)
			}
		}
		impl $tr<&Sf> for Sf {
			type Output = Sf;
			fn $f(self, rhs: &Sf) -> Sf {
				Sf(self.0 $op rhs.0)
			}
		}
		impl $tr<Sf> for &Sf {
			type Output = Sf;
			fn $f(self, rhs: Sf) -> Sf {
				Sf(self.0 $op rhs.0)
			}
		}
		impl $tr<&Sf> for &Sf {
			type Output = Sf;
			fn $f(self, rhs: &Sf) -> Sf {
				Sf(self.0 $op rhs.0)
			}
		}
		impl $atr for Sf {
			fn $af(&mut self, rhs: Sf) {
				self.0 = self.0 $op rhs.0;
			}
		}
		impl $atr<&Sf> for Sf {
			fn $af(&mut self, rhs: &Sf) {
				self.0 = self.0 $op rhs.0;
			}
		}
	};
}
sf_binop!(Add, add, AddAssign, add_assign, +);
sf_binop!(Sub, sub, SubAssign, sub_assign, -);
sf_binop!(Mul, mul, MulAssign, mul_assign, *);
sf_binop!(Div, div, DivAssign, div_assign, /);
sf_binop!(Rem, rem, RemAssign, rem_assign, %);

impl Neg for Sf {
	type Output = Sf;
	fn neg(self) -> Sf {
		Sf(-self.0)
	}
}
impl Neg for &Sf {
	type Output = Sf;
	fn neg(self) -> Sf {
		Sf(-self.0)
	}
}
impl num_traits::Zero for Sf {
	fn zero() -> Self {
		Sf(0.0)
	}
	fn is_zero(&self) -> bool {
		self.0 == 0.0
	}
}
impl num_traits::One for Sf {
	fn one() -> Self {
		Sf(1.0)
	}
}
impl num_traits::Num for Sf {
	type FromStrRadixErr = num_traits::ParseFloatError;
	fn from_str_radix(
		s: &str,
		radix: u32,
	) -> Result<Self, Self::FromStrRadixErr> {
		f64::from_str_radix(s, radix).map(Sf)
	}
}
impl Precision for Sf {
	fn eps() -> Self {
		Sf(f64::EPSILON)
	}
	fn safe_min() -> Self {
		Sf(f64::MIN_POSITIVE)
	}
	fn safe_max() -> Self {
		Sf(1.0 / f64::MIN_POSITIVE)
	}
}

/// The whole S0 implementation of `Sf`: scalar functions only, no SIMD.
impl ComplexField for Sf {
	const IS_REAL: bool = true;
	type Real = Sf;
	type Simd = NoSimd;
	type Kernels = crate::GenericKernels;

	fn zero_impl() -> Self {
		Sf(0.0)
	}
	fn one_impl() -> Self {
		Sf(1.0)
	}
	fn nan_impl() -> Self {
		Sf(f64::NAN)
	}
	fn infinity_impl() -> Self {
		Sf(f64::INFINITY)
	}
	fn from_real_impl(real: &Sf) -> Self {
		real.clone()
	}
	fn from_f64_impl(real: f64) -> Self {
		Sf(real)
	}
	fn real_part_impl(value: &Self) -> Sf {
		value.clone()
	}
	fn imag_part_impl(_: &Self) -> Sf {
		Sf(0.0)
	}
	fn copy_impl(value: &Self) -> Self {
		value.clone()
	}
	fn conj_impl(value: &Self) -> Self {
		value.clone()
	}
	fn recip_impl(value: &Self) -> Self {
		Sf(1.0 / value.0)
	}
	fn sqrt_impl(value: &Self) -> Self {
		Sf(value.0.sqrt())
	}
	fn abs_impl(value: &Self) -> Sf {
		Sf(value.0.abs())
	}
	fn abs1_impl(value: &Self) -> Sf {
		Sf(value.0.abs())
	}
	fn abs2_impl(value: &Self) -> Sf {
		Sf(value.0 * value.0)
	}
	fn mul_real_impl(lhs: &Self, rhs: &Sf) -> Self {
		Sf(lhs.0 * rhs.0)
	}
	fn mul_pow2_impl(lhs: &Self, rhs: &Sf) -> Self {
		Sf(lhs.0 * rhs.0)
	}
	fn is_finite_impl(value: &Self) -> bool {
		value.0.is_finite()
	}
}
impl RealField for Sf {}

/// Builds test values of any scalar from two `f64`s (the imaginary part is
/// dropped for real types).
pub trait TestValue: ComplexField {
	/// The precision parameter of the real type, as `f64`.
	const EPS: f64;
	/// The spacing of subnormals: the absolute error floor of one rounding.
	const TINY: f64;
	fn make(re: f64, im: f64) -> Self;
	/// Real and imaginary parts, as `f64` (exact for every test scalar).
	fn parts(&self) -> (f64, f64);
}
impl TestValue for f64 {
	const EPS: f64 = f64::EPSILON;
	const TINY: f64 = f64::MIN_POSITIVE * f64::EPSILON;
	fn make(re: f64, _: f64) -> Self {
		re
	}
	fn parts(&self) -> (f64, f64) {
		(*self, 0.0)
	}
}
impl TestValue for f32 {
	const EPS: f64 = f32::EPSILON as f64;
	const TINY: f64 = (f32::MIN_POSITIVE * f32::EPSILON) as f64;
	fn make(re: f64, _: f64) -> Self {
		re as f32
	}
	fn parts(&self) -> (f64, f64) {
		(*self as f64, 0.0)
	}
}
impl TestValue for Sf {
	const EPS: f64 = f64::EPSILON;
	const TINY: f64 = f64::MIN_POSITIVE * f64::EPSILON;
	fn make(re: f64, _: f64) -> Self {
		Sf(re)
	}
	fn parts(&self) -> (f64, f64) {
		(self.0, 0.0)
	}
}
impl<T: TestValue + RealField> TestValue for Complex<T> {
	const EPS: f64 = T::EPS;
	const TINY: f64 = T::TINY;
	fn make(re: f64, im: f64) -> Self {
		Complex {
			re: T::make(re, 0.0),
			im: T::make(im, 0.0),
		}
	}
	fn parts(&self) -> (f64, f64) {
		(self.re.parts().0, self.im.parts().0)
	}
}

/// Small exactly representable test values, distinct for distinct `i`.
pub fn value<T: TestValue>(i: usize) -> T {
	let i = i as f64;
	T::make(1.5 + i, 0.25 - 2.0 * i)
}
