//! Precision values and the scaled complex functions.
//!
//! The complex `abs`/`recip`/`sqrt` tests are ported from colmat-rs
//! `faer-traits/src/lib.rs` at `71337cdc` (lines 5880-6087), without the
//! `ComplexImpl` cases, which this crate does not have.

use super::*;
use crate::scalar::complex_recip;
use crate::test_support::Sf;
use crate::{Complex, Precision};

#[test]
fn native_precision_keeps_colmat_values() {
	assert_eq!(eps::<f64>(), f64::EPSILON);
	assert_eq!(safe_min::<f64>(), f64::MIN_POSITIVE);
	assert_eq!(safe_max::<f64>(), 1.0 / f64::MIN_POSITIVE);
	assert_eq!(eps::<f32>(), f32::EPSILON);
	assert_eq!(safe_min::<f32>(), f32::MIN_POSITIVE);
	assert_eq!(safe_max::<f32>(), 1.0 / f32::MIN_POSITIVE);
	// `safe_max` is finite and its reciprocal is `safe_min`.
	assert!(safe_max::<f64>().is_finite());
	assert_eq!(1.0 / safe_max::<f64>(), safe_min::<f64>());
}

#[test]
fn digits_from_eps() {
	assert_eq!(digits::<f64>(), f64::MANTISSA_DIGITS);
	assert_eq!(digits::<f32>(), f32::MANTISSA_DIGITS);
	assert_eq!(digits::<Sf>(), f64::MANTISSA_DIGITS);
}

#[test]
fn sqrt_safe_range() {
	assert_eq!(sqrt_safe_min::<f64>(), f64::MIN_POSITIVE.sqrt());
	assert_eq!(sqrt_safe_max::<f64>(), (1.0 / f64::MIN_POSITIVE).sqrt());
}

/// The scalar-only type gets the complex type and its scaled functions from
/// the blanket implementation.
#[test]
fn scalar_only_type_gets_complex_arithmetic() {
	let z = Complex::new(Sf(3.0), Sf(-4.0));
	assert_eq!(abs(&z), Sf(5.0));
	assert_eq!(abs2(&z), Sf(25.0));
	assert_eq!(conj(&z), Complex::new(Sf(3.0), Sf(4.0)));
	assert_eq!(
		sqrt(&Complex::new(Sf(3.0), Sf(4.0))),
		Complex::new(Sf(2.0), Sf(1.0))
	);
	assert_eq!(
		recip(&Complex::new(Sf(0.0), Sf(2.0))),
		Complex::new(Sf(0.0), Sf(-0.5))
	);
	assert_eq!(<Sf as Precision>::eps(), Sf(f64::EPSILON));
	const { assert!(!<Complex<Sf> as ComplexField>::IS_REAL) };
}

/// The reciprocal of a subnormal complex value is finite whenever the exact
/// result is (`tensor4all/faer#31`).
#[test]
fn recip_of_subnormal() {
	let (re, im) = complex_recip(0.0f64, f64::MIN_POSITIVE / 2.0);
	assert_eq!((re, im), (0.0, -2.0 / f64::MIN_POSITIVE));
	let (re, im) = complex_recip(f64::MIN_POSITIVE / 4.0, 0.0);
	assert_eq!((re, im), (4.0 / f64::MIN_POSITIVE, 0.0));
	let (re, im) = complex_recip(0.0f32, f32::MIN_POSITIVE / 2.0);
	assert_eq!((re, im), (0.0, -2.0 / f32::MIN_POSITIVE));
	// 1 / 2^-1074 exceeds the largest finite value.
	let (re, _) = complex_recip(f64::from_bits(1), 0.0);
	assert_eq!(re, f64::INFINITY);
}

/// `abs` of each `(re, im)`, bit for bit against the platform `hypot`.
macro_rules! assert_abs_is_hypot {
	($t:ty, $cases:expr) => {
		for (re, im) in $cases {
			let expected = <$t>::hypot(re, im).to_bits();
			let z = Complex::<$t> { re, im };
			assert_eq!(abs(&z).to_bits(), expected, "abs({re:e}, {im:e})");
			assert_eq!(
				hypot(&re, &im).to_bits(),
				expected,
				"hypot({re:e}, {im:e})"
			);
		}
	};
}

/// The modulus of a nonzero subnormal is exact and nonzero where `hypot`'s
/// is (`tensor4all/faer#274`).
// Subnormal inputs: the assertion compares the last bit, which Miri's floating
// point does not reproduce. Not a UB report, so nothing is lost there.
#[cfg_attr(miri, ignore)]
#[test]
fn abs_of_subnormal_complex() {
	let s = f32::from_bits;
	let d = f64::from_bits;
	assert_abs_is_hypot!(
		f32,
		[
			(0.0, s(1 << 19)),
			(0.0, s(1 << 9)),
			(-0.0, -s(1)),
			(s(3), s(4)),
			(s(3 * 2_372_667), s(4 * 2_372_667)),
			(-s(3 << 12), s(4 << 12)),
			(s(3 * 1_900_000), s(4 * 1_900_000)),
		]
	);
	assert_abs_is_hypot!(
		f64,
		[
			(0.0, d(1 << 44)),
			(0.0, d(1 << 4)),
			(d(3), -d(4)),
			(d(3 << 50), d(4 << 50)),
			(d(3 * 1_000_000_000_000_000), d(4 * 1_000_000_000_000_000)),
		]
	);
}

/// The modulus near `safe_max` is finite where `hypot`'s is
/// (`tensor4all/faer#274`).
#[cfg_attr(miri, ignore)]
#[test]
fn abs_near_max_positive() {
	assert_abs_is_hypot!(
		f32,
		[
			(f32::MAX, 0.0),
			(0.0, -f32::MAX),
			(3.0 * 2.0f32.powi(125), 4.0 * 2.0f32.powi(125)),
		]
	);
	assert_abs_is_hypot!(
		f64,
		[
			(f64::MAX, 0.0),
			(-3.0 * 2.0f64.powi(1021), 4.0 * 2.0f64.powi(1021))
		]
	);
	assert_eq!(abs(&Complex::new(f64::MAX, f64::MAX)), f64::INFINITY);
}

/// Zero, infinite and NaN inputs.
#[test]
fn abs_of_zero_inf_nan() {
	assert_eq!(abs(&Complex::new(0.0f64, -0.0)).to_bits(), 0);
	assert_eq!(abs(&Complex::new(f32::INFINITY, 1.0)), f32::INFINITY);
	assert_eq!(abs(&Complex::new(1.0, f64::NEG_INFINITY)), f64::INFINITY);
	assert!(abs(&Complex::new(f64::NAN, 1e-320)).is_nan());
	assert!(abs(&Complex::new(f32::MAX, f32::NAN)).is_nan());
}

/// `sqrt(s² (3 + 4i)) = s (2 + i)` for `s²` deep in the subnormals
/// (`tensor4all/faer#274`).
#[cfg_attr(miri, ignore)]
#[test]
fn sqrt_of_subnormal_complex() {
	let s = 2.0f32.powi(-74);
	let z = Complex::new(3.0 * s * s, 4.0 * s * s);
	assert_eq!(sqrt(&z), Complex::new(2.0 * s, s));
	let s = 2.0f64.powi(-537);
	let z = Complex::new(3.0 * s * s, -4.0 * s * s);
	assert_eq!(sqrt(&z), Complex::new(2.0 * s, -s));
}
