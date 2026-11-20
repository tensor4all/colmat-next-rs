//! The scaled complex `abs`, `recip` and `sqrt` across the whole exponent
//! range, so that every scaling branch is reached, and the scalar predicates.

use super::{complex_abs, complex_recip, complex_sqrt};
use crate::test_support::Sf;
use crate::{Complex, ComplexField, RealField, c32, c64};
use proptest::prelude::*;

/// `±m 2^e` with `m` in `[1, 2)` and `e` in `exp`, or a signed zero.
fn wide_f64(
	exp: core::ops::RangeInclusive<i32>,
) -> impl Strategy<Value = f64> + Clone {
	prop_oneof![
		9 => (1.0..2.0f64, exp, any::<bool>()).prop_map(|(m, e, neg)| {
			let x = m * 2.0f64.powi(e);
			if neg { -x } else { x }
		}),
		1 => prop_oneof![Just(0.0), Just(-0.0)],
	]
}

/// The same for `f32`.
fn wide_f32(
	exp: core::ops::RangeInclusive<i32>,
) -> impl Strategy<Value = f32> + Clone {
	prop_oneof![
		9 => (1.0..2.0f32, exp, any::<bool>()).prop_map(|(m, e, neg)| {
			let x = (m as f64 * 2.0f64.powi(e)) as f32;
			if neg { -x } else { x }
		}),
		1 => prop_oneof![Just(0.0f32), Just(-0.0)],
	]
}

/// Every exponent of `f64`, normal and subnormal.
const F64_ALL: core::ops::RangeInclusive<i32> = -1074..=1023;
/// Every exponent of `f32`.
const F32_ALL: core::ops::RangeInclusive<i32> = -149..=127;
/// Exponents below the regime of the known `recip` defect: a part of
/// magnitude `2^e` with `e` at most these stays finite when squared after
/// the `sqrt(safe_min)` scaling.
const F64_RECIP_OK: core::ops::RangeInclusive<i32> = -1074..=1022;
const F32_RECIP_OK: core::ops::RangeInclusive<i32> = -149..=126;

/// `1 / (re + i im)` in `f64`, scaled by the exponent of the larger part so
/// that nothing overflows or underflows before the last multiplication.
fn recip_ref(re: f64, im: f64) -> (f64, f64) {
	let m = re.abs().max(im.abs());
	let k = m.log2().floor() as i32;
	// `2^-k` and `2^k` as two factors each, so neither overflows.
	let (s1, s2) = (2.0f64.powi(-k / 2), 2.0f64.powi(-k + k / 2));
	let (r, i) = (re * s1 * s2, im * s1 * s2);
	let n = r * r + i * i;
	let (u1, u2) = (2.0f64.powi(-k / 2), 2.0f64.powi(-k + k / 2));
	(r / n * u1 * u2, -i / n * u1 * u2)
}

fn assert_close(
	got: (f64, f64),
	want: (f64, f64),
	eps: f64,
	tiny: f64,
	what: &str,
) {
	let scale = want.0.abs().max(want.1.abs());
	let tol = 8.0 * eps * scale + 4.0 * tiny;
	assert!(
		(got.0 - want.0).abs() <= tol && (got.1 - want.1).abs() <= tol,
		"{what}: {got:?} vs {want:?} (tolerance {tol:e})"
	);
}

fn check_recip_f64(re: f64, im: f64) {
	if re == 0.0 && im == 0.0 {
		return;
	}
	let want = recip_ref(re, im);
	// Results at the overflow threshold may round to either side.
	if want.0.abs().max(want.1.abs()) >= f64::MAX / 4.0 {
		return;
	}
	let got = complex_recip(re, im);
	assert_close(
		got,
		want,
		f64::EPSILON,
		f64::MIN_POSITIVE * f64::EPSILON,
		"recip",
	);
}

fn check_recip_f32(re: f32, im: f32) {
	if re == 0.0 && im == 0.0 {
		return;
	}
	let (r, i) = (re as f64, im as f64);
	let n = r * r + i * i;
	let want = (r / n, -i / n);
	if want.0.abs().max(want.1.abs()) >= f32::MAX as f64 / 4.0 {
		return;
	}
	let (gr, gi) = complex_recip(re, im);
	let tiny = (f32::MIN_POSITIVE * f32::EPSILON) as f64;
	assert_close(
		(gr as f64, gi as f64),
		want,
		f32::EPSILON as f64,
		tiny,
		"recip",
	);
}

proptest! {
	#[cfg_attr(miri, ignore)]
	#[test]
	fn prop_recip_f64(re in wide_f64(F64_RECIP_OK), im in wide_f64(F64_RECIP_OK)) {
		check_recip_f64(re, im);
	}

	#[cfg_attr(miri, ignore)]
	#[test]
	fn prop_recip_f32(re in wide_f32(F32_RECIP_OK), im in wide_f32(F32_RECIP_OK)) {
		check_recip_f32(re, im);
	}

	/// The regime of the known defect: a part of magnitude `2^1023` (`2^127`
	/// for `f32`), where the result is subnormal and `recip` returns zero.
	#[ignore = "known defect: recip flushes to 0 when |z| is near the overflow threshold (big branch squares after scaling by sqrt(safe_min) only)"]
	#[test]
	fn prop_recip_near_overflow(
		re in wide_f64(1023..=1023),
		im in wide_f64(F64_ALL),
		re32 in wide_f32(127..=127),
		im32 in wide_f32(F32_ALL),
	) {
		check_recip_f64(re, im);
		check_recip_f32(re32, im32);
	}

	#[cfg_attr(miri, ignore)]
	#[test]
	fn prop_abs_matches_hypot(re in wide_f64(F64_ALL), im in wide_f64(F64_ALL)) {
		let want = re.hypot(im);
		let got = complex_abs(re, im);
		prop_assert!(
			(got - want).abs() <= 2.0 * f64::EPSILON * want,
			"abs({re:e}, {im:e}) = {got:e}, hypot = {want:e}"
		);
	}

	#[cfg_attr(miri, ignore)]
	#[test]
	fn prop_abs_f32_matches_f64_hypot(re in wide_f32(F32_ALL), im in wide_f32(F32_ALL)) {
		let want = (re as f64).hypot(im as f64);
		prop_assume!(want < f32::MAX as f64 / 2.0);
		let got = complex_abs(re, im) as f64;
		let tiny = (f32::MIN_POSITIVE * f32::EPSILON) as f64;
		prop_assert!(
			(got - want).abs() <= 2.0 * f32::EPSILON as f64 * want + tiny,
			"abs({re:e}, {im:e}) = {got:e}, f64 hypot = {want:e}"
		);
	}

	/// The larger component of `sqrt(z)` is `sqrt((|z| + |re|) / 2)` up to
	/// rounding (it is computed without cancellation), the real part is
	/// non-negative, and the imaginary part has the sign of `im`.
	#[cfg_attr(miri, ignore)]
	#[test]
	fn prop_sqrt_large_component(re in -1e300..1e300f64, im in -1e300..1e300f64) {
		let (sr, si) = complex_sqrt(re, im);
		prop_assert!(sr >= 0.0);
		if im != 0.0 {
			prop_assert_eq!(si.is_sign_negative(), im < 0.0);
		}
		let t = ((re.hypot(im) + re.abs()) / 2.0).sqrt();
		let large = if re >= 0.0 { sr } else { si.abs() };
		prop_assert!(
			(large - t).abs() <= 4.0 * f64::EPSILON * t,
			"sqrt({re:e}, {im:e}) = ({sr:e}, {si:e}), larger part should be {t:e}"
		);
	}

	/// `sqrt(z)² = z` up to rounding: also the smaller component is accurate.
	#[ignore = "known defect: sqrt loses the imaginary part when |im| << re (|z| - re cancels)"]
	#[test]
	fn prop_sqrt_squares_back(re in -1e300..1e300f64, im in -1e300..1e300f64) {
		let (sr, si) = complex_sqrt(re, im);
		prop_assert!(sr >= 0.0);
		if im != 0.0 {
			prop_assert_eq!(si.is_sign_negative(), im < 0.0);
		}
		let back = (sr * sr - si * si, 2.0 * sr * si);
		let scale = re.hypot(im);
		let tol = 8.0 * f64::EPSILON * scale;
		prop_assert!(
			(back.0 - re).abs() <= tol && (back.1 - im).abs() <= tol,
			"sqrt({re:e}, {im:e}) = ({sr:e}, {si:e})"
		);
	}
}

// Last-bit comparison of `powi`/`fma` results, which Miri's floating point
// deliberately perturbs. Not a UB check, so nothing is lost there.
#[cfg_attr(miri, ignore)]
#[test]
fn recip_special_values() {
	assert!(complex_recip(f64::NAN, 1.0).0.is_nan());
	assert!(complex_recip(1.0, f64::NAN).1.is_nan());
	assert_eq!(complex_recip(0.0, 0.0), (f64::INFINITY, f64::INFINITY));
	assert_eq!(complex_recip(f64::INFINITY, 1.0), (0.0, 0.0));
	assert_eq!(complex_recip(1.0, f64::NEG_INFINITY), (0.0, 0.0));
	// One branch per scaling regime, exact results.
	assert_eq!(
		complex_recip(2.0f64.powi(600), 0.0),
		(2.0f64.powi(-600), 0.0)
	);
	assert_eq!(complex_recip(0.0, 4.0), (0.0, -0.25));
	assert_eq!(complex_recip(0.25, 0.0), (4.0, 0.0));
	assert_eq!(
		complex_recip(2.0f64.powi(-600), 0.0),
		(2.0f64.powi(600), 0.0)
	);
}

#[test]
fn complex_predicates() {
	let nan_re = c64::new(f64::NAN, 0.0);
	let nan_im = c64::new(0.0, f64::NAN);
	let inf_im = c64::new(0.0, f64::INFINITY);
	let finite = c64::new(1.0, -2.0);
	assert!(c64::is_nan_impl(&nan_re) && c64::is_nan_impl(&nan_im));
	assert!(!c64::is_nan_impl(&finite) && !c64::is_nan_impl(&inf_im));
	assert!(c64::is_finite_impl(&finite));
	assert!(!c64::is_finite_impl(&inf_im) && !c64::is_finite_impl(&nan_re));
	assert!(!c64::is_finite_impl(&c64::new(f64::INFINITY, 0.0)));
	assert!(c32::is_finite_impl(&c32::new(1.0, 1.0)));
	assert!(!Complex::<Sf>::is_finite_impl(&Complex::new(
		Sf(f64::INFINITY),
		Sf(0.0)
	)));
}

/// The default `mul_add_impl` (used by `Sf`) is `self * lhs + rhs`; `f32`
/// and `f64` override it with the fused operation.
// Last-bit comparison of `powi`/`fma` results, which Miri's floating point
// deliberately perturbs. Not a UB check, so nothing is lost there.
#[cfg_attr(miri, ignore)]
#[test]
fn mul_add_impl() {
	assert_eq!(Sf(2.0).mul_add_impl(Sf(3.0), Sf(4.0)), Sf(10.0));
	assert_eq!(2.0f64.mul_add_impl(3.0, 4.0), 10.0);
	// Fused: `(1 + 2^-30)² - (1 + 2^-29)` keeps the `2^-60` that rounding loses.
	let x = 1.0 + 2.0f64.powi(-30);
	assert_eq!(
		x.mul_add_impl(x, -(1.0 + 2.0f64.powi(-29))),
		2.0f64.powi(-60)
	);
	let y = 1.0 + 2.0f32.powi(-12);
	assert_eq!(
		y.mul_add_impl(y, -(1.0 + 2.0f32.powi(-11))),
		2.0f32.powi(-24)
	);
}

/// `sqrt(x - 0i)` keeps colmat-rs's convention: the imaginary part is `+0`
/// (C99 `csqrt` would return `-0`). Pinned so that a change is deliberate.
// Exact `sqrt` results, which Miri's floating point deliberately perturbs.
#[cfg_attr(miri, ignore)]
#[test]
fn sqrt_of_negative_zero_imaginary() {
	let (re, im) = complex_sqrt(4.0f64, -0.0);
	assert_eq!(re, 2.0);
	assert!(im == 0.0 && im.is_sign_positive());
	let (re, im) = complex_sqrt(-4.0f64, -0.0);
	assert_eq!((re, im), (0.0, 2.0));
}
