//! The precision contract (S0).
//!
//! Replaces colmat-rs's six `RealField` constants (`epsilon_impl`,
//! `nbits_impl`, `min_positive_impl`, `max_positive_impl`,
//! `sqrt_min_positive_impl`, `sqrt_max_positive_impl`; `faer-traits/src/lib.rs`
//! at `71337cdc`, lines 1625-1647) with the three values of the design
//! document's section 5. The others are derived in [`crate::math_utils`]
//! (`digits`, `sqrt_safe_min`, `sqrt_safe_max`).

/// The precision parameter and safe range of a real scalar type.
///
/// The type declares its values; they are not derived from `f64`. Which
/// values an extended-precision type should declare is open (design document
/// section 5); the native types keep colmat-rs's values.
pub trait Precision: Sized {
	/// The precision parameter: for `f32` and `f64`, the machine epsilon
	/// (the distance from 1 to the next representable value).
	///
	/// Algorithm tolerances are multiples of this value. Routines that need
	/// the error bound of a particular operation or kernel take it from that
	/// operation's documentation, not from here.
	fn eps() -> Self;
	/// The smallest positive value with full relative precision whose
	/// reciprocal does not overflow.
	fn safe_min() -> Self;
	/// The reciprocal of [`Precision::safe_min`]: values above it may
	/// overflow when squared or inverted.
	fn safe_max() -> Self;
}

impl Precision for f64 {
	#[inline(always)]
	fn eps() -> Self {
		f64::EPSILON
	}

	#[inline(always)]
	fn safe_min() -> Self {
		f64::MIN_POSITIVE
	}

	#[inline(always)]
	fn safe_max() -> Self {
		1.0 / f64::MIN_POSITIVE
	}
}

impl Precision for f32 {
	#[inline(always)]
	fn eps() -> Self {
		f32::EPSILON
	}

	#[inline(always)]
	fn safe_min() -> Self {
		f32::MIN_POSITIVE
	}

	#[inline(always)]
	fn safe_max() -> Self {
		1.0 / f32::MIN_POSITIVE
	}
}
