//! Conjugation as a type-level view.
//!
//! Ported from colmat-rs `faer-traits/src/lib.rs` at `71337cdc` (lines
//! 863-883 and 1161-1178). faer's MIT notice: `LICENSE-faer`.

use crate::ops::{RefOps, imply::Imply};
use crate::scalar::{ComplexField, RealField};
use num_complex::Complex;

/// A scalar type, or the conjugate view of one.
///
/// Matrix views reinterpret their elements as `Self::Conj` or
/// `Self::Canonical` to conjugate lazily, without touching memory.
///
/// # Safety
///
/// `Self`, `Self::Conj` and `Self::Canonical` must have the same size,
/// alignment and bit validity: callers reinterpret a pointer to one as a
/// pointer to another, so the same bits must be a valid value of each.
/// `Self::Conj` reads those bits as the complex conjugate of `Self`, and
/// `IS_CANONICAL` must be `true` exactly when `Self` is `Self::Canonical`.
pub unsafe trait Conjugate:
	Send + Sync + core::fmt::Debug + Imply<Self::Canonical, Is: RefOps>
{
	/// Whether `Self` is its own canonical type.
	const IS_CANONICAL: bool;
	/// The conjugate view of `Self`.
	type Conj: Conjugate<Conj = Self, Canonical = Self::Canonical>;
	/// The scalar type that arithmetic is done in.
	type Canonical: Conjugate<Canonical = Self::Canonical> + ComplexField;
}

/// The real type of a scalar or of its conjugate view.
pub type Real<T> = <<T as Conjugate>::Canonical as ComplexField>::Real;

/// The bits of a `Complex<T>` read as its conjugate: the stored imaginary part
/// is the negated imaginary part of the value.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct ComplexConj<T> {
	/// Real part.
	pub re: T,
	/// Negated imaginary part.
	pub im_neg: T,
}

// SAFETY: a real type is its own conjugate and canonical type.
unsafe impl<T: RealField> Conjugate for T {
	type Canonical = T;
	type Conj = T;

	const IS_CANONICAL: bool = true;
}
// SAFETY: `Complex<T>` and `ComplexConj<T>` are both `#[repr(C)]` pairs of `T`
// (num-complex documents `Complex` as `#[repr(C)]`), so they share size,
// alignment and bit validity.
unsafe impl<T: RealField> Conjugate for Complex<T> {
	type Canonical = Complex<T>;
	type Conj = ComplexConj<T>;

	const IS_CANONICAL: bool = true;
}
// SAFETY: as for `Complex<T>` above.
unsafe impl<T: RealField> Conjugate for ComplexConj<T> {
	type Canonical = Complex<T>;
	type Conj = Complex<T>;

	const IS_CANONICAL: bool = false;
}
