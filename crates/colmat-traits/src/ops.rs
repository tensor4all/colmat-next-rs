//! Operator bounds shared by every scalar.
//!
//! Ported from colmat-rs `faer-traits/src/lib.rs` at `71337cdc` (`imply`,
//! `RefOps`, `ByRef` and the `*ByRef` traits, lines 9-20, 257-336, 420-434,
//! 1194-1256). The `imply` module there is marked "taken from imply-hack (MIT
//! license)". faer's MIT notice: `LICENSE-faer`.

/// Implied bounds on associated types (the imply-hack pattern), so that a
/// bound written once on a trait is visible at every use site.
pub mod imply {
	/// `Self` implies a bound on `T`: write `Imply<T, Is: Bound>` in a
	/// supertrait list to make `T: Bound` available wherever `Self` is used.
	pub trait Imply<T: ?Sized>: sealed::ImplyInner<T, Is = T> {}
	impl<T: ?Sized, U: ?Sized> Imply<T> for U {}
	mod sealed {
		pub trait ImplyInner<T: ?Sized> {
			type Is: ?Sized;
		}
		impl<T: ?Sized, U: ?Sized> ImplyInner<T> for U {
			type Is = T;
		}
	}
}

use imply::Imply;

/// `&lhs + &rhs` as a method, for generic code that holds references.
pub trait AddByRef<Rhs = Self> {
	/// Result type.
	type Output;
	/// Returns `self + rhs`.
	fn add_by_ref(&self, rhs: &Rhs) -> Self::Output;
}
/// `&lhs - &rhs` as a method.
pub trait SubByRef<Rhs = Self> {
	/// Result type.
	type Output;
	/// Returns `self - rhs`.
	fn sub_by_ref(&self, rhs: &Rhs) -> Self::Output;
}
/// `-&value` as a method.
pub trait NegByRef {
	/// Result type.
	type Output;
	/// Returns `-self`.
	fn neg_by_ref(&self) -> Self::Output;
}
/// `&lhs * &rhs` as a method.
pub trait MulByRef<Rhs = Self> {
	/// Result type.
	type Output;
	/// Returns `self * rhs`.
	fn mul_by_ref(&self, rhs: &Rhs) -> Self::Output;
}
/// `&lhs / &rhs` as a method.
pub trait DivByRef<Rhs = Self> {
	/// Result type.
	type Output;
	/// Returns `self / rhs`.
	fn div_by_ref(&self, rhs: &Rhs) -> Self::Output;
}

macro_rules! impl_by_ref {
	($tr:ident, $method:ident, $op:ident, $sym:tt) => {
		impl<Rhs, Lhs, Output> $tr<Rhs> for Lhs
		where
			for<'a> &'a Lhs: core::ops::$op<&'a Rhs, Output = Output>,
		{
			type Output = Output;

			#[inline]
			#[track_caller]
			fn $method(&self, rhs: &Rhs) -> Self::Output {
				self $sym rhs
			}
		}
	};
}
impl_by_ref!(AddByRef, add_by_ref, Add, +);
impl_by_ref!(SubByRef, sub_by_ref, Sub, -);
impl_by_ref!(MulByRef, mul_by_ref, Mul, *);
impl_by_ref!(DivByRef, div_by_ref, Div, /);

impl<T, Output> NegByRef for T
where
	for<'a> &'a T: core::ops::Neg<Output = Output>,
{
	type Output = Output;

	#[inline]
	#[track_caller]
	fn neg_by_ref(&self) -> Self::Output {
		-self
	}
}

/// Accepts either a value or a reference where a reference is needed.
pub trait ByRef<T> {
	/// Returns a reference to the value.
	fn by_ref(&self) -> &T;
}
impl<T> ByRef<T> for T {
	#[inline]
	fn by_ref(&self) -> &T {
		self
	}
}
impl<T> ByRef<T> for &T {
	#[inline]
	fn by_ref(&self) -> &T {
		self
	}
}

/// The arithmetic operators a scalar supports, on owned values and on
/// references, so that generic code never has to clone to add.
pub trait RefOps:
	Sized
	+ core::ops::Neg<Output = Self>
	+ core::ops::Add<Output = Self>
	+ core::ops::Sub<Output = Self>
	+ core::ops::Mul<Output = Self>
	+ core::ops::Div<Output = Self>
	+ core::ops::AddAssign
	+ core::ops::SubAssign
	+ core::ops::MulAssign
	+ core::ops::DivAssign
	+ for<'a> core::ops::Add<&'a Self, Output = Self>
	+ for<'a> core::ops::Sub<&'a Self, Output = Self>
	+ for<'a> core::ops::Mul<&'a Self, Output = Self>
	+ for<'a> core::ops::Div<&'a Self, Output = Self>
	+ for<'a> core::ops::AddAssign<&'a Self>
	+ for<'a> core::ops::SubAssign<&'a Self>
	+ for<'a> core::ops::MulAssign<&'a Self>
	+ for<'a> core::ops::DivAssign<&'a Self>
	+ for<'a> Imply<
		&'a Self,
		Is: core::ops::Neg<Output = Self>
		        + core::ops::Add<Output = Self>
		        + core::ops::Sub<Output = Self>
		        + core::ops::Mul<Output = Self>
		        + core::ops::Div<Output = Self>
		        + core::ops::Add<Self, Output = Self>
		        + core::ops::Sub<Self, Output = Self>
		        + core::ops::Mul<Self, Output = Self>
		        + core::ops::Div<Self, Output = Self>,
	>
{
}
impl<T> RefOps for T
where
	T: core::ops::Neg<Output = Self>
		+ core::ops::Add<Output = Self>
		+ core::ops::Sub<Output = Self>
		+ core::ops::Mul<Output = Self>
		+ core::ops::Div<Output = Self>
		+ core::ops::AddAssign
		+ core::ops::SubAssign
		+ core::ops::MulAssign
		+ core::ops::DivAssign
		+ for<'a> core::ops::Add<&'a T, Output = Self>
		+ for<'a> core::ops::Sub<&'a T, Output = Self>
		+ for<'a> core::ops::Mul<&'a T, Output = Self>
		+ for<'a> core::ops::Div<&'a T, Output = Self>
		+ for<'a> core::ops::AddAssign<&'a T>
		+ for<'a> core::ops::SubAssign<&'a T>
		+ for<'a> core::ops::MulAssign<&'a T>
		+ for<'a> core::ops::DivAssign<&'a T>,
	for<'a> &'a T: core::ops::Neg<Output = T>
		+ core::ops::Add<Output = T>
		+ core::ops::Sub<Output = T>
		+ core::ops::Mul<Output = T>
		+ core::ops::Div<Output = T>
		+ core::ops::Add<T, Output = T>
		+ core::ops::Sub<T, Output = T>
		+ core::ops::Mul<T, Output = T>
		+ core::ops::Div<T, Output = T>,
{
}
