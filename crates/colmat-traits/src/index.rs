//! Index types for sparse structures and SIMD lane indices.
//!
//! Ported unchanged in behaviour from colmat-rs `faer-traits/src/lib.rs` at
//! `71337cdc` (lines 901-1160), with documentation added. faer's MIT notice:
//! `LICENSE-faer`.

mod seal {
	// Only the integer types below may implement the index traits.
	pub trait Seal {}
	impl Seal for u8 {}
	impl Seal for u16 {}
	impl Seal for u32 {}
	impl Seal for u64 {}
	impl Seal for u128 {}
	impl Seal for usize {}
	impl Seal for i32 {}
	impl Seal for i64 {}
	impl Seal for isize {}
}
/// Sealed marker: only the built-in integer types are indices.
pub trait Seal: seal::Seal {}
impl<T: seal::Seal> Seal for T {}
/// Trait for signed integers corresponding to the ones satisfying [`Index`].
///
/// Always smaller than or equal to `isize`.
pub trait SignedIndex:
	Seal
	+ core::fmt::Debug
	+ core::ops::Neg<Output = Self>
	+ core::ops::Add<Output = Self>
	+ core::ops::Sub<Output = Self>
	+ core::ops::AddAssign
	+ core::ops::SubAssign
	+ bytemuck::Pod
	+ Eq
	+ Ord
	+ Send
	+ Sync
{
	/// Maximum representable value.
	/// Maximum representable value.
	const MAX: Self;
	/// Truncate `value` to type [`Self`].
	#[must_use]
	fn truncate(value: usize) -> Self;
	/// Zero extend `self`.
	#[must_use]
	fn zx(self) -> usize;
	/// Sign extend `self`.
	#[must_use]
	fn sx(self) -> usize;
	/// Sum nonnegative values while checking for overflow.
	fn sum_nonnegative(slice: &[Self]) -> Option<Self> {
		let mut acc = Self::zeroed();
		for &i in slice {
			if Self::MAX - i < acc {
				return None;
			}
			acc += i;
		}
		Some(acc)
	}
}
impl SignedIndex for i32 {
	const MAX: Self = Self::MAX;

	#[inline(always)]
	fn truncate(value: usize) -> Self {
		value as isize as Self
	}

	#[inline(always)]
	fn zx(self) -> usize {
		self as u32 as usize
	}

	#[inline(always)]
	fn sx(self) -> usize {
		self as isize as usize
	}
}
impl SignedIndex for i64 {
	const MAX: Self = Self::MAX;

	#[inline(always)]
	fn truncate(value: usize) -> Self {
		value as isize as Self
	}

	#[inline(always)]
	fn zx(self) -> usize {
		self as u64 as usize
	}

	#[inline(always)]
	fn sx(self) -> usize {
		self as isize as usize
	}
}
impl SignedIndex for isize {
	const MAX: Self = Self::MAX;

	#[inline(always)]
	fn truncate(value: usize) -> Self {
		value as isize
	}

	#[inline(always)]
	fn zx(self) -> usize {
		self as usize
	}

	#[inline(always)]
	fn sx(self) -> usize {
		self as usize
	}
}
/// Unsigned integer usable as an index.
pub trait IndexCore:
	Seal
	+ core::fmt::Debug
	+ core::ops::Not<Output = Self>
	+ core::ops::BitAnd<Output = Self>
	+ core::ops::BitOr<Output = Self>
	+ core::ops::BitXor<Output = Self>
	+ core::ops::Add<Output = Self>
	+ core::ops::Sub<Output = Self>
	+ core::ops::AddAssign
	+ core::ops::SubAssign
	+ bytemuck::Pod
	+ Eq
	+ Ord
	+ Send
	+ Sync
	+ Ord
{
	/// Maximum representable value.
	const MAX: Self;
	/// Truncate `value` to type [`Self`].
	#[must_use]
	fn truncate(value: usize) -> Self;
	/// Zero extend `self`.
	#[must_use]
	fn zx(self) -> usize;
}
/// Unsigned index type with a signed counterpart of the same width.
pub trait Index: IndexCore {
	/// Equally-sized index type with a fixed size (no `usize`).
	type FixedWidth: Index;
	/// Equally-sized signed index type.
	type Signed: SignedIndex;
	/// Width in bits.
	const BITS: u32 = core::mem::size_of::<Self>() as u32 * 8;
	/// Convert a reference to a slice of [`Self`] to fixed width types.
	#[inline(always)]
	fn canonicalize(slice: &[Self]) -> &[Self::FixedWidth] {
		bytemuck::cast_slice(slice)
	}
	/// Convert a mutable reference to a slice of [`Self`] to fixed width types.
	#[inline(always)]
	fn canonicalize_mut(slice: &mut [Self]) -> &mut [Self::FixedWidth] {
		bytemuck::cast_slice_mut(slice)
	}
	/// Convert a signed value to an unsigned one.
	#[inline(always)]
	fn from_signed(value: Self::Signed) -> Self {
		bytemuck::cast(value)
	}
	/// Convert an unsigned value to a signed one.
	#[inline(always)]
	fn to_signed(self) -> Self::Signed {
		bytemuck::cast(self)
	}
	/// Sum values while checking for overflow.
	#[inline]
	fn sum_nonnegative(slice: &[Self]) -> Option<Self> {
		Self::Signed::sum_nonnegative(bytemuck::cast_slice(slice))
			.map(Self::from_signed)
	}
}
impl IndexCore for u8 {
	const MAX: Self = Self::MAX;

	#[inline(always)]
	fn truncate(value: usize) -> Self {
		value as _
	}

	#[inline(always)]
	fn zx(self) -> usize {
		self as _
	}
}
impl IndexCore for u16 {
	const MAX: Self = Self::MAX;

	#[inline(always)]
	fn truncate(value: usize) -> Self {
		value as _
	}

	#[inline(always)]
	fn zx(self) -> usize {
		self as _
	}
}
impl IndexCore for u32 {
	const MAX: Self = Self::MAX;

	#[inline(always)]
	fn truncate(value: usize) -> Self {
		value as _
	}

	#[inline(always)]
	fn zx(self) -> usize {
		self as _
	}
}
impl IndexCore for u64 {
	const MAX: Self = Self::MAX;

	#[inline(always)]
	fn truncate(value: usize) -> Self {
		value as _
	}

	#[inline(always)]
	fn zx(self) -> usize {
		self as _
	}
}
impl IndexCore for u128 {
	const MAX: Self = Self::MAX;

	#[inline(always)]
	fn truncate(value: usize) -> Self {
		value as _
	}

	#[inline(always)]
	fn zx(self) -> usize {
		self as _
	}
}
impl IndexCore for usize {
	const MAX: Self = Self::MAX;

	#[inline(always)]
	fn truncate(value: usize) -> Self {
		value
	}

	#[inline(always)]
	fn zx(self) -> usize {
		self
	}
}
impl Index for u32 {
	type FixedWidth = u32;
	type Signed = i32;
}
impl Index for u64 {
	type FixedWidth = u64;
	type Signed = i64;
}
impl Index for usize {
	#[cfg(target_pointer_width = "32")]
	type FixedWidth = u32;
	#[cfg(target_pointer_width = "64")]
	type FixedWidth = u64;
	type Signed = isize;
}
