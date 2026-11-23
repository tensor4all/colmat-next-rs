//! Scalar contract and optional SIMD providers for colmat-next (trial).
//!
//! The layers follow [#2] and the design document's section 7a (S0–S3):
//!
//! - **S0, required.** [`ComplexField`] and [`RealField`] hold scalar
//!   arithmetic, operators and comparisons. [`Precision`] holds the precision
//!   parameter and the safe range. A scalar type that implements these, and
//!   names [`NoSimd`] as its provider, reaches every generic path.
//! - **S1, optional.** The SIMD surface is not part of the scalar trait. Each
//!   scalar names a provider ([`ComplexField::Simd`]) that implements
//!   [`SimdLayout`] (data movement: element loads and stores, masks, splats,
//!   selects, lane indices) and [`SimdField`] (arithmetic and reductions on
//!   registers). [`NoSimd`] is the one-lane provider built on S0 alone;
//!   [`NativeSimd`] serves `f32` and `f64`; [`ComplexSimd`] derives the complex
//!   provider from the real one through [`SimdPairs`].
//! - **S3, optional.** [`ComplexField::COST`] states the type's
//!   [`CostWeight`] against `f64`, which the cost model of `colmat-core`
//!   multiplies its measured rates by. The default prices the type as `f64`.
//! - **S2, optional.** [`ComplexField::Kernels`] names the type's kernel
//!   provider ([`GenericKernels`] if it has none of its own; [`NativeKernels`]
//!   for `f32`, `f64` and their complex types). The kernel hooks themselves
//!   live in `colmat-core`, implemented for the provider and the scalar.
//!
//! Dispatch is by capability, never by type identity: nothing here branches on
//! `TypeId`, `size_of` or "is this `f64`" constants.
//!
//! # Provenance
//!
//! Ported from colmat-rs (`faer-traits/src/lib.rs` at `71337cdc`), a fork of
//! faer (<https://codeberg.org/sarah-quinones/faer>), and restructured as
//! described above. faer is Copyright (c) 2026 sarah quiñones el kazdadi; the
//! modifications in colmat-rs are Copyright (c) tensor4all contributors; both
//! are distributed under the MIT licence, whose text is in this crate's
//! `LICENSE-faer` (see also the repository's `NOTICE`). Each module states
//! what it ports.
//!
//! [#2]: https://github.com/tensor4all/colmat-next-rs/issues/2
#![cfg_attr(not(feature = "std"), no_std)]
#![allow(non_camel_case_types)]

mod conjugate;
mod cost;
mod index;
mod kernels;
pub mod math_utils;
mod ops;
mod precision;
mod scalar;
mod simd;
#[cfg(test)]
mod test_support;

pub use conjugate::{ComplexConj, Conjugate, Real};
pub use cost::CostWeight;
pub use index::{Index, IndexCore, Seal, SignedIndex};
pub use kernels::{GenericKernels, NativeKernels};
pub use num_complex::Complex;
pub use ops::{
	AddByRef, ByRef, DivByRef, MulByRef, NegByRef, RefOps, SubByRef, imply,
};
pub use precision::Precision;
pub use pulp;
pub use scalar::{ComplexField, RealField};
pub use simd::{
	ComplexSimd, NativeSimd, NoSimd, SimdArch, SimdField, SimdLayout, SimdPairs,
};

/// Double-precision complex number.
pub type c64 = Complex<f64>;
/// Single-precision complex number.
pub type c32 = Complex<f32>;

/// Extension methods on [`ComplexField`] and [`RealField`] values, so that
/// generic code can write `x.abs()` or `T::zero()`.
pub mod ext {
	pub use crate::math_utils::{ComplexFieldExt, RealFieldExt};
}
