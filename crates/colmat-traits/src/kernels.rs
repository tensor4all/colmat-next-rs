//! The kernel provider a scalar selects (S2, optional).
//!
//! New in this trial (#23). colmat-rs chose its native matrix-product kernels by testing
//! `IS_NATIVE_F64`/`IS_NATIVE_C64`/... and reached them through `transmute_copy`
//! (`faer/src/linalg/matmul/mod.rs` at `71337cdc`, lines 1396-1411 and 1567-1595): a type that
//! set one of those constants reached undefined behaviour from safe code. Here the scalar names a
//! provider type in [`ComplexField::Kernels`](crate::ComplexField::Kernels), and the kernel hooks
//! (`colmat-core`'s `Gemm<T>`, ...) are implemented for the provider and the scalar together. A
//! native kernel is then implemented for exactly the concrete types it serves, so nothing
//! reinterprets one type as another.
//!
//! The hook traits live in `colmat-core`, next to the views their contracts are written in; this
//! crate only names the providers, so it does not depend on `colmat-core`.

/// The provider of the generic kernels: every hook is implemented for every scalar, on the
/// scalar's arithmetic and its SIMD provider ([`ComplexField::Simd`](crate::ComplexField::Simd)).
/// The choice of a scalar that has no kernel of its own.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct GenericKernels;

/// The provider of the native kernels of `f32`, `f64`, `Complex<f32>` and `Complex<f64>` (the
/// `gemm` and `nano-gemm` crates and their instruction-set dispatch). Its hooks are implemented
/// for those four types only.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct NativeKernels;
