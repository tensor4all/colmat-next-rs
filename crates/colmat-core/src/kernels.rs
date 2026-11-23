//! Kernel hooks (S2): the operations a scalar's kernel provider may specialise.
//!
//! A scalar names its provider in [`ComplexField::Kernels`]; each hook here is a trait that the
//! provider implements for that scalar. [`GenericKernels`] implements every hook for every
//! scalar, so a type that names it reaches every path through its arithmetic and its SIMD
//! provider alone. [`NativeKernels`] implements them for `f32`, `f64`, `c32` and `c64` with the
//! kernels colmat-rs uses. A driver asks for [`KernelField`], which bundles the bounds, and never
//! asks which type it has.
//!
//! The first hook is [`Gemm`] (P1, the matrix product; design record §5). Its contract is what a
//! driver may rely on whichever provider runs it:
//!
//! - `dst` is `m x n`, `lhs` is `m x k`, `rhs` is `k x n`, with any strides; `dst` does not alias
//!   `lhs` or `rhs` (the borrows guarantee it);
//! - the result is `dst ← α·conj?(lhs)·conj?(rhs)` for [`Accum::Replace`] and
//!   `dst ← dst + α·conj?(lhs)·conj?(rhs)` for [`Accum::Add`]; with `Replace`, `dst` is not read;
//! - `stack` holds at least [`Gemm::gemm_scratch`] for the same shape and parallelism, and the
//!   call allocates nothing beyond it, except where the provider's documentation says otherwise;
//! - the call runs on the participants of `par` only.
//!
//! The driver ([`crate::linalg::matmul`]) handles the empty, vector and rank-one cases before it
//! calls the hook, so a provider that supplies only a packed product still gets the skinny and
//! strided paths (design document §7a, "Candidate boundary").
//!
//! New in this trial (#23); replaces the `IS_NATIVE_*` selection of colmat-rs
//! (`faer/src/linalg/matmul/mod.rs` at `71337cdc`, lines 1396-1595, debts K1 and K2).

use crate::mat::{MatMut, MatRef};
use crate::{Accum, Conj, Par, ParShape};
use colmat_traits::ComplexField;
pub use colmat_traits::{GenericKernels, NativeKernels};
use dyn_stack::{MemStack, StackReq};

/// P1, the matrix product: `dst ← α·op(lhs)·op(rhs)` or `dst ← dst + α·op(lhs)·op(rhs)`.
///
/// See the [module documentation](self) for the contract.
pub trait Gemm<T: ComplexField>:
	Copy + Default + Send + Sync + 'static
{
	/// Serial nanoseconds per multiply-add of `T` (a complex one for a complex `T`) of this
	/// provider on the host, for the cost model's width decisions.
	fn ns_per_mac() -> f64;

	/// The scratch [`Gemm::gemm`] needs for an `m x n` product of depth `k` with this
	/// parallelism.
	fn gemm_scratch(m: usize, n: usize, k: usize, par: ParShape) -> StackReq;

	/// The product. See the [module documentation](self).
	#[allow(clippy::too_many_arguments)]
	fn gemm(
		dst: MatMut<'_, T, usize, usize>,
		accum: Accum,
		lhs: MatRef<'_, T, usize, usize>,
		conj_lhs: Conj,
		rhs: MatRef<'_, T, usize, usize>,
		conj_rhs: Conj,
		alpha: &T,
		par: Par<'_>,
		stack: &mut MemStack,
	);
}

/// A scalar whose kernel provider implements every hook: what the drivers of this crate ask for.
/// Implemented for every [`ComplexField`] whose provider does, so a scalar writes nothing for it.
pub trait KernelField: ComplexField<Kernels: Gemm<Self>> {}

impl<T: ComplexField<Kernels: Gemm<T>>> KernelField for T {}
