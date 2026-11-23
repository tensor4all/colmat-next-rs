//! The matrix product (P1): `dst ← α·op(lhs)·op(rhs)` or `dst ← dst + α·op(lhs)·op(rhs)`.
//!
//! Two entries:
//!
//! - [`matmul_with_scratch`] takes its working memory from a [`MemStack`] sized with
//!   [`matmul_scratch`], and allocates nothing else (design record §5, "Workspace"), with one
//!   exception for now: the packed products of `f32`, `f64`, `c32` and `c64` (see its
//!   "Allocation" section);
//! - [`matmul`] and [`matmul_with_conj`] size and allocate that memory themselves, once per call.
//!
//! The driver here handles the empty, vector and rank-one cases with the generic kernels of
//! [`generic`], and hands every other product to the scalar's kernel provider
//! ([`Gemm`], through [`ComplexField::Kernels`]).
//!
//! Ported from colmat-rs `faer/src/linalg/matmul/mod.rs` at `71337cdc` (`matmul_imp`,
//! `matmul`, `matmul_with_conj` and `precondition`, lines 1260-1680 and 2019-2200; MIT, faer's
//! notice in `LICENSE-faer`). Changed in this trial (#23): the choice of a native kernel moved
//! from `IS_NATIVE_*` tests and `transmute_copy` to the provider (debts K1, K2), the entries take
//! or size scratch, and the triangular products and the uninitialised-output entry are not
//! ported in this step.

pub mod generic;
mod native;

use crate::kernels::{Gemm, KernelField};
use crate::mat::{AsMatMut, AsMatRef, MatMut, MatRef};
use crate::{Accum, Conj, Par, ParShape, Shape};
use colmat_traits::math_utils::zero;
use colmat_traits::{ComplexField, Conjugate};
use dyn_stack::{MemBuffer, MemStack, StackReq};
use reborrow::*;

#[track_caller]
fn precondition<M: Shape, N: Shape, K: Shape>(
	dst_nrows: M,
	dst_ncols: N,
	lhs_nrows: M,
	lhs_ncols: K,
	rhs_nrows: K,
	rhs_ncols: N,
) {
	equator::assert!(all(
		dst_nrows == lhs_nrows,
		dst_ncols == rhs_ncols,
		lhs_ncols == rhs_nrows,
	));
}

/// The scratch [`matmul_with_scratch`] needs for an `m x n` product of depth `k` on `par`.
///
/// It depends on the shape and the parallelism only, not on the strides or the conjugation.
pub fn matmul_scratch<T: KernelField>(
	m: usize,
	n: usize,
	k: usize,
	par: ParShape,
) -> StackReq {
	if m == 0 || n == 0 || k <= 1 || m == 1 || n == 1 {
		return StackReq::EMPTY;
	}
	<T::Kernels as Gemm<T>>::gemm_scratch(m, n, k, par)
}

/// `dst ← α·op(lhs)·op(rhs)` (for [`Accum::Replace`]; `dst` is then not read) or
/// `dst ← dst + α·op(lhs)·op(rhs)` (for [`Accum::Add`]), where `op` conjugates an operand when
/// its `Conj` says so, with all working memory taken from `stack`.
///
/// `stack` must hold [`matmul_scratch::<T>(m, n, k, par.shape())`](matmul_scratch). The call
/// allocates nothing else, from the first call on, except as below.
///
/// # Allocation
///
/// Nothing is allocated, from the first call on, by the paths that need no packing:
///
/// - the empty cases;
/// - the matrix-vector and rank-one cases whose vector and matrix are contiguous along the
///   product (a strided one is computed as a general product);
/// - products of at most `16^3` multiply-adds;
/// - every product of a scalar whose kernel provider is
///   [`GenericKernels`](crate::kernels::GenericKernels).
///
/// **The packed products of `f32`, `f64`, `c32` and `c64` allocate**, until their kernels take
/// scratch from the caller. They run `t4a-gemm` (`t4a-private-gemm-x86` on x86-64), which keeps
/// its packing memory in thread-locals:
///
/// - `t4a-gemm` allocates on a thread's first call (an L2-sized buffer and the pack buffer),
///   and on every call whose pack exceeds the L2 size;
/// - `t4a-private-gemm-x86` allocates on a thread's first call, and when a product needs more
///   than the memory it holds.
///
/// After one call of the same shape on the same threads, a product whose pack fits in L2
/// allocates nothing on aarch64. The entries that remove this exception are
/// tensor4all/t4a-gemm#21 and tensor4all/t4a-private-gemm-x86#14; [`matmul_scratch`] stays empty
/// for these products until then.
///
/// # Panics
///
/// When the shapes do not agree (`dst` is `m x n`, `lhs` is `m x k`, `rhs` is `k x n`), or when
/// `stack` cannot hold the scratch; both before `dst` is written.
#[track_caller]
#[allow(clippy::too_many_arguments)]
pub fn matmul_with_scratch<T: KernelField, M: Shape, N: Shape, K: Shape>(
	dst: impl AsMatMut<T = T, Rows = M, Cols = N>,
	accum: Accum,
	lhs: impl AsMatRef<T = T, Rows = M, Cols = K>,
	conj_lhs: Conj,
	rhs: impl AsMatRef<T = T, Rows = K, Cols = N>,
	conj_rhs: Conj,
	alpha: T,
	par: Par,
	stack: &mut MemStack,
) {
	let mut dst = dst;
	let dst = dst.as_mat_mut();
	let lhs = lhs.as_mat_ref();
	let rhs = rhs.as_mat_ref();
	precondition(
		dst.nrows(),
		dst.ncols(),
		lhs.nrows(),
		lhs.ncols(),
		rhs.nrows(),
		rhs.ncols(),
	);
	let (m, n, k) = (
		dst.nrows().unbound(),
		dst.ncols().unbound(),
		lhs.ncols().unbound(),
	);
	let req = matmul_scratch::<T>(m, n, k, par.shape());
	equator::assert!(stack.can_hold(req));
	matmul_imp(
		dst.as_dyn_mut().as_dyn_stride_mut(),
		accum,
		lhs.as_dyn().as_dyn_stride(),
		conj_lhs,
		rhs.as_dyn().as_dyn_stride(),
		conj_rhs,
		&alpha,
		par,
		stack,
	);
}

/// [`matmul_with_scratch`] with the scratch allocated by the call.
///
/// # Allocation
///
/// This entry allocates the scratch of [`matmul_scratch`] once per call (nothing when it is
/// empty: the vector and rank-one cases and the generic kernels need none). Code that must not
/// allocate calls [`matmul_with_scratch`] instead.
///
/// # Panics
///
/// When the shapes do not agree, before `dst` is written.
#[track_caller]
#[allow(clippy::too_many_arguments)]
pub fn matmul_with_conj<T: KernelField, M: Shape, N: Shape, K: Shape>(
	dst: impl AsMatMut<T = T, Rows = M, Cols = N>,
	accum: Accum,
	lhs: impl AsMatRef<T = T, Rows = M, Cols = K>,
	conj_lhs: Conj,
	rhs: impl AsMatRef<T = T, Rows = K, Cols = N>,
	conj_rhs: Conj,
	alpha: T,
	par: Par,
) {
	let mut dst = dst;
	let dst = dst.as_mat_mut();
	let lhs = lhs.as_mat_ref();
	let rhs = rhs.as_mat_ref();
	precondition(
		dst.nrows(),
		dst.ncols(),
		lhs.nrows(),
		lhs.ncols(),
		rhs.nrows(),
		rhs.ncols(),
	);
	let (m, n, k) = (
		dst.nrows().unbound(),
		dst.ncols().unbound(),
		lhs.ncols().unbound(),
	);
	let req = matmul_scratch::<T>(m, n, k, par.shape());
	// no buffer at all for an empty request: the common case, and a small product should not
	// pay for the allocator's call
	let mut empty = [];
	let mut buf = (req.size_bytes() != 0).then(|| MemBuffer::new(req));
	let stack = match &mut buf {
		Some(buf) => MemStack::new(buf),
		None => MemStack::new(&mut empty),
	};
	matmul_with_scratch(
		dst, accum, lhs, conj_lhs, rhs, conj_rhs, alpha, par, stack,
	);
}

/// [`matmul_with_conj`] for operands whose conjugation is in their type: `lhs` and `rhs` may be
/// views of `T` or of its conjugate (`.conjugate()`).
///
/// # Allocation
///
/// As [`matmul_with_conj`]: the scratch of [`matmul_scratch`] once per call. Code that must not
/// allocate calls [`matmul_with_scratch`] instead.
///
/// # Panics
///
/// When the shapes do not agree, before `dst` is written.
///
/// # Example
///
/// ```
/// use colmat_core::linalg::matmul::matmul;
/// use colmat_core::{Accum, Mat, Par, mat};
/// let lhs = mat![[0.0, 2.0], [1.0, 3.0]];
/// let rhs = mat![[4.0, 6.0], [5.0, 7.0]];
/// let mut dst = Mat::<f64>::zeros(2, 2);
/// matmul(&mut dst, Accum::Replace, &lhs, &rhs, 2.5, Par::Seq);
/// assert_eq!(dst, mat![[25.0, 35.0], [47.5, 67.5]]);
/// ```
#[track_caller]
pub fn matmul<
	T: KernelField,
	LhsT: Conjugate<Canonical = T>,
	RhsT: Conjugate<Canonical = T>,
	M: Shape,
	N: Shape,
	K: Shape,
>(
	dst: impl AsMatMut<T = T, Rows = M, Cols = N>,
	accum: Accum,
	lhs: impl AsMatRef<T = LhsT, Rows = M, Cols = K>,
	rhs: impl AsMatRef<T = RhsT, Rows = K, Cols = N>,
	alpha: T,
	par: Par,
) {
	let lhs = lhs.as_mat_ref();
	let rhs = rhs.as_mat_ref();
	matmul_with_conj(
		dst,
		accum,
		lhs.canonical(),
		const { Conj::get::<LhsT>() },
		rhs.canonical(),
		const { Conj::get::<RhsT>() },
		alpha,
		par,
	)
}

/// The driver: shapes agree and `stack` holds [`matmul_scratch`].
#[allow(clippy::too_many_arguments)]
fn matmul_imp<T: KernelField>(
	dst: MatMut<'_, T, usize, usize>,
	accum: Accum,
	lhs: MatRef<'_, T, usize, usize>,
	conj_lhs: Conj,
	rhs: MatRef<'_, T, usize, usize>,
	conj_rhs: Conj,
	alpha: &T,
	par: Par,
	stack: &mut MemStack,
) {
	let mut dst = dst;
	let (m, n) = dst.shape();
	let k = lhs.ncols();
	if m == 0 || n == 0 {
		return;
	}
	if k == 0 {
		if accum == Accum::Replace {
			dst.fill(zero());
		}
		return;
	}
	let (mut lhs, mut rhs) = (lhs, rhs);
	// Positive strides, so that a contiguous direction is found whichever way the views run.
	if dst.row_stride() < 0 {
		dst = dst.reverse_rows_mut();
		lhs = lhs.reverse_rows();
	}
	if dst.col_stride() < 0 {
		dst = dst.reverse_cols_mut();
		rhs = rhs.reverse_cols();
	}
	if lhs.col_stride() < 0 {
		lhs = lhs.reverse_cols();
		rhs = rhs.reverse_rows();
	}
	let mut par = par;
	if n == 1 {
		match matvec(
			dst.rb_mut(),
			accum,
			lhs,
			conj_lhs,
			rhs,
			conj_rhs,
			alpha,
			par,
		) {
			None => return,
			Some(p) => par = p,
		}
	}
	if m == 1 {
		let d = dst.rb_mut().transpose_mut();
		match matvec(
			d,
			accum,
			rhs.transpose(),
			conj_rhs,
			lhs.transpose(),
			conj_lhs,
			alpha,
			par,
		) {
			None => return,
			Some(p) => par = p,
		}
	}
	if k == 1 {
		crate::with_dim!(M, m);
		crate::with_dim!(N, n);
		crate::with_dim!(K, 1usize);
		let z = K.idx(0);
		let l = lhs.as_shape(M, K);
		let r = rhs.as_shape(K, N);
		if let (Some(d), Some(l)) = (
			dst.rb_mut().as_shape_mut(M, N).try_as_col_major_mut(),
			l.try_as_col_major(),
		) {
			generic::rank_update(
				d,
				accum,
				l.col(z),
				conj_lhs,
				r.row(z),
				conj_rhs,
				alpha,
				par,
			);
			return;
		}
		if let (Some(d), Some(r)) = (
			dst.rb_mut().as_shape_mut(M, N).try_as_row_major_mut(),
			r.try_as_row_major(),
		) {
			generic::rank_update(
				d.transpose_mut(),
				accum,
				r.row(z).transpose(),
				conj_rhs,
				l.col(z).transpose(),
				conj_lhs,
				alpha,
				par,
			);
			return;
		}
	}
	<T::Kernels as Gemm<T>>::gemm(
		dst, accum, lhs, conj_lhs, rhs, conj_rhs, alpha, par, stack,
	)
}

/// The product with a single column `dst` on a contiguous layout, if it has one. Hands `par`
/// back when it has none and did not run.
#[allow(clippy::too_many_arguments)]
fn matvec<'p, T: ComplexField>(
	dst: MatMut<'_, T, usize, usize>,
	accum: Accum,
	lhs: MatRef<'_, T, usize, usize>,
	conj_lhs: Conj,
	rhs: MatRef<'_, T, usize, usize>,
	conj_rhs: Conj,
	alpha: &T,
	par: Par<'p>,
) -> Option<Par<'p>> {
	crate::with_dim!(M, dst.nrows());
	crate::with_dim!(K, lhs.ncols());
	crate::with_dim!(N, 1usize);
	let first = N.idx(0);
	let mut dst = dst.as_shape_mut(M, N);
	let lhs = lhs.as_shape(M, K);
	let rhs = rhs.as_shape(K, N);
	if let (Some(d), Some(l)) =
		(dst.rb_mut().try_as_col_major_mut(), lhs.try_as_col_major())
	{
		generic::matvec_colmajor(
			d.col_mut(first),
			accum,
			l,
			conj_lhs,
			rhs.col(first),
			conj_rhs,
			alpha,
			par,
		);
		return None;
	}
	if let (Some(r), Some(l)) = (rhs.try_as_col_major(), lhs.try_as_row_major())
	{
		generic::matvec_rowmajor(
			dst.col_mut(first),
			accum,
			l,
			conj_lhs,
			r.col(first),
			conj_rhs,
			alpha,
			par,
		);
		return None;
	}
	Some(par)
}

#[cfg(test)]
mod tests;
