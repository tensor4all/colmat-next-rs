//! The matrix product (P1): `dst ← α·op(lhs)·op(rhs)` or `dst ← dst + α·op(lhs)·op(rhs)`.
//!
//! Two entries:
//!
//! - [`matmul_with_scratch`] takes all its working memory from a [`MemStack`] sized with
//!   [`matmul_scratch`], and allocates nothing else (design record §5, "Workspace"), from the
//!   first call on (#34);
//! - [`matmul`] and [`matmul_with_conj`] allocate the driver's own scratch, once per call, and
//!   let the kernel provider find its memory itself ([`Gemm::gemm_alloc`]): the packed kernels of
//!   `f32`, `f64`, `c32` and `c64` keep theirs in thread-locals (see their "Allocation" sections).
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
/// It holds the kernel provider's own ([`Gemm::gemm_scratch`]), and the partial sums of a
/// column-major matrix-vector product whose depth a team splits (`nrows x width` elements,
/// #28): see [`matmul_with_scratch`], "Rounding on a team". On x86-64 the packed products of
/// `f32`, `f64`, `c32` and `c64` ask for about twice the L3 size per participant whatever
/// their shape: see [`matmul_with_scratch`], "Allocation".
pub fn matmul_scratch<T: KernelField>(
	m: usize,
	n: usize,
	k: usize,
	par: ParShape,
) -> StackReq {
	if m == 0 || n == 0 || k == 0 {
		return StackReq::EMPTY;
	}
	// The general product: every shape reaches it when its layout has no shortcut (a strided
	// vector, a rank-one product whose `dst` is neither column- nor row-major), and the query
	// does not see the layout.
	let general = <T::Kernels as Gemm<T>>::gemm_scratch(m, n, k, par);
	driver_scratch::<T>(m, n, k, par).or(general)
}

/// The scratch of the driver itself, without the kernel provider's: the partial sums of a
/// column-major matrix-vector product whose depth a team splits. What the allocating entries
/// allocate, since their provider finds its own memory ([`Gemm::gemm_alloc`]).
fn driver_scratch<T: KernelField>(
	m: usize,
	n: usize,
	k: usize,
	par: ParShape,
) -> StackReq {
	if m == 0 || n == 0 || k <= 1 {
		return StackReq::EMPTY;
	}
	let col = if n == 1 {
		generic::matvec_colmajor_scratch::<T>(m, k, par)
	} else {
		StackReq::EMPTY
	};
	let row = if m == 1 {
		generic::matvec_colmajor_scratch::<T>(n, k, par)
	} else {
		StackReq::EMPTY
	};
	col.or(row)
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
/// Nothing is allocated, from the first call on, on any path: the packed products of `f32`,
/// `f64`, `c32` and `c64` take their packing memory from `stack` too (#34). The entry computes
/// [`matmul_scratch`] once, to check `stack`; a packed product on a team then sizes one part of
/// `stack` per tile for the width it runs at.
///
/// **On x86-64 the request does not shrink with the shape.** `t4a-private-gemm-x86` asks for
/// its L3-sized packing buffers whatever the shape, so for these four types [`matmul_scratch`]
/// is about twice the L3 size per participant for every product above `16^3` multiply-adds: on
/// a host with a 32 MiB L3, about 64 MiB for a `17^3` product, and four times that on a team of
/// four. [`matmul`] and [`matmul_with_conj`] do not take their memory from a stack, so this
/// reaches only a caller of this entry.
///
/// # Rounding on a team
///
/// On a team the output is split and every element is computed as on one thread, so the result
/// is bitwise that of `Par::Seq`, except for a column-major matrix-vector product (`n == 1`
/// with a column-major `lhs`, or `m == 1` with a row-major `rhs`) of depth at least 1024 whose
/// matrix is larger than 8 MiB or whose columns are at most 512 bytes. Its depth is split, as
/// colmat-rs did, and the partial sums are added in order: the result rounds differently from
/// `Par::Seq`, within the same error bound, and is the same on every call of the same width.
///
/// The packed native products call their kernel once per tile, so this also needs the kernel to
/// group the depth of every element the same way whatever the tile's shape. `t4a-gemm` does since
/// tensor4all/t4a-gemm#25 (#45); for `t4a-private-gemm-x86` on x86-64 it is read from the code
/// but not tested (tensor4all/t4a-private-gemm-x86#24).
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
		Packing::Stack,
	);
}

/// [`matmul_with_scratch`] with the scratch allocated by the call.
///
/// # Allocation
///
/// This entry allocates, once per call, the driver's own scratch: the partial sums of a
/// column-major matrix-vector product whose depth a team splits, and nothing for any other
/// product. The kernel provider finds its memory itself ([`Gemm::gemm_alloc`]): the packed
/// products of `f32`, `f64`, `c32` and `c64` keep theirs in thread-locals, so
///
/// - `t4a-gemm` allocates on a thread's first call (an L2-sized buffer and the pack buffer),
///   and on every call whose pack exceeds the L2 size;
/// - `t4a-private-gemm-x86` (x86-64) allocates on a thread's first call, and when a product
///   needs more than the memory it holds.
///
/// Code that must not allocate calls [`matmul_with_scratch`] instead.
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
	// the driver's own scratch only: the kernel provider finds its memory itself
	let req = driver_scratch::<T>(m, n, k, par.shape());
	// no buffer at all for an empty request: the common case, and a small product should not
	// pay for the allocator's call
	let mut empty = [];
	let mut buf = (req.size_bytes() != 0).then(|| MemBuffer::new(req));
	let stack = match &mut buf {
		Some(buf) => MemStack::new(buf),
		None => MemStack::new(&mut empty),
	};
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
		Packing::Provider,
	);
}

/// [`matmul_with_conj`] for operands whose conjugation is in their type: `lhs` and `rhs` may be
/// views of `T` or of its conjugate (`.conjugate()`).
///
/// # Allocation
///
/// As [`matmul_with_conj`]. Code that must not allocate calls [`matmul_with_scratch`] instead.
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

/// Where the kernel provider's working memory comes from.
#[derive(Clone, Copy)]
enum Packing {
	/// From the stack ([`Gemm::gemm`]), which holds [`matmul_scratch`].
	Stack,
	/// From the provider ([`Gemm::gemm_alloc`]); the stack holds [`driver_scratch`].
	Provider,
}

/// The driver: shapes agree and `stack` holds what `packing` says.
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
	packing: Packing,
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
			stack,
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
			stack,
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
	match packing {
		Packing::Stack => <T::Kernels as Gemm<T>>::gemm(
			dst, accum, lhs, conj_lhs, rhs, conj_rhs, alpha, par, stack,
		),
		Packing::Provider => <T::Kernels as Gemm<T>>::gemm_alloc(
			dst, accum, lhs, conj_lhs, rhs, conj_rhs, alpha, par,
		),
	}
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
	stack: &mut MemStack,
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
			stack,
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
