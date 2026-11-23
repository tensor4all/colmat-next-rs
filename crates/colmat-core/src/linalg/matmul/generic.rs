//! The generic kernels: every path of the product on a scalar's arithmetic and its SIMD provider.
//!
//! [`GenericKernels`] implements [`Gemm`] for every [`ComplexField`] with these, and the driver
//! uses the vector and rank-one kernels for every scalar. None of them needs scratch or
//! allocates.
//!
//! Ported from colmat-rs `faer/src/linalg/matmul/mod.rs` at `71337cdc` (`matmul_vertical`,
//! `matmul_horizontal`, `dot`, `matvec_rowmajor`, `matvec_colmajor`, `rank_update` and the
//! schoolbook loop of `matmul_imp`, lines 23-1260 and 1594-1670; MIT, faer's notice in
//! `LICENSE-faer`). Changed in this trial (#23):
//!
//! - registers are reached through [`SimdCtx`] on the provider's [`SimdLayout`](colmat_traits::SimdLayout)/[`SimdField`](colmat_traits::SimdField),
//!   and every scalar, `NoSimd` ones included, takes the same kernels (a `NoSimd` register holds
//!   one element), so the separate schoolbook route for non-SIMD types is gone;
//! - the micro-kernel shape is 2 registers by 2 columns for every scalar (colmat-rs derived it
//!   from the size of a scalar's unit, which this trial's scalars do not have);
//! - parallel regions split the output, never the depth, so they need no buffer of partial sums:
//!   colmat-rs's column-major matrix-vector product split the depth and kept the partial sums in
//!   a thread-local (`with_partials`), which the workspace contract of this trial forbids
//!   (design record §5, "Workspace");
//! - the dot product and the kernels run their arithmetic as colmat-rs does (four accumulators
//!   for a dot product, `KC`/`NC` blocking), so their rounding is the same.

use super::{Accum, Gemm};
use crate::col::{ColMut, ColRef};
use crate::mat::{MatMut, MatRef};
use crate::row::RowRef;
use crate::utils::bound::Dim;
use crate::utils::simd::{SimdCtx, SimdOps, dispatch};
use crate::utils::thread::Producer;
use crate::{Conj, ContiguousFwd, Par, ParShape};
use colmat_traits::math_utils::zero;
use colmat_traits::pulp::{Simd, WithSimd};
use colmat_traits::{ComplexField, GenericKernels};
use dyn_stack::{MemStack, StackReq};
use reborrow::*;

/// Columns of `dst` per outer block of the generic product.
const NC: usize = 2048;
/// Depth per block of the generic product (in registers for the horizontal kernel).
const KC: usize = 128;
/// Registers per column of the micro-kernel.
const MR_DIV_N: usize = 2;
/// Columns of the micro-kernel.
const NR: usize = 2;

#[inline(always)]
fn conj_apply<T: ComplexField>(conj: Conj, value: &T) -> T {
	if conj.is_conj() {
		T::conj_impl(value)
	} else {
		T::copy_impl(value)
	}
}

/// `alpha * acc`, added to `dst` for [`Accum::Add`].
#[inline(always)]
fn finish<T: ComplexField>(dst: &mut T, accum: Accum, alpha: &T, acc: &T) {
	let v = alpha * acc;
	*dst = match accum {
		Accum::Add => &*dst + &v,
		Accum::Replace => v,
	};
}

// ---------------------------------------------------------------------------------------------
// dot product

/// `sum_k conj?(lhs[k]) * conj?(rhs[k])` over any strides.
pub(crate) fn inner_prod<'K, T: ComplexField>(
	lhs: RowRef<'_, T, Dim<'K>>,
	conj_lhs: Conj,
	rhs: ColRef<'_, T, Dim<'K>>,
	conj_rhs: Conj,
) -> T {
	if let (Some(lhs), Some(rhs)) =
		(lhs.try_as_row_major(), rhs.try_as_col_major())
	{
		inner_prod_contiguous(lhs.transpose(), conj_lhs, rhs, conj_rhs)
	} else {
		inner_prod_schoolbook(lhs, conj_lhs, rhs, conj_rhs)
	}
}

fn inner_prod_contiguous<'K, T: ComplexField>(
	lhs: ColRef<'_, T, Dim<'K>, ContiguousFwd>,
	conj_lhs: Conj,
	rhs: ColRef<'_, T, Dim<'K>, ContiguousFwd>,
	conj_rhs: Conj,
) -> T {
	struct Impl<'a, 'K, T> {
		lhs: ColRef<'a, T, Dim<'K>, ContiguousFwd>,
		conj_lhs: Conj,
		rhs: ColRef<'a, T, Dim<'K>, ContiguousFwd>,
		conj_rhs: Conj,
	}
	impl<T: ComplexField> WithSimd for Impl<'_, '_, T> {
		type Output = T;

		#[inline(always)]
		fn with_simd<S: Simd>(self, simd: S) -> T {
			let Self {
				lhs,
				conj_lhs,
				rhs,
				conj_rhs,
			} = self;
			let simd = SimdCtx::<T, S>::new(simd, lhs.nrows());
			let tmp = inner_prod_simd(simd, lhs, conj_lhs != conj_rhs, rhs);
			if conj_rhs.is_conj() {
				T::conj_impl(&tmp)
			} else {
				tmp
			}
		}
	}
	dispatch::<T, _>(Impl {
		lhs,
		conj_lhs,
		rhs,
		conj_rhs,
	})
}

/// `sum_k op(lhs[k]) * rhs[k]` with `op` the conjugate when `conj_lhs`, in four accumulators.
#[inline(always)]
pub(crate) fn inner_prod_simd<'K, T: ComplexField, S: Simd>(
	simd: SimdCtx<'K, T, S>,
	lhs: ColRef<'_, T, Dim<'K>, ContiguousFwd>,
	conj_lhs: bool,
	rhs: ColRef<'_, T, Dim<'K>, ContiguousFwd>,
) -> T {
	let mut acc: [_; 4] = core::array::from_fn(|_| simd.zero());
	let (batches, body, tail) = simd.batch_indices::<4>();
	macro_rules! step {
		($acc:expr, $i:expr) => {{
			let l = simd.read(lhs, $i);
			let r = simd.read(rhs, $i);
			let a = core::mem::replace(&mut $acc, simd.zero());
			$acc = if conj_lhs {
				simd.conj_mul_add(l, r, a)
			} else {
				simd.mul_add(l, r, a)
			};
		}};
	}
	for batch in batches {
		for (k, i) in batch.into_iter().enumerate() {
			step!(acc[k], i);
		}
	}
	for (k, i) in body.enumerate() {
		step!(acc[k], i);
	}
	if let Some(i) = tail {
		step!(acc[3], i);
	}
	let [a0, a1, a2, a3] = acc;
	let a0 = simd.add(a0, a1);
	let a2 = simd.add(a2, a3);
	simd.reduce_sum(simd.add(a0, a2))
}

pub(crate) fn inner_prod_schoolbook<'K, T: ComplexField>(
	lhs: RowRef<'_, T, Dim<'K>>,
	conj_lhs: Conj,
	rhs: ColRef<'_, T, Dim<'K>>,
	conj_rhs: Conj,
) -> T {
	let mut acc = zero::<T>();
	for k in lhs.ncols().indices() {
		let prod = match (conj_lhs, conj_rhs) {
			(Conj::No, Conj::No) => &lhs[k] * &rhs[k],
			(Conj::No, Conj::Yes) => &lhs[k] * &T::conj_impl(&rhs[k]),
			(Conj::Yes, Conj::No) => &T::conj_impl(&lhs[k]) * &rhs[k],
			(Conj::Yes, Conj::Yes) => T::conj_impl(&(&lhs[k] * &rhs[k])),
		};
		acc = &prod + &acc;
	}
	acc
}

// ---------------------------------------------------------------------------------------------
// matrix-vector products

/// `dst ← [dst +] alpha * op(lhs) * op(rhs)` for a row-major `lhs` and a contiguous `rhs`: one
/// dot product per row. A team splits the rows.
#[allow(clippy::too_many_arguments)]
pub(crate) fn matvec_rowmajor<'M, 'K, T: ComplexField>(
	dst: ColMut<'_, T, Dim<'M>>,
	accum: Accum,
	lhs: MatRef<'_, T, Dim<'M>, Dim<'K>, isize, ContiguousFwd>,
	conj_lhs: Conj,
	rhs: ColRef<'_, T, Dim<'K>, ContiguousFwd>,
	conj_rhs: Conj,
	alpha: &T,
	par: Par,
) {
	let width = matvec_width::<T>(&par, *lhs.nrows(), *lhs.ncols());
	crate::utils::thread::with_width(par, width, |par| match par {
		Par::Seq => {
			matvec_rowmajor_seq(dst, accum, lhs, conj_lhs, rhs, conj_rhs, alpha)
		},
		par => {
			let m = *lhs.nrows();
			let chunk = m.div_ceil(par.degree());
			let lhs = lhs.as_dyn_rows();
			let dst = dst.as_dyn_rows_mut().as_mat_mut();
			crate::utils::thread::for_each(
				par,
				m.div_ceil(chunk),
				dst.par_row_chunks_mut(chunk)
					.zip_eq(lhs.par_row_chunks(chunk)),
				|(dst, lhs)| {
					crate::with_dim!(M, lhs.nrows());
					matvec_rowmajor_seq(
						dst.col_mut(0).as_row_shape_mut(M),
						accum,
						lhs.as_row_shape(M),
						conj_lhs,
						rhs,
						conj_rhs,
						alpha,
					)
				},
			)
		},
	})
}

fn matvec_rowmajor_seq<'M, 'K, T: ComplexField>(
	dst: ColMut<'_, T, Dim<'M>>,
	accum: Accum,
	lhs: MatRef<'_, T, Dim<'M>, Dim<'K>, isize, ContiguousFwd>,
	conj_lhs: Conj,
	rhs: ColRef<'_, T, Dim<'K>, ContiguousFwd>,
	conj_rhs: Conj,
	alpha: &T,
) {
	struct Impl<'a, 'M, 'K, T> {
		dst: ColMut<'a, T, Dim<'M>>,
		accum: Accum,
		lhs: MatRef<'a, T, Dim<'M>, Dim<'K>, isize, ContiguousFwd>,
		conj_lhs: Conj,
		rhs: ColRef<'a, T, Dim<'K>, ContiguousFwd>,
		conj_rhs: Conj,
		alpha: &'a T,
	}
	impl<T: ComplexField> WithSimd for Impl<'_, '_, '_, T> {
		type Output = ();

		#[inline(always)]
		fn with_simd<S: Simd>(self, simd: S) {
			let Self {
				mut dst,
				accum,
				lhs,
				conj_lhs,
				rhs,
				conj_rhs,
				alpha,
			} = self;
			let simd = SimdCtx::<T, S>::new(simd, lhs.ncols());
			for i in lhs.nrows().indices() {
				let mut tmp = inner_prod_simd(
					simd,
					lhs.row(i).transpose(),
					conj_lhs != conj_rhs,
					rhs,
				);
				if conj_rhs.is_conj() {
					tmp = T::conj_impl(&tmp);
				}
				finish(&mut dst[i], accum, alpha, &tmp);
			}
		}
	}
	dispatch::<T, _>(Impl {
		dst,
		accum,
		lhs,
		conj_lhs,
		rhs,
		conj_rhs,
		alpha,
	})
}

/// `dst ← [dst +] alpha * op(lhs) * op(rhs)` for a column-major `lhs`: one axpy per column. A
/// team splits the rows, so no partial sums are kept.
#[allow(clippy::too_many_arguments)]
pub(crate) fn matvec_colmajor<'M, 'K, T: ComplexField>(
	dst: ColMut<'_, T, Dim<'M>, ContiguousFwd>,
	accum: Accum,
	lhs: MatRef<'_, T, Dim<'M>, Dim<'K>, ContiguousFwd, isize>,
	conj_lhs: Conj,
	rhs: ColRef<'_, T, Dim<'K>>,
	conj_rhs: Conj,
	alpha: &T,
	par: Par,
) {
	let width = matvec_width::<T>(&par, *lhs.nrows(), *lhs.ncols());
	crate::utils::thread::with_width(par, width, |par| match par {
		Par::Seq => {
			matvec_colmajor_seq(dst, accum, lhs, conj_lhs, rhs, conj_rhs, alpha)
		},
		par => {
			let m = *lhs.nrows();
			let chunk = m.div_ceil(par.degree());
			let lhs = lhs.as_dyn_rows();
			let dst = dst.as_dyn_rows_mut().as_mat_mut();
			crate::utils::thread::for_each(
				par,
				m.div_ceil(chunk),
				dst.par_row_chunks_mut(chunk)
					.zip_eq(lhs.par_row_chunks(chunk)),
				|(dst, lhs)| {
					crate::with_dim!(M, lhs.nrows());
					matvec_colmajor_seq(
						dst.col_mut(0).as_row_shape_mut(M),
						accum,
						lhs.as_row_shape(M),
						conj_lhs,
						rhs,
						conj_rhs,
						alpha,
					)
				},
			)
		},
	})
}

fn matvec_colmajor_seq<'M, 'K, T: ComplexField>(
	dst: ColMut<'_, T, Dim<'M>, ContiguousFwd>,
	accum: Accum,
	lhs: MatRef<'_, T, Dim<'M>, Dim<'K>, ContiguousFwd, isize>,
	conj_lhs: Conj,
	rhs: ColRef<'_, T, Dim<'K>>,
	conj_rhs: Conj,
	alpha: &T,
) {
	struct Impl<'a, 'M, 'K, T> {
		dst: ColMut<'a, T, Dim<'M>, ContiguousFwd>,
		accum: Accum,
		lhs: MatRef<'a, T, Dim<'M>, Dim<'K>, ContiguousFwd, isize>,
		conj_lhs: Conj,
		rhs: ColRef<'a, T, Dim<'K>>,
		conj_rhs: Conj,
		alpha: &'a T,
	}
	impl<T: ComplexField> WithSimd for Impl<'_, '_, '_, T> {
		type Output = ();

		#[inline(always)]
		fn with_simd<S: Simd>(self, simd: S) {
			let Self {
				mut dst,
				accum,
				lhs,
				conj_lhs,
				rhs,
				conj_rhs,
				alpha,
			} = self;
			let simd = SimdCtx::<T, S>::new(simd, lhs.nrows());
			let (body, tail) = simd.indices();
			if accum == Accum::Replace {
				for i in body.clone() {
					simd.write(dst.rb_mut(), i, simd.zero());
				}
				if let Some(i) = tail {
					simd.write(dst.rb_mut(), i, simd.zero());
				}
			}
			for j in lhs.ncols().indices() {
				let col = lhs.col(j);
				let r = alpha * &conj_apply(conj_rhs, &rhs[j]);
				let vr = simd.splat(&r);
				macro_rules! axpy {
					($i:expr) => {{
						let i = $i;
						let y = simd.read(dst.rb(), i);
						let x = simd.read(col, i);
						let v = if conj_lhs.is_conj() {
							simd.conj_mul_add(x, vr.clone(), y)
						} else {
							simd.mul_add(x, vr.clone(), y)
						};
						simd.write(dst.rb_mut(), i, v);
					}};
				}
				for i in body.clone() {
					axpy!(i);
				}
				if let Some(i) = tail {
					axpy!(i);
				}
			}
		}
	}
	dispatch::<T, _>(Impl {
		dst,
		accum,
		lhs,
		conj_lhs,
		rhs,
		conj_rhs,
		alpha,
	})
}

fn matvec_width<T: ComplexField>(par: &Par, m: usize, k: usize) -> usize {
	// One section, split by rows; the unit is a matrix element, read once (the streaming rate).
	let width = crate::utils::cost::par_width::<T>(
		par,
		m * k,
		crate::utils::cost::rate::matvec(),
		1,
	);
	width.min(m.max(1))
}

// ---------------------------------------------------------------------------------------------
// rank-one update

/// `dst ← [dst +] alpha * op(lhs) * op(rhs)` for a column `lhs` and a row `rhs`. A team splits
/// the columns.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rank_update<'M, 'N, T: ComplexField>(
	dst: MatMut<'_, T, Dim<'M>, Dim<'N>, ContiguousFwd>,
	accum: Accum,
	lhs: ColRef<'_, T, Dim<'M>, ContiguousFwd>,
	conj_lhs: Conj,
	rhs: RowRef<'_, T, Dim<'N>>,
	conj_rhs: Conj,
	alpha: &T,
	par: Par,
) {
	let (m, n) = (*dst.nrows(), *dst.ncols());
	let width = matvec_width::<T>(&par, n, m).min(n);
	crate::utils::thread::with_width(par, width, |par| match par {
		Par::Seq => {
			rank_update_seq(dst, accum, lhs, conj_lhs, rhs, conj_rhs, alpha)
		},
		par => {
			let chunk = n.div_ceil(par.degree());
			let dst = dst.as_dyn_cols_mut();
			let rhs = rhs.as_dyn_cols().as_mat();
			crate::utils::thread::for_each(
				par,
				n.div_ceil(chunk),
				dst.par_col_chunks_mut(chunk)
					.zip_eq(rhs.par_col_chunks(chunk)),
				|(dst, rhs)| {
					crate::with_dim!(N, dst.ncols());
					rank_update_seq(
						dst.as_col_shape_mut(N),
						accum,
						lhs,
						conj_lhs,
						rhs.row(0).as_col_shape(N),
						conj_rhs,
						alpha,
					)
				},
			)
		},
	})
}

fn rank_update_seq<'M, 'N, T: ComplexField>(
	dst: MatMut<'_, T, Dim<'M>, Dim<'N>, ContiguousFwd>,
	accum: Accum,
	lhs: ColRef<'_, T, Dim<'M>, ContiguousFwd>,
	conj_lhs: Conj,
	rhs: RowRef<'_, T, Dim<'N>>,
	conj_rhs: Conj,
	alpha: &T,
) {
	struct Impl<'a, 'M, 'N, T> {
		dst: MatMut<'a, T, Dim<'M>, Dim<'N>, ContiguousFwd>,
		accum: Accum,
		lhs: ColRef<'a, T, Dim<'M>, ContiguousFwd>,
		conj_lhs: Conj,
		rhs: RowRef<'a, T, Dim<'N>>,
		conj_rhs: Conj,
		alpha: &'a T,
	}
	impl<T: ComplexField> WithSimd for Impl<'_, '_, '_, T> {
		type Output = ();

		#[inline(always)]
		fn with_simd<S: Simd>(self, simd: S) {
			let Self {
				mut dst,
				accum,
				lhs,
				conj_lhs,
				rhs,
				conj_rhs,
				alpha,
			} = self;
			let simd = SimdCtx::<T, S>::new(simd, dst.nrows());
			let (body, tail) = simd.indices();
			for j in dst.ncols().indices() {
				let mut dst = dst.rb_mut().col_mut(j);
				let r = alpha * &conj_apply(conj_rhs, &rhs[j]);
				let vr = simd.splat(&r);
				macro_rules! step {
					($i:expr) => {{
						let i = $i;
						let x = simd.read(lhs, i);
						let acc = match accum {
							Accum::Add => simd.read(dst.rb(), i),
							Accum::Replace => simd.zero(),
						};
						let v = if conj_lhs.is_conj() {
							simd.conj_mul_add(x, vr.clone(), acc)
						} else {
							simd.mul_add(x, vr.clone(), acc)
						};
						simd.write(dst.rb_mut(), i, v);
					}};
				}
				for i in body.clone() {
					step!(i);
				}
				if let Some(i) = tail {
					step!(i);
				}
			}
		}
	}
	dispatch::<T, _>(Impl {
		dst,
		accum,
		lhs,
		conj_lhs,
		rhs,
		conj_rhs,
		alpha,
	})
}

// ---------------------------------------------------------------------------------------------
// the product

impl<T: ComplexField> Gemm<T> for GenericKernels {
	#[inline]
	fn ns_per_mac() -> f64 {
		crate::utils::cost::rate::gemm_mac().of::<T>()
	}

	#[inline]
	fn gemm_scratch(_: usize, _: usize, _: usize, _: ParShape) -> StackReq {
		StackReq::EMPTY
	}

	fn gemm(
		dst: MatMut<'_, T, usize, usize>,
		accum: Accum,
		lhs: MatRef<'_, T, usize, usize>,
		conj_lhs: Conj,
		rhs: MatRef<'_, T, usize, usize>,
		conj_rhs: Conj,
		alpha: &T,
		par: Par<'_>,
		_: &mut MemStack,
	) {
		gemm_generic(dst, accum, lhs, conj_lhs, rhs, conj_rhs, alpha, par)
	}
}

/// The generic product: [`split_product`] over [`gemm_generic_seq`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_generic<T: ComplexField>(
	dst: MatMut<'_, T, usize, usize>,
	accum: Accum,
	lhs: MatRef<'_, T, usize, usize>,
	conj_lhs: Conj,
	rhs: MatRef<'_, T, usize, usize>,
	conj_rhs: Conj,
	alpha: &T,
	par: Par<'_>,
) {
	let ns = <GenericKernels as Gemm<T>>::ns_per_mac();
	split_product(dst, lhs, rhs, par, ns, |dst, lhs, rhs| {
		gemm_generic_seq(dst, accum, lhs, conj_lhs, rhs, conj_rhs, alpha)
	})
}

/// Runs a product on the width of `par` that its serial time (`ns_per_mac` per multiply-add)
/// pays for: on one thread, `seq(dst, lhs, rhs)`; on a team, `seq` on each participant's part,
/// the columns of `dst` and `rhs` (the rows of `dst` and `lhs` when `dst` has fewer columns than
/// participants). The depth is never split, so every element is computed as on one thread.
pub(crate) fn split_product<T: ComplexField>(
	dst: MatMut<'_, T, usize, usize>,
	lhs: MatRef<'_, T, usize, usize>,
	rhs: MatRef<'_, T, usize, usize>,
	par: Par<'_>,
	ns_per_mac: f64,
	seq: impl Sync
	+ Fn(
		MatMut<'_, T, usize, usize>,
		MatRef<'_, T, usize, usize>,
		MatRef<'_, T, usize, usize>,
	),
) {
	let (m, n) = dst.shape();
	let k = lhs.ncols();
	let macs = m.saturating_mul(n).saturating_mul(k);
	let width = crate::utils::cost::par_width_time(
		&par,
		crate::utils::cost::serial_time(macs, ns_per_mac),
		1,
	);
	crate::utils::thread::with_width(par, width, |par| match par {
		Par::Seq => seq(dst, lhs, rhs),
		par => {
			let degree = par.degree();
			if n >= degree {
				let chunk = n.div_ceil(degree);
				crate::utils::thread::for_each(
					par,
					n.div_ceil(chunk),
					dst.par_col_chunks_mut(chunk)
						.zip_eq(rhs.par_col_chunks(chunk)),
					|(dst, rhs)| seq(dst, lhs, rhs),
				)
			} else {
				let chunk = m.div_ceil(degree);
				crate::utils::thread::for_each(
					par,
					m.div_ceil(chunk),
					dst.par_row_chunks_mut(chunk)
						.zip_eq(lhs.par_row_chunks(chunk)),
					|(dst, lhs)| seq(dst, lhs, rhs),
				)
			}
		},
	})
}

/// The sequential generic product: the vertical kernel for a column-major `dst` and `lhs` (or,
/// transposed, a row-major `dst` and `rhs`), the horizontal kernel for a row-major `lhs` and a
/// column-major `rhs`, and one dot product per element otherwise.
pub(crate) fn gemm_generic_seq<T: ComplexField>(
	dst: MatMut<'_, T, usize, usize>,
	accum: Accum,
	lhs: MatRef<'_, T, usize, usize>,
	conj_lhs: Conj,
	rhs: MatRef<'_, T, usize, usize>,
	conj_rhs: Conj,
	alpha: &T,
) {
	let mut dst = dst;
	if dst.nrows() == 0 || dst.ncols() == 0 {
		return;
	}
	if let (Some(d), Some(l)) =
		(dst.rb_mut().try_as_col_major_mut(), lhs.try_as_col_major())
	{
		return vertical(d, accum, l, conj_lhs, rhs, conj_rhs, alpha);
	}
	if let (Some(d), Some(r)) =
		(dst.rb_mut().try_as_row_major_mut(), rhs.try_as_row_major())
	{
		return vertical(
			d.transpose_mut(),
			accum,
			r.transpose(),
			conj_rhs,
			lhs.transpose(),
			conj_lhs,
			alpha,
		);
	}
	if let (Some(l), Some(r)) = (lhs.try_as_row_major(), rhs.try_as_col_major())
	{
		return horizontal(dst, accum, l, conj_lhs, r, conj_rhs, alpha);
	}
	crate::with_dim!(M, dst.nrows());
	crate::with_dim!(N, dst.ncols());
	crate::with_dim!(K, lhs.ncols());
	let mut dst = dst.as_shape_mut(M, N);
	let lhs = lhs.as_shape(M, K);
	let rhs = rhs.as_shape(K, N);
	for j in N.indices() {
		for i in M.indices() {
			let acc = inner_prod(lhs.row(i), conj_lhs, rhs.col(j), conj_rhs);
			finish(&mut dst[(i, j)], accum, alpha, &acc);
		}
	}
}

/// Runs `job(row, col, nrows, ncols)` over the `mr x nr` tiles of an `m x n` block.
#[inline(always)]
fn tiles(
	m: usize,
	n: usize,
	mr: usize,
	nr: usize,
	mut job: impl FnMut(usize, usize, usize, usize),
) {
	let mut col = 0;
	while col < n {
		let ncols = Ord::min(n - col, nr);
		let mut row = 0;
		while row < m {
			let nrows = Ord::min(m - row, mr);
			job(row, col, nrows, ncols);
			row += nrows;
		}
		col += ncols;
	}
}

/// Column-major `dst` and `lhs`: registers run down the columns, `rhs` is broadcast.
fn vertical<T: ComplexField>(
	dst: MatMut<'_, T, usize, usize, ContiguousFwd>,
	accum: Accum,
	lhs: MatRef<'_, T, usize, usize, ContiguousFwd>,
	conj_lhs: Conj,
	rhs: MatRef<'_, T, usize, usize>,
	conj_rhs: Conj,
	alpha: &T,
) {
	let (m, n) = dst.shape();
	let k = lhs.ncols();
	let lanes = crate::utils::simd::lane_count::<T>();
	let mr = MR_DIV_N * lanes;
	let mut dst = dst;
	let mut col_outer = 0;
	while col_outer < n {
		let n_chunk = Ord::min(n - col_outer, NC);
		let mut accum = accum;
		let mut depth = 0;
		while depth < k {
			let k_chunk = Ord::min(k - depth, KC);
			tiles(m, n_chunk, mr, NR, |row, col, nrows, ncols| {
				let dst = dst.rb_mut().submatrix_mut(
					row,
					col_outer + col,
					nrows,
					ncols,
				);
				let a = lhs.submatrix(row, depth, nrows, k_chunk);
				let b = rhs.submatrix(depth, col_outer + col, k_chunk, ncols);
				let ukr = VerticalUkr {
					dst,
					a,
					b,
					conj_lhs,
					conj_rhs,
					alpha,
					accum,
				};
				match (nrows.div_ceil(lanes), ncols) {
					(2, 2) => dispatch::<T, _>(ukr.shape::<2, 2>()),
					(2, 1) => dispatch::<T, _>(ukr.shape::<2, 1>()),
					(1, 2) => dispatch::<T, _>(ukr.shape::<1, 2>()),
					(1, 1) => dispatch::<T, _>(ukr.shape::<1, 1>()),
					_ => unreachable!(),
				}
			});
			accum = Accum::Add;
			depth += k_chunk;
		}
		col_outer += n_chunk;
	}
}

struct VerticalUkr<'a, T> {
	dst: MatMut<'a, T, usize, usize, ContiguousFwd>,
	a: MatRef<'a, T, usize, usize, ContiguousFwd>,
	b: MatRef<'a, T, usize, usize>,
	conj_lhs: Conj,
	conj_rhs: Conj,
	alpha: &'a T,
	accum: Accum,
}

impl<'a, T> VerticalUkr<'a, T> {
	#[inline(always)]
	fn shape<const MR: usize, const NR_: usize>(
		self,
	) -> VerticalShape<'a, MR, NR_, T> {
		VerticalShape(self)
	}
}

struct VerticalShape<'a, const MR: usize, const NR_: usize, T>(
	VerticalUkr<'a, T>,
);

impl<const MR: usize, const NR_: usize, T: ComplexField> WithSimd
	for VerticalShape<'_, MR, NR_, T>
{
	type Output = ();

	#[inline(always)]
	fn with_simd<S: Simd>(self, simd: S) {
		let VerticalUkr {
			dst,
			a,
			b,
			conj_lhs,
			conj_rhs,
			alpha,
			accum,
		} = self.0;
		crate::with_dim!(M, a.nrows());
		crate::with_dim!(N, b.ncols());
		crate::with_dim!(K, a.ncols());
		let a = a.as_shape(M, K);
		let b = b.as_shape(K, N);
		let mut dst = dst.as_shape_mut(M, N);
		let simd = SimdCtx::<T, S>::new_force_mask(simd, M);
		let (body, tail) = simd.indices();
		let tail = tail.unwrap();
		// `conj(b) * a` when exactly one operand is conjugated, then the sum is conjugated when
		// `lhs` is: that gives `a * conj(b)` for `rhs` alone, `conj(a) * b` for `lhs` alone and
		// `conj(a * b)` for both.
		let conj_a = conj_lhs != conj_rhs;
		let mut acc: [[_; MR]; NR_] =
			core::array::from_fn(|_| core::array::from_fn(|_| simd.zero()));
		for depth in K.indices() {
			let col = a.col(depth);
			let mut body = body.clone();
			let av: [_; MR] = core::array::from_fn(|i| {
				if i + 1 < MR {
					simd.read(col, body.next().unwrap())
				} else {
					simd.read(col, tail)
				}
			});
			for (j, acc) in N.indices().zip(acc.iter_mut()) {
				let bv = simd.splat(&b[(depth, j)]);
				for (acc, av) in acc.iter_mut().zip(av.iter()) {
					let x = core::mem::replace(acc, simd.zero());
					*acc = if conj_a {
						simd.conj_mul_add(bv.clone(), av.clone(), x)
					} else {
						simd.mul_add(av.clone(), bv.clone(), x)
					};
				}
			}
		}
		if conj_lhs.is_conj() {
			for acc in acc.iter_mut().flatten() {
				let x = core::mem::replace(acc, simd.zero());
				*acc = simd.conj(x);
			}
		}
		let alpha = simd.splat(alpha);
		for (j, acc) in N.indices().zip(acc) {
			let mut body = body.clone();
			for (i, acc) in acc.into_iter().enumerate() {
				macro_rules! out {
					($at:expr) => {{
						let at = $at;
						let v = match accum {
							Accum::Add => {
								let y = simd.read(dst.rb().col(j), at);
								simd.mul_add(alpha.clone(), acc, y)
							},
							Accum::Replace => simd.mul(alpha.clone(), acc),
						};
						simd.write(dst.rb_mut().col_mut(j), at, v);
					}};
				}
				if i + 1 < MR {
					out!(body.next().unwrap());
				} else {
					out!(tail);
				}
			}
		}
	}
}

/// Row-major `lhs` and column-major `rhs`: registers run along the depth, one dot product per
/// element of a `2 x 2` tile.
fn horizontal<T: ComplexField>(
	dst: MatMut<'_, T, usize, usize>,
	accum: Accum,
	lhs: MatRef<'_, T, usize, usize, isize, ContiguousFwd>,
	conj_lhs: Conj,
	rhs: MatRef<'_, T, usize, usize, ContiguousFwd, isize>,
	conj_rhs: Conj,
	alpha: &T,
) {
	let (m, n) = dst.shape();
	let k = lhs.ncols();
	let kc = KC * crate::utils::simd::lane_count::<T>();
	let mut dst = dst;
	let mut col_outer = 0;
	while col_outer < n {
		let n_chunk = Ord::min(n - col_outer, NC);
		let mut accum = accum;
		let mut depth = 0;
		while depth < k {
			let k_chunk = Ord::min(k - depth, kc);
			tiles(m, n_chunk, MR_DIV_N, NR, |row, col, nrows, ncols| {
				let dst = dst.rb_mut().submatrix_mut(
					row,
					col_outer + col,
					nrows,
					ncols,
				);
				let a = lhs.submatrix(row, depth, nrows, k_chunk);
				let b = rhs.submatrix(depth, col_outer + col, k_chunk, ncols);
				let ukr = HorizontalUkr {
					dst,
					a,
					b,
					conj_lhs,
					conj_rhs,
					alpha,
					accum,
				};
				match (nrows, ncols) {
					(2, 2) => dispatch::<T, _>(ukr.shape::<2, 2>()),
					(2, 1) => dispatch::<T, _>(ukr.shape::<2, 1>()),
					(1, 2) => dispatch::<T, _>(ukr.shape::<1, 2>()),
					(1, 1) => dispatch::<T, _>(ukr.shape::<1, 1>()),
					_ => unreachable!(),
				}
			});
			accum = Accum::Add;
			depth += k_chunk;
		}
		col_outer += n_chunk;
	}
}

struct HorizontalUkr<'a, T> {
	dst: MatMut<'a, T, usize, usize>,
	a: MatRef<'a, T, usize, usize, isize, ContiguousFwd>,
	b: MatRef<'a, T, usize, usize, ContiguousFwd, isize>,
	conj_lhs: Conj,
	conj_rhs: Conj,
	alpha: &'a T,
	accum: Accum,
}

impl<'a, T> HorizontalUkr<'a, T> {
	#[inline(always)]
	fn shape<const MR: usize, const NR_: usize>(
		self,
	) -> HorizontalShape<'a, MR, NR_, T> {
		HorizontalShape(self)
	}
}

struct HorizontalShape<'a, const MR: usize, const NR_: usize, T>(
	HorizontalUkr<'a, T>,
);

impl<const MR: usize, const NR_: usize, T: ComplexField> WithSimd
	for HorizontalShape<'_, MR, NR_, T>
{
	type Output = ();

	#[inline(always)]
	fn with_simd<S: Simd>(self, simd: S) {
		let HorizontalUkr {
			dst,
			a,
			b,
			conj_lhs,
			conj_rhs,
			alpha,
			accum,
		} = self.0;
		crate::with_dim!(M, a.nrows());
		crate::with_dim!(N, b.ncols());
		crate::with_dim!(K, a.ncols());
		let a = a.as_shape(M, K);
		let b = b.as_shape(K, N);
		let mut dst = dst.as_shape_mut(M, N);
		let simd = SimdCtx::<T, S>::new(simd, K);
		let (body, tail) = simd.indices();
		let is: [_; MR] = core::array::from_fn(|i| M.idx(i));
		let js: [_; NR_] = core::array::from_fn(|j| N.idx(j));
		let conj_a = conj_lhs != conj_rhs;
		let mut acc: [[_; MR]; NR_] =
			core::array::from_fn(|_| core::array::from_fn(|_| simd.zero()));
		macro_rules! step {
			($at:expr) => {{
				let at = $at;
				let av = is.map(|i| simd.read(a.row(i).transpose(), at));
				let bv = js.map(|j| simd.read(b.col(j), at));
				for (acc, bv) in acc.iter_mut().zip(bv.iter()) {
					for (acc, av) in acc.iter_mut().zip(av.iter()) {
						let x = core::mem::replace(acc, simd.zero());
						*acc = if conj_a {
							simd.conj_mul_add(bv.clone(), av.clone(), x)
						} else {
							simd.mul_add(av.clone(), bv.clone(), x)
						};
					}
				}
			}};
		}
		for at in body {
			step!(at);
		}
		if let Some(at) = tail {
			step!(at);
		}
		for (acc, j) in acc.into_iter().zip(js) {
			for (acc, i) in acc.into_iter().zip(is) {
				let mut r = simd.reduce_sum(acc);
				if conj_lhs.is_conj() {
					r = T::conj_impl(&r);
				}
				finish(&mut dst[(i, j)], accum, alpha, &r);
			}
		}
	}
}
