//! The generic kernels: every path of the product on a scalar's arithmetic and its SIMD provider.
//!
//! [`GenericKernels`] implements [`Gemm`] for every [`ComplexField`] with these, and the driver
//! uses the vector and rank-one kernels for every scalar. None of them allocates; only the
//! depth split of the column-major matrix-vector product takes scratch
//! (`matvec_colmajor_scratch`).
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
//! - parallel regions split the output, except the column-major matrix-vector product above the
//!   size of `depth_split` (#28), which splits its depth as colmat-rs did; colmat-rs kept the
//!   partial sums in a thread-local (`with_partials`), which the workspace contract of this trial
//!   forbids (design record §5, "Workspace"), so they come from the caller's scratch here;
//! - the dot product and the kernels run their arithmetic as colmat-rs does (four accumulators
//!   for a dot product, `KC`/`NC` blocking), so their rounding is the same.

use super::{Accum, Gemm};
use crate::col::{ColMut, ColRef};
use crate::mat::{AsMatMut, MatMut, MatRef};
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

/// `dst ← [dst +] alpha * op(lhs) * op(rhs)` for a column-major `lhs`: one axpy per column.
///
/// A team splits the rows while [`depth_split`] says no: each participant runs every axpy on
/// its rows, so the result is bitwise that of `Par::Seq`. Above that size it splits the depth,
/// as colmat-rs did: each participant sums its block of columns into its own column of an
/// `nrows x jobs` buffer taken from `stack` ([`matvec_colmajor_scratch`]), and the columns are
/// then added to `dst` in order. That result differs from `Par::Seq` in rounding, and is the
/// same for every call on the same width.
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
	stack: &mut MemStack,
) {
	let (m, k) = (*lhs.nrows(), *lhs.ncols());
	let width = matvec_width::<T>(&par, m, k);
	crate::utils::thread::with_width(par, width, |par| match par {
		Par::Seq => {
			matvec_colmajor_seq(dst, accum, lhs, conj_lhs, rhs, conj_rhs, alpha)
		},
		mut par if depth_split::<T>(m, k) => {
			let chunk = k.div_ceil(par.degree());
			let jobs = k.div_ceil(chunk);
			let (mut tmp, _) = crate::workspace::temp_mat_zeroed::<T, _, _>(
				lhs.nrows(),
				jobs,
				stack,
			);
			let mut tmp = tmp
				.as_mat_mut()
				.try_as_col_major_mut()
				.expect("a temporary matrix is column-major");
			crate::utils::thread::for_each(
				par.reborrow(),
				jobs,
				tmp.rb_mut()
					.par_col_chunks_mut(1)
					.zip_eq(lhs.par_col_chunks(chunk))
					.zip_eq(rhs.as_mat().par_row_chunks(chunk)),
				|((mut tmp, lhs), rhs)| {
					crate::with_dim!(J, lhs.ncols());
					matvec_colmajor_seq(
						tmp.rb_mut().col_mut(0),
						Accum::Replace,
						lhs.as_col_shape(J),
						conj_lhs,
						rhs.col(0).as_row_shape(J),
						conj_rhs,
						alpha,
					)
				},
			);
			// the partials are added by rows, on the whole team; each element still adds
			// them in order
			let chunk = m.div_ceil(par.degree());
			crate::utils::thread::for_each(
				par,
				m.div_ceil(chunk),
				dst.as_dyn_rows_mut()
					.as_mat_mut()
					.par_row_chunks_mut(chunk)
					.zip_eq(tmp.rb().as_dyn_rows().par_row_chunks(chunk)),
				|(dst, tmp)| {
					crate::with_dim!(M, tmp.nrows());
					add_cols_seq(
						dst.col_mut(0).as_row_shape_mut(M),
						accum,
						tmp.as_row_shape(M),
					)
				},
			);
		},
		par => {
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

/// The scratch of [`matvec_colmajor`] for `nrows` rows and depth `k` on `par`: the partial sums
/// of the depth split, one column per participant, when the split can be taken.
pub(crate) fn matvec_colmajor_scratch<T: ComplexField>(
	nrows: usize,
	k: usize,
	par: ParShape,
) -> StackReq {
	let degree = par.degree();
	if degree < 2 || !depth_split::<T>(nrows, k) {
		return StackReq::EMPTY;
	}
	crate::workspace::temp_mat_scratch::<T>(nrows, degree.min(k))
}

/// The depth below which a team always splits the rows of a column-major matrix-vector
/// product.
const DEPTH_SPLIT_MIN_DEPTH: usize = 1024;
/// The size of the matrix, in bytes, above which the depth is split.
const DEPTH_SPLIT_MIN_BYTES: usize = 8 << 20;
/// The size of a column, in bytes, at or below which the depth is split.
const DEPTH_SPLIT_MAX_COL_BYTES: usize = 512;

/// Whether a team splits the depth of a column-major `nrows x k` matrix-vector product rather
/// than its rows: when the depth is at least [`DEPTH_SPLIT_MIN_DEPTH`], and the matrix is
/// larger than [`DEPTH_SPLIT_MIN_BYTES`] or its columns no longer than
/// [`DEPTH_SPLIT_MAX_COL_BYTES`].
///
/// Measured on an Apple M1 Pro, widths 2 and 4 (#28). The row split gives every participant a
/// segment of every column. It stays ahead while the matrix is cache resident, by up to 2x at
/// width 4, but falls behind once the matrix streams from memory (up to 2x slower than one
/// thread on `f64`, `1024 x 4096`), and when the segments are a few cache lines long. A
/// shallow depth split pays for its partial sums: 1.2-4x slower than the row split below a
/// depth of 256. The rule depends on the shape and the scalar only, so [`matvec_colmajor_scratch`]
/// can tell from the shape whether the partials are needed.
pub(crate) fn depth_split<T: ComplexField>(nrows: usize, k: usize) -> bool {
	let size = core::mem::size_of::<T>();
	k >= DEPTH_SPLIT_MIN_DEPTH
		&& (nrows.saturating_mul(k).saturating_mul(size)
			> DEPTH_SPLIT_MIN_BYTES
			|| nrows.saturating_mul(size) <= DEPTH_SPLIT_MAX_COL_BYTES)
}

/// `dst ← [dst +] Σ_j src[:, j]`, the columns added in order.
fn add_cols_seq<'M, T: ComplexField>(
	dst: ColMut<'_, T, Dim<'M>, ContiguousFwd>,
	accum: Accum,
	src: MatRef<'_, T, Dim<'M>, usize, ContiguousFwd, isize>,
) {
	struct Impl<'a, 'M, T> {
		dst: ColMut<'a, T, Dim<'M>, ContiguousFwd>,
		accum: Accum,
		src: MatRef<'a, T, Dim<'M>, usize, ContiguousFwd, isize>,
	}
	impl<T: ComplexField> WithSimd for Impl<'_, '_, T> {
		type Output = ();

		#[inline(always)]
		fn with_simd<S: Simd>(self, simd: S) {
			let Self {
				mut dst,
				accum,
				src,
			} = self;
			let simd = SimdCtx::<T, S>::new(simd, src.nrows());
			let (body, tail) = simd.indices();
			let ncols = src.ncols();
			macro_rules! sum {
				($i:expr) => {{
					let i = $i;
					let mut j = 0;
					let mut acc = if accum == Accum::Replace {
						j = 1;
						simd.read(src.col(0), i)
					} else {
						simd.read(dst.rb(), i)
					};
					while j < ncols {
						acc = simd.add(acc, simd.read(src.col(j), i));
						j += 1;
					}
					simd.write(dst.rb_mut(), i, acc);
				}};
			}
			for i in body {
				sum!(i);
			}
			if let Some(i) = tail {
				sum!(i);
			}
		}
	}
	dispatch::<T, _>(Impl { dst, accum, src })
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
	split_product(
		dst,
		lhs,
		rhs,
		par,
		ns,
		|_, _| StackReq::EMPTY,
		MemStack::new(&mut []),
		|dst, lhs, rhs, _| {
			gemm_generic_seq(dst, accum, lhs, conj_lhs, rhs, conj_rhs, alpha)
		},
	)
}

/// The width [`split_product`] runs a product of serial time `estimate` at, on a team of
/// `par`: 1 when no width pays, otherwise one of `2..=degree` (the query side cannot know
/// which, so [`split_product_scratch`] covers them all).
fn split_widths(
	par: ParShape,
	estimate: core::time::Duration,
) -> core::ops::RangeInclusive<usize> {
	let degree = par.degree();
	// the same rejection as `par_width_time` (the cheapest width of the same cost table)
	if degree < 2
		|| !crate::utils::cost::pays_at(estimate, 1, par.min_dispatch_cost())
	{
		return 1..=1;
	}
	1..=degree
}

/// The tiles of [`split_product`] at `width`: the row height and column width of a tile, and
/// the number of tile rows and columns; one tile covering `dst` below two participants.
fn split_tiles(
	m: usize,
	n: usize,
	width: usize,
) -> (usize, usize, usize, usize) {
	use crate::utils::partition::{COL_GRAIN, ROW_GRAIN, tile_grid};
	let (pm, pn) = tile_grid(m, n, width.max(1));
	if width < 2 || pm * pn <= 1 {
		return (m, n, 1, 1);
	}
	let rh = m.div_ceil(pm).next_multiple_of(ROW_GRAIN).min(m);
	let cw = n.div_ceil(pn).next_multiple_of(COL_GRAIN).min(n);
	(rh, cw, m.div_ceil(rh), n.div_ceil(cw))
}

/// The bytes one tile's part of the stack takes at `width` (every tile shape of the grid
/// fits, wherever its part starts), and the number of tiles; `(0, tiles)` when `tile_req` is
/// empty for every shape.
fn split_part_bytes(
	m: usize,
	n: usize,
	width: usize,
	mut tile_req: impl FnMut(usize, usize) -> StackReq,
) -> (usize, usize) {
	let (rh, cw, nrow, ncol) = split_tiles(m, n, width);
	let tiles = nrow * ncol;
	if tiles == 1 {
		return (0, 1);
	}
	// the tile shapes: full ones, and the last row and column, which may be shorter
	let (rl, cl) = (m - (nrow - 1) * rh, n - (ncol - 1) * cw);
	let mut req = tile_req(rh, cw);
	for (r, c) in [(rl, cw), (rh, cl), (rl, cl)] {
		req = req.or(tile_req(r, c));
	}
	if req.size_bytes() == 0 {
		return (0, tiles);
	}
	// a part starts anywhere in the stack: count the alignment in
	let bytes = req.unaligned_bytes_required().next_multiple_of(64);
	(bytes, tiles)
}

/// The scratch of [`split_product`] for an `m x n` product of depth `k` on `par`, when one
/// call of `seq` on an `r x c` tile takes `tile_req(r, c)`: the larger of the whole product
/// on one thread and one part per tile at every width the product may run at.
pub(crate) fn split_product_scratch(
	m: usize,
	n: usize,
	k: usize,
	par: ParShape,
	ns_per_mac: f64,
	tile_req: impl Fn(usize, usize) -> StackReq,
) -> StackReq {
	let macs = m.saturating_mul(n).saturating_mul(k);
	let estimate = crate::utils::cost::serial_time(macs, ns_per_mac);
	// One evaluation of `tile_req` per distinct tile shape over all the widths: the grids of
	// neighbouring widths often share their tiles, and a request is not cheap (#34). The
	// shapes past the cache's size are evaluated each time.
	let mut seen = [(0usize, 0usize, StackReq::EMPTY); 32];
	let mut len = 0;
	let mut cached = |r: usize, c: usize| {
		if let Some(&(_, _, req)) =
			seen[..len].iter().find(|&&(sr, sc, _)| (sr, sc) == (r, c))
		{
			return req;
		}
		let req = tile_req(r, c);
		if len < seen.len() {
			seen[len] = (r, c, req);
			len += 1;
		}
		req
	};
	let mut req = cached(m, n);
	for width in split_widths(par, estimate) {
		let (bytes, tiles) = split_part_bytes(m, n, width, &mut cached);
		if bytes != 0 {
			req = req.or(StackReq::new::<u8>(bytes.saturating_mul(tiles)));
		}
	}
	req
}

/// Runs a product on the width of `par` that its serial time (`ns_per_mac` per multiply-add)
/// pays for: on one thread, `seq(dst, lhs, rhs)`; on a team, `seq` on each tile of a grid over
/// `dst` ([`tile_grid`]: the grid that packs the least operand per participant, with edges on
/// [`ROW_GRAIN`]/[`COL_GRAIN`], as colmat-rs's `gemm_tiled`). The columns of the grid are dealt
/// to sub-teams, each of which deals its rows. The depth is never split, so every element is
/// computed as on one thread.
#[allow(clippy::too_many_arguments)]
pub(crate) fn split_product<T: ComplexField>(
	dst: MatMut<'_, T, usize, usize>,
	lhs: MatRef<'_, T, usize, usize>,
	rhs: MatRef<'_, T, usize, usize>,
	par: Par<'_>,
	ns_per_mac: f64,
	tile_req: impl Fn(usize, usize) -> StackReq,
	stack: &mut MemStack,
	seq: impl Sync
	+ Fn(
		MatMut<'_, T, usize, usize>,
		MatRef<'_, T, usize, usize>,
		MatRef<'_, T, usize, usize>,
		&mut MemStack,
	),
) {
	use crate::utils::thread::{for_each, for_each_with_par};
	let (m, n) = dst.shape();
	let k = lhs.ncols();
	let macs = m.saturating_mul(n).saturating_mul(k);
	let width = crate::utils::cost::par_width_time(
		&par,
		crate::utils::cost::serial_time(macs, ns_per_mac),
		1,
	);
	crate::utils::thread::with_width(par, width, |par| {
		let width = if matches!(par, Par::Seq) {
			1
		} else {
			par.degree()
		};
		let (rh, cw, nrow, ncol) = split_tiles(m, n, width);
		if nrow * ncol == 1 {
			return seq(dst, lhs, rhs, stack);
		}
		// one part of the stack per tile, in the order of the tiles (column-major over the
		// grid), each large enough for every tile shape
		let (bytes, tiles) = split_part_bytes(m, n, width, &tile_req);
		let (parts, _) = stack.make_aligned_uninit::<u8>(bytes * tiles, 1);
		let parts = StackParts {
			buf: parts,
			bytes,
			len: tiles,
		};
		if ncol == 1 {
			for_each(
				par,
				nrow,
				dst.par_row_chunks_mut(rh)
					.zip_eq(lhs.par_row_chunks(rh))
					.zip_eq(parts),
				|((dst, lhs), part)| seq(dst, lhs, rhs, MemStack::new(part)),
			)
		} else if nrow == 1 {
			for_each(
				par,
				ncol,
				dst.par_col_chunks_mut(cw)
					.zip_eq(rhs.par_col_chunks(cw))
					.zip_eq(parts),
				|((dst, rhs), part)| seq(dst, lhs, rhs, MemStack::new(part)),
			)
		} else {
			let col_parts = StackParts {
				buf: parts.buf,
				bytes: bytes * nrow,
				len: ncol,
			};
			for_each_with_par(
				par,
				ncol,
				dst.par_col_chunks_mut(cw)
					.zip_eq(rhs.par_col_chunks(cw))
					.zip_eq(col_parts),
				|((dst, rhs), col_part), par| {
					let parts = StackParts {
						buf: col_part,
						bytes,
						len: nrow,
					};
					for_each(
						par,
						nrow,
						dst.par_row_chunks_mut(rh)
							.zip_eq(lhs.par_row_chunks(rh))
							.zip_eq(parts),
						|((dst, lhs), part)| {
							seq(dst, lhs, rhs, MemStack::new(part))
						},
					)
				},
			)
		}
	})
}

/// `len` consecutive parts of `bytes` bytes of a stack buffer, dealt out with the tiles of
/// [`split_product`] (a part of no bytes when the tiles take no scratch).
struct StackParts<'a> {
	buf: &'a mut [core::mem::MaybeUninit<u8>],
	bytes: usize,
	len: usize,
}

impl<'a> Producer for StackParts<'a> {
	type Item = &'a mut [core::mem::MaybeUninit<u8>];
	type IntoIter = StackPartsIter<'a>;

	#[inline]
	fn len(&self) -> usize {
		self.len
	}

	#[inline]
	fn split_at(self, index: usize) -> (Self, Self) {
		equator::assert!(index <= self.len);
		let (a, b) = self.buf.split_at_mut(index * self.bytes);
		(
			StackParts {
				buf: a,
				bytes: self.bytes,
				len: index,
			},
			StackParts {
				buf: b,
				bytes: self.bytes,
				len: self.len - index,
			},
		)
	}

	#[inline]
	fn into_items(self) -> Self::IntoIter {
		StackPartsIter(self)
	}
}

/// The parts of a [`StackParts`], in order.
struct StackPartsIter<'a>(StackParts<'a>);

impl<'a> Iterator for StackPartsIter<'a> {
	type Item = &'a mut [core::mem::MaybeUninit<u8>];

	#[inline]
	fn next(&mut self) -> Option<Self::Item> {
		if self.0.len == 0 {
			return None;
		}
		let buf = core::mem::take(&mut self.0.buf);
		let (part, rest) = buf.split_at_mut(self.0.bytes);
		self.0.buf = rest;
		self.0.len -= 1;
		Some(part)
	}
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
