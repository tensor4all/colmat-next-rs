//! Element-wise equality of matrices, vectors, diagonals and permutations.
//!
//! Ported from colmat-rs `faer/src/linalg/mat_ops.rs` at `71337cdc` (lines
//! 52-180, the `PartialEq` impls only; the arithmetic operators stay with the
//! products in a later crate). Changed: matrices of the same row count and
//! different column counts compare unequal; colmat-rs compared the row counts
//! only and then panicked in `as_shape`. faer's MIT notice: `LICENSE-faer`.

use crate::internal_prelude::*;
use crate::{col, diag, mat, perm, row};

impl<
	LT: PartialEq<RT>,
	LRows: Shape,
	LCols: Shape,
	LRStride: Stride,
	LCStride: Stride,
	RT,
	RRows: Shape,
	RCols: Shape,
	RRStride: Stride,
	RCStride: Stride,
	L: for<'a> Reborrow<
			'a,
			Target = mat::Ref<'a, LT, LRows, LCols, LRStride, LCStride>,
		>,
	R: for<'a> Reborrow<
			'a,
			Target = mat::Ref<'a, RT, RRows, RCols, RRStride, RCStride>,
		>,
> PartialEq<mat::generic::Mat<R>> for mat::generic::Mat<L>
{
	fn eq(&self, other: &mat::generic::Mat<R>) -> bool {
		fn imp<LT: PartialEq<RT>, RT>(
			l: MatRef<'_, LT>,
			r: MatRef<'_, RT>,
		) -> bool {
			if l.nrows() != r.nrows() || l.ncols() != r.ncols() {
				return false;
			}
			with_dim!(M, l.nrows());
			with_dim!(N, l.ncols());
			let l = l.as_shape(M, N);
			let r = r.as_shape(M, N);
			for j in N.indices() {
				for i in M.indices() {
					if l[(i, j)] != r[(i, j)] {
						return false;
					}
				}
			}
			true
		}
		imp(
			self.rb().as_dyn().as_dyn_stride(),
			other.rb().as_dyn().as_dyn_stride(),
		)
	}
}
impl<
	LT: PartialEq<RT>,
	LRows: Shape,
	LRStride: Stride,
	RT,
	RRows: Shape,
	RRStride: Stride,
	L: for<'a> Reborrow<'a, Target = col::Ref<'a, LT, LRows, LRStride>>,
	R: for<'a> Reborrow<'a, Target = col::Ref<'a, RT, RRows, RRStride>>,
> PartialEq<col::generic::Col<R>> for col::generic::Col<L>
{
	fn eq(&self, other: &col::generic::Col<R>) -> bool {
		fn imp<LT: PartialEq<RT>, RT>(
			l: ColRef<'_, LT>,
			r: ColRef<'_, RT>,
		) -> bool {
			if l.nrows() != r.nrows() {
				return false;
			}
			with_dim!(N, l.nrows());
			let l = l.as_row_shape(N);
			let r = r.as_row_shape(N);
			for i in N.indices() {
				if l[i] != r[i] {
					return false;
				}
			}
			true
		}
		imp(
			self.rb().as_dyn_rows().as_dyn_stride(),
			other.rb().as_dyn_rows().as_dyn_stride(),
		)
	}
}
impl<
	LT: PartialEq<RT>,
	LCols: Shape,
	LCStride: Stride,
	RT,
	RCols: Shape,
	RCStride: Stride,
	L: for<'a> Reborrow<'a, Target = row::Ref<'a, LT, LCols, LCStride>>,
	R: for<'a> Reborrow<'a, Target = row::Ref<'a, RT, RCols, RCStride>>,
> PartialEq<row::generic::Row<R>> for row::generic::Row<L>
{
	fn eq(&self, other: &row::generic::Row<R>) -> bool {
		self.rb().transpose() == other.rb().transpose()
	}
}
impl<
	LT: PartialEq<RT>,
	LDim: Shape,
	LStride: Stride,
	RT,
	RDim: Shape,
	RStride: Stride,
	L: for<'a> Reborrow<'a, Target = diag::Ref<'a, LT, LDim, LStride>>,
	R: for<'a> Reborrow<'a, Target = diag::Ref<'a, RT, RDim, RStride>>,
> PartialEq<diag::generic::Diag<R>> for diag::generic::Diag<L>
{
	fn eq(&self, other: &diag::generic::Diag<R>) -> bool {
		self.rb().column_vector() == other.rb().column_vector()
	}
}
impl<
	LI: Index,
	LN: Shape,
	RI: Index,
	RN: Shape,
	L: for<'a> Reborrow<'a, Target = perm::Ref<'a, LI, LN>>,
	R: for<'a> Reborrow<'a, Target = perm::Ref<'a, RI, RN>>,
> PartialEq<perm::generic::Perm<R>> for perm::generic::Perm<L>
where
	LN::Idx<LI>: PartialEq<RN::Idx<RI>>,
{
	#[inline]
	fn eq(&self, other: &perm::generic::Perm<R>) -> bool {
		self.rb().arrays().0 == other.rb().arrays().0
	}
}
