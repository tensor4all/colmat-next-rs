use super::*;
use crate::internal_prelude::*;
use crate::into_range::IntoRange;
use crate::{Idx, IdxInc, assert, debug_assert};
impl<
	'a,
	R: Shape,
	C: Shape,
	T,
	Rs: Stride,
	Cs: Stride,
	RowRange: IntoRange<IdxInc<R>, Len<R>: 'a>,
	ColRange: IntoRange<IdxInc<C>, Len<C>: 'a>,
> MatIndex<RowRange, ColRange> for MatRef<'a, T, R, C, Rs, Cs>
{
	type Target = MatRef<'a, T, RowRange::Len<R>, ColRange::Len<C>, Rs, Cs>;

	#[track_caller]
	#[inline]
	fn get(this: Self, row: RowRange, col: ColRange) -> Self::Target {
		let row = row.into_range(R::start(), this.nrows().end());
		let col = col.into_range(C::start(), this.ncols().end());
		assert!(all(
			row.start <= row.end,
			row.end <= this.nrows(),
			col.start <= col.end,
			col.end <= this.ncols(),
		));
		// SAFETY: the assert above gives `row.start <= row.end <=
		// this.nrows()`, so the subtraction does not underflow and the length
		// fits the view; `Len<R>` is `usize` (no invariant) for every range
		// type except `..`, where it is `R` and the range is
		// `R::start()..this.nrows()`, so the length is exactly `this.nrows()`,
		// a valid `R`.
		let nrows = unsafe {
			RowRange::Len::<R>::new_unbound(
				row.end.unbound() - row.start.unbound(),
			)
		};
		// SAFETY: as for `nrows`, using the column half of the assert above
		// (`..` gives `Len<C> = C` with length `this.ncols()`).
		let ncols = unsafe {
			ColRange::Len::<C>::new_unbound(
				col.end.unbound() - col.start.unbound(),
			)
		};
		this.submatrix(row.start, col.start, nrows, ncols)
	}

	#[track_caller]
	#[inline]
	unsafe fn get_unchecked(
		this: Self,
		row: RowRange,
		col: ColRange,
	) -> Self::Target {
		let row = row.into_range(R::start(), this.nrows().end());
		let col = col.into_range(C::start(), this.ncols().end());
		debug_assert!(all(
			row.start <= row.end,
			row.end <= this.nrows(),
			col.start <= col.end,
			col.end <= this.ncols(),
		));
		// SAFETY: the caller guarantees the ranges are in bounds (# Safety of
		// `MatIndex::get_unchecked`), so `row.start <= row.end <= this.nrows()`
		// and the subtraction does not underflow; `Len<R>` is `usize` (no
		// invariant) for every range type except `..`, where it is `R` and the
		// range is `R::start()..this.nrows()`, so the length is exactly
		// `this.nrows()`, a valid `R`.
		let nrows = unsafe {
			RowRange::Len::<R>::new_unbound(
				row.end.unbound() - row.start.unbound(),
			)
		};
		// SAFETY: as for `nrows`, using the caller's guarantee for the column
		// range (`..` gives `Len<C> = C` with length `this.ncols()`).
		let ncols = unsafe {
			ColRange::Len::<C>::new_unbound(
				col.end.unbound() - col.start.unbound(),
			)
		};
		this.submatrix(row.start, col.start, nrows, ncols)
	}
}
impl<
	'a,
	R: Shape,
	C: Shape,
	T,
	Rs: Stride,
	Cs: Stride,
	RowRange: IntoRange<IdxInc<R>, Len<R>: 'a>,
	ColRange: IntoRange<IdxInc<C>, Len<C>: 'a>,
> MatIndex<RowRange, ColRange> for MatMut<'a, T, R, C, Rs, Cs>
{
	type Target = MatMut<'a, T, RowRange::Len<R>, ColRange::Len<C>, Rs, Cs>;

	#[track_caller]
	#[inline]
	fn get(this: Self, row: RowRange, col: ColRange) -> Self::Target {
		let row = row.into_range(R::start(), this.nrows().end());
		let col = col.into_range(C::start(), this.ncols().end());
		assert!(all(
			row.start <= row.end,
			row.end <= this.nrows(),
			col.start <= col.end,
			col.end <= this.ncols(),
		));
		// SAFETY: the assert above gives `row.start <= row.end <=
		// this.nrows()`, so the subtraction does not underflow and the length
		// fits the view; `Len<R>` is `usize` (no invariant) for every range
		// type except `..`, where it is `R` and the range is
		// `R::start()..this.nrows()`, so the length is exactly `this.nrows()`,
		// a valid `R`.
		let nrows = unsafe {
			RowRange::Len::<R>::new_unbound(
				row.end.unbound() - row.start.unbound(),
			)
		};
		// SAFETY: as for `nrows`, using the column half of the assert above
		// (`..` gives `Len<C> = C` with length `this.ncols()`).
		let ncols = unsafe {
			ColRange::Len::<C>::new_unbound(
				col.end.unbound() - col.start.unbound(),
			)
		};
		this.submatrix_mut(row.start, col.start, nrows, ncols)
	}

	#[track_caller]
	#[inline]
	unsafe fn get_unchecked(
		this: Self,
		row: RowRange,
		col: ColRange,
	) -> Self::Target {
		let row = row.into_range(R::start(), this.nrows().end());
		let col = col.into_range(C::start(), this.ncols().end());
		debug_assert!(all(
			row.start <= row.end,
			row.end <= this.nrows(),
			col.start <= col.end,
			col.end <= this.ncols(),
		));
		// SAFETY: the caller guarantees the ranges are in bounds (# Safety of
		// `MatIndex::get_unchecked`), so `row.start <= row.end <= this.nrows()`
		// and the subtraction does not underflow; `Len<R>` is `usize` (no
		// invariant) for every range type except `..`, where it is `R` and the
		// range is `R::start()..this.nrows()`, so the length is exactly
		// `this.nrows()`, a valid `R`.
		let nrows = unsafe {
			RowRange::Len::<R>::new_unbound(
				row.end.unbound() - row.start.unbound(),
			)
		};
		// SAFETY: as for `nrows`, using the caller's guarantee for the column
		// range (`..` gives `Len<C> = C` with length `this.ncols()`).
		let ncols = unsafe {
			ColRange::Len::<C>::new_unbound(
				col.end.unbound() - col.start.unbound(),
			)
		};
		this.submatrix_mut(row.start, col.start, nrows, ncols)
	}
}
macro_rules! row_impl {
    ($R:ty $(,$tt:tt)?) => {
        impl <'a $(, $tt)?, C : Shape, T, Rs : Stride, Cs : Stride, ColRange : IntoRange
        < IdxInc < C >, Len < C >: 'a >> MatIndex < Idx <$R >, ColRange > for MatRef <'a,
        T, $R, C, Rs, Cs > { type Target = RowRef <'a, T, ColRange::Len < C >, Cs >;
        #[track_caller] #[inline] fn get(this : Self, row : Idx <$R >, col : ColRange) ->
        Self::Target { let col = col.into_range(C::start(), this.ncols().end());
        assert!(all(row < this.nrows(), col.start <= col.end, col.end <= this.ncols(),));
        // SAFETY: the assert above gives `col.start <= col.end <= this.ncols()`, so the
        // subtraction does not underflow; `Len<C>` is `usize` (no invariant) except for
        // `..`, where the length is exactly `this.ncols()`, a valid `C`.
        let ncols = unsafe { ColRange::Len::< C >::new_unbound(col.end.unbound() - col
        .start.unbound()) }; this.subcols(col.start, ncols).row(row) } #[track_caller]
        #[inline] unsafe fn get_unchecked(this : Self, row : Idx <$R >, col : ColRange)
        -> Self::Target { let col = col.into_range(C::start(), this.ncols().end());
        debug_assert!(all(row < this.nrows(), col.start <= col.end, col.end <= this
        .ncols(),));
        // SAFETY: the caller guarantees the column range is in bounds (# Safety of
        // `MatIndex::get_unchecked`), so `col.start <= col.end <= this.ncols()`; `Len<C>`
        // is `usize` (no invariant) except for `..`, where the length is exactly
        // `this.ncols()`, a valid `C`.
        let ncols = unsafe { ColRange::Len::< C >::new_unbound(col.end
        .unbound() - col.start.unbound()) }; this.subcols(col.start, ncols).row(row) } }
        impl <'a $(, $tt)?, C : Shape, T, Rs : Stride, Cs : Stride, ColRange : IntoRange
        < IdxInc < C >, Len < C >: 'a >> MatIndex < Idx <$R >, ColRange > for MatMut <'a,
        T, $R, C, Rs, Cs > { type Target = RowMut <'a, T, ColRange::Len < C >, Cs >;
        #[track_caller] #[inline] fn get(this : Self, row : Idx <$R >, col : ColRange) ->
        Self::Target { let col = col.into_range(C::start(), this.ncols().end());
        assert!(all(row < this.nrows(), col.start <= col.end, col.end <= this.ncols(),));
        // SAFETY: the assert above gives `col.start <= col.end <= this.ncols()`, so the
        // subtraction does not underflow; `Len<C>` is `usize` (no invariant) except for
        // `..`, where the length is exactly `this.ncols()`, a valid `C`.
        let ncols = unsafe { ColRange::Len::< C >::new_unbound(col.end.unbound() - col
        .start.unbound()) }; this.subcols_mut(col.start, ncols).row_mut(row) }
        #[track_caller] #[inline] unsafe fn get_unchecked(this : Self, row : Idx <$R >,
        col : ColRange) -> Self::Target { let col = col.into_range(C::start(), this
        .ncols().end()); debug_assert!(all(row < this.nrows(), col.start <= col.end, col
        .end <= this.ncols(),));
        // SAFETY: the caller guarantees the column range is in bounds (# Safety of
        // `MatIndex::get_unchecked`), so `col.start <= col.end <= this.ncols()`; `Len<C>`
        // is `usize` (no invariant) except for `..`, where the length is exactly
        // `this.ncols()`, a valid `C`.
        let ncols = unsafe { ColRange::Len::< C
        >::new_unbound(col.end.unbound() - col.start.unbound()) }; this.subcols_mut(col
        .start, ncols).row_mut(row) } }
    };
}
macro_rules! col_impl {
    ($C:ty $(,$tt:tt)?) => {
        impl <'a $(, $tt)?, R : Shape, T, Rs : Stride, Cs : Stride, RowRange : IntoRange
        < IdxInc < R >, Len < R >: 'a >> MatIndex < RowRange, Idx <$C >> for MatRef <'a,
        T, R, $C, Rs, Cs > { type Target = ColRef <'a, T, RowRange::Len < R >, Rs >;
        #[track_caller] #[inline] fn get(this : Self, row : RowRange, col : Idx <$C >) ->
        Self::Target { let row = row.into_range(R::start(), this.nrows().end());
        assert!(all(col < this.ncols(), row.start <= row.end, row.end <= this.nrows(),));
        // SAFETY: the assert above gives `row.start <= row.end <= this.nrows()`, so the
        // subtraction does not underflow; `Len<R>` is `usize` (no invariant) except for
        // `..`, where the length is exactly `this.nrows()`, a valid `R`.
        let nrows = unsafe { RowRange::Len::< R >::new_unbound(row.end.unbound() - row
        .start.unbound()) }; this.subrows(row.start, nrows).col(col) } #[track_caller]
        #[inline] unsafe fn get_unchecked(this : Self, row : RowRange, col : Idx <$C >)
        -> Self::Target { let row = row.into_range(R::start(), this.nrows().end());
        assert!(all(col < this.ncols(), row.start <= row.end, row.end <= this.nrows(),));
        // SAFETY: the assert above gives `row.start <= row.end <= this.nrows()`, so the
        // subtraction does not underflow; `Len<R>` is `usize` (no invariant) except for
        // `..`, where the length is exactly `this.nrows()`, a valid `R`.
        let nrows = unsafe { RowRange::Len::< R >::new_unbound(row.end.unbound() - row
        .start.unbound()) }; this.subrows(row.start, nrows).col(col) } } impl <'a $(,
        $tt)?, R : Shape, T, Rs : Stride, Cs : Stride, RowRange : IntoRange < IdxInc < R
        >, Len < R >: 'a >> MatIndex < RowRange, Idx <$C >> for MatMut <'a, T, R, $C, Rs,
        Cs > { type Target = ColMut <'a, T, RowRange::Len < R >, Rs >; #[track_caller]
        #[inline] fn get(this : Self, row : RowRange, col : Idx <$C >) -> Self::Target {
        let row = row.into_range(R::start(), this.nrows().end()); assert!(all(col < this
        .ncols(), row.start <= row.end, row.end <= this.nrows(),));
        // SAFETY: the assert above gives `row.start <= row.end <= this.nrows()`, so the
        // subtraction does not underflow; `Len<R>` is `usize` (no invariant) except for
        // `..`, where the length is exactly `this.nrows()`, a valid `R`.
        let nrows = unsafe {
        RowRange::Len::< R >::new_unbound(row.end.unbound() - row.start.unbound()) };
        this.subrows_mut(row.start, nrows).col_mut(col) } #[track_caller] #[inline]
        unsafe fn get_unchecked(this : Self, row : RowRange, col : Idx <$C >) ->
        Self::Target { let row = row.into_range(R::start(), this.nrows().end());
        assert!(all(col < this.ncols(), row.start <= row.end, row.end <= this.nrows(),));
        // SAFETY: the assert above gives `row.start <= row.end <= this.nrows()`, so the
        // subtraction does not underflow; `Len<R>` is `usize` (no invariant) except for
        // `..`, where the length is exactly `this.nrows()`, a valid `R`.
        let nrows = unsafe { RowRange::Len::< R >::new_unbound(row.end.unbound() - row
        .start.unbound()) }; this.subrows_mut(row.start, nrows).col_mut(col) } }
    };
}
macro_rules! idx_impl {
    (($R:ty $(,$Rtt:tt)?), ($C:ty $(,$Ctt:tt)?)) => {
        impl <'a $(, $Rtt)? $(, $Ctt)?, T, Rs : Stride, Cs : Stride > MatIndex < Idx <$R
        >, Idx <$C >> for MatRef <'a, T, $R, $C, Rs, Cs > { type Target = &'a T;
        #[track_caller] #[inline] fn get(this : Self, row : Idx <$R >, col : Idx <$C >)
        -> Self::Target { this.at(row, col) } #[track_caller] #[inline] unsafe fn
        get_unchecked(this : Self, row : Idx <$R >, col : Idx <$C >) -> Self::Target {
        // SAFETY: the caller guarantees `row < this.nrows()` and `col < this.ncols()`
        // (# Safety of `MatIndex::get_unchecked`), which is the precondition of
        // `at_unchecked`.
        unsafe {
        this.at_unchecked(row, col) }} } impl <'a $(, $Rtt)? $(, $Ctt)?, T, Rs : Stride,
        Cs : Stride > MatIndex < Idx <$R >, Idx <$C >> for MatMut <'a, T, $R, $C, Rs, Cs
        > { type Target = &'a mut T; #[track_caller] #[inline] fn get(this : Self, row :
        Idx <$R >, col : Idx <$C >) -> Self::Target { this.at_mut(row, col) }
        #[track_caller] #[inline] unsafe fn get_unchecked(this : Self, row : Idx <$R >,
        col : Idx <$C >) -> Self::Target {
        // SAFETY: the caller guarantees `row < this.nrows()` and `col < this.ncols()`
        // (# Safety of `MatIndex::get_unchecked`), which is the precondition of
        // `at_mut_unchecked`.
        unsafe { this.at_mut_unchecked(row, col) }} }
    };
}
row_impl!(usize);
row_impl!(Dim <'N >, 'N);
col_impl!(usize);
col_impl!(Dim <'N >, 'N);
idx_impl!((usize), (usize));
idx_impl!((usize), (Dim <'N >, 'N));
idx_impl!((Dim <'M >, 'M), (usize));
idx_impl!((Dim <'M >, 'M), (Dim <'N >, 'N));
