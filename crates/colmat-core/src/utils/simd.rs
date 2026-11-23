//! Register-sized walks over a contiguous column.
//!
//! [`SimdCtx`] splits a contiguous column of length `len` into full registers (the body) and at
//! most one partial register (the tail), and reads and writes them through the scalar's SIMD
//! provider ([`ComplexField::Simd`]). A kernel written on it runs for every scalar: with
//! [`NoSimd`](colmat_traits::NoSimd) a register holds one element and the walk is a scalar loop.
//!
//! The column's length is branded (`Dim<'N>`), and every body and tail index is built by the
//! context from that length, so a read or a write through one stays inside the column without a
//! bounds check.
//!
//! Ported from colmat-rs `faer/src/utils/simd.rs` at `71337cdc` (MIT, faer's notice in
//! `LICENSE-faer`). Changed in this trial (#23):
//!
//! - registers move through [`SimdLayout::load`]/[`SimdLayout::mask_load`] on elements, not
//!   through a cast of the column's pointer to a register type, so a provider whose registers
//!   hold elements split (complex, double-word) needs no layout assumption here;
//! - the aligned head of `new_align` is not ported: the products of this step do not use it;
//! - the arithmetic is reached through methods of the context that forward to
//!   [`SimdField`], since the provider is no longer a set of `ComplexField` methods;
//! - `vzeroupper` handling (`clear_upper_vector_state`) waits for the x86 kernels.

use crate::ContiguousFwd;
use crate::col::{ColMut, ColRef};
use crate::utils::bound::Dim;
use colmat_traits::pulp::Simd;
use colmat_traits::{ComplexField, SimdField, SimdLayout};
use core::marker::PhantomData;

/// The provider of `T`.
pub type P<T> = <T as ComplexField>::Simd;
/// A register of `T` for the instruction set `S`.
pub type Vec<T, S> = <P<T> as SimdLayout<T>>::Vec<S>;
/// A mask over the lanes of a register of `T`.
pub type Mask<T, S> = <P<T> as SimdLayout<T>>::Mask<S>;
type MemMask<T, S> = <P<T> as SimdLayout<T>>::MemMask<S>;
/// The provider's context for `S`.
pub type Ctx<T, S> = <P<T> as SimdLayout<T>>::Ctx<S>;

/// Elements per register of `T` on the instruction set that `T`'s dispatcher selects.
#[inline]
pub fn lane_count<T: ComplexField>() -> usize {
	struct Lanes<T>(PhantomData<fn() -> T>);
	impl<T: ComplexField> colmat_traits::pulp::WithSimd for Lanes<T> {
		type Output = usize;

		#[inline(always)]
		fn with_simd<S: Simd>(self, _: S) -> usize {
			P::<T>::lanes::<S>()
		}
	}
	dispatch::<T, _>(Lanes::<T>(PhantomData))
}

/// Runs `f` on the instruction set that `T`'s provider selects.
#[inline(always)]
pub fn dispatch<T: ComplexField, F: colmat_traits::pulp::WithSimd>(
	f: F,
) -> F::Output {
	colmat_traits::SimdArch::dispatch(
		<<P<T> as SimdLayout<T>>::Arch as Default>::default(),
		f,
	)
}

/// A walk over a contiguous column of length `len`, in registers of `T`.
pub struct SimdCtx<'N, T: ComplexField, S: Simd> {
	/// The provider's context.
	pub ctx: Ctx<T, S>,
	/// The length of the column.
	pub len: Dim<'N>,
	lanes: usize,
	body_end: usize,
	tail_end: usize,
	tail_mask: Mask<T, S>,
	tail_mem_mask: MemMask<T, S>,
}

impl<T: ComplexField, S: Simd> Copy for SimdCtx<'_, T, S> {}
impl<T: ComplexField, S: Simd> Clone for SimdCtx<'_, T, S> {
	#[inline]
	fn clone(&self) -> Self {
		*self
	}
}
impl<T: ComplexField, S: Simd> core::fmt::Debug for SimdCtx<'_, T, S> {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		f.debug_struct("SimdCtx")
			.field("len", &*self.len)
			.field("lanes", &self.lanes)
			.field("body_end", &self.body_end)
			.field("tail_end", &self.tail_end)
			.finish_non_exhaustive()
	}
}

/// A full register of a column: elements `start..start + lanes`.
#[derive(Debug)]
pub struct SimdBody<'N> {
	start: usize,
	__marker: PhantomData<Dim<'N>>,
}
/// The partial register at the end of a column: elements `start..len`, with `len - start` in
/// `1..=lanes`.
#[derive(Debug)]
pub struct SimdTail<'N> {
	start: usize,
	__marker: PhantomData<Dim<'N>>,
}
impl Copy for SimdBody<'_> {}
impl Clone for SimdBody<'_> {
	#[inline]
	fn clone(&self) -> Self {
		*self
	}
}
impl Copy for SimdTail<'_> {}
impl Clone for SimdTail<'_> {
	#[inline]
	fn clone(&self) -> Self {
		*self
	}
}
impl SimdBody<'_> {
	/// The index of the first element.
	#[inline]
	pub fn start(self) -> usize {
		self.start
	}
}
impl SimdTail<'_> {
	/// The index of the first element.
	#[inline]
	pub fn start(self) -> usize {
		self.start
	}
}

/// A register position in a column of length `'N`: a [`SimdBody`] or a [`SimdTail`].
pub trait SimdIndex<'N, T: ComplexField, S: Simd>: Copy {
	/// Reads the register at `self`.
	fn read(
		simd: &SimdCtx<'N, T, S>,
		col: ColRef<'_, T, Dim<'N>, ContiguousFwd>,
		at: Self,
	) -> Vec<T, S>;
	/// Writes the register at `self`; for a tail, only the elements inside the column.
	fn write(
		simd: &SimdCtx<'N, T, S>,
		col: ColMut<'_, T, Dim<'N>, ContiguousFwd>,
		at: Self,
		value: Vec<T, S>,
	);
}

impl<'N, T: ComplexField, S: Simd> SimdIndex<'N, T, S> for SimdBody<'N> {
	#[inline(always)]
	fn read(
		simd: &SimdCtx<'N, T, S>,
		col: ColRef<'_, T, Dim<'N>, ContiguousFwd>,
		at: Self,
	) -> Vec<T, S> {
		let col = col.as_slice();
		// SAFETY: a `SimdBody<'N>` is built only by `SimdCtx::<'N, ..>::indices`/`batch_indices`,
		// with `start + lanes <= len`, and `col` has the branded length `len`.
		let src = unsafe { col.get_unchecked(at.start..at.start + simd.lanes) };
		P::<T>::load(&simd.ctx, src)
	}

	#[inline(always)]
	fn write(
		simd: &SimdCtx<'N, T, S>,
		col: ColMut<'_, T, Dim<'N>, ContiguousFwd>,
		at: Self,
		value: Vec<T, S>,
	) {
		let col = col.as_slice_mut();
		// SAFETY: as in `read`.
		let dst =
			unsafe { col.get_unchecked_mut(at.start..at.start + simd.lanes) };
		P::<T>::store(&simd.ctx, dst, value)
	}
}

impl<'N, T: ComplexField, S: Simd> SimdIndex<'N, T, S> for SimdTail<'N> {
	#[inline(always)]
	fn read(
		simd: &SimdCtx<'N, T, S>,
		col: ColRef<'_, T, Dim<'N>, ContiguousFwd>,
		at: Self,
	) -> Vec<T, S> {
		// SAFETY: a `SimdTail<'N>` is built only by `SimdCtx::<'N, ..>::indices`/`batch_indices`,
		// with `start <= len`, so the pointer stays inside the column or one past it; the mask
		// enables exactly `0..len - start` (`SimdLayout` contract of `mem_mask_between`), so every
		// element read is one of the column's, initialised and aligned (view invariant).
		unsafe {
			P::<T>::mask_load(
				&simd.ctx,
				simd.tail_mem_mask,
				col.as_ptr().add(at.start),
			)
		}
	}

	#[inline(always)]
	fn write(
		simd: &SimdCtx<'N, T, S>,
		col: ColMut<'_, T, Dim<'N>, ContiguousFwd>,
		at: Self,
		value: Vec<T, S>,
	) {
		// SAFETY: as in `read`; `col` is a unique view, so the writes alias nothing. The
		// overwritten elements are not dropped, which at worst leaks for a scalar that owns
		// memory (as in colmat-rs).
		unsafe {
			P::<T>::mask_store(
				&simd.ctx,
				simd.tail_mem_mask,
				col.as_ptr_mut().add(at.start),
				value,
			)
		}
	}
}

impl<'N, T: ComplexField, S: Simd> SimdCtx<'N, T, S> {
	/// A walk over `len` elements: `len / lanes` full registers, then a tail when `lanes` does
	/// not divide `len`.
	#[inline(always)]
	pub fn new(simd: S, len: Dim<'N>) -> Self {
		let lanes = P::<T>::lanes::<S>();
		let rem = *len % lanes;
		Self::with_split(simd, len, lanes, *len / lanes, rem)
	}

	/// A walk whose tail is never empty for `len > 0`: `(len - 1) / lanes` full registers, then
	/// a tail of `1..=lanes` elements. Kernels that keep their last register in the tail use it.
	///
	/// # Panics
	///
	/// When `len == 0`.
	#[inline(always)]
	#[track_caller]
	pub fn new_force_mask(simd: S, len: Dim<'N>) -> Self {
		equator::assert!(*len != 0);
		let lanes = P::<T>::lanes::<S>();
		let body = (*len - 1) / lanes;
		Self::with_split(simd, len, lanes, body, *len - body * lanes)
	}

	#[inline(always)]
	fn with_split(
		simd: S,
		len: Dim<'N>,
		lanes: usize,
		body_end: usize,
		tail_len: usize,
	) -> Self {
		let ctx = P::<T>::ctx(simd);
		Self {
			ctx,
			len,
			lanes,
			body_end,
			tail_end: body_end + (tail_len != 0) as usize,
			tail_mask: P::<T>::mask_between(&ctx, 0, tail_len),
			tail_mem_mask: P::<T>::mem_mask_between(&ctx, 0, tail_len),
		}
	}

	/// Elements per register.
	#[inline(always)]
	pub fn lanes(&self) -> usize {
		self.lanes
	}

	/// The lanes of the tail register that hold elements of the column.
	#[inline(always)]
	pub fn tail_mask(&self) -> Mask<T, S> {
		self.tail_mask
	}

	/// The full registers, then the tail if there is one.
	#[inline(always)]
	#[allow(clippy::type_complexity)]
	pub fn indices(
		&self,
	) -> (
		impl Clone + ExactSizeIterator + DoubleEndedIterator<Item = SimdBody<'N>>,
		Option<SimdTail<'N>>,
	) {
		let lanes = self.lanes;
		(
			(0..self.body_end).map(
				#[inline(always)]
				move |i| SimdBody {
					start: i * lanes,
					__marker: PhantomData,
				},
			),
			self.tail(),
		)
	}

	/// As [`indices`](Self::indices), with the full registers grouped by `BATCH` and the
	/// `body_end % BATCH` left over ones apart.
	#[inline(always)]
	#[allow(clippy::type_complexity)]
	pub fn batch_indices<const BATCH: usize>(
		&self,
	) -> (
		impl Clone
		+ ExactSizeIterator
		+ DoubleEndedIterator<Item = [SimdBody<'N>; BATCH]>,
		impl Clone + ExactSizeIterator + DoubleEndedIterator<Item = SimdBody<'N>>,
		Option<SimdTail<'N>>,
	) {
		let lanes = self.lanes;
		let batched = self.body_end / BATCH * BATCH;
		(
			(0..self.body_end / BATCH).map(
				#[inline(always)]
				move |b| {
					core::array::from_fn(
						#[inline(always)]
						|k| SimdBody {
							start: (b * BATCH + k) * lanes,
							__marker: PhantomData,
						},
					)
				},
			),
			(batched..self.body_end).map(
				#[inline(always)]
				move |i| SimdBody {
					start: i * lanes,
					__marker: PhantomData,
				},
			),
			self.tail(),
		)
	}

	#[inline(always)]
	fn tail(&self) -> Option<SimdTail<'N>> {
		(self.body_end != self.tail_end).then_some(SimdTail {
			start: self.body_end * self.lanes,
			__marker: PhantomData,
		})
	}

	/// Reads the register at `at` of `col`.
	#[inline(always)]
	pub fn read<I: SimdIndex<'N, T, S>>(
		&self,
		col: ColRef<'_, T, Dim<'N>, ContiguousFwd>,
		at: I,
	) -> Vec<T, S> {
		I::read(self, col, at)
	}

	/// Writes the register at `at` of `col`.
	#[inline(always)]
	pub fn write<I: SimdIndex<'N, T, S>>(
		&self,
		col: ColMut<'_, T, Dim<'N>, ContiguousFwd>,
		at: I,
		value: Vec<T, S>,
	) {
		I::write(self, col, at, value)
	}
}

/// The register operations the kernels use, forwarded to `T`'s provider so a kernel reads as
/// `simd.mul_add(a, b, acc)`.
pub trait SimdOps<T: ComplexField, S: Simd> {
	/// The provider's context.
	fn ctx(&self) -> &Ctx<T, S>;

	/// A register of zeros.
	#[inline(always)]
	fn zero(&self) -> Vec<T, S> {
		P::<T>::splat(self.ctx(), &colmat_traits::math_utils::zero::<T>())
	}
	/// Every lane set to `value`.
	#[inline(always)]
	fn splat(&self, value: &T) -> Vec<T, S> {
		P::<T>::splat(self.ctx(), value)
	}
	/// `lhs + rhs`.
	#[inline(always)]
	fn add(&self, lhs: Vec<T, S>, rhs: Vec<T, S>) -> Vec<T, S> {
		P::<T>::add(self.ctx(), lhs, rhs)
	}
	/// `lhs * rhs`.
	#[inline(always)]
	fn mul(&self, lhs: Vec<T, S>, rhs: Vec<T, S>) -> Vec<T, S> {
		P::<T>::mul(self.ctx(), lhs, rhs)
	}
	/// `lhs * rhs + acc`.
	#[inline(always)]
	fn mul_add(
		&self,
		lhs: Vec<T, S>,
		rhs: Vec<T, S>,
		acc: Vec<T, S>,
	) -> Vec<T, S> {
		P::<T>::mul_add(self.ctx(), lhs, rhs, acc)
	}
	/// `conj(lhs) * rhs + acc`.
	#[inline(always)]
	fn conj_mul_add(
		&self,
		lhs: Vec<T, S>,
		rhs: Vec<T, S>,
		acc: Vec<T, S>,
	) -> Vec<T, S> {
		P::<T>::conj_mul_add(self.ctx(), lhs, rhs, acc)
	}
	/// `conj(value)`.
	#[inline(always)]
	fn conj(&self, value: Vec<T, S>) -> Vec<T, S> {
		P::<T>::conj(self.ctx(), value)
	}
	/// The sum of the lanes.
	#[inline(always)]
	fn reduce_sum(&self, value: Vec<T, S>) -> T {
		P::<T>::reduce_sum(self.ctx(), value)
	}
}

impl<T: ComplexField, S: Simd> SimdOps<T, S> for SimdCtx<'_, T, S> {
	#[inline(always)]
	fn ctx(&self) -> &Ctx<T, S> {
		&self.ctx
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{Col, c64};
	use colmat_traits::pulp::WithSimd;

	/// Copies `src` to `dst` through the walk of `force_mask`, then checks that every element
	/// arrived and that the body and tail cover `0..len` once, in order.
	fn round_trip<T: ComplexField>(src: &[T], force_mask: bool) {
		struct Go<'a, T> {
			src: &'a [T],
			force_mask: bool,
		}
		impl<T: ComplexField> WithSimd for Go<'_, T> {
			type Output = Vec2<T>;

			fn with_simd<S: Simd>(self, simd: S) -> Vec2<T> {
				let src = Col::from_fn(self.src.len(), |i| self.src[i].clone());
				let mut dst = Col::from_fn(self.src.len(), |_| {
					colmat_traits::math_utils::zero::<T>()
				});
				crate::with_dim!(N, self.src.len());
				let src =
					src.as_ref().as_row_shape(N).try_as_col_major().unwrap();
				let mut dst_v = dst
					.as_mut()
					.as_row_shape_mut(N)
					.try_as_col_major_mut()
					.unwrap();
				let ctx = if self.force_mask {
					SimdCtx::<T, S>::new_force_mask(simd, N)
				} else {
					SimdCtx::<T, S>::new(simd, N)
				};
				let (body, tail) = ctx.indices();
				let mut starts = alloc::vec::Vec::new();
				for i in body {
					starts.push(i.start());
					let v = ctx.read(src, i);
					ctx.write(dst_v.rb_mut(), i, v);
				}
				if let Some(i) = tail {
					starts.push(i.start());
					let v = ctx.read(src, i);
					ctx.write(dst_v.rb_mut(), i, v);
				}
				let expected: alloc::vec::Vec<usize> =
					(0..self.src.len().div_ceil(ctx.lanes()))
						.map(|i| i * ctx.lanes())
						.collect();
				assert_eq!(starts, expected);
				Vec2(dst.as_ref().iter().cloned().collect())
			}
		}
		pub struct Vec2<T>(alloc::vec::Vec<T>);
		let out = dispatch::<T, _>(Go { src, force_mask });
		assert!(out.0 == src);
	}
	use reborrow::ReborrowMut;

	#[test]
	fn walks_cover_every_length() {
		for len in 0..40usize {
			let f: alloc::vec::Vec<f64> =
				(0..len).map(|i| i as f64 + 0.5).collect();
			let c: alloc::vec::Vec<c64> = (0..len)
				.map(|i| c64::new(i as f64, -(i as f64) - 0.25))
				.collect();
			let s: alloc::vec::Vec<f32> = (0..len).map(|i| i as f32).collect();
			round_trip(&f, false);
			round_trip(&c, false);
			round_trip(&s, false);
			if len > 0 {
				round_trip(&f, true);
				round_trip(&c, true);
				round_trip(&s, true);
			}
		}
	}
}
