use super::*;
use crate::utils::bound::{Array, Dim};
use crate::{Idx, assert};
/// see [`super::PermRef`]
#[derive(Debug)]
pub struct Ref<'a, I: Index, N: Shape = usize> {
	pub(super) forward: &'a [N::Idx<I>],
	pub(super) inverse: &'a [N::Idx<I>],
}
impl<I: Index, N: Shape> Copy for Ref<'_, I, N> {}
impl<I: Index, N: Shape> Clone for Ref<'_, I, N> {
	#[inline]
	fn clone(&self) -> Self {
		*self
	}
}
impl<'short, I: Index, N: Shape> Reborrow<'short> for Ref<'_, I, N> {
	type Target = Ref<'short, I, N>;

	#[inline]
	fn rb(&'short self) -> Self::Target {
		*self
	}
}
impl<'short, I: Index, N: Shape> ReborrowMut<'short> for Ref<'_, I, N> {
	type Target = Ref<'short, I, N>;

	#[inline]
	fn rb_mut(&'short mut self) -> Self::Target {
		*self
	}
}
impl<'a, I: Index, N: Shape> IntoConst for Ref<'a, I, N> {
	type Target = Ref<'a, I, N>;

	#[inline]
	fn into_const(self) -> Self::Target {
		self
	}
}
impl<
	I: Index,
	N: Shape,
	Inner: for<'short> Reborrow<'short, Target = Ref<'short, I, N>>,
> generic::Perm<Inner>
{
	/// convert `self` to a permutation view
	#[inline]
	pub fn as_ref(&self) -> PermRef<'_, I, N> {
		PermRef { 0: self.0.rb() }
	}
}
impl<'a, I: Index, N: Shape> PermRef<'a, I, N> {
	/// returns the input permutation with the given shape after checking that
	/// it matches the current shape
	#[inline]
	pub fn as_shape<M: Shape>(self, dim: M) -> PermRef<'a, I, M> {
		assert!(self.len().unbound() == dim.unbound());
		// SAFETY: both slices have length `dim` (checked above) and are
		// borrowed for `'a`; `N::Idx<I>` and `M::Idx<I>` both have the layout
		// of `I`, and every entry is `< dim` by the permutation invariant, so
		// the entries are valid `M::Idx<I>` and the invariant holds for shape
		// `M`.
		// TODO(review): the layout claim fails when `N` is `One`
		// (`One::Idx<I>` is the zero-sized `Zero`): a `PermRef<'_, I, One>`
		// built by `new_unchecked` over real `&[Zero]` slices, then
		// `as_shape(1usize)`, reads an `I` from a zero-sized allocation.
		unsafe {
			PermRef {
				0: Ref {
					forward: core::slice::from_raw_parts(
						self.forward.as_ptr() as _,
						dim.unbound(),
					),
					inverse: core::slice::from_raw_parts(
						self.inverse.as_ptr() as _,
						dim.unbound(),
					),
				},
			}
		}
	}

	/// creates a new permutation, by checking the validity of the inputs
	///
	/// # panics
	///
	/// the function panics if any of the following conditions are violated:
	/// * `forward` and `inverse` must have the same length which must be less
	///   than or equal to
	///   `I::Signed::MAX`, be valid permutations, and be inverse permutations of
	///   each other
	#[inline]
	#[track_caller]
	pub fn new_checked(
		forward: &'a [Idx<N, I>],
		inverse: &'a [Idx<N, I>],
		dim: N,
	) -> Self {
		#[track_caller]
		fn check<I: Index>(forward: &[I], inverse: &[I], n: usize) {
			assert!(all(
				n <= I::Signed::MAX.zx(),
				forward.len() == n,
				inverse.len() == n,
			));
			for (i, &p) in forward.iter().enumerate() {
				let p = p.to_signed().zx();
				assert!(p < n);
				assert!(inverse[p].to_signed().zx() == i);
			}
		}
		check(
			I::canonicalize(N::cast_idx_slice(forward)),
			I::canonicalize(N::cast_idx_slice(inverse)),
			dim.unbound(),
		);
		Self(Ref { forward, inverse })
	}

	/// creates a new permutation reference, without checking the validity of
	/// the inputs
	///
	/// # Safety
	///
	/// `forward` and `inverse` must have the same length which must be less
	/// than or equal to `I::Signed::MAX`, be valid permutations, and be
	/// inverse permutations of each other
	#[inline]
	#[track_caller]
	pub unsafe fn new_unchecked(
		forward: &'a [Idx<N, I>],
		inverse: &'a [Idx<N, I>],
		dim: N,
	) -> Self {
		assert!(all(
			dim.unbound() <= I::Signed::MAX.zx(),
			forward.len() == dim.unbound(),
			inverse.len() == dim.unbound(),
		));
		Self(Ref { forward, inverse })
	}

	#[allow(clippy::type_complexity)] // faer signature shape, kept explicit to match upstream
	/// returns the permutation as an array
	#[inline]
	pub fn arrays(self) -> (&'a [Idx<N, I>], &'a [Idx<N, I>]) {
		(self.forward, self.inverse)
	}

	/// returns the dimension of the permutation
	#[inline]
	pub fn len(&self) -> N {
		// SAFETY: by the permutation invariant, `forward.len()` is a valid
		// value of `N`.
		unsafe { N::new_unbound(self.forward.len()) }
	}

	/// returns the inverse permutation
	#[inline]
	pub fn inverse(self) -> Self {
		Self(Ref {
			forward: self.inverse,
			inverse: self.forward,
		})
	}

	/// returns the inverse permutation
	#[inline]
	pub fn transpose(self) -> Self {
		self.inverse()
	}

	/// cast the permutation to the fixed width index type
	#[inline(always)]
	pub fn canonicalized(self) -> PermRef<'a, I::FixedWidth, N> {
		// SAFETY: `I::FixedWidth` is the unsigned integer of the same size
		// (hence alignment) as `I` (`Index` is implemented only for `u32`,
		// `u64` and `usize`), so `N::Idx<I>` and `N::Idx<I::FixedWidth>` have
		// the same layout, and each entry keeps its value; the slices keep
		// their lengths and lifetime `'a`, so the permutation invariant holds.
		unsafe {
			PermRef {
				0: Ref {
					forward: core::slice::from_raw_parts(
						self.forward.as_ptr() as _,
						self.forward.len(),
					),
					inverse: core::slice::from_raw_parts(
						self.inverse.as_ptr() as _,
						self.inverse.len(),
					),
				},
			}
		}
	}

	/// cast the permutation from the fixed width index type
	#[inline(always)]
	pub fn uncanonicalized<J: Index>(self) -> PermRef<'a, J, N> {
		assert!(core::mem::size_of::<J>() == core::mem::size_of::<I>());
		// SAFETY: `I` and `J` are unsigned integers of the same size (checked
		// above; `Index` is implemented only for `u32`, `u64` and `usize`),
		// hence of the same alignment, so `N::Idx<I>` and `N::Idx<J>` have the
		// same layout and each entry keeps its value; `J::Signed::MAX ==
		// I::Signed::MAX`, so the permutation invariant holds for `J`.
		unsafe {
			PermRef {
				0: Ref {
					forward: core::slice::from_raw_parts(
						self.forward.as_ptr() as _,
						self.forward.len(),
					),
					inverse: core::slice::from_raw_parts(
						self.inverse.as_ptr() as _,
						self.inverse.len(),
					),
				},
			}
		}
	}
}
impl<'a, 'N, I: Index> PermRef<'a, I, Dim<'N>> {
	#[allow(clippy::type_complexity)] // faer signature shape, kept explicit to match upstream
	/// returns the permutation as an array
	#[inline]
	pub fn bound_arrays(
		self,
	) -> (
		&'a Array<'N, Idx<Dim<'N>, I>>,
		&'a Array<'N, Idx<Dim<'N>, I>>,
	) {
		// SAFETY: `Array<'N, T>` is `#[repr(transparent)]` over `[T]`, and its
		// invariant, length equal to the value tied to `'N`, holds for both
		// slices by the permutation invariant; the borrows keep lifetime `'a`.
		unsafe {
			(
				&*(self.forward as *const [Idx<Dim<'N>, I>]
					as *const Array<'N, Idx<Dim<'N>, I>>),
				&*(self.inverse as *const [Idx<Dim<'N>, I>]
					as *const Array<'N, Idx<Dim<'N>, I>>),
			)
		}
	}
}
