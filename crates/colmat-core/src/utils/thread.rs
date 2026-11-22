//! Splittable sequences of items: what the team skeletons deal out to the members of a team.
//!
//! The skeletons themselves (`join_raw`, `for_each`, ...) are ported from colmat-rs
//! `faer/src/utils/mod.rs` at `71337cdc` in the following commit; this commit adds only the
//! [`Producer`] trait they take in place of rayon's `IndexedParallelIterator` (debt K7).

/// A sequence of items that can be cut in two at any position, and walked in
/// order on one thread.
///
/// This is what the item loops (`for_each`, `for_each_with_par`) deal out
/// to the members of a team: the producer is split on the leader into one
/// contiguous block per job, and each block is walked by the leader of its
/// part. Nothing is collected, so dealing allocates nothing whatever the
/// team's width. It replaces rayon's `Producer`, which colmat-rs used as the
/// same interface (design-debt work log, K7).
///
/// Implemented for index ranges, for the chunks of a matrix view
/// ([`MatRef::par_col_chunks`](crate::MatRef::par_col_chunks) and its
/// relatives), and for two producers of the same length walked together
/// ([`Producer::zip_eq`]).
pub trait Producer: Send + Sized {
	/// The item.
	type Item;
	/// The walk over the items, on one thread.
	type IntoIter: Iterator<Item = Self::Item>;

	/// The number of items.
	fn len(&self) -> usize;

	/// Whether there is no item.
	#[inline]
	fn is_empty(&self) -> bool {
		self.len() == 0
	}

	/// The first `index` items and the rest. `index <= self.len()`.
	fn split_at(self, index: usize) -> (Self, Self);

	/// The items, in order.
	fn into_items(self) -> Self::IntoIter;

	/// The pairs of the items of `self` and `other`.
	///
	/// # Panics
	///
	/// Panics if the two lengths differ.
	#[track_caller]
	#[inline]
	fn zip_eq<P: Producer>(self, other: P) -> Zip<Self, P> {
		equator::assert!(self.len() == other.len());
		Zip(self, other)
	}
}

impl Producer for core::ops::Range<usize> {
	type IntoIter = core::ops::Range<usize>;
	type Item = usize;

	#[inline]
	fn len(&self) -> usize {
		ExactSizeIterator::len(self)
	}

	#[inline]
	#[track_caller]
	fn split_at(self, index: usize) -> (Self, Self) {
		equator::assert!(index <= Producer::len(&self));
		let mid = self.start + index;
		(self.start..mid, mid..self.end)
	}

	#[inline]
	fn into_items(self) -> Self::IntoIter {
		self
	}
}

/// Two producers of the same length, walked together ([`Producer::zip_eq`]).
#[derive(Debug)]
pub struct Zip<A, B>(A, B);

impl<A: Producer, B: Producer> Producer for Zip<A, B> {
	type IntoIter = core::iter::Zip<A::IntoIter, B::IntoIter>;
	type Item = (A::Item, B::Item);

	#[inline]
	fn len(&self) -> usize {
		self.0.len()
	}

	#[inline]
	fn split_at(self, index: usize) -> (Self, Self) {
		let (a0, a1) = self.0.split_at(index);
		let (b0, b1) = self.1.split_at(index);
		(Zip(a0, b0), Zip(a1, b1))
	}

	#[inline]
	fn into_items(self) -> Self::IntoIter {
		self.0.into_items().zip(self.1.into_items())
	}
}
