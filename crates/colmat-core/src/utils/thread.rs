//! The team skeletons: joins, splits and item loops on a [`Par`].
//!
//! Every function here takes the parallelism by value and hands each
//! concurrent child its own part of it, so a child can never use more
//! participants than it was given. With [`Par::Seq`] everything runs on the
//! caller, in order.
//!
//! # Provenance
//!
//! Ported from colmat-rs `faer/src/utils/mod.rs` at `71337cdc` (lines
//! 11-291, the `thread` module). Changed: the item loops ([`for_each`],
//! [`for_each_with_par`]) take a [`Producer`] of this crate instead of
//! rayon's `IndexedParallelIterator` (debt K7), and are available without the
//! `threadteam` feature; `parallelism_degree` and `Ptr` are not ported
//! (`Par::degree` and the call sites' own pointer wrappers replace them).
//! faer's MIT notice: `LICENSE-faer`.

use crate::Par;

/// A sequence of items that can be cut in two at any position, and walked in
/// order on one thread.
///
/// This is what the item loops ([`for_each`], [`for_each_with_par`]) deal out
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

/// A closure handed to exactly one concurrent child: each index is taken once, which is what
/// the split below guarantees. `core` only, so the hand-off does not need `std`.
#[cfg(feature = "threadteam")]
pub(crate) struct Slot<T>(core::cell::UnsafeCell<Option<T>>);
// SAFETY: the value is moved out once, by the one child that owns the slot
// (`Slot::take`'s contract); no two threads access the cell, so sharing a
// `&Slot` only needs the value itself to be sendable to that child.
#[cfg(feature = "threadteam")]
unsafe impl<T: Send> Sync for Slot<T> {}
#[cfg(feature = "threadteam")]
impl<T> Slot<T> {
	pub(crate) fn new(value: T) -> Self {
		Self(core::cell::UnsafeCell::new(Some(value)))
	}

	/// # Safety
	/// Each slot is taken at most once, and by one thread: no other access to
	/// the cell runs concurrently.
	pub(crate) unsafe fn take(&self) -> T {
		// SAFETY: the caller guarantees exclusive access to the cell for this
		// call (one taker per slot).
		unsafe { (*self.0.get()).take() }.expect("one child per slot")
	}
}

/// executes the two operations, possibly in parallel, while splitting the
/// amount of parallelism between the two
#[inline]
pub fn join_raw(
	op_a: impl Send + FnOnce(Par),
	op_b: impl Send + FnOnce(Par),
	parallelism: Par,
) {
	fn implementation<'a>(
		op_a: &'a mut (dyn Send + FnMut(Par)),
		op_b: &'a mut (dyn Send + FnMut(Par)),
		mut parallelism: Par,
	) {
		match parallelism {
			Par::Seq => {
				op_a(parallelism.reborrow());
				op_b(parallelism);
			},
			#[cfg(feature = "threadteam")]
			Par::Team(team) => {
				let n_threads = team.size().get();
				if n_threads == 1 {
					op_a(Par::Team(team));
					op_b(Par::Seq);
				} else {
					// each half runs on the leader of a sub-team of its width and owns that
					// sub-team
					let half = n_threads / 2;
					let ops = [Slot::new(op_a), Slot::new(op_b)];
					super::team::note_split(2);
					let _ =
						team.split(&[n_threads - half, half], |c, child| {
							// SAFETY: the split runs its callback once per child index
							// `c < 2`, so each slot is taken once, by one thread
							unsafe { (ops[c].take())(Par::Team(child)) };
						});
				}
			},
			#[cfg(not(feature = "threadteam"))]
			Par::__Lifetime(x, _) => match x {},
		};
	}
	let mut op_a = Some(op_a);
	let mut op_b = Some(op_b);
	implementation(
		&mut |parallelism| op_a.take().unwrap()(parallelism),
		&mut |parallelism| op_b.take().unwrap()(parallelism),
		parallelism,
	)
}

/// Runs `f` on a part of `par` of `width` participants: `Par::Seq` when `width` or the degree
/// of `par` is below two (a team of one participant included), `par` itself when `width` is
/// its degree or more, otherwise a sub-team of `width` (a split into one child, which is led by
/// the caller and dispatches nothing; the members it leaves out stay idle). A region decides its
/// width with [`cost::par_width`](crate::utils::cost::par_width) and runs here, so only that
/// region is narrowed and the caller keeps its whole `par`.
pub fn with_width<R>(par: Par, width: usize, f: impl FnOnce(Par) -> R) -> R {
	// before the `width >= degree` return: a held team of one would otherwise reach the
	// parallel branches with one thread (colmat-rs#168)
	if width < 2 || par.degree() < 2 {
		return f(Par::Seq);
	}
	if width >= par.degree() {
		return f(par);
	}
	match par {
		Par::Seq => f(Par::Seq),
		#[cfg(feature = "threadteam")]
		Par::Team(team) => {
			super::team::note_width(width);
			struct Once<F, R>(core::cell::UnsafeCell<(Option<F>, Option<R>)>);
			// SAFETY: a split into the single child `[width]` runs its callback once, on the
			// calling thread (child 0 is led by the caller), so the cell is never accessed
			// from another thread and neither `F` nor `R` crosses one.
			unsafe impl<F, R> Sync for Once<F, R> {}
			impl<F, R> Once<F, R> {
				// a method, so that the closure below captures the whole `Once` (which is
				// `Sync`) rather than its field (which is not)
				fn get(&self) -> *mut (Option<F>, Option<R>) {
					self.0.get()
				}
			}
			let once = Once(core::cell::UnsafeCell::new((Some(f), None)));
			team.split(&[width], |_, child| {
				// SAFETY: see `Once`: this is the only access to the cell while the split runs.
				let (f, out) = unsafe { &mut *once.get() };
				*out = Some((f.take().expect("one child"))(Par::Team(child)));
			})
			.expect("2 <= width < size is a valid split");
			once.0.into_inner().1.expect("the child ran")
		},
		#[cfg(not(feature = "threadteam"))]
		Par::__Lifetime(x, _) => match x {},
	}
}

/// A flat item loop on the team of `par`: the items of `producer` are dealt out in contiguous
/// blocks over `min(n_jobs, items, degree)` jobs, each walked by the leader of its part of the
/// team; the caller alone walks them, in order, for `Seq`.
pub fn for_each<P: Producer>(
	par: Par,
	n_jobs: usize,
	producer: P,
	f: impl Sync + Fn(P::Item),
) {
	match par {
		Par::Seq => producer.into_items().for_each(f),
		#[cfg(feature = "threadteam")]
		Par::Team(team) => super::team::for_each(team, n_jobs, producer, f),
		#[cfg(not(feature = "threadteam"))]
		Par::__Lifetime(x, _) => {
			let _ = (n_jobs, producer, f);
			match x {}
		},
	}
}

/// [`for_each`] whose closure also receives the parallelism of its job: the share of `par`
/// that one of `n_jobs` concurrent jobs may use (sequential below two threads)
pub fn for_each_with_par<P: Producer>(
	par: Par,
	n_jobs: usize,
	producer: P,
	f: impl Sync + Fn(P::Item, Par),
) {
	match par {
		Par::Seq => {
			for_each(Par::Seq, n_jobs, producer, |item| f(item, Par::Seq))
		},
		#[cfg(feature = "threadteam")]
		Par::Team(team) => {
			// the item runs on the leader of its own sub-team and receives it
			super::team::for_each_child(
				team,
				n_jobs,
				producer,
				|item, child| f(item, Par::Team(child)),
			)
		},
		#[cfg(not(feature = "threadteam"))]
		Par::__Lifetime(x, _) => {
			let _ = (n_jobs, producer, f);
			match x {}
		},
	}
}

/// runs the two operations concurrently on a split of `par` in which the first receives
/// `size_a` threads (clamped so that both get at least one) and the second the rest
pub fn join_sized(
	par: Par,
	size_a: usize,
	op_a: impl Send + FnOnce(Par),
	op_b: impl Send + FnOnce(Par),
) {
	let degree = par.degree();
	if degree < 2 {
		op_a(Par::Seq);
		op_b(Par::Seq);
		return;
	}
	#[cfg(feature = "threadteam")]
	{
		let size_a = size_a.clamp(1, degree - 1);
		// a typed slot per closure, so nothing is boxed (colmat-rs#111: no allocation per
		// region)
		let (a, b) = (Slot::new(op_a), Slot::new(op_b));
		split(par, &[size_a, degree - size_a], &|c, child| {
			// SAFETY: the split runs its callback once per child index `c < 2`, so each
			// slot is taken once, by one thread
			unsafe {
				if c == 0 {
					(a.take())(child)
				} else {
					(b.take())(child)
				}
			}
		});
	}
	// a degree of two or more needs a team
	#[cfg(not(feature = "threadteam"))]
	{
		let _ = (size_a, op_a, op_b);
		unreachable!("a degree of {degree} without the parallel route")
	}
}

/// runs `f(c, child)` for every `c < sizes.len()` concurrently, where `child` is the part of
/// `par` of width `sizes[c]` (sequential for a width of one). The widths must be non-zero and
/// add up to at most the degree of `par`; the children never use more than they are handed.
///
/// # Panics
///
/// Panics if a width is zero.
#[track_caller]
pub fn split(par: Par, sizes: &[usize], f: &(dyn Sync + Fn(usize, Par))) {
	equator::assert!(sizes.iter().all(|&s| s > 0));
	match par {
		Par::Seq => {
			for c in 0..sizes.len() {
				f(c, Par::Seq);
			}
		},
		#[cfg(feature = "threadteam")]
		Par::Team(team) => {
			if sizes.iter().sum::<usize>() <= team.size().get() {
				super::team::note_split(sizes.len());
				let _ = team.split(sizes, |c, child| {
					f(c, Par::Team(child));
				});
			} else {
				// the team is too small for the requested widths: run the parts in turn
				for c in 0..sizes.len() {
					f(c, Par::Seq);
				}
			}
		},
		#[cfg(not(feature = "threadteam"))]
		Par::__Lifetime(x, _) => match x {},
	}
}

/// returns the start and length of a subsegment of `0..n`, split between
/// `chunk_count` consumers, for the consumer at index `idx`
///
/// for the same `n` and `chunk_count`, different values of `idx` between in
/// `0..chunk_count` will represent distinct subsegments
#[inline]
pub fn par_split_indices(
	n: usize,
	idx: usize,
	chunk_count: usize,
) -> (usize, usize) {
	let chunk_size = n / chunk_count;
	let rem = n % chunk_count;
	let idx_to_col_start = move |idx| {
		if idx < rem {
			idx * (chunk_size + 1)
		} else {
			rem + idx * chunk_size
		}
	};
	let start = idx_to_col_start(idx);
	let end = idx_to_col_start(idx + 1);
	(start, end - start)
}
