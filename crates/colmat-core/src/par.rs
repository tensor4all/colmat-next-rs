//! The parallelism token [`Par`] and its shape [`ParShape`].
//!
//! # Provenance
//!
//! Ported from colmat-rs `faer/src/lib.rs` at `71337cdc` (lines 1016-1194:
//! `Par`, `ParShape` and their methods). Changed: one feature, `threadteam`,
//! gates the parallel route (colmat-rs had `rayon` and `threadteam`); the
//! doctests name this crate. faer's MIT notice: `LICENSE-faer`.

#[cfg(feature = "threadteam")]
use core::num::NonZeroUsize;

/// determines the parallelization configuration, and carries the right to execute with it
///
/// `Par` is neither `Copy` nor `Clone`: a `Team` owns exclusive authority over reserved
/// threads, so a duplicate could hand the same threads to two concurrent children. Every
/// function that runs work takes `Par` by value; a caller that needs another phase lends the
/// parallelism with [`Par::reborrow`], and concurrent children are created by splitting
/// (see [`crate::utils::thread::split`]), never by copying. Queries that only need the size
/// (workspace requirements, thresholds) use [`ParShape`] or [`Par::degree`].
///
/// ```
/// use colmat_core::{Par, ParShape};
/// // sizes are plain values
/// fn is_copy<T: Copy>() {}
/// is_copy::<ParShape>();
/// // phases lend, they do not copy
/// fn phase(_: Par<'_>) {}
/// let mut par = Par::Seq;
/// phase(par.reborrow());
/// phase(par);
/// ```
///
/// Not `Copy`:
///
/// ```compile_fail
/// fn is_copy<T: Copy>() {}
/// is_copy::<colmat_core::Par<'static>>();
/// ```
///
/// Not `Clone`:
///
/// ```compile_fail
/// fn is_clone<T: Clone>() {}
/// is_clone::<colmat_core::Par<'static>>();
/// ```
///
/// A phase that takes the parallelism consumes it:
///
/// ```compile_fail
/// fn phase(_: colmat_core::Par<'_>) {}
/// let par = colmat_core::Par::Seq;
/// phase(par);
/// phase(par);
/// ```
///
/// Concurrent children cannot both use the parent:
///
/// ```compile_fail
/// fn phase(_: colmat_core::Par<'_>) {}
/// let par = colmat_core::Par::Seq;
/// colmat_core::utils::thread::join_raw(|_| phase(par), |_| phase(par), colmat_core::Par::Seq);
/// ```
///
/// A nested child cannot also take its borrowed parent:
///
/// ```compile_fail,E0505
/// use colmat_core::Par;
///
/// fn nested_work(_child: Par<'_>, _parent: Par<'_>) {}
///
/// fn main() {
///     let mut parent = Par::Seq;
///     let child = parent.reborrow();
///     nested_work(child, parent);
/// }
/// ```
#[derive(Debug)]
pub enum Par<'team> {
	/// sequential, non portable across different platforms
	Seq,
	/// parallelized on the reserved members of a threadteam team. The value is the exclusive
	/// authority over them for as long as it lives; the number of participants is the size of the
	/// team. This is the only parallel route: see [`crate::utils::team::with_team`].
	#[cfg(feature = "threadteam")]
	Team(threadteam::TeamExec<'team>),
	/// owns the lifetime when the parallel route is not compiled (no `threadteam` feature): a
	/// build without it has no way to make a team, so this variant is never constructed
	#[cfg(not(feature = "threadteam"))]
	#[doc(hidden)]
	__Lifetime(
		core::convert::Infallible,
		core::marker::PhantomData<&'team ()>,
	),
}

/// the shape of a [`Par`]: which kind of parallelism and how many threads, without the
/// right to run anything. `Copy`, so workspace queries and thresholds can use it freely; it
/// cannot be passed to any function that executes work.
#[allow(clippy::large_enum_variant)] // public enum shape; boxing a variant would change the API
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ParShape {
	/// sequential
	Seq,
	/// a threadteam team of this many participants, with the dispatch costs of the pool it
	/// belongs to (see [`ParShape::dispatch_cost`])
	#[cfg(feature = "threadteam")]
	Team(NonZeroUsize, threadteam::DispatchCosts),
}

impl ParShape {
	/// sequential
	#[inline]
	pub const fn seq() -> Self {
		Self::Seq
	}

	/// `Team(nthreads)` with the host default dispatch costs, zero treated as one
	#[cfg(feature = "threadteam")]
	#[inline]
	pub fn team(nthreads: usize) -> Self {
		Self::Team(
			NonZeroUsize::new(nthreads.max(1)).unwrap(),
			threadteam::DispatchCosts::host_default(),
		)
	}

	/// the number of threads that should ideally execute an operation of this shape
	#[inline]
	pub fn degree(self) -> usize {
		match self {
			Self::Seq => 1,
			#[cfg(feature = "threadteam")]
			Self::Team(n, _) => n.get(),
		}
	}
}

impl Par<'_> {
	/// lends the parallelism for one phase. The parent is exclusively borrowed and unusable
	/// while the lent value lives, so a phase sequence never holds the authority twice.
	#[inline]
	pub fn reborrow(&mut self) -> Par<'_> {
		match self {
			Par::Seq => Par::Seq,
			#[cfg(feature = "threadteam")]
			Par::Team(team) => Par::Team(team.reborrow()),
			#[cfg(not(feature = "threadteam"))]
			Par::__Lifetime(x, _) => match *x {},
		}
	}

	/// the shape of this parallelism, without the right to execute with it
	#[inline]
	pub fn shape(&self) -> ParShape {
		match self {
			Par::Seq => ParShape::Seq,
			#[cfg(feature = "threadteam")]
			Par::Team(team) => ParShape::Team(team.size(), team.dispatch_costs()),
			#[cfg(not(feature = "threadteam"))]
			Par::__Lifetime(x, _) => match *x {},
		}
	}

	/// the number of threads that should ideally execute an operation with the
	/// given parallelism
	#[inline]
	pub fn degree(&self) -> usize {
		match self {
			Par::Seq => 1,
			#[cfg(feature = "threadteam")]
			Par::Team(team) => team.size().get(),
			#[cfg(not(feature = "threadteam"))]
			Par::__Lifetime(x, _) => match *x {},
		}
	}
}
