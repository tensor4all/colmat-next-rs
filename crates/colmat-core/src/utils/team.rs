//! The route of the parallel regions onto a threadteam team (feature `threadteam`).
//!
//! A `Par::Team(TeamExec)` is the exclusive authority over the reserved members of a team. The
//! functions here consume that authority (or lend it for the call), so the type system, not a
//! thread local, decides who may dispatch to the members:
//!
//! * flat sections ([`run`]): every participant runs a closure at once; the team is consumed;
//! * splits ([`for_each_child`]): each child is a `TeamExec` of its own, handed to the callback
//!   that runs on its leader, concurrently with its siblings;
//! * a team of one runs everything inline on the caller.
//!
//! Nothing here starts a Rayon job: the team members are threadteam members that wait while the
//! reservation is held, and the section dispatch is a write to their slots.
//!
//! # Provenance
//!
//! Ported from colmat-rs `faer/src/utils/team.rs` at `71337cdc` (lines 1-450: `with_team`,
//! the test gate, `stats`, `run`, `for_each`, `for_each_child`, `deal`). Changed: the dealing
//! splits a [`Producer`] of this crate instead of rayon's (debt K7); `end_of_job` (the
//! `vzeroupper` of the x86 assembly kernels), the x86 kernel adapter (`kernel`, `job_tables`,
//! `x86_kernel`) and the GEMM counters are left for the matrix product (core-3); `tile_grid`
//! and `grain_range` moved to [`crate::utils::partition`]; the counters are test-only. faer's
//! MIT notice: `LICENSE-faer`.

use super::thread::Producer;
use core::cell::UnsafeCell;
use threadteam::TeamExec;

/// Runs `f` with the exclusive authority over a team of `width` participants.
///
/// This is how a caller owns the parallelism: the team is created for the duration of the closure,
/// `Par::Team` carries the authority to the calls inside it, and the members are returned when the
/// closure ends. A caller with a thread budget wraps its work once, rather than handing a number to
/// every call:
///
/// ```
/// # #[cfg(not(miri))] { // threadteam's pool calls foreign functions Miri does not support
/// use colmat_core::Mat;
/// let src = Mat::<f64>::from_fn(64, 64, |i, j| (i + j) as f64);
/// let mut dst = Mat::<f64>::zeros(64, 64);
/// colmat_core::utils::team::with_team(2, |par| {
///     // every call in here runs its regions on 2 participants: the calling thread and
///     // 1 reserved worker
///     colmat_core::layout::copy_across_layouts(dst.as_mut(), src.transpose(), par);
/// });
/// assert!(dst == src.transpose());
/// # }
/// ```
///
/// `width` is clamped to at least one; a width of one is [`crate::Par::Seq`] and reserves nothing.
/// The team has `width` participants: the calling thread and `width - 1` reserved workers. The pool
/// is the host's own ([`threadteam::Pool::builder`], `width` threads): `into_team` clamps its width
/// to the pool's, so a pool of `width - 1` threads would hand out one participant fewer, and the
/// `width`-th worker parks unused (threadteam-rs#12). A caller that already owns a pool or a team
/// should borrow from it directly instead, because two pools do not share their members.
///
/// `with_team` is for a one-off call: every call builds a new pool and acquires its members again.
/// The threads themselves are not new: a pool with threadteam's default thread settings returns
/// its threads to a process-wide cache when it is dropped, and the next pool of the same width
/// takes them (threadteam 0.2), so repeated calls at one width reuse running threads instead of
/// spawning and joining them. A caller that runs parallel work repeatedly should still own the
/// [`threadteam::Pool`] and [`threadteam::Team`] itself, build them once, and lend a `Par` from
/// the team to each call:
///
/// ```
/// # #[cfg(not(miri))] { // as above
/// use colmat_core::{Mat, Par};
/// let src = Mat::<f64>::from_fn(64, 64, |i, j| (i + j) as f64);
/// let mut dst = Mat::<f64>::zeros(64, 64);
///
/// // once: 4 participants, the calling thread and 3 reserved workers. `into_team`
/// // fails when a member is not idle yet, which happens while a doctest elsewhere in this
/// // crate holds a team, so a pool built here waits for its own members.
/// let pool = threadteam::Pool::builder()
///     .threads(4)
///     .acquire_timeout(std::time::Duration::from_secs(30))
///     .build()
///     .unwrap();
/// let team = pool.into_team(4).ok().expect("an idle pool");
///
/// // per call: lend the team for the duration of the closure
/// for _ in 0..3 {
///     let mut exec = team.exec(4);
///     exec.team(|t| {
///         colmat_core::layout::copy_across_layouts(dst.as_mut(), src.transpose(), Par::Team(t));
///     })
///     .ok()
///     .expect("a team token for an idle team");
/// }
/// assert!(dst == src.transpose());
/// # }
/// ```
///
/// While the team lives, the calling thread is placed: `into_team` pins it to the one CPU no member
/// takes, for the whole scope (`threadteam` 0.5), wherever threadteam could place the team (pinning
/// on, Linux, a mask at least as wide as the team, a team wider than one); elsewhere nothing is
/// pinned. The mask comes back when the `Team` is dropped, or turned back into its pool on that same
/// thread.
///
/// # Panics
///
/// Panics if the team cannot be built or acquired: the closure's members wake asynchronously, so
/// the acquisition waits up to 30 s instead of threadteam's 2 ms default, which a loaded machine
/// can miss.
pub fn with_team<R>(width: usize, f: impl FnOnce(crate::Par<'_>) -> R) -> R {
	// the dispatch counters are process wide: in tests, one team at a time keeps a test that reads
	// them from seeing another test's regions
	#[cfg(test)]
	let _gate = team_turn();
	let width = width.max(1);
	if width == 1 {
		return f(crate::Par::Seq);
	}
	let pool = threadteam::Pool::builder()
		.threads(width)
		.acquire_timeout(core::time::Duration::from_secs(30))
		.build()
		.expect("a thread pool for the team");
	let team = pool
		.into_team(width)
		.expect("an idle pool: it was just created");
	let mut exec = team.exec(width);
	exec.team(|team| f(crate::Par::Team(team)))
		.expect("a team token for an idle team")
}

/// Records a region narrowed to a sub-team of `width` (tests only).
#[inline]
pub(crate) fn note_width(width: usize) {
	#[cfg(test)]
	stats::hist(&stats::WIDTH, width);
	let _ = width;
}

/// Records a split of the team into `parts` concurrent children (tests only).
#[inline]
pub(crate) fn note_split(parts: usize) {
	#[cfg(test)]
	stats::hist(&stats::SPLIT, parts);
	let _ = parts;
}

/// Runs `f(i, m)` on `m = min(n, team size)` participants at once; inline `f(0, 1)` when
/// `m == 1`. The team is consumed; members the section does not use stay parked.
pub fn run(team: TeamExec<'_>, n: usize, f: &(dyn Fn(usize, usize) + Sync)) {
	let size = team.size().get();
	let m = n.min(size);
	#[cfg(test)]
	stats::hist(&stats::RUN, m.max(1));
	if m <= 1 {
		return f(0, 1);
	}
	if m == size {
		team.run(f);
	} else {
		let _ = team.split(&[m], |_c, child| {
			child.run(f);
		});
	}
}

/// A flat item loop on the team: the items of `producer` are dealt out in contiguous static
/// blocks over `min(n_jobs, items, team size)` parts, each part processed by the leader of a
/// sub-team.
pub fn for_each<P: Producer>(
	team: TeamExec<'_>,
	n_jobs: usize,
	producer: P,
	f: impl Sync + Fn(P::Item),
) {
	for_each_child(team, n_jobs, producer, |item, _child| f(item))
}

/// [`for_each`] whose closure also receives the sub-team of its block, lent for the call.
///
/// The items are not collected: the producer is split on the leader into the blocks
/// ([`Producer::split_at`]), and each block is walked by the leader of its part, so the dispatch
/// allocates nothing whatever the team's width (colmat-rs#111).
pub fn for_each_child<P: Producer>(
	team: TeamExec<'_>,
	n_jobs: usize,
	producer: P,
	f: impl Sync + for<'c> Fn(P::Item, TeamExec<'c>),
) {
	let len = producer.len();
	if len == 0 {
		return;
	}
	let jobs = n_jobs.max(1).min(len).min(team.size().get());
	note_split(jobs);
	deal(team, producer, len, jobs, &f)
}

/// Blocks a single fork deals out of an inline array: up to this many, and only while the array
/// stays small (`FLAT_BYTES` per slot, 16 KiB in all), since a producer can carry state of any
/// size. Wider or heavier splits halve recursively instead, which costs a section per level
/// and allocates nothing either.
const FLAT_JOBS: usize = 64;
const FLAT_BYTES: usize = 256;

/// One producer piece, taken by one participant.
struct Piece<P>(UnsafeCell<Option<P>>);
// SAFETY: each piece is taken by exactly one participant (`Piece::take`'s contract), so the cell
// is never accessed from two threads; the piece itself moves to that participant, hence `P: Send`.
unsafe impl<P: Send> Sync for Piece<P> {}
impl<P> Piece<P> {
	/// # Safety
	/// Called once per piece, by the participant that owns it, with no other access to the
	/// piece running concurrently.
	unsafe fn take(&self) -> P {
		// SAFETY: exclusive access for this call, by the contract above.
		unsafe { (*self.0.get()).take() }.expect("each piece is taken once")
	}
}

/// Runs the `len` items of `producer` as `jobs` contiguous blocks (the boundaries of
/// `par_split_indices`), block `c` on the leader of the `c`-th part of `team`.
fn deal<P: Producer, F: Sync + for<'c> Fn(P::Item, TeamExec<'c>)>(
	mut team: TeamExec<'_>,
	producer: P,
	len: usize,
	jobs: usize,
	f: &F,
) {
	if jobs <= 1 {
		for item in producer.into_items() {
			f(item, team.reborrow());
		}
		return;
	}
	if jobs <= FLAT_JOBS && core::mem::size_of::<Option<P>>() <= FLAT_BYTES {
		let mut pieces: [Piece<P>; FLAT_JOBS] =
			core::array::from_fn(|_| Piece(UnsafeCell::new(None)));
		let mut rest = producer;
		for (c, piece) in pieces.iter_mut().enumerate().take(jobs - 1) {
			// split relative to what is left: the earlier blocks are already taken off
			let (_, n) = super::thread::par_split_indices(len, c, jobs);
			let (head, tail) = rest.split_at(n);
			*piece.0.get_mut() = Some(head);
			rest = tail;
		}
		*pieces[jobs - 1].0.get_mut() = Some(rest);
		team.fork(jobs, |c, mut child| {
			// SAFETY: the fork runs its callback once per child index `c < jobs`, so each
			// piece is taken once, by one thread
			let piece = unsafe { pieces[c].take() };
			for item in piece.into_items() {
				f(item, child.reborrow());
			}
		});
		return;
	}
	// two halves, each with the participants of its blocks
	let left = jobs / 2;
	let (left_len, _) = super::thread::par_split_indices(len, left, jobs);
	let size = team.size().get();
	let (left_size, _) = super::thread::par_split_indices(size, left, jobs);
	let (a, b) = producer.split_at(left_len);
	let halves = [
		Piece(UnsafeCell::new(Some(a))),
		Piece(UnsafeCell::new(Some(b))),
	];
	note_split(2);
	team.split(&[left_size, size - left_size], |c, child| {
		// SAFETY: the split runs its callback once per child index `c < 2`, so each half is
		// taken once, by one thread
		let half = unsafe { halves[c].take() };
		if c == 0 {
			deal(child, half, left_len, left, f)
		} else {
			deal(child, half, len - left_len, jobs - left, f)
		}
	})
	.expect("two non-empty parts within the team");
}

/// Serializes the tests that hold a team, so the process-wide dispatch counters belong to one
/// test at a time.
///
/// Reentrant on one thread: the outer acquisition takes the mutex, an inner one (a `with_team`
/// inside a `with_team`, or a test that takes the gate itself) only counts its depth, so a nested
/// scope cannot deadlock on a non-reentrant lock.
#[cfg(test)]
pub(crate) static TEAM_GATE: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The gate's guard: holds the mutex only at depth one on this thread.
#[cfg(test)]
pub(crate) struct TeamTurn(Option<std::sync::MutexGuard<'static, ()>>);
#[cfg(test)]
std::thread_local! {
	static TEAM_DEPTH: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}
#[cfg(test)]
impl Drop for TeamTurn {
	fn drop(&mut self) {
		TEAM_DEPTH.with(|d| d.set(d.get() - 1));
		drop(self.0.take());
	}
}
/// Takes the team turn for the current scope.
#[cfg(test)]
pub(crate) fn team_turn() -> TeamTurn {
	let nested = TEAM_DEPTH.with(|d| {
		d.set(d.get() + 1);
		d.get() > 1
	});
	if nested {
		TeamTurn(None)
	} else {
		TeamTurn(Some(TEAM_GATE.lock().unwrap_or_else(|e| e.into_inner())))
	}
}

/// Counters of how the team regions were dispatched, for tests that ask "did the parallel path
/// engage". Off by default (one relaxed load per region).
///
/// The counters are process wide, so a test that reads them must be the only one dispatching a
/// team while they are on: `TEAM_GATE` is what serializes that (see `with_team`).
#[cfg(test)]
pub(crate) mod stats {
	use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
	pub(super) static ON: AtomicBool = AtomicBool::new(false);
	const W: usize = 17;
	#[allow(clippy::declare_interior_mutable_const)]
	const Z: AtomicUsize = AtomicUsize::new(0);
	/// flat sections by participant count (index 1 = ran inline on the caller)
	pub(super) static RUN: [AtomicUsize; W] = [Z; W];
	/// splits and forks by number of parts (index 1 = ran inline on the caller)
	pub(super) static SPLIT: [AtomicUsize; W] = [Z; W];
	/// regions narrowed to a sub-team, by width (`thread::with_width`)
	pub(super) static WIDTH: [AtomicUsize; W] = [Z; W];
	pub(super) fn hist<const N: usize>(h: &[AtomicUsize; N], n: usize) {
		if ON.load(Relaxed) {
			h[n.min(N - 1)].fetch_add(1, Relaxed);
		}
	}
	/// Counters since the last `take`.
	#[derive(Debug, Clone, Default)]
	pub(crate) struct Stats {
		/// `run` sections, index = participants
		pub run: Vec<usize>,
		/// splits and forks, index = parts
		pub split: Vec<usize>,
		/// regions narrowed to a sub-team, index = its width
		pub width: Vec<usize>,
	}
	/// Enables or disables counting.
	pub(crate) fn enable(on: bool) {
		ON.store(on, Relaxed);
	}
	/// Returns and resets the counters.
	pub(crate) fn take() -> Stats {
		fn t<const N: usize>(h: &[AtomicUsize; N]) -> Vec<usize> {
			h.iter().map(|c| c.swap(0, Relaxed)).collect()
		}
		Stats {
			run: t(&RUN),
			split: t(&SPLIT),
			width: t(&WIDTH),
		}
	}
}
