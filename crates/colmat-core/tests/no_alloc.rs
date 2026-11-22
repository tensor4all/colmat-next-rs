//! The workspace contract (#20): once the destination, the team and the scratch are supplied,
//! the call allocates nothing, on any thread (design document §11, "Workspace checks").
//!
//! A counting global allocator counts the allocations of the participating threads: the test's
//! own thread and the members of the team, which the pool marks when it starts them. (Counting
//! every thread of the process is not reliable: the test harness's own thread allocates at
//! times.) Each check builds its inputs and its team first, then counts across the call only, at
//! first use and on repeated calls of growing shape.

use colmat_core::layout::copy_across_layouts;
use colmat_core::perm::{
	Perm, permute_cols_in_place, permute_rows_in_place,
	permute_rows_in_place_scratch,
};
use colmat_core::utils::thread::for_each;
use colmat_core::workspace::{MemBuffer, MemStack, StackReq, temp_mat_scratch};
use colmat_core::{Mat, Par};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct Counting;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicUsize = AtomicUsize::new(0);

std::thread_local! {
	/// Whether this thread takes part in the counted calls. Const-initialised and without drop
	/// glue, so reading it from the allocator neither allocates nor fails during thread exit.
	static PARTICIPANT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn counts_here() -> bool {
	COUNTING.load(Ordering::Relaxed)
		&& PARTICIPANT.try_with(|p| p.get()).unwrap_or(false)
}

// SAFETY: every method forwards to `System` with the caller's arguments unchanged; the counter
// is a side effect that touches no memory the allocator hands out.
unsafe impl GlobalAlloc for Counting {
	unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
		if counts_here() {
			ALLOCS.fetch_add(1, Ordering::Relaxed);
		}
		// SAFETY: forwarded under the caller's contract.
		unsafe { System.alloc(layout) }
	}

	unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
		// SAFETY: forwarded under the caller's contract.
		unsafe { System.dealloc(ptr, layout) }
	}

	unsafe fn realloc(
		&self,
		ptr: *mut u8,
		layout: Layout,
		new_size: usize,
	) -> *mut u8 {
		if counts_here() {
			ALLOCS.fetch_add(1, Ordering::Relaxed);
		}
		// SAFETY: forwarded under the caller's contract.
		unsafe { System.realloc(ptr, layout, new_size) }
	}
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// The allocations, on every thread, while `f` runs.
fn allocations(f: impl FnOnce()) -> usize {
	ALLOCS.store(0, Ordering::Relaxed);
	COUNTING.store(true, Ordering::SeqCst);
	f();
	COUNTING.store(false, Ordering::SeqCst);
	ALLOCS.load(Ordering::Relaxed)
}

fn perm(n: usize) -> Perm<usize> {
	let fwd: Vec<usize> = (0..n).rev().collect();
	let inv = fwd.clone();
	Perm::new_checked(fwd.into_boxed_slice(), inv.into_boxed_slice(), n)
}

fn check(mut par: Par<'_>) {
	let degree = par.degree();
	// shapes in growing order: first use, then larger ones on the same buffer
	let shapes = [(17usize, 9usize), (64, 64), (130, 70), (130, 70)];
	let (mmax, nmax) = (130, 70);
	let req = StackReq::any_of(&[
		permute_rows_in_place_scratch::<usize, f64>(mmax, nmax),
		temp_mat_scratch::<f64>(mmax, nmax),
	]);
	let mut buf = MemBuffer::new(req);
	for (m, n) in shapes {
		let src = Mat::from_fn(m, n, |i, j| (i * 1000 + j) as f64);
		let mut dst = Mat::<f64>::zeros(n, m);
		let mut a = src.clone();
		let (pm, pn) = (perm(m), perm(n));

		let k = allocations(|| {
			copy_across_layouts(
				dst.as_mut().transpose_mut(),
				src.as_ref(),
				par.reborrow(),
			)
		});
		assert_eq!(k, 0, "copy {m}x{n} on {degree}");
		assert!(dst.transpose() == src.as_ref());

		let k = allocations(|| {
			let chunks = a.as_mut().par_col_chunks_mut(8);
			for_each(par.reborrow(), degree, chunks, |mut c| {
				for j in 0..c.ncols() {
					for i in 0..c.nrows() {
						c[(i, j)] *= 2.0;
					}
				}
			})
		});
		assert_eq!(k, 0, "chunks {m}x{n} on {degree}");

		let k = allocations(|| {
			permute_rows_in_place(
				a.as_mut(),
				pm.as_ref(),
				MemStack::new(&mut buf),
			);
			permute_cols_in_place(
				a.as_mut(),
				pn.as_ref(),
				MemStack::new(&mut buf),
			);
		});
		assert_eq!(k, 0, "permutations {m}x{n}");
		assert!(a[(0, 0)] == 2.0 * src[(m - 1, n - 1)]);
	}
}

/// Runs `f` on a team of `width` participants whose members are marked as participating.
#[cfg(feature = "threadteam")]
fn with_marked_team(width: usize, f: impl FnOnce(Par<'_>)) {
	let pool = threadteam::Pool::builder()
		.threads(width)
		.on_start(|_| PARTICIPANT.with(|p| p.set(true)))
		.acquire_timeout(std::time::Duration::from_secs(30))
		.build()
		.expect("a pool");
	let team = pool.into_team(width).expect("an idle pool");
	let mut exec = team.exec(width);
	exec.team(|mut team| {
		// A member allocates three times (48 bytes and less) the first time it goes idle after
		// the pool starts it: threadteam's own one-time set-up, measured on threadteam 0.6.0,
		// not something a call does. Let every member pass that point before counting.
		team.reborrow().run(|_, _| {});
		std::thread::sleep(std::time::Duration::from_millis(50));
		f(Par::Team(team))
	})
	.expect("a team token");
}

#[test]
fn supplied_destination_team_and_scratch_allocate_nothing() {
	PARTICIPANT.with(|p| p.set(true));
	// the counter sees an allocation on this thread
	assert_eq!(allocations(|| drop(std::hint::black_box(vec![1u8; 10]))), 1);
	check(Par::Seq);
	#[cfg(feature = "threadteam")]
	for width in [2usize, 4] {
		with_marked_team(width, |mut par| {
			assert_eq!(par.degree(), width);
			// the counter sees an allocation on every member
			let Par::Team(team) = par.reborrow() else {
				unreachable!()
			};
			let k = allocations(|| {
				team.run(|_, _| drop(std::hint::black_box(vec![1u8; 10])))
			});
			assert_eq!(k, width);
			check(par)
		});
	}
}
