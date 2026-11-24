//! The workspace contract (#20): once the destination, the team and the scratch are supplied,
//! the call allocates nothing, on any thread (design document §11, "Workspace checks").
//!
//! A counting global allocator counts the allocations of the participating threads: the test's
//! own thread and the members of the team, which the pool marks when it starts them. (Counting
//! every thread of the process is not reliable: the test harness's own thread allocates at
//! times.) Each check builds its inputs and its team first, then counts across the call only, at
//! first use and on repeated calls of growing shape.

use colmat_core::layout::copy_across_layouts;
use colmat_core::linalg::matmul::{matmul_scratch, matmul_with_scratch};
use colmat_core::perm::{
	Perm, permute_cols_in_place, permute_rows_in_place,
	permute_rows_in_place_scratch,
};
use colmat_core::utils::thread::for_each;
use colmat_core::workspace::{MemBuffer, MemStack, StackReq, temp_mat_scratch};
use colmat_core::{Accum, Conj, Mat, Par, c64};
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
	check_matmul::<f64>(par.reborrow(), |i| i as f64);
	check_matmul::<f32>(par.reborrow(), |i| i as f32);
	check_matmul::<c64>(par, |i| c64::new(i as f64, 1.0 - i as f64));
}

/// The paths of the matrix product (#23) that allocate nothing from the first call: the
/// driver's matrix-vector (both layouts) and rank-one kernels, and `nano-gemm` for small
/// products. The operands are laid out so that each shape takes the path it names.
fn check_matmul<T: colmat_core::kernels::KernelField + Copy>(
	mut par: Par<'_>,
	v: impl Fn(usize) -> T,
) {
	let degree = par.degree();
	// (m, n, k, lhs row-major): `nano-gemm`; column-major matrix-vector; row-major
	// matrix-vector (a row `dst` times a column-major `rhs`); rank one; and the column-major
	// matrix-vector product whose depth a team splits, with its partial sums in the scratch
	// (#28; larger than 8 MiB for every scalar here, not run under Miri)
	let shapes = [
		(17usize, 9usize, 13usize, false),
		(300, 1, 200, false),
		(300, 1, 200, true),
		(1, 300, 200, false),
		(300, 200, 1, false),
		(2100, 1, 1024, false),
	];
	for (m, n, k, lhs_row_major) in shapes {
		if cfg!(miri) && m * k > 1 << 20 {
			continue;
		}
		let a = Mat::from_fn(m, k, |i, j| v(i + 3 * j));
		let at = Mat::from_fn(k, m, |j, i| v(i + 3 * j));
		let b = Mat::from_fn(k, n, |i, j| v(2 * i + j));
		// a `1 x n` destination is the transpose of an `n x 1` column, so it is contiguous
		let mut c = Mat::from_fn(m, n, |i, j| v(i + j));
		let mut ct = Mat::from_fn(n, m, |j, i| v(i + j));
		let mut buf = MemBuffer::new(matmul_scratch::<T>(m, n, k, par.shape()));
		for accum in [Accum::Replace, Accum::Add] {
			let lhs = if lhs_row_major {
				at.as_ref().transpose()
			} else {
				a.as_ref()
			};
			let lhs = if m == 1 { at.as_ref().transpose() } else { lhs };
			let dst = if m == 1 {
				ct.as_mut().transpose_mut()
			} else {
				c.as_mut()
			};
			let n_alloc = allocations(|| {
				matmul_with_scratch(
					dst,
					accum,
					lhs,
					Conj::No,
					b.as_ref(),
					Conj::Yes,
					v(2),
					par.reborrow(),
					MemStack::new(&mut buf),
				)
			});
			assert_eq!(
				n_alloc,
				0,
				"matmul {} {m}x{n}x{k} {accum:?} on {degree}",
				core::any::type_name::<T>()
			);
		}
	}
}

/// The packed native products, `t4a-gemm` (and `t4a-private-gemm-x86` on x86-64). They keep
/// their packing memory in thread-locals, so the first call on a thread allocates and a product
/// whose pack exceeds L2 allocates on every call (tensor4all/t4a-gemm#21,
/// tensor4all/t4a-private-gemm-x86#14). `first_call` asks for nothing to be allocated even then;
/// otherwise each shape is warmed up once on every participant before it is counted.
fn check_native_matmul(mut par: Par<'_>, first_call: bool) {
	let degree = par.degree();
	// below L2 for the pack on every host this runs on (a 130 x 90 lhs block is 94 KB)
	let shapes = [(130usize, 70usize, 90usize), (64, 200, 64)];
	for (m, n, k) in shapes {
		let a = Mat::from_fn(m, k, |i, j| (i + 3 * j) as f64);
		let b = Mat::from_fn(k, n, |i, j| (2 * i + j) as f64);
		let mut c = Mat::<f64>::zeros(m, n);
		let mut buf =
			MemBuffer::new(matmul_scratch::<f64>(m, n, k, par.shape()));
		let mut run = |par: Par<'_>| {
			matmul_with_scratch(
				c.as_mut(),
				Accum::Replace,
				a.as_ref(),
				Conj::No,
				b.as_ref(),
				Conj::No,
				2.0,
				par,
				MemStack::new(&mut buf),
			)
		};
		if !first_call {
			run(par.reborrow());
		}
		let n_alloc = allocations(|| run(par.reborrow()));
		assert_eq!(
			n_alloc, 0,
			"native matmul f64 {m}x{n}x{k} on {degree}, first call: {first_call}"
		);
	}
}

/// Runs `f` on `Par::Seq` and on marked teams of 2 and 4, each on fresh threads (a new thread
/// for `Seq`, a new pool for a team), so that nothing a previous check warmed up is reused.
fn on_fresh_threads(f: fn(Par<'_>)) {
	let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
	std::thread::spawn(move || {
		PARTICIPANT.with(|p| p.set(true));
		f(Par::Seq)
	})
	.join()
	.unwrap();
	#[cfg(feature = "threadteam")]
	for width in [2usize, 4] {
		std::thread::spawn(move || {
			PARTICIPANT.with(|p| p.set(true));
			with_marked_team(width, f)
		})
		.join()
		.unwrap();
	}
}

/// The tests of this file count on shared counters: one at a time.
static TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
#[cfg_attr(
	target_arch = "x86_64",
	ignore = "t4a-private-gemm-x86 grows its packing memory per thread: tensor4all/t4a-private-gemm-x86#14"
)]
fn native_matmul_allocates_nothing_after_a_warm_up() {
	on_fresh_threads(|par| check_native_matmul(par, false));
}

#[test]
#[ignore = "the packed native kernels allocate on a thread's first call: tensor4all/t4a-gemm#21, tensor4all/t4a-private-gemm-x86#14"]
fn native_matmul_allocates_nothing_from_the_first_call() {
	on_fresh_threads(|par| check_native_matmul(par, true));
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
	let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
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
