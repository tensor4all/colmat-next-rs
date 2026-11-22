//! Tests of the execution layer (#20): the producers, the team skeletons and the workspace.
//!
//! The team tests are ported from colmat-rs `faer/src/utils/team.rs` at `71337cdc`
//! (`with_team_has_width_participants`, `with_team_below_two_is_seq`,
//! `par_width_takes_the_widest_width_that_pays`, `with_width_runs_on_a_sub_team_of_that_width`,
//! `for_each_child_deals_every_item_once`, `for_each_child_unwinds_and_keeps_the_team`,
//! `par_pays_time_is_the_shape_gate`, `for_each_deals_every_item_once_and_nests`,
//! `join_raw_leaves_match_the_width`, `join_sized_on_a_team_hands_out_exactly_the_team`,
//! `a_team_of_one_behaves_exactly_like_seq`, `with_team_calls_of_one_width_reuse_the_threads`)
//! with the rayon iterators replaced by producers of this crate and the matrix product parts
//! left out. faer's MIT notice: `LICENSE-faer`.
//!
//! Under Miri only the tests without a pool run: threadteam's pool calls foreign functions
//! (thread placement and QoS) that Miri does not support. `Exec::serial` (a team of one) does
//! run there.

use crate::internal_prelude::*;
use crate::utils::thread::{Producer, Zip};
#[cfg(feature = "threadteam")]
use core::sync::atomic::{AtomicUsize, Ordering};
use std::assert;

#[cfg(feature = "threadteam")]
/// A producer of `0..len` whose items carry a payload, to exercise the dealing of a producer
/// heavier than `FLAT_BYTES` and of items that own a resource.
struct Mapped<F, const N: usize> {
	range: core::ops::Range<usize>,
	_payload: [u8; N],
	f: F,
}

#[cfg(feature = "threadteam")]
impl<R, F: Clone + Send + Fn(usize) -> R, const N: usize> Producer
	for Mapped<F, N>
{
	type IntoIter = core::iter::Map<core::ops::Range<usize>, F>;
	type Item = R;

	fn len(&self) -> usize {
		Producer::len(&self.range)
	}

	fn split_at(self, index: usize) -> (Self, Self) {
		let (a, b) = self.range.split_at(index);
		(
			Self {
				range: a,
				_payload: self._payload,
				f: self.f.clone(),
			},
			Self {
				range: b,
				_payload: self._payload,
				f: self.f,
			},
		)
	}

	fn into_items(self) -> Self::IntoIter {
		self.range.map(self.f)
	}
}

#[cfg(feature = "threadteam")]
fn mapped<R, F: Clone + Send + Fn(usize) -> R, const N: usize>(
	len: usize,
	f: F,
) -> Mapped<F, N> {
	Mapped {
		range: 0..len,
		_payload: [0; N],
		f,
	}
}

#[test]
fn ranges_split_anywhere_and_walk_in_order() {
	let r = 3..10usize;
	assert!(Producer::len(&r) == 7);
	for k in 0..=7 {
		let (a, b) = (3..10usize).split_at(k);
		assert!(a == (3..3 + k));
		assert!(b == (3 + k..10));
	}
	let items: Vec<_> = (2..5usize).zip_eq(10..13usize).into_items().collect();
	assert!(items == [(2, 10), (3, 11), (4, 12)]);
}

#[test]
#[should_panic]
fn zip_eq_rejects_different_lengths() {
	let _: Zip<_, _> = (0..3usize).zip_eq(0..4usize);
}

/// The chunks of a view: the same boundaries as colmat-rs's `par_*_chunks`, whatever the split.
#[test]
fn chunks_split_on_chunk_boundaries() {
	let a = Mat::from_fn(7, 10, |i, j| (10 * i + j) as f64);
	for size in 1..=11 {
		let p = a.as_ref().par_col_chunks(size);
		assert!(p.len() == 10usize.div_ceil(size));
		for k in 0..=p.len() {
			let (l, r) = a.as_ref().par_col_chunks(size).split_at(k);
			let mut col = 0;
			for c in l.into_items().chain(r.into_items()) {
				assert!(c.nrows() == 7 && c.ncols() >= 1 && c.ncols() <= size);
				for j in 0..c.ncols() {
					for i in 0..7 {
						assert!(c[(i, j)] == a[(i, col + j)]);
					}
				}
				col += c.ncols();
			}
			assert!(col == 10);
		}
		let p = a.as_ref().par_row_chunks(size);
		assert!(p.len() == 7usize.div_ceil(size));
		let rows: usize = p.into_items().map(|c| c.nrows()).sum();
		assert!(rows == 7);
	}
	// mutable chunks write disjoint parts that cover the matrix
	let mut b = Mat::<f64>::zeros(7, 10);
	let (l, r) = b.as_mut().par_col_chunks_mut(3).split_at(2);
	for (k, mut c) in l.into_items().chain(r.into_items()).enumerate() {
		for j in 0..c.ncols() {
			for i in 0..c.nrows() {
				c[(i, j)] += (k + 1) as f64;
			}
		}
	}
	for j in 0..10 {
		for i in 0..7 {
			assert!(b[(i, j)] == (j / 3 + 1) as f64);
		}
	}
	let mut b = Mat::<f64>::zeros(7, 10);
	for (k, mut c) in b.as_mut().par_row_chunks_mut(2).into_items().enumerate()
	{
		for j in 0..c.ncols() {
			for i in 0..c.nrows() {
				c[(i, j)] += (k + 1) as f64;
			}
		}
	}
	for j in 0..10 {
		for i in 0..7 {
			assert!(b[(i, j)] == (i / 2 + 1) as f64);
		}
	}
	// empty views give no chunk
	let e = Mat::<f64>::zeros(0, 5);
	assert!(e.as_ref().par_row_chunks(2).len() == 0);
	assert!(e.as_ref().par_col_chunks(2).len() == 3);
}

/// The skeletons with `Par::Seq` run everything on the caller, in order.
#[test]
fn sequential_skeletons_run_in_order_on_the_caller() {
	use crate::utils::thread::*;
	let me = std::thread::current().id();
	let seen = std::sync::Mutex::new(Vec::new());
	for_each(Par::Seq, 4, 0..6usize, |i| {
		assert!(std::thread::current().id() == me);
		seen.lock().unwrap().push(i);
	});
	assert!(*seen.lock().unwrap() == [0, 1, 2, 3, 4, 5]);
	let seen = std::sync::Mutex::new(Vec::new());
	join_raw(
		|p| seen.lock().unwrap().push((0, p.degree())),
		|p| seen.lock().unwrap().push((1, p.degree())),
		Par::Seq,
	);
	join_sized(
		Par::Seq,
		1,
		|p| seen.lock().unwrap().push((2, p.degree())),
		|p| seen.lock().unwrap().push((3, p.degree())),
	);
	split(Par::Seq, &[2, 1], &|c, p| {
		seen.lock().unwrap().push((4 + c, p.degree()))
	});
	assert!(
		*seen.lock().unwrap()
			== [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]
	);
	assert!(with_width(Par::Seq, 8, |p| p.degree()) == 1);
	for (n, chunks) in [(10usize, 3usize), (3, 5), (0, 2), (7, 7)] {
		let mut next = 0;
		for c in 0..chunks {
			let (start, len) = par_split_indices(n, c, chunks);
			assert!(start == next);
			next += len;
		}
		assert!(next == n);
	}
}

#[cfg(feature = "threadteam")]
mod team {
	use super::*;
	use crate::utils::team::{for_each_child, stats, team_turn, with_team};
	use core::time::Duration;
	use threadteam::{Pool, TeamExec};

	fn team_of(width: usize) -> threadteam::Team {
		Pool::builder()
			.threads(width)
			.acquire_timeout(Duration::from_secs(30))
			.build()
			.unwrap()
			.into_team(width)
			.expect("an idle pool")
	}

	/// Runs `f` with the authority over a team of `width` participants.
	fn with_par<R>(width: usize, f: impl FnOnce(Par<'_>) -> R) -> R {
		let _turn = team_turn();
		let team = team_of(width);
		let mut exec = team.exec(width);
		exec.team(|team| f(Par::Team(team))).expect("a team token")
	}

	/// A team whose dispatch cost row has the given anchors.
	fn with_row<R>(
		width: usize,
		anchors: &[(usize, Duration)],
		f: impl FnOnce(Par<'_>) -> R,
	) -> R {
		use threadteam::{DispatchCosts, DispatchKind, WidthCosts};
		let _turn = team_turn();
		let costs = DispatchCosts::for_class(threadteam::HostClass::Generic)
			.with(
				DispatchKind::Team,
				WidthCosts::from_anchors(anchors).expect("a valid row"),
			);
		let team = Pool::builder()
			.threads(width)
			.acquire_timeout(Duration::from_secs(30))
			.dispatch_costs(costs)
			.build()
			.unwrap()
			.into_team(width)
			.expect("an idle pool");
		let mut exec = team.exec(width);
		exec.team(|team| f(Par::Team(team))).expect("a team token")
	}

	/// `with_team(width)` hands out `width` participants: the calling thread and `width - 1`
	/// reserved workers.
	#[test]
	#[cfg_attr(
		miri,
		ignore = "threadteam's pool calls foreign functions Miri does not support"
	)]
	fn with_team_has_width_participants() {
		for width in [2usize, 3, 4] {
			let size = with_team(width, |par| match par {
				Par::Team(t) => t.size().get(),
				_ => 1,
			});
			assert!(size == width);
		}
	}

	/// A width of zero or one is `Par::Seq`, on the calling thread.
	#[test]
	fn with_team_below_two_is_seq() {
		let me = std::thread::current().id();
		for width in [0usize, 1] {
			let (seq, on_caller) = with_team(width, |par| {
				(matches!(par, Par::Seq), std::thread::current().id() == me)
			});
			assert!(seq, "width {width} is not Par::Seq");
			assert!(on_caller, "width {width} left the calling thread");
		}
	}

	/// The width of a region is the widest that pays, every width considered.
	#[test]
	#[cfg_attr(
		miri,
		ignore = "threadteam's pool calls foreign functions Miri does not support"
	)]
	fn par_width_takes_the_widest_width_that_pays() {
		use crate::utils::cost::{par_width_time, par_width_time_upto};
		let us = Duration::from_micros;
		// 1 us at 2 and 4, 100 us at 8: 6 us of work pays for one section at 4 (3 * 2 us) but not
		// at 5 to 8
		with_row(8, &[(2, us(1)), (4, us(2)), (8, us(100))], |par| {
			assert!(par_width_time(&par, us(6), 1) == 4);
			// 5 us pays at 3 (interpolated 1.5 us) but not at 4
			assert!(par_width_time(&par, us(5), 1) == 3);
			assert!(par_width_time(&par, us(2), 1) == 1);
			assert!(par_width_time(&par, us(2000), 1) == 8);
		});
		// a falling row: 3 us pays at 8 only, 9 us at 2 inside a cap of 5
		with_row(
			8,
			&[(2, us(3)), (4, us(9)), (6, us(1)), (8, us(1))],
			|par| {
				assert!(par_width_time(&par, us(3), 1) == 8);
				assert!(par_width_time_upto(&par, us(3), 1, 5) == 1);
				assert!(par_width_time_upto(&par, us(9), 1, 5) == 2);
			},
		);
		assert!(par_width_time(&Par::Seq, us(1_000_000), 1) == 1);
	}

	/// `with_width` hands the region a sub-team of exactly that width, the whole token at its
	/// degree or more, and `Seq` below two, and the parent keeps its width afterwards.
	#[test]
	#[cfg_attr(
		miri,
		ignore = "threadteam's pool calls foreign functions Miri does not support"
	)]
	fn with_width_runs_on_a_sub_team_of_that_width() {
		use crate::utils::thread::with_width;
		with_par(8, |mut par| {
			stats::enable(true);
			let _ = stats::take();
			for (w, want) in [(3, 3), (8, 8), (9, 8), (1, 1), (0, 1)] {
				let seen = AtomicUsize::new(0);
				let degree = with_width(par.reborrow(), w, |par| {
					let degree = par.degree();
					if let Par::Team(team) = par {
						team.run(|_, _| {
							seen.fetch_add(1, Ordering::Relaxed);
						});
					}
					degree
				});
				assert!(degree == want, "width {w}");
				if want > 1 {
					assert!(seen.load(Ordering::Relaxed) == want, "width {w}");
				}
			}
			assert!(par.degree() == 8);
			stats::enable(false);
			let s = stats::take();
			// only the narrowing to 3 is a sub-team; 8 and 9 are the token itself
			assert!(s.width[3] == 1, "{s:?}");
			assert!(s.width.iter().sum::<usize>() == 1, "{s:?}");
		});
	}

	/// Every item is dealt exactly once, on a child whose width is the team's share of its
	/// block, for flat and recursive dealing and for a producer heavier than the flat array
	/// allows.
	#[test]
	#[cfg_attr(
		miri,
		ignore = "threadteam's pool calls foreign functions Miri does not support"
	)]
	fn for_each_child_deals_every_item_once() {
		for (width, len, n_jobs, big) in [
			(8usize, 1000usize, 7usize, false),
			(8, 1000, 7, true),
			(70, 1000, 70, false),
			(70, 333, 69, true),
			(5, 3, 5, false),
		] {
			with_par(width, |par| {
				let Par::Team(team) = par else { unreachable!() };
				let seen: Vec<AtomicUsize> =
					(0..len).map(|_| AtomicUsize::new(0)).collect();
				let widths: Vec<AtomicUsize> =
					(0..len).map(|_| AtomicUsize::new(0)).collect();
				let f = |i: usize, child: TeamExec<'_>| {
					seen[i].fetch_add(1, Ordering::Relaxed);
					widths[i].store(child.size().get(), Ordering::Relaxed);
				};
				if big {
					for_each_child(
						team,
						n_jobs,
						mapped::<_, _, 512>(len, |i| i),
						f,
					);
				} else {
					for_each_child(team, n_jobs, 0..len, f);
				}
				let jobs = n_jobs.min(len).min(width);
				let (lo, hi) = (width / jobs, width.div_ceil(jobs));
				for i in 0..len {
					assert!(
						seen[i].load(Ordering::Relaxed) == 1,
						"{width} {len} {n_jobs} {big}: item {i}"
					);
					let w = widths[i].load(Ordering::Relaxed);
					assert!(
						lo <= w && w <= hi,
						"{width} {len} {n_jobs} {big}: item {i} on a child of {w}"
					);
				}
			});
		}
	}

	/// A panic in one block reaches the caller after every block has finished, the items are
	/// dropped once each, and the team stays usable.
	#[test]
	#[cfg_attr(
		miri,
		ignore = "threadteam's pool calls foreign functions Miri does not support"
	)]
	fn for_each_child_unwinds_and_keeps_the_team() {
		use std::panic::{AssertUnwindSafe, catch_unwind};
		struct Counted<'a>(&'a AtomicUsize);
		impl Drop for Counted<'_> {
			fn drop(&mut self) {
				self.0.fetch_add(1, Ordering::Relaxed);
			}
		}
		for width in [8usize, 70] {
			with_par(width, |mut par| {
				let dropped = AtomicUsize::new(0);
				let made = AtomicUsize::new(0);
				let r = catch_unwind(AssertUnwindSafe(|| {
					let Par::Team(team) = par.reborrow() else {
						unreachable!()
					};
					for_each_child(
						team,
						width,
						mapped::<_, _, 0>(1000, |i| {
							made.fetch_add(1, Ordering::Relaxed);
							(i, Counted(&dropped))
						}),
						|(i, _c), _| {
							if i == 500 {
								panic!("block failure");
							}
						},
					);
				}));
				assert!(r.is_err());
				assert!(
					dropped.load(Ordering::Relaxed)
						== made.load(Ordering::Relaxed)
				);
				// the team still works
				let Par::Team(team) = par.reborrow() else {
					unreachable!()
				};
				let hits = AtomicUsize::new(0);
				for_each_child(team, width, 0..100usize, |_, _| {
					hits.fetch_add(1, Ordering::Relaxed);
				});
				assert!(hits.load(Ordering::Relaxed) == 100);
			});
		}
	}

	/// `par_pays_time` is `ParShape::pays` without the shape.
	#[test]
	#[cfg_attr(
		miri,
		ignore = "threadteam's pool calls foreign functions Miri does not support"
	)]
	fn par_pays_time_is_the_shape_gate() {
		use crate::utils::cost::par_pays_time;
		with_par(6, |par| {
			let shape = par.shape();
			for &us in &[0u64, 1, 3, 5, 8, 12, 30, 1000] {
				for sections in 1..=3 {
					for width in 1..=8usize {
						let est = Duration::from_micros(us);
						let w =
							core::num::NonZeroUsize::new(width.min(6)).unwrap();
						let want = width >= 2 && shape.pays(est, sections, w);
						assert!(
							par_pays_time(&par, est, sections, width) == want,
							"{us} us, {sections}, {width}"
						);
					}
				}
			}
		});
		assert!(!par_pays_time(&Par::Seq, Duration::from_secs(1), 1, 4));
	}

	#[test]
	#[cfg_attr(
		miri,
		ignore = "threadteam's pool calls foreign functions Miri does not support"
	)]
	fn for_each_deals_every_item_once_and_nests() {
		with_par(6, |mut par| {
			for jobs in [1usize, 2, 3, 6, 9] {
				let hits: Vec<AtomicUsize> =
					(0..13).map(|_| AtomicUsize::new(0)).collect();
				crate::utils::thread::for_each_with_par(
					par.reborrow(),
					jobs,
					0..13usize,
					|i, child| {
						hits[i].fetch_add(1, Ordering::Relaxed);
						// a job receives a share of the team no wider than the team
						assert!(child.degree() <= 6);
					},
				);
				assert!(
					hits.iter().all(|h| h.load(Ordering::Relaxed) == 1),
					"jobs = {jobs}"
				);
			}
		});
	}

	#[test]
	#[cfg_attr(
		miri,
		ignore = "threadteam's pool calls foreign functions Miri does not support"
	)]
	fn join_raw_leaves_match_the_width() {
		for n in [2usize, 3, 5, 6] {
			let widths = std::sync::Mutex::new(Vec::new());
			with_par(n, |par| {
				crate::utils::thread::join_raw(
					|p| widths.lock().unwrap().push(p.degree()),
					|p| widths.lock().unwrap().push(p.degree()),
					par,
				)
			});
			let mut w = widths.into_inner().unwrap();
			w.sort_unstable();
			let half = n / 2;
			assert!(w == vec![half, n - half], "n = {n}");
		}
	}

	#[test]
	#[cfg_attr(
		miri,
		ignore = "threadteam's pool calls foreign functions Miri does not support"
	)]
	fn join_sized_on_a_team_hands_out_exactly_the_team() {
		for left in 0..=7usize {
			let active = AtomicUsize::new(0);
			let peak = AtomicUsize::new(0);
			let widths = [AtomicUsize::new(0), AtomicUsize::new(0)];
			with_par(6, |par| {
				let side = |which: usize, par: Par<'_>| {
					let jobs = par.degree();
					widths[which].store(jobs, Ordering::Relaxed);
					crate::utils::thread::for_each_with_par(
						par,
						jobs,
						0..jobs,
						|_, _| {
							let now = active.fetch_add(1, Ordering::AcqRel) + 1;
							peak.fetch_max(now, Ordering::AcqRel);
							std::thread::sleep(Duration::from_millis(3));
							active.fetch_sub(1, Ordering::AcqRel);
						},
					);
				};
				crate::utils::thread::join_sized(
					par,
					left,
					|p| side(0, p),
					|p| side(1, p),
				);
			});
			assert!(
				widths[0].load(Ordering::Relaxed)
					+ widths[1].load(Ordering::Relaxed)
					== 6,
				"left = {left}"
			);
			assert!(peak.load(Ordering::Relaxed) <= 6, "left = {left}");
		}
	}

	/// A split asking for more participants than the team has runs its parts in turn, each
	/// sequential.
	#[test]
	#[cfg_attr(
		miri,
		ignore = "threadteam's pool calls foreign functions Miri does not support"
	)]
	fn an_oversized_split_runs_the_parts_in_turn() {
		with_par(3, |par| {
			let seen = std::sync::Mutex::new(Vec::new());
			crate::utils::thread::split(par, &[2, 2], &|c, p| {
				seen.lock().unwrap().push((c, p.degree()))
			});
			assert!(*seen.lock().unwrap() == [(0, 1), (1, 1)]);
		});
	}

	#[test]
	fn a_team_of_one_behaves_exactly_like_seq() {
		use crate::utils::cost::{ParThreshold, join_pays, par_width, rate};
		let _turn = team_turn();
		threadteam::Exec::serial()
			.team(|team| {
				let par = Par::Team(team);
				assert!(par.degree() == 1);
				// no gate pays on a single participant, however large the work
				assert!(
					par_width::<f64>(&par, 1 << 40, rate::gemm_mac(), 1) == 1
				);
				assert!(!join_pays::<f64>(&par, 1 << 40, rate::gemm_mac(), 1));
				assert!(
					ParThreshold::Derived.region_width::<f64>(
						&par,
						1 << 40,
						rate::gemm_mac(),
						1
					) == 1
				);
				// and a copy is the sequential one, with no team section at all
				let src = Mat::from_fn(40, 33, |i, j| (i * 100 + j) as f64);
				let mut dst = Mat::<f64>::zeros(33, 40);
				stats::enable(true);
				let _ = stats::take();
				crate::layout::copy_across_layouts(
					dst.as_mut().transpose_mut(),
					src.as_ref(),
					par,
				);
				let st = stats::take();
				stats::enable(false);
				assert!(dst.transpose() == src.as_ref());
				assert!(st.split.iter().skip(2).sum::<usize>() == 0, "{st:?}");
				assert!(st.run.iter().skip(2).sum::<usize>() == 0, "{st:?}");
			})
			.unwrap();
	}

	/// The parallel copy opens one section, of the participants that get a tile each.
	#[test]
	#[cfg_attr(
		miri,
		ignore = "threadteam's pool calls foreign functions Miri does not support"
	)]
	fn the_parallel_copy_uses_one_section_of_the_team() {
		with_par(4, |mut par| {
			let src = Mat::from_fn(64, 100, |i, j| (i * 1000 + j) as f64);
			for (m, parts) in [(64usize, 4usize), (40, 3), (16, 1)] {
				let src = src.as_ref().subrows(0, m);
				let mut dst = Mat::<f64>::zeros(100, m);
				stats::enable(true);
				let _ = stats::take();
				crate::layout::copy_across_layouts(
					dst.as_mut().transpose_mut(),
					src,
					par.reborrow(),
				);
				let st = stats::take();
				stats::enable(false);
				assert!(dst.transpose() == src);
				// the destination is row-major: `m` rows in tiles of 16
				for (k, &count) in st.split.iter().enumerate() {
					let want = usize::from(k == parts && parts > 1);
					assert!(count == want, "m = {m}: {st:?}");
				}
			}
		});
	}

	#[test]
	#[cfg_attr(
		miri,
		ignore = "threadteam's pool calls foreign functions Miri does not support"
	)]
	fn with_team_calls_of_one_width_reuse_the_threads() {
		use std::collections::HashSet;
		use std::sync::Mutex;
		let _turn = team_turn();
		let me = std::thread::current().id();
		// the threads of the members, without the calling thread
		let workers = || {
			with_team(5, |par| {
				let Par::Team(mut team) = par else {
					panic!("a width of 5 is a team")
				};
				let ids = Mutex::new(HashSet::new());
				team.reborrow().run(|_, _| {
					ids.lock().unwrap().insert(std::thread::current().id());
				});
				let mut ids = ids.into_inner().unwrap();
				ids.remove(&me);
				ids
			})
		};
		let first = workers();
		let second = workers();
		assert!((first.len(), second.len()) == (4, 4));
		let all = first.union(&second).count();
		assert!(
			all <= 5,
			"two calls ran on {all} threads: the second pool spawned new ones"
		);
	}
}

mod workspace {
	use super::*;
	use crate::perm::{
		Perm, permute_cols, permute_cols_in_place,
		permute_cols_in_place_scratch, permute_rows, permute_rows_in_place,
		permute_rows_in_place_scratch,
	};
	use crate::workspace::{
		MemBuffer, MemStack, StackReq, temp_mat_scratch, temp_mat_uninit,
		temp_mat_zeroed,
	};

	fn perm(n: usize) -> Perm<usize> {
		// a fixed permutation with no fixed point for `n > 1`
		let fwd: Vec<usize> = (0..n).map(|i| (i * 3 + 1) % n).collect();
		let fwd = if n > 1
			&& (0..n).all(|i| fwd.iter().filter(|&&x| x == i).count() == 1)
		{
			fwd
		} else {
			(0..n).rev().collect()
		};
		let mut inv = vec![0; n];
		for (i, &f) in fwd.iter().enumerate() {
			inv[f] = i;
		}
		Perm::new_checked(fwd.into_boxed_slice(), inv.into_boxed_slice(), n)
	}

	/// The in-place permutations equal the out-of-place ones, with a buffer of exactly the
	/// queried size, in both layouts.
	#[test]
	fn in_place_permutations_match_and_need_only_the_queried_scratch() {
		for (m, n) in [(0usize, 3usize), (1, 1), (5, 4), (9, 17), (33, 2)] {
			for transposed in [false, true] {
				let a = Mat::from_fn(m, n, |i, j| (i * 100 + j) as f64);
				let mut want_r = Mat::<f64>::zeros(m, n);
				permute_rows(want_r.as_mut(), a.as_ref(), perm(m).as_ref());
				let mut want_c = Mat::<f64>::zeros(m, n);
				permute_cols(want_c.as_mut(), a.as_ref(), perm(n).as_ref());

				let mut store = Mat::<f64>::zeros(
					if transposed { n } else { m },
					if transposed { m } else { n },
				);
				let mut view = if transposed {
					store.as_mut().transpose_mut()
				} else {
					store.as_mut()
				};
				view.copy_from(a.as_ref());
				let mut buf = MemBuffer::new(permute_rows_in_place_scratch::<
					usize,
					f64,
				>(m, n));
				permute_rows_in_place(
					view.rb_mut(),
					perm(m).as_ref(),
					MemStack::new(&mut buf),
				);
				assert!(
					view.rb() == want_r.as_ref(),
					"rows {m}x{n} {transposed}"
				);

				view.copy_from(a.as_ref());
				let mut buf = MemBuffer::new(permute_cols_in_place_scratch::<
					usize,
					f64,
				>(m, n));
				permute_cols_in_place(
					view.rb_mut(),
					perm(n).as_ref(),
					MemStack::new(&mut buf),
				);
				assert!(
					view.rb() == want_c.as_ref(),
					"cols {m}x{n} {transposed}"
				);
			}
		}
	}

	/// A stack too small for the requirement is refused before the matrix is changed.
	#[test]
	fn a_short_stack_panics_before_the_output_changes() {
		use std::panic::{AssertUnwindSafe, catch_unwind};
		let a = Mat::from_fn(8, 8, |i, j| (i * 10 + j) as f64);
		let mut b = a.clone();
		let req = permute_rows_in_place_scratch::<usize, f64>(8, 8);
		let mut buf = MemBuffer::new(StackReq::new::<u8>(req.size_bytes() / 2));
		let r = catch_unwind(AssertUnwindSafe(|| {
			permute_rows_in_place(
				b.as_mut(),
				perm(8).as_ref(),
				MemStack::new(&mut buf),
			)
		}));
		let msg = r.expect_err("a half-size stack is refused");
		let msg = msg
			.downcast_ref::<String>()
			.map(String::as_str)
			.or_else(|| msg.downcast_ref::<&str>().copied())
			.unwrap_or("");
		assert!(msg.contains("temporary matrix needs"), "{msg}");
		assert!(b == a);
	}

	/// The temporaries are aligned like an owned matrix, zero-filled when asked, and fit a
	/// buffer of exactly the queried size; two of them fit `all_of` the two requirements.
	#[test]
	fn temporaries_fit_their_requirement() {
		for (m, n) in [(0usize, 0usize), (1, 1), (5, 3), (505, 2), (8, 0)] {
			let req = temp_mat_scratch::<f64>(m, n);
			let mut buf = MemBuffer::new(req);
			let stack = MemStack::new(&mut buf);
			let (mut z, _) = temp_mat_zeroed::<f64, _, _>(m, n, stack);
			let z = z.as_mat_mut();
			assert!(z.shape() == (m, n));
			assert!((z.as_ptr() as usize).is_multiple_of(64));
			assert!(z.col_stride() as usize >= m);
			assert!((0..n).all(|j| (0..m).all(|i| z[(i, j)] == 0.0)));
		}
		let req = StackReq::all_of(&[
			temp_mat_scratch::<f64>(7, 5),
			temp_mat_scratch::<crate::c64>(3, 9),
		]);
		let mut buf = MemBuffer::new(req);
		let stack = MemStack::new(&mut buf);
		let (mut a, stack) = temp_mat_zeroed::<f64, _, _>(7, 5, stack);
		// SAFETY: `b` is written in full below before it is read.
		let (mut b, _) =
			unsafe { temp_mat_uninit::<crate::c64, _, _>(3, 9, stack) };
		let mut a = a.as_mat_mut();
		let mut b = b.as_mat_mut();
		a.fill(1.0);
		b.fill(crate::c64::new(2.0, 0.0));
		assert!(a[(6, 4)] == 1.0 && b[(2, 8)] == crate::c64::new(2.0, 0.0));
	}

	/// The macros' `@alloc` arms and `stack_mat!` give the same temporaries.
	#[test]
	fn alloc_macros_and_stack_mat() {
		let mut buf = MemBuffer::new(StackReq::all_of(&[
			temp_mat_scratch::<f64>(3, 2),
			temp_mat_scratch::<f64>(4, 1),
			temp_mat_scratch::<f64>(5, 1),
		]));
		let stack = MemStack::new(&mut buf);
		crate::mat!(@alloc (stack) (mut a) (zero::<f64>, 3, 2));
		crate::col!(@alloc (stack) (mut c) (zero::<f64>, 4));
		// SAFETY: `r` is written in full before it is read.
		crate::row!(@alloc unsafe (stack) (mut r) (uninit::<f64>, 5));
		let _ = stack;
		a[(2, 1)] = 1.0;
		c[3] = 2.0;
		r.fill(3.0);
		assert!(a[(0, 0)] == 0.0 && a[(2, 1)] == 1.0);
		assert!(c[0] == 0.0 && c[3] == 2.0);
		assert!(r.ncols() == 5 && r[4] == 3.0);

		crate::stack_mat!(s, 3, 2, 4, 4, f64);
		assert!(s.shape() == (3, 2) && s[(2, 1)] == 0.0);
		s[(2, 1)] = 1.0;
		assert!(s[(2, 1)] == 1.0);
	}
}
