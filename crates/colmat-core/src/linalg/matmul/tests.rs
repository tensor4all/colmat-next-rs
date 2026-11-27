//! Tests of the matrix product against a reference of this crate (#23, answer (4): no
//! BLAS-derived tests).
//!
//! The reference sums the products of each element in `f64`. The product is held to
//! `|got - ref| <= (k + 2) * eps(T) * |alpha| * sum_p |a_ip| |b_pj| + eps(T) * (|dst_ij| +
//! |ref|)`: the standard bound of a sum of `k` products in any order (Higham, *Accuracy and
//! Stability of Numerical Algorithms*, 2nd ed., §3.1), one rounding for `alpha`, one for the
//! addition to `dst` and one for the result. For `f32` and `c32` the reference's own error
//! (`f32` products are exact in `f64`; at most `k * 2^-53` relative to the sum of magnitudes) is
//! far below that bound. For `f64` and `c64` the reference rounds like the product, so the bound
//! is doubled. A complex product rounds each component of each multiplication, which the bound
//! covers by a further factor of two.

use super::*;
use crate::test_support::{Strided, layouts};
use crate::{Mat, c32, c64};
use colmat_traits::Complex;

/// A small deterministic generator (SplitMix64), so the tests need no dependency.
struct Rng(u64);
impl Rng {
	fn next(&mut self) -> f64 {
		self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
		let mut z = self.0;
		z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
		z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
		z ^= z >> 31;
		(z >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
	}
}

/// What the tests need of a scalar: a value from two `f64`, its two components as `f64`, and
/// the precision of the type.
trait TestScalar:
	KernelField + Copy + PartialEq + core::fmt::Debug + Send + Sync
{
	const EPS: f64;
	fn new(re: f64, im: f64) -> Self;
	fn parts(self) -> (f64, f64);
}
impl TestScalar for f64 {
	const EPS: f64 = f64::EPSILON;
	fn new(re: f64, _: f64) -> Self {
		re
	}
	fn parts(self) -> (f64, f64) {
		(self, 0.0)
	}
}
impl TestScalar for f32 {
	const EPS: f64 = f32::EPSILON as f64;
	fn new(re: f64, _: f64) -> Self {
		re as f32
	}
	fn parts(self) -> (f64, f64) {
		(self as f64, 0.0)
	}
}
impl TestScalar for c64 {
	const EPS: f64 = f64::EPSILON;
	fn new(re: f64, im: f64) -> Self {
		Complex::new(re, im)
	}
	fn parts(self) -> (f64, f64) {
		(self.re, self.im)
	}
}
impl TestScalar for c32 {
	const EPS: f64 = f32::EPSILON as f64;
	fn new(re: f64, im: f64) -> Self {
		Complex::new(re as f32, im as f32)
	}
	fn parts(self) -> (f64, f64) {
		(self.re as f64, self.im as f64)
	}
}

type C = (f64, f64);
fn mul((a, b): C, (c, d): C) -> C {
	(a * c - b * d, a * d + b * c)
}
fn conj_if((a, b): C, conj: Conj) -> C {
	if conj.is_conj() { (a, -b) } else { (a, b) }
}
fn abs((a, b): C) -> f64 {
	a.hypot(b)
}

/// The reference value and the error bound of element `(i, j)`.
#[allow(clippy::too_many_arguments)]
fn reference<T: TestScalar>(
	dst0: T,
	accum: Accum,
	lhs: impl Fn(usize) -> T,
	conj_lhs: Conj,
	rhs: impl Fn(usize) -> T,
	conj_rhs: Conj,
	alpha: T,
	k: usize,
) -> (C, f64) {
	let mut acc = (0.0, 0.0);
	let mut mag = 0.0;
	for p in 0..k {
		let (a, b) = (
			conj_if(lhs(p).parts(), conj_lhs),
			conj_if(rhs(p).parts(), conj_rhs),
		);
		let ab = mul(a, b);
		acc = (acc.0 + ab.0, acc.1 + ab.1);
		mag += abs(a) * abs(b);
	}
	let mut out = mul(alpha.parts(), acc);
	let mut bound = (k as f64 + 2.0) * T::EPS * abs(alpha.parts()) * mag;
	if accum == Accum::Add {
		let d = dst0.parts();
		out = (out.0 + d.0, out.1 + d.1);
		bound += T::EPS * abs(d);
	}
	bound += T::EPS * abs(out);
	if T::EPS == f64::EPSILON {
		bound *= 2.0;
	}
	if !T::IS_REAL {
		bound *= 2.0;
	}
	(out, bound)
}

/// Which kernels compute the product: the scalar's own (`matmul_with_scratch`, through
/// [`KernelField::Kernels`]), or [`generic::gemm_generic`] called directly. The tests' scalars
/// all name `NativeKernels`, so without the second the generic packed product, the one a
/// scalar naming `GenericKernels` takes, runs only under Miri, where the native types fall back
/// to it.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Kernel {
	Own,
	Generic,
}

/// Runs `dst ← [dst +] alpha * op(lhs) * op(rhs)` on strided buffers of the given layouts and
/// checks every element against the reference, the padding of all three buffers, and, for
/// `Replace`, that `dst` was not read (it starts as NaN).
#[allow(clippy::too_many_arguments)]
fn check<T: TestScalar>(
	kernel: Kernel,
	rng: &mut Rng,
	(m, n, k): (usize, usize, usize),
	(ld, ll, lr): ((isize, isize), (isize, isize), (isize, isize)),
	accum: Accum,
	conj_lhs: Conj,
	conj_rhs: Conj,
	par: Par,
) {
	let pad = T::new(7.25, -3.5);
	let mut v = || T::new(rng.next(), rng.next());
	let lhs_vals: Vec<T> = (0..m * k).map(|_| v()).collect();
	let rhs_vals: Vec<T> = (0..k * n).map(|_| v()).collect();
	let dst_vals: Vec<T> = (0..m * n)
		.map(|_| {
			if accum == Accum::Replace {
				T::new(f64::NAN, f64::NAN)
			} else {
				v()
			}
		})
		.collect();
	let alpha = v();
	let lhs = Strided::new(m, k, ll, pad, |i, p| lhs_vals[i + m * p]);
	let rhs = Strided::new(k, n, lr, pad, |p, j| rhs_vals[p + k * j]);
	let mut dst = Strided::new(m, n, ld, pad, |i, j| dst_vals[i + m * j]);
	match kernel {
		Kernel::Own => {
			let mut stack_buf =
				MemBuffer::new(matmul_scratch::<T>(m, n, k, par.shape()));
			matmul_with_scratch(
				dst.view_mut(),
				accum,
				lhs.view(),
				conj_lhs,
				rhs.view(),
				conj_rhs,
				alpha,
				par,
				MemStack::new(&mut stack_buf),
			);
		},
		Kernel::Generic => generic::gemm_generic(
			dst.view_mut(),
			accum,
			lhs.view(),
			conj_lhs,
			rhs.view(),
			conj_rhs,
			&alpha,
			par,
		),
	}
	let ctx = format!(
		"{kernel:?} {} {m}x{n}x{k} {ld:?} {ll:?} {lr:?} {accum:?} {conj_lhs:?} {conj_rhs:?}",
		core::any::type_name::<T>()
	);
	for j in 0..n {
		for i in 0..m {
			let (want, bound) = reference(
				dst_vals[i + m * j],
				accum,
				|p| lhs_vals[i + m * p],
				conj_lhs,
				|p| rhs_vals[p + k * j],
				conj_rhs,
				alpha,
				k,
			);
			let got = dst.get(i, j).parts();
			let err = abs((got.0 - want.0, got.1 - want.1));
			assert!(
				err <= bound,
				"{ctx}: ({i}, {j}) got {got:?}, want {want:?}, err {err:e} > {bound:e}"
			);
		}
	}
	dst.assert_padding_untouched(&ctx);
	lhs.assert_padding_untouched(&ctx);
	rhs.assert_padding_untouched(&ctx);
}

/// Shapes that reach every path of the driver and the kernels: empty, depth 0, the vector and
/// rank-one cases, tiles smaller and larger than a micro-kernel, and depths past one block.
fn shapes() -> Vec<(usize, usize, usize)> {
	let mut s = vec![
		(0, 3, 2),
		(3, 0, 2),
		(3, 2, 0),
		(1, 1, 1),
		(1, 7, 5),
		(7, 1, 5),
		(5, 6, 1),
		(2, 2, 2),
		(17, 9, 13),
		(3, 40, 7),
	];
	if !cfg!(miri) {
		s.extend([(33, 5, 300), (9, 9, 260), (70, 3, 3)]);
	}
	s
}

/// Every layout of `dst`, `lhs` and `rhs` (one in three triples; one in nine for
/// [`Kernel::Generic`], which reaches the same lines and is run only for them), each conjugation
/// and both accumulation modes. Under Miri, which runs these about a thousand times slower, each layout
/// of each operand appears once (16 triples per shape) with one of the eight modes, so the
/// unsafe loads and stores of every layout are still checked within the scheduled Miri jobs' limit (#21).
fn exhaustive<T: TestScalar>(kernel: Kernel) {
	let mut rng = Rng(0x5eed);
	let every = if kernel == Kernel::Generic { 9 } else { 3 };
	let modes = |t: usize| {
		let accum = if t & 1 == 0 {
			Accum::Replace
		} else {
			Accum::Add
		};
		let conj = |b: usize| if b == 0 { Conj::No } else { Conj::Yes };
		(accum, conj((t >> 1) & 1), conj((t >> 2) & 1))
	};
	for (m, n, k) in shapes() {
		// The driver handles depth 0 itself and never passes it to a hook (`kernels`), and
		// `gemm_generic` leaves `dst` as it was there.
		if kernel == Kernel::Generic && k == 0 {
			continue;
		}
		let (ds, ls, rs) = (layouts(m, n), layouts(m, k), layouts(k, n));
		if cfg!(miri) {
			for t in 0..ds.len() {
				let (accum, conj_lhs, conj_rhs) = modes(t);
				let lay =
					(ds[t], ls[(5 * t) % ls.len()], rs[(11 * t) % rs.len()]);
				check::<T>(
					kernel,
					&mut rng,
					(m, n, k),
					lay,
					accum,
					conj_lhs,
					conj_rhs,
					Par::Seq,
				);
			}
			continue;
		}
		let mut idx = 0usize;
		for ld in &ds {
			for ll in &ls {
				for lr in &rs {
					idx += 1;
					if !idx.is_multiple_of(every) {
						continue;
					}
					for t in 0..8 {
						let (accum, conj_lhs, conj_rhs) = modes(t);
						check::<T>(
							kernel,
							&mut rng,
							(m, n, k),
							(*ld, *ll, *lr),
							accum,
							conj_lhs,
							conj_rhs,
							Par::Seq,
						);
					}
				}
			}
		}
	}
}

#[test]
fn matmul_matches_the_reference_f64() {
	exhaustive::<f64>(Kernel::Own);
}
#[test]
// The same kernels as `f64`/`c64` at another lane count, which `utils::simd`'s walk test covers
// under Miri; skipped there to keep the scheduled Miri jobs within their limit (#21).
#[cfg_attr(
	miri,
	ignore = "covered by the f64/c64 cases and the simd walk test under Miri"
)]
fn matmul_matches_the_reference_f32() {
	exhaustive::<f32>(Kernel::Own);
}
#[test]
fn matmul_matches_the_reference_c64() {
	exhaustive::<c64>(Kernel::Own);
}
#[test]
// The same kernels as `f64`/`c64` at another lane count, which `utils::simd`'s walk test covers
// under Miri; skipped there to keep the scheduled Miri jobs within their limit (#21).
#[cfg_attr(
	miri,
	ignore = "covered by the f64/c64 cases and the simd walk test under Miri"
)]
fn matmul_matches_the_reference_c32() {
	exhaustive::<c32>(Kernel::Own);
}

// The generic kernels on the native types, whose SIMD registers they then use: the product of
// a scalar that names `GenericKernels`. Under Miri the native types already take these kernels
// (the tests above), so they are skipped there.
#[test]
#[cfg_attr(
	miri,
	ignore = "the native tests above take these kernels under Miri"
)]
fn generic_matmul_matches_the_reference_f64() {
	exhaustive::<f64>(Kernel::Generic);
}
#[test]
#[cfg_attr(
	miri,
	ignore = "the native tests above take these kernels under Miri"
)]
fn generic_matmul_matches_the_reference_f32() {
	exhaustive::<f32>(Kernel::Generic);
}
#[test]
#[cfg_attr(
	miri,
	ignore = "the native tests above take these kernels under Miri"
)]
fn generic_matmul_matches_the_reference_c64() {
	exhaustive::<c64>(Kernel::Generic);
}
#[test]
#[cfg_attr(
	miri,
	ignore = "the native tests above take these kernels under Miri"
)]
fn generic_matmul_matches_the_reference_c32() {
	exhaustive::<c32>(Kernel::Generic);
}

/// The convenience entries agree with [`matmul_with_scratch`], and `matmul` reads the
/// conjugation from the operand type.
#[test]
fn entries_agree() {
	let mut rng = Rng(7);
	let a = Mat::<c64>::from_fn(6, 4, |_, _| c64::new(rng.next(), rng.next()));
	let b = Mat::<c64>::from_fn(4, 5, |_, _| c64::new(rng.next(), rng.next()));
	let alpha = c64::new(0.5, -1.25);
	let mut want = Mat::<c64>::zeros(6, 5);
	let mut buf = MemBuffer::new(matmul_scratch::<c64>(6, 5, 4, ParShape::Seq));
	matmul_with_scratch(
		&mut want,
		Accum::Replace,
		&a,
		Conj::Yes,
		&b,
		Conj::No,
		alpha,
		Par::Seq,
		MemStack::new(&mut buf),
	);
	let mut got = Mat::<c64>::zeros(6, 5);
	matmul_with_conj(
		&mut got,
		Accum::Replace,
		&a,
		Conj::Yes,
		&b,
		Conj::No,
		alpha,
		Par::Seq,
	);
	assert_eq!(got, want);
	let mut got = Mat::<c64>::zeros(6, 5);
	matmul(&mut got, Accum::Replace, a.conjugate(), &b, alpha, Par::Seq);
	assert_eq!(got, want);
}

#[test]
#[should_panic]
fn mismatched_shapes_panic() {
	let a = Mat::<f64>::zeros(3, 4);
	let b = Mat::<f64>::zeros(5, 2);
	let mut c = Mat::<f64>::zeros(3, 2);
	matmul(&mut c, Accum::Replace, &a, &b, 1.0, Par::Seq);
}

/// On a team the product splits the output, never the depth, so every element is computed in
/// the same order as on one thread: the result is bitwise that of `Par::Seq` at every width, and
/// the parallel path did run. This needs the kernels' blocking of the depth not to depend on the
/// tile's shape: `(257, 3, 513)` and `(130, 130, 1024)` cut tiles of at most `64 x 64` from a
/// product deeper than 512, which `t4a-gemm` blocked differently from the whole product until
/// tensor4all/t4a-gemm#25 (#45). The column-major matrix-vector product splits its depth above
/// the size of `generic::depth_split` (`team_matvec_depth_split_*` below); the shapes here stay
/// below it.
#[cfg(all(feature = "threadteam", not(miri)))]
#[test]
fn team_result_is_bitwise_seq() {
	use crate::utils::team::{stats, with_team};
	let mut rng = Rng(11);
	// gemm (both splits), matrix-vector (both layouts) and rank-one shapes, large enough to pay
	for &(m, n, k) in &[
		(256usize, 256usize, 256usize),
		(256, 3, 256),
		(4096, 1, 512),
		(1, 2048, 2048),
		(2048, 2048, 1),
		(257, 3, 513),
		(130, 130, 1024),
	] {
		let a = Mat::<f64>::from_fn(m, k, |_, _| rng.next());
		let b = Mat::<f64>::from_fn(k, n, |_, _| rng.next());
		let mut seq = Mat::<f64>::zeros(m, n);
		matmul(&mut seq, Accum::Replace, &a, &b, 1.5, Par::Seq);
		for width in 2..=8 {
			let mut team = Mat::<f64>::zeros(m, n);
			let ran = with_team(width, |par| {
				stats::enable(true);
				let _ = stats::take();
				matmul(&mut team, Accum::Replace, &a, &b, 1.5, par);
				stats::enable(false);
				let s = stats::take();
				s.run.iter().skip(2).sum::<usize>()
					+ s.split.iter().skip(2).sum::<usize>()
			});
			// The cost model keeps `256 x 3 x 256` on one thread below width 4; every other
			// shape here, and these two at every width, take the team path.
			if width >= 4 || k > 512 {
				assert!(
					ran > 0,
					"{m}x{n}x{k} width {width}: the team path did not run"
				);
			}
			assert!(
				team == seq,
				"{m}x{n}x{k} width {width}: team result differs from Seq"
			);
			// the packing memory from the stack (`Gemm::gemm`) rather than from the kernels'
			// thread-locals (`Gemm::gemm_alloc`, which `matmul` takes): the same result
			let mut stacked = Mat::<f64>::zeros(m, n);
			with_team(width, |par| {
				let mut buf =
					MemBuffer::new(matmul_scratch::<f64>(m, n, k, par.shape()));
				matmul_with_scratch(
					&mut stacked,
					Accum::Replace,
					&a,
					Conj::No,
					&b,
					Conj::No,
					1.5,
					par,
					MemStack::new(&mut buf),
				);
			});
			assert!(
				stacked == seq,
				"{m}x{n}x{k} width {width}: stack result differs from Seq"
			);
		}
	}
}

/// [`generic::gemm_generic`] on a team splits the output as the native product does, so its
/// result is bitwise that of `Par::Seq` as well, and the parallel path did run.
#[cfg(all(feature = "threadteam", not(miri)))]
#[test]
fn generic_team_result_is_bitwise_seq() {
	use crate::utils::team::{stats, with_team};
	let mut rng = Rng(13);
	// both splits of the grid, a depth past one block, and tiles that are not whole micro-kernels
	for &(m, n, k) in
		&[(192usize, 160usize, 96usize), (300, 7, 70), (5, 260, 33)]
	{
		let a =
			Mat::<c64>::from_fn(m, k, |_, _| c64::new(rng.next(), rng.next()));
		let b =
			Mat::<c64>::from_fn(k, n, |_, _| c64::new(rng.next(), rng.next()));
		let d =
			Mat::<c64>::from_fn(m, n, |_, _| c64::new(rng.next(), rng.next()));
		let alpha = c64::new(0.75, -1.25);
		let run = |dst: &mut Mat<c64>, par: Par| {
			generic::gemm_generic(
				dst.as_mut(),
				Accum::Add,
				a.as_ref(),
				Conj::Yes,
				b.as_ref(),
				Conj::No,
				&alpha,
				par,
			)
		};
		let mut seq = d.clone();
		run(&mut seq, Par::Seq);
		let mut team = d.clone();
		let ran = with_team(4, |par| {
			stats::enable(true);
			let _ = stats::take();
			run(&mut team, par);
			stats::enable(false);
			let s = stats::take();
			s.run.iter().skip(2).sum::<usize>()
				+ s.split.iter().skip(2).sum::<usize>()
		});
		assert!(ran > 0, "{m}x{n}x{k}: the team path did not run");
		assert!(team == seq, "{m}x{n}x{k}: team result differs from Seq");
	}
}

/// The shapes of the depth split of the column-major matrix-vector product for `T`: an
/// `m x 1` product of depth `k` (`rhs_row_major` false) or a `1 x m` product whose `rhs` is
/// row-major (true), which the driver computes transposed. Larger than 8 MiB, and with columns
/// of 512 bytes.
#[cfg(all(feature = "threadteam", not(miri)))]
fn depth_split_shapes<T>() -> [(usize, usize, bool); 3] {
	let short = 512 / core::mem::size_of::<T>();
	[
		(2100, 1024, false),
		(short, 8193, false),
		(2100, 1024, true),
	]
}

/// Above the size of `generic::depth_split`, a team splits the depth of the column-major
/// matrix-vector product, so its result rounds differently from `Par::Seq`. It is held to the
/// reference bound, for both accumulations and every conjugation, and runs the team path.
#[cfg(all(feature = "threadteam", not(miri)))]
fn team_matvec_depth_split<T: TestScalar>() {
	use crate::utils::team::{stats, with_team};
	let mut rng = Rng(23);
	for (m, k, rhs_row_major) in depth_split_shapes::<T>() {
		assert!(
			generic::depth_split::<T>(m, k),
			"{m}x{k} is below the split"
		);
		let a = Mat::<T>::from_fn(m, k, |_, _| T::new(rng.next(), rng.next()));
		let x = Mat::<T>::from_fn(k, 1, |_, _| T::new(rng.next(), rng.next()));
		let xt = Mat::<T>::from_fn(1, k, |_, j| x[(j, 0)]);
		let d0 = Mat::<T>::from_fn(m, 1, |_, _| T::new(rng.next(), rng.next()));
		let alpha = T::new(0.75, -0.5);
		for width in [2, 3, 4] {
			for accum in [Accum::Replace, Accum::Add] {
				for (conj_lhs, conj_rhs) in [
					(Conj::No, Conj::No),
					(Conj::Yes, Conj::No),
					(Conj::No, Conj::Yes),
				] {
					let mut d = d0.clone();
					let mut dt = Mat::<T>::from_fn(1, m, |_, i| d0[(i, 0)]);
					let ran = with_team(width, |par| {
						let mut buf =
							dyn_stack::MemBuffer::new(if rhs_row_major {
								matmul_scratch::<T>(1, m, k, par.shape())
							} else {
								matmul_scratch::<T>(m, 1, k, par.shape())
							});
						stats::enable(true);
						let _ = stats::take();
						if rhs_row_major {
							// `dt = x^T * a^T`, with `a^T` row-major
							matmul_with_scratch(
								&mut dt,
								accum,
								&xt,
								conj_rhs,
								a.transpose(),
								conj_lhs,
								alpha,
								par,
								MemStack::new(&mut buf),
							);
						} else {
							matmul_with_scratch(
								&mut d,
								accum,
								&a,
								conj_lhs,
								&x,
								conj_rhs,
								alpha,
								par,
								MemStack::new(&mut buf),
							);
						}
						stats::enable(false);
						let s = stats::take();
						s.run.iter().skip(2).sum::<usize>()
							+ s.split.iter().skip(2).sum::<usize>()
					});
					assert!(
						ran > 0,
						"{m}x{k} on {width}: the team path did not run"
					);
					for i in 0..m {
						let got =
							if rhs_row_major { dt[(0, i)] } else { d[(i, 0)] };
						let (want, bound) = reference(
							d0[(i, 0)],
							accum,
							|p| a[(i, p)],
							conj_lhs,
							|p| x[(p, 0)],
							conj_rhs,
							alpha,
							k,
						);
						let (g, w) = (got.parts(), want);
						let err = abs((g.0 - w.0, g.1 - w.1));
						assert!(
							err <= bound,
							"{m}x{k} on {width} {accum:?} {conj_lhs:?} {conj_rhs:?} row {i}: \
							 {err} > {bound}"
						);
					}
				}
			}
		}
	}
}

/// The depth split is the same on every call of the same width: the partition of the depth and
/// the order in which the partial sums are added depend on the shape and the width only.
#[cfg(all(feature = "threadteam", not(miri)))]
#[test]
fn team_matvec_depth_split_is_deterministic_per_width() {
	use crate::utils::team::with_team;
	let mut rng = Rng(29);
	let (m, k) = (2100, 1024);
	let a = Mat::<f64>::from_fn(m, k, |_, _| rng.next());
	let x = Mat::<f64>::from_fn(k, 1, |_, _| rng.next());
	for width in [2, 3, 4] {
		let run = || {
			let mut d = Mat::<f64>::zeros(m, 1);
			with_team(width, |par| {
				matmul(&mut d, Accum::Replace, &a, &x, 1.5, par)
			});
			d
		};
		let first = run();
		for _ in 0..3 {
			assert!(run() == first, "width {width}: the result changed");
		}
	}
}

#[cfg(all(feature = "threadteam", not(miri)))]
#[test]
fn team_matvec_depth_split_f64() {
	team_matvec_depth_split::<f64>();
}

#[cfg(all(feature = "threadteam", not(miri)))]
#[test]
fn team_matvec_depth_split_f32() {
	team_matvec_depth_split::<f32>();
}

#[cfg(all(feature = "threadteam", not(miri)))]
#[test]
fn team_matvec_depth_split_c64() {
	team_matvec_depth_split::<c64>();
}

#[cfg(all(feature = "threadteam", not(miri)))]
#[test]
fn team_matvec_depth_split_c32() {
	team_matvec_depth_split::<c32>();
}

/// The rule of the depth split, on the shapes it was measured at (#28).
#[test]
fn depth_split_rule() {
	use generic::depth_split;
	// shallow: always the rows
	assert!(!depth_split::<f64>(16384, 256));
	assert!(!depth_split::<f32>(64, 1023));
	// cache resident with long columns: the rows
	assert!(!depth_split::<f64>(256, 4096));
	assert!(!depth_split::<f64>(1024, 1024));
	assert!(!depth_split::<c32>(1024, 1024));
	// larger than 8 MiB
	assert!(depth_split::<f64>(1024, 1025));
	assert!(depth_split::<f32>(1024, 4096));
	assert!(depth_split::<c64>(256, 4096 + 1));
	// columns of at most 512 bytes
	assert!(depth_split::<f64>(64, 1024));
	assert!(depth_split::<c64>(32, 1024));
	assert!(!depth_split::<c64>(33, 1024));
}
