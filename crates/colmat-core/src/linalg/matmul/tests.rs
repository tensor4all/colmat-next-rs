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

/// Runs `dst ← [dst +] alpha * op(lhs) * op(rhs)` on strided buffers of the given layouts and
/// checks every element against the reference, the padding of all three buffers, and, for
/// `Replace`, that `dst` was not read (it starts as NaN).
#[allow(clippy::too_many_arguments)]
fn check<T: TestScalar>(
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
	let ctx = format!(
		"{} {m}x{n}x{k} {ld:?} {ll:?} {lr:?} {accum:?} {conj_lhs:?} {conj_rhs:?}",
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

/// Every layout of `dst`, `lhs` and `rhs` (one in three triples), each conjugation and both
/// accumulation modes. Under Miri, which runs these about a thousand times slower, each layout
/// of each operand appears once (16 triples per shape) with one of the eight modes, so the
/// unsafe loads and stores of every layout are still checked within the nightly budget (#21).
fn exhaustive<T: TestScalar>() {
	let mut rng = Rng(0x5eed);
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
		let (ds, ls, rs) = (layouts(m, n), layouts(m, k), layouts(k, n));
		if cfg!(miri) {
			for t in 0..ds.len() {
				let (accum, conj_lhs, conj_rhs) = modes(t);
				let lay =
					(ds[t], ls[(5 * t) % ls.len()], rs[(11 * t) % rs.len()]);
				check::<T>(
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
					if !idx.is_multiple_of(3) {
						continue;
					}
					for t in 0..8 {
						let (accum, conj_lhs, conj_rhs) = modes(t);
						check::<T>(
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
	exhaustive::<f64>();
}
#[test]
// The same kernels as `f64`/`c64` at another lane count, which `utils::simd`'s walk test covers
// under Miri; skipped there to keep the nightly Miri jobs within their limit (#21).
#[cfg_attr(
	miri,
	ignore = "covered by the f64/c64 cases and the simd walk test under Miri"
)]
fn matmul_matches_the_reference_f32() {
	exhaustive::<f32>();
}
#[test]
fn matmul_matches_the_reference_c64() {
	exhaustive::<c64>();
}
#[test]
// The same kernels as `f64`/`c64` at another lane count, which `utils::simd`'s walk test covers
// under Miri; skipped there to keep the nightly Miri jobs within their limit (#21).
#[cfg_attr(
	miri,
	ignore = "covered by the f64/c64 cases and the simd walk test under Miri"
)]
fn matmul_matches_the_reference_c32() {
	exhaustive::<c32>();
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
/// the same order as on one thread: the result is bitwise that of `Par::Seq`, and the parallel
/// path did run.
#[cfg(all(feature = "threadteam", not(miri)))]
#[test]
fn team_result_is_bitwise_seq() {
	use crate::utils::team::{stats, with_team};
	let mut rng = Rng(11);
	// gemm (both splits), matrix-vector (both layouts) and rank-one shapes, large enough to pay
	for &(m, n, k) in &[
		(256usize, 256usize, 256usize),
		(256, 3, 256),
		(2048, 1, 2048),
		(1, 2048, 2048),
		(2048, 2048, 1),
	] {
		let a = Mat::<f64>::from_fn(m, k, |_, _| rng.next());
		let b = Mat::<f64>::from_fn(k, n, |_, _| rng.next());
		let mut seq = Mat::<f64>::zeros(m, n);
		matmul(&mut seq, Accum::Replace, &a, &b, 1.5, Par::Seq);
		let mut team = Mat::<f64>::zeros(m, n);
		let ran = with_team(4, |par| {
			stats::enable(true);
			let _ = stats::take();
			matmul(&mut team, Accum::Replace, &a, &b, 1.5, par);
			stats::enable(false);
			let s = stats::take();
			s.run.iter().skip(2).sum::<usize>()
				+ s.split.iter().skip(2).sum::<usize>()
		});
		assert!(ran > 0, "{m}x{n}x{k}: the team path did not run");
		assert!(team == seq, "{m}x{n}x{k}: team result differs from Seq");
	}
}
