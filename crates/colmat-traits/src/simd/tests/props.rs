//! Property tests: register operations against scalar operations on generated
//! inputs, including NaN, ±0, ±inf, subnormals and random partial masks.
//!
//! - Operations that IEEE arithmetic computes identically lane by lane are
//!   compared bit for bit (NaN matches any NaN).
//! - A real `mul_add` may be fused or not, so its result must equal one of
//!   the two.
//! - Complex products and reductions reorder or fuse roundings, so they are
//!   checked on bounded finite inputs against an `f64` reference with a stated
//!   bound.
//!
//! The case count is proptest's (`PROPTEST_CASES`, 256 by default); the
//! nightly heavy tests raise it. Miri skips these tests: its lane runs the
//! fixed tests, and generated cases would take hours there.

// Indices address several arrays in step; clippy's iterator form would hide
// that.
#![allow(clippy::needless_range_loop)]

use super::{DataCheck, Inspect, for_all_types};
use crate::math_utils::*;
use crate::test_support::TestValue;
use crate::{ComplexField, RealField, SimdField, SimdLayout};
use proptest::prelude::*;
use pulp::Simd;

type P<T> = <T as ComplexField>::Simd;

/// The most lanes any provider under test has (f32 on AVX-512 would be 16).
const MAX_LANES: usize = 16;

fn special() -> impl Strategy<Value = f64> + Clone {
	prop_oneof![
		Just(0.0),
		Just(-0.0),
		Just(f64::INFINITY),
		Just(f64::NEG_INFINITY),
		Just(f64::NAN),
		Just(1.0),
		Just(-1.0),
		Just(f64::MIN_POSITIVE),
		Just(-f64::MIN_POSITIVE / 4.0),
		Just(f64::MAX),
		Just(f32::MIN_POSITIVE as f64 / 4.0),
		Just(f32::MAX as f64),
	]
}

/// Any value, specials weighted up.
fn any_value() -> impl Strategy<Value = f64> + Clone {
	prop_oneof![
		3 => special(),
		5 => prop::num::f64::NORMAL | prop::num::f64::SUBNORMAL | prop::num::f64::ZERO,
		2 => -1e3..1e3f64,
	]
}

/// Finite values whose products and sums stay finite in `f32`.
fn bounded() -> impl Strategy<Value = f64> + Clone {
	prop_oneof![
		6 => -1e15..1e15f64,
		1 => Just(0.0),
		1 => Just(-0.0),
		2 => -1e-20..1e-20f64,
	]
}

fn pairs(
	value: impl Strategy<Value = f64> + Clone,
) -> impl Strategy<Value = Vec<(f64, f64)>> {
	prop::collection::vec((value.clone(), value), MAX_LANES)
}

fn take<T: TestValue>(src: &[(f64, f64)], n: usize) -> Vec<T> {
	src[..n].iter().map(|&(re, im)| T::make(re, im)).collect()
}

fn same_f64(a: f64, b: f64) -> bool {
	a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

fn same<T: TestValue>(a: &T, b: &T) -> bool {
	let (a, b) = (a.parts(), b.parts());
	same_f64(a.0, b.0) && same_f64(a.1, b.1)
}

fn elems<T: TestValue, S: Simd>(
	ctx: &<P<T> as SimdLayout<T>>::Ctx<S>,
	v: <P<T> as SimdLayout<T>>::Vec<S>,
) -> Vec<T> {
	let n = P::<T>::lanes::<S>();
	let mut out: Vec<T> = (0..n).map(|_| zero()).collect();
	P::<T>::store(ctx, &mut out, v);
	out
}

fn real_f64<T: TestValue>(x: &T::Real) -> f64 {
	T::from_real_impl(x).parts().0
}

/// Exact operations, layout and masks, on any values.
#[derive(Debug)]
struct ExactOps {
	a: Vec<(f64, f64)>,
	b: Vec<(f64, f64)>,
	c: Vec<(f64, f64)>,
	start: usize,
	end: usize,
}

impl DataCheck for ExactOps {
	fn run<T: TestValue, S: Simd>(&self, simd: S)
	where
		T::Simd: Inspect<T>,
	{
		let ctx = P::<T>::ctx(simd);
		let n = P::<T>::lanes::<S>();
		let (a, b, c) = (
			take::<T>(&self.a, n),
			take::<T>(&self.b, n),
			take::<T>(&self.c, n),
		);
		let (va, vb, vc) = (
			P::<T>::load(&ctx, &a),
			P::<T>::load(&ctx, &b),
			P::<T>::load(&ctx, &c),
		);
		let e = |v| elems::<T, S>(&ctx, v);
		let check = |name: &str, got: Vec<T>, want: &dyn Fn(usize) -> T| {
			for i in 0..n {
				let w = want(i);
				assert!(
					same(&got[i], &w),
					"{name}[{i}]: {:?} vs {:?}",
					got[i],
					w
				);
			}
		};
		let check_real =
			|name: &str, got: Vec<T>, want: &dyn Fn(usize) -> f64| {
				for i in 0..n {
					let g = got[i].parts().0;
					assert!(
						same_f64(g, want(i)),
						"{name}[{i}]: {g:e} vs {:e}",
						want(i)
					);
				}
			};

		check("add", e(P::<T>::add(&ctx, va.clone(), vb.clone())), &|i| {
			&a[i] + &b[i]
		});
		check("sub", e(P::<T>::sub(&ctx, va.clone(), vb.clone())), &|i| {
			&a[i] - &b[i]
		});
		check("neg", e(P::<T>::neg(&ctx, va.clone())), &|i| -&a[i]);
		check("conj", e(P::<T>::conj(&ctx, va.clone())), &|i| conj(&a[i]));
		if T::IS_REAL {
			check("mul", e(P::<T>::mul(&ctx, va.clone(), vb.clone())), &|i| {
				&a[i] * &b[i]
			});
		}

		// Real registers of exact values.
		let abs1_b = P::<T>::abs1(&ctx, vb.clone());
		check_real("abs1", e(abs1_b.clone()), &|i| real_f64::<T>(&abs1(&b[i])));
		check_real("abs_max", e(P::<T>::abs_max(&ctx, va.clone())), &|i| {
			let (re, im) = (abs1(&real(&a[i])), abs1(&imag(&a[i])));
			real_f64::<T>(if re > im || is_nan(&re) { &re } else { &im })
		});
		check(
			"mul_real",
			e(P::<T>::mul_real(&ctx, va.clone(), abs1_b.clone())),
			&|i| mul_real(&a[i], &abs1(&b[i])),
		);
		let half = P::<T>::splat_real(&ctx, &from_f64(0.5));
		check(
			"mul_pow2",
			e(P::<T>::mul_pow2(&ctx, va.clone(), half)),
			&|i| mul_pow2(&a[i], &from_f64(0.5)),
		);

		// `abs2`, and for real types the fused-or-not operations.
		check_real("abs2", e(P::<T>::abs2(&ctx, va.clone())), &|i| {
			let (re, im) = (real(&a[i]), imag(&a[i]));
			let unfused = real_f64::<T>(&(&re * &re + &im * &im));
			let fused = real_f64::<T>(&re.mul_add_impl(re.clone(), &im * &im));
			let got = elems::<T, S>(&ctx, P::<T>::abs2(&ctx, va.clone()))[i]
				.parts()
				.0;
			if same_f64(got, fused) { fused } else { unfused }
		});
		if T::IS_REAL {
			let fused_or_not =
				|name: &str,
				 got: Vec<T>,
				 x: &dyn Fn(usize) -> (T::Real, T::Real, T::Real)| {
					for i in 0..n {
						let (p, q, r) = x(i);
						let unfused = real_f64::<T>(&(&p * &q + &r));
						let fused = real_f64::<T>(&p.mul_add_impl(q, r));
						let g = got[i].parts().0;
						assert!(
							same_f64(g, unfused) || same_f64(g, fused),
							"{name}[{i}]: {g:e} is neither {unfused:e} nor {fused:e}"
						);
					}
				};
			let r = |x: &T| real(x);
			fused_or_not(
				"mul_add",
				e(P::<T>::mul_add(&ctx, va.clone(), vb.clone(), vc.clone())),
				&|i| (r(&a[i]), r(&b[i]), r(&c[i])),
			);
			fused_or_not(
				"conj_mul_add",
				e(P::<T>::conj_mul_add(
					&ctx,
					va.clone(),
					vb.clone(),
					vc.clone(),
				)),
				&|i| (r(&a[i]), r(&b[i]), r(&c[i])),
			);
			let abs2_c = P::<T>::abs2(&ctx, vc.clone());
			fused_or_not(
				"abs2_add",
				e(P::<T>::abs2_add(&ctx, va.clone(), abs2_c)),
				&|i| (r(&a[i]), r(&a[i]), r(&c[i]) * r(&c[i])),
			);
		}

		// Comparisons, by logical lane.
		let iota = P::<T>::index_lanes::<S>(P::<T>::iota(&ctx));
		let abs1_a = P::<T>::abs1(&ctx, va.clone());
		let lanes_of = |m| P::<T>::mask_lanes::<S>(&ctx, m);
		let cmp = |name: &str, m, f: &dyn Fn(&T::Real, &T::Real) -> bool| {
			let lanes = lanes_of(m);
			for p in 0..n {
				let i = iota[p];
				let want = f(&abs1(&a[i]), &abs1(&b[i]));
				assert_eq!(lanes[p], want, "{name} at element {i}");
			}
		};
		cmp(
			"lt",
			P::<T>::less_than(&ctx, abs1_a.clone(), abs1_b.clone()),
			&|x, y| x < y,
		);
		cmp(
			"le",
			P::<T>::less_than_or_equal(&ctx, abs1_a.clone(), abs1_b.clone()),
			&|x, y| x <= y,
		);
		cmp(
			"gt",
			P::<T>::greater_than(&ctx, abs1_a.clone(), abs1_b.clone()),
			&|x, y| x > y,
		);
		cmp(
			"ge",
			P::<T>::greater_than_or_equal(&ctx, abs1_a.clone(), abs1_b.clone()),
			&|x, y| x >= y,
		);
		let eq = lanes_of(P::<T>::equal(&ctx, va.clone(), vb.clone()));
		for p in 0..n {
			assert_eq!(
				eq[p],
				a[iota[p]] == b[iota[p]],
				"equal at element {}",
				iota[p]
			);
		}

		// Select and masked memory access with a random range.
		let (start, end) = (self.start, self.end);
		let inside = |i: usize| start <= i && i < end;
		let m = P::<T>::mask_between(&ctx, start, end);
		check(
			"select",
			e(P::<T>::select(&ctx, m, va.clone(), vb.clone())),
			&|i| {
				if inside(i) {
					a[i].clone()
				} else {
					b[i].clone()
				}
			},
		);
		let mem = P::<T>::mem_mask_between(&ctx, start, end);
		// SAFETY: `a` holds `lanes` initialised elements.
		let loaded = unsafe { P::<T>::mask_load(&ctx, mem, a.as_ptr()) };
		let mut dst = b.clone();
		// SAFETY: `dst` holds `lanes` elements; the test scalars own no
		// resources, so overwriting without a drop leaks nothing.
		unsafe { P::<T>::mask_store(&ctx, mem, dst.as_mut_ptr(), loaded) };
		check("masked load/store", dst, &|i| {
			if inside(i) {
				a[i].clone()
			} else {
				b[i].clone()
			}
		});
	}
}

/// Rounded operations on bounded finite values, against an `f64` reference:
/// each component within `8 eps (|a|₁ |b|₁ + |c|₁)` (products), and sums
/// within `2 n eps Σ|x|₁`, plus the same multiples of the subnormal spacing
/// for results that underflow.
#[derive(Debug)]
struct RoundedOps {
	a: Vec<(f64, f64)>,
	b: Vec<(f64, f64)>,
	c: Vec<(f64, f64)>,
}

fn norm1(x: (f64, f64)) -> f64 {
	x.0.abs() + x.1.abs()
}

impl DataCheck for RoundedOps {
	fn run<T: TestValue, S: Simd>(&self, simd: S)
	where
		T::Simd: Inspect<T>,
	{
		let ctx = P::<T>::ctx(simd);
		let n = P::<T>::lanes::<S>();
		let (a, b, c) = (
			take::<T>(&self.a, n),
			take::<T>(&self.b, n),
			take::<T>(&self.c, n),
		);
		let (va, vb, vc) = (
			P::<T>::load(&ctx, &a),
			P::<T>::load(&ctx, &b),
			P::<T>::load(&ctx, &c),
		);
		let e = |v| elems::<T, S>(&ctx, v);
		let eps = T::EPS;
		// Exact complex arithmetic on the stored (already rounded) inputs.
		let cmul = |x: (f64, f64), y: (f64, f64)| {
			(x.0 * y.0 - x.1 * y.1, x.0 * y.1 + x.1 * y.0)
		};
		let conj = |x: (f64, f64)| (x.0, -x.1);
		let close = |name: &str,
		             got: Vec<T>,
		             want: &dyn Fn(usize) -> ((f64, f64), f64)| {
			for i in 0..n {
				let (w, scale) = want(i);
				let g = got[i].parts();
				let tol = 8.0 * eps * scale + 8.0 * T::TINY;
				assert!(
					(g.0 - w.0).abs() <= tol && (g.1 - w.1).abs() <= tol,
					"{name}[{i}]: {g:?} vs {w:?}, tolerance {tol:e}"
				);
			}
		};
		let pa = |i: usize| a[i].parts();
		let pb = |i: usize| b[i].parts();
		let pc = |i: usize| c[i].parts();
		let add = |x: (f64, f64), y: (f64, f64)| (x.0 + y.0, x.1 + y.1);

		close("mul", e(P::<T>::mul(&ctx, va.clone(), vb.clone())), &|i| {
			(cmul(pa(i), pb(i)), norm1(pa(i)) * norm1(pb(i)))
		});
		close(
			"conj_mul",
			e(P::<T>::conj_mul(&ctx, va.clone(), vb.clone())),
			&|i| (cmul(conj(pa(i)), pb(i)), norm1(pa(i)) * norm1(pb(i))),
		);
		close(
			"mul_add",
			e(P::<T>::mul_add(&ctx, va.clone(), vb.clone(), vc.clone())),
			&|i| {
				(
					add(cmul(pa(i), pb(i)), pc(i)),
					norm1(pa(i)) * norm1(pb(i)) + norm1(pc(i)),
				)
			},
		);
		close(
			"conj_mul_add",
			e(P::<T>::conj_mul_add(
				&ctx,
				va.clone(),
				vb.clone(),
				vc.clone(),
			)),
			&|i| {
				(
					add(cmul(conj(pa(i)), pb(i)), pc(i)),
					norm1(pa(i)) * norm1(pb(i)) + norm1(pc(i)),
				)
			},
		);
		let abs2 = |x: (f64, f64)| x.0 * x.0 + x.1 * x.1;
		let real_close =
			|name: &str, got: Vec<T>, want: &dyn Fn(usize) -> (f64, f64)| {
				for i in 0..n {
					let (w, scale) = want(i);
					let g = got[i].parts().0;
					let tol = 8.0 * eps * scale + 8.0 * T::TINY;
					assert!(
						(g - w).abs() <= tol,
						"{name}[{i}]: {g:e} vs {w:e}, tolerance {tol:e}"
					);
				}
			};
		real_close("abs2", e(P::<T>::abs2(&ctx, va.clone())), &|i| {
			(abs2(pa(i)), abs2(pa(i)))
		});
		let abs2_c = P::<T>::abs2(&ctx, vc.clone());
		real_close(
			"abs2_add",
			e(P::<T>::abs2_add(&ctx, va.clone(), abs2_c)),
			&|i| (abs2(pa(i)) + abs2(pc(i)), abs2(pa(i)) + abs2(pc(i))),
		);

		// Reductions.
		let s = P::<T>::reduce_sum(&ctx, va.clone()).parts();
		let w = (0..n).fold((0.0, 0.0), |acc, i| add(acc, pa(i)));
		let scale: f64 = (0..n).map(|i| norm1(pa(i))).sum();
		let tol = 2.0 * n as f64 * eps * scale + 2.0 * n as f64 * T::TINY;
		assert!(
			(s.0 - w.0).abs() <= tol && (s.1 - w.1).abs() <= tol,
			"reduce_sum: {s:?} vs {w:?}, tolerance {tol:e}"
		);
		let m = P::<T>::reduce_max(&ctx, P::<T>::abs1(&ctx, va.clone()))
			.parts()
			.0;
		let want = (0..n)
			.map(|i| real_f64::<T>(&abs1(&a[i])))
			.fold(0.0, f64::max);
		assert_eq!(m, want, "reduce_max of abs1");
	}
}

proptest! {
	#[cfg_attr(miri, ignore)]
	#[test]
	fn prop_exact_ops_match_scalar_ops(
		a in pairs(any_value()),
		b in pairs(any_value()),
		c in pairs(any_value()),
		start in 0..=MAX_LANES + 1,
		len in 0..=MAX_LANES + 1,
	) {
		for_all_types(&ExactOps { a, b, c, start, end: start + len });
	}

	#[cfg_attr(miri, ignore)]
	#[test]
	fn prop_rounded_ops_within_bounds(
		a in pairs(bounded()),
		b in pairs(bounded()),
		c in pairs(bounded()),
	) {
		for_all_types(&RoundedOps { a, b, c });
	}
}
