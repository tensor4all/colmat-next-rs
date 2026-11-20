//! Arithmetic: every register operation against the scalar operation, on
//! values whose results are exact, so that the comparison is bitwise.

use super::{Check, Inspect, run_all};
use crate::math_utils::*;
use crate::test_support::{TestValue, value};
use crate::{Complex, ComplexField, SimdField, SimdLayout};
use pulp::Simd;

type P<T> = <T as ComplexField>::Simd;

/// Stores a register and returns its elements in logical order.
fn elems<T: TestValue, S: Simd>(
	ctx: &<P<T> as SimdLayout<T>>::Ctx<S>,
	v: <P<T> as SimdLayout<T>>::Vec<S>,
) -> Vec<T> {
	let n = P::<T>::lanes::<S>();
	let mut out: Vec<T> = (0..n).map(|_| zero()).collect();
	P::<T>::store(ctx, &mut out, v);
	out
}

/// The real parts of a real register's elements.
fn reals<T: TestValue, S: Simd>(
	ctx: &<P<T> as SimdLayout<T>>::Ctx<S>,
	v: <P<T> as SimdLayout<T>>::Vec<S>,
) -> Vec<T::Real> {
	elems::<T, S>(ctx, v).iter().map(real).collect()
}

#[test]
fn lane_wise_ops_match_scalar_ops() {
	struct C;
	impl Check for C {
		fn run<T: TestValue, S: Simd>(simd: S)
		where
			T::Simd: Inspect<T>,
		{
			let ctx = P::<T>::ctx(simd);
			let n = P::<T>::lanes::<S>();
			let a: Vec<T> = (0..n).map(value).collect();
			let b: Vec<T> = (0..n).map(|i| value(i + 7)).collect();
			let c: Vec<T> = (0..n).map(|i| value(i + 13)).collect();
			let (va, vb, vc) = (
				P::<T>::load(&ctx, &a),
				P::<T>::load(&ctx, &b),
				P::<T>::load(&ctx, &c),
			);
			let map2 = |f: &dyn Fn(&T, &T) -> T| -> Vec<T> {
				a.iter().zip(&b).map(|(x, y)| f(x, y)).collect()
			};
			let map3 = |f: &dyn Fn(&T, &T, &T) -> T| -> Vec<T> {
				(0..n).map(|i| f(&a[i], &b[i], &c[i])).collect()
			};
			let e = |v| elems::<T, S>(&ctx, v);

			assert_eq!(
				e(P::<T>::add(&ctx, va.clone(), vb.clone())),
				map2(&|x, y| x + y)
			);
			assert_eq!(
				e(P::<T>::sub(&ctx, va.clone(), vb.clone())),
				map2(&|x, y| x - y)
			);
			assert_eq!(
				e(P::<T>::mul(&ctx, va.clone(), vb.clone())),
				map2(&|x, y| x * y)
			);
			assert_eq!(
				e(P::<T>::conj_mul(&ctx, va.clone(), vb.clone())),
				map2(&|x, y| conj(x) * y)
			);
			assert_eq!(
				e(P::<T>::mul_add(&ctx, va.clone(), vb.clone(), vc.clone())),
				map3(&|x, y, z| x * y + z)
			);
			assert_eq!(
				e(P::<T>::conj_mul_add(
					&ctx,
					va.clone(),
					vb.clone(),
					vc.clone()
				)),
				map3(&|x, y, z| conj(x) * y + z)
			);
			assert_eq!(
				e(P::<T>::neg(&ctx, va.clone())),
				a.iter().map(|x| -x).collect::<Vec<_>>()
			);
			assert_eq!(
				e(P::<T>::conj(&ctx, va.clone())),
				a.iter().map(conj).collect::<Vec<_>>()
			);

			// Real registers: compare the real parts.
			let r = |v| reals::<T, S>(&ctx, v);
			assert_eq!(
				r(P::<T>::abs1(&ctx, va.clone())),
				a.iter().map(abs1).collect::<Vec<_>>()
			);
			assert_eq!(
				r(P::<T>::abs2(&ctx, va.clone())),
				a.iter().map(abs2).collect::<Vec<_>>()
			);
			let abs2_c = P::<T>::abs2(&ctx, vc.clone());
			assert_eq!(
				r(P::<T>::abs2_add(&ctx, va.clone(), abs2_c.clone())),
				(0..n)
					.map(|i| abs2(&a[i]) + abs2(&c[i]))
					.collect::<Vec<_>>()
			);
			assert_eq!(
				r(P::<T>::abs_max(&ctx, va.clone())),
				a.iter()
					.map(|x| max(&abs1(&real(x)), &abs1(&imag(x))))
					.collect::<Vec<_>>()
			);

			// `mul_real` and `mul_pow2` read the real part of the right operand.
			let half = P::<T>::splat_real(&ctx, &from_f64(0.5));
			assert_eq!(
				e(P::<T>::mul_real(&ctx, va.clone(), abs2_c.clone())),
				(0..n)
					.map(|i| mul_real(&a[i], &abs2(&c[i])))
					.collect::<Vec<_>>()
			);
			assert_eq!(
				e(P::<T>::mul_pow2(&ctx, va.clone(), half)),
				a.iter()
					.map(|x| mul_pow2(x, &from_f64(0.5)))
					.collect::<Vec<_>>()
			);

			// Reductions (exact values, so the order does not matter).
			let sum = a.iter().fold(zero::<T>(), |acc, x| acc + x);
			assert_eq!(P::<T>::reduce_sum(&ctx, va.clone()), sum);
			let max2 = a
				.iter()
				.map(abs2)
				.fold(zero::<T::Real>(), |m, x| max(&x, &m));
			assert_eq!(
				real(&P::<T>::reduce_max(&ctx, P::<T>::abs2(&ctx, va.clone()))),
				max2
			);

			// Comparisons on real registers, lane by lane through `iota`.
			let iota = P::<T>::index_lanes::<S>(P::<T>::iota(&ctx));
			let (ra, rb) = (
				P::<T>::abs2(&ctx, va.clone()),
				P::<T>::abs2(&ctx, vb.clone()),
			);
			let cmp = |m, f: &dyn Fn(&T::Real, &T::Real) -> bool| {
				let lanes = P::<T>::mask_lanes::<S>(&ctx, m);
				for p in 0..n {
					let i = iota[p];
					assert_eq!(
						lanes[p],
						f(&abs2(&a[i]), &abs2(&b[i])),
						"lane {p}"
					);
				}
			};
			cmp(P::<T>::less_than(&ctx, ra.clone(), rb.clone()), &|x, y| {
				x < y
			});
			cmp(P::<T>::less_than(&ctx, rb.clone(), ra.clone()), &|x, y| {
				y < x
			});
			cmp(
				P::<T>::less_than_or_equal(&ctx, ra.clone(), ra.clone()),
				&|_, _| true,
			);
			cmp(
				P::<T>::greater_than(&ctx, rb.clone(), ra.clone()),
				&|x, y| y > x,
			);
			cmp(
				P::<T>::greater_than_or_equal(&ctx, ra.clone(), rb.clone()),
				&|x, y| x >= y,
			);
		}
	}
	run_all::<C>();
}

/// A NaN in either part of a complex value survives `abs_max`
/// (colmat-rs `tensor4all/faer#320`, ported test): the unordered `re > im`
/// alone would report the finite part.
#[test]
fn complex_abs_max_propagates_nan() {
	struct AbsMax {
		re: f64,
		im: f64,
	}
	impl pulp::WithSimd for AbsMax {
		type Output = f64;
		fn with_simd<S: Simd>(self, simd: S) -> f64 {
			type C = Complex<f64>;
			type Q = <C as ComplexField>::Simd;
			let ctx = <Q as SimdLayout<C>>::ctx(simd);
			let v =
				<Q as SimdLayout<C>>::splat(&ctx, &C::new(self.re, self.im));
			let v = <Q as SimdField<C>>::abs_max(&ctx, v);
			<Q as SimdField<C>>::reduce_max(&ctx, v).re
		}
	}
	for abs_max in [
		|re, im| pulp::Arch::new().dispatch(AbsMax { re, im }),
		|re, im| {
			pulp::WithSimd::with_simd(AbsMax { re, im }, pulp::Scalar::new())
		},
	] {
		assert_eq!(abs_max(1.0, 2.0), 2.0);
		assert_eq!(abs_max(-3.0, 2.0), 3.0);
		assert_eq!(abs_max(-3.0, -4.0), 4.0);
		assert!(abs_max(f64::NAN, 1.0).is_nan());
		assert!(abs_max(1.0, f64::NAN).is_nan());
		assert!(abs_max(f64::NAN, f64::NAN).is_nan());
	}
}
