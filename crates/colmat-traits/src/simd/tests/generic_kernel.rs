//! A kernel written once against `SimdField` serves native, complex and
//! scalar-only types alike: the S1 contract of the design document §7a
//! ("if absent, P4–P8 run as scalar loops: correct, slower").

use super::{Check, Inspect, run_all};
use crate::math_utils::*;
use crate::test_support::{TestValue, value};
use crate::{ComplexField, SimdField, SimdLayout};
use pulp::Simd;

/// `sum_i conj(a_i) b_i` with full registers and one masked tail.
fn dot<T: ComplexField, S: Simd>(simd: S, a: &[T], b: &[T]) -> T {
	type P<T> = <T as ComplexField>::Simd;
	assert_eq!(a.len(), b.len());
	let ctx = P::<T>::ctx(simd);
	let n = P::<T>::lanes::<S>();
	let mut acc = P::<T>::splat(&ctx, &zero());
	let full = a.len() / n * n;
	for (x, y) in a[..full].chunks_exact(n).zip(b[..full].chunks_exact(n)) {
		let x = P::<T>::load(&ctx, x);
		let y = P::<T>::load(&ctx, y);
		acc = P::<T>::conj_mul_add(&ctx, x, y, acc);
	}
	let rest = a.len() - full;
	if rest > 0 {
		let mask = P::<T>::mem_mask_between(&ctx, 0, rest);
		// SAFETY: the mask enables `rest` elements, and `a[full..]` and
		// `b[full..]` hold exactly `rest` initialised elements each.
		let (x, y) = unsafe {
			(
				P::<T>::mask_load(&ctx, mask, a[full..].as_ptr()),
				P::<T>::mask_load(&ctx, mask, b[full..].as_ptr()),
			)
		};
		// Disabled lanes are unspecified: clear them before accumulating.
		let keep = P::<T>::mask_between(&ctx, 0, rest);
		let zero = P::<T>::splat(&ctx, &zero());
		let x = P::<T>::select(&ctx, keep, x, zero);
		acc = P::<T>::conj_mul_add(&ctx, x, y, acc);
	}
	P::<T>::reduce_sum(&ctx, acc)
}

#[test]
fn one_dot_kernel_for_every_scalar() {
	struct C;
	impl Check for C {
		fn run<T: TestValue, S: Simd>(simd: S)
		where
			T::Simd: Inspect<T>,
		{
			for len in 0..40 {
				let a: Vec<T> = (0..len).map(value).collect();
				let b: Vec<T> = (0..len).map(|i| value(i + 3)).collect();
				let expected = a
					.iter()
					.zip(&b)
					.fold(zero::<T>(), |acc, (x, y)| acc + conj(x) * y);
				assert_eq!(dot(simd, &a, &b), expected, "length {len}");
			}
		}
	}
	run_all::<C>();
}
