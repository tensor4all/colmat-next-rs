//! Data movement: round trips, partial masks and logical lane indices.

use super::{Check, Inspect, run_all};
use crate::test_support::{TestValue, value};
use crate::{SimdField, SimdLayout};
use pulp::Simd;

/// A full load followed by a full store copies the elements in order.
#[test]
fn load_store_round_trip() {
	struct C;
	impl Check for C {
		fn run<T: TestValue, S: Simd>(simd: S)
		where
			T::Simd: Inspect<T>,
		{
			type P<T> = <T as crate::ComplexField>::Simd;
			let ctx = P::<T>::ctx(simd);
			let n = P::<T>::lanes::<S>();
			let src: Vec<T> = (0..n).map(value).collect();
			let mut dst: Vec<T> = (0..n).map(|i| value(100 + i)).collect();
			let v = P::<T>::load(&ctx, &src);
			P::<T>::store(&ctx, &mut dst, v);
			assert_eq!(dst, src);
		}
	}
	run_all::<C>();
}

/// `iota` is a permutation of `0..lanes`, and the lane holding element `i`
/// after a load is the lane whose `iota` is `i`; `first_true_mask` finds it.
#[test]
fn iota_names_the_element_of_each_lane() {
	struct C;
	impl Check for C {
		fn run<T: TestValue, S: Simd>(simd: S)
		where
			T::Simd: Inspect<T>,
		{
			type P<T> = <T as crate::ComplexField>::Simd;
			let ctx = P::<T>::ctx(simd);
			let n = P::<T>::lanes::<S>();
			let iota = P::<T>::index_lanes::<S>(P::<T>::iota(&ctx));
			let mut sorted = iota.clone();
			sorted.sort();
			assert_eq!(sorted, (0..n).collect::<Vec<_>>());

			let src: Vec<T> = (0..n).map(value).collect();
			let v = P::<T>::load(&ctx, &src);
			for (i, x) in src.iter().enumerate() {
				let eq = P::<T>::equal(&ctx, v.clone(), P::<T>::splat(&ctx, x));
				let lanes = P::<T>::mask_lanes::<S>(&ctx, eq);
				let hits: Vec<usize> = (0..n).filter(|&p| lanes[p]).collect();
				assert_eq!(hits.len(), 1, "element {i} is in one lane");
				assert_eq!(iota[hits[0]], i, "iota names element {i}");
				assert_eq!(P::<T>::first_true_mask(&ctx, eq), hits[0]);
			}
		}
	}
	run_all::<C>();
}

/// `mask_between(start, end)` enables exactly the lanes whose element index
/// lies in `start..end`, with `end` clamped to the lane count.
#[test]
fn mask_between_follows_iota() {
	struct C;
	impl Check for C {
		fn run<T: TestValue, S: Simd>(simd: S)
		where
			T::Simd: Inspect<T>,
		{
			type P<T> = <T as crate::ComplexField>::Simd;
			let ctx = P::<T>::ctx(simd);
			let n = P::<T>::lanes::<S>();
			let iota = P::<T>::index_lanes::<S>(P::<T>::iota(&ctx));
			for start in 0..=n + 1 {
				for end in start..=n + 2 {
					let m = P::<T>::mask_between(&ctx, start, end);
					let lanes = P::<T>::mask_lanes::<S>(&ctx, m);
					for p in 0..n {
						let expected = start <= iota[p] && iota[p] < end;
						assert_eq!(
							lanes[p], expected,
							"{start}..{end}, lane {p}"
						);
					}
				}
			}
		}
	}
	run_all::<C>();
}

/// A masked load reads the enabled elements, and a masked store writes them
/// and nothing else.
#[test]
fn masked_load_and_store() {
	struct C;
	impl Check for C {
		fn run<T: TestValue, S: Simd>(simd: S)
		where
			T::Simd: Inspect<T>,
		{
			type P<T> = <T as crate::ComplexField>::Simd;
			let ctx = P::<T>::ctx(simd);
			let n = P::<T>::lanes::<S>();
			let src: Vec<T> = (0..n).map(value).collect();
			let sentinel: Vec<T> = (0..n).map(|i| value(100 + i)).collect();
			for start in 0..=n {
				for end in start..=n + 1 {
					let mask = P::<T>::mem_mask_between(&ctx, start, end);
					// SAFETY: `src` holds `lanes` initialised elements.
					let v =
						unsafe { P::<T>::mask_load(&ctx, mask, src.as_ptr()) };

					let mut full = sentinel.clone();
					P::<T>::store(&ctx, &mut full, v.clone());
					let hi = end.min(n);
					assert_eq!(
						full[start..hi],
						src[start..hi],
						"load {start}..{end}"
					);

					let mut dst = sentinel.clone();
					// SAFETY: `dst` holds `lanes` elements; the test scalars own
					// no resources, so overwriting without a drop leaks nothing.
					unsafe {
						P::<T>::mask_store(&ctx, mask, dst.as_mut_ptr(), v)
					};
					for i in 0..n {
						let expected = if start <= i && i < end {
							&src[i]
						} else {
							&sentinel[i]
						};
						assert_eq!(
							&dst[i], expected,
							"store {start}..{end}, element {i}"
						);
					}
				}
			}
		}
	}
	run_all::<C>();
}
