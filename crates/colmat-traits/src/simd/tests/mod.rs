//! Provider tests, run for every scalar of the crate and for the test-only
//! scalar-only type, under the detected instruction set and under
//! `pulp::Scalar`.

mod arith;
mod generic_kernel;
mod layout;

use crate::test_support::{Sf, TestValue};
use crate::{
	Complex, ComplexField, ComplexSimd, NativeSimd, NoSimd, RealField,
	SimdField, SimdLayout, SimdPairs,
};
use pulp::Simd;

/// Reads the lanes of index and mask registers in physical order, so that
/// tests can check them against `iota`.
pub(super) trait Inspect<T: ComplexField>: SimdField<T> {
	fn index_lanes<S: Simd>(v: Self::IndexVec<S>) -> Vec<usize>;

	fn mask_lanes<S: Simd>(ctx: &Self::Ctx<S>, m: Self::Mask<S>) -> Vec<bool> {
		let one = Self::index_splat(ctx, Self::index_one());
		let zero = Self::index_splat(ctx, Self::index_zero());
		Self::index_lanes::<S>(Self::index_select(ctx, m, one, zero))
			.into_iter()
			.map(|x| x != 0)
			.collect()
	}

	fn index_one() -> Self::Index;
	fn index_zero() -> Self::Index;
}

macro_rules! inspect_native {
	($ty:ty, $index:ty) => {
		impl Inspect<$ty> for NativeSimd {
			fn index_lanes<S: Simd>(v: Self::IndexVec<S>) -> Vec<usize> {
				bytemuck::cast_slice::<u8, $index>(bytemuck::bytes_of(&v))
					.iter()
					.map(|&x| x as usize)
					.collect()
			}
			fn index_one() -> $index {
				1
			}
			fn index_zero() -> $index {
				0
			}
		}
	};
}
inspect_native!(f64, u64);
inspect_native!(f32, u32);

impl<T: ComplexField> Inspect<T> for NoSimd {
	fn index_lanes<S: Simd>(v: usize) -> Vec<usize> {
		vec![v]
	}
	fn index_one() -> usize {
		1
	}
	fn index_zero() -> usize {
		0
	}
}

impl<T, P> Inspect<Complex<T>> for ComplexSimd<P>
where
	T: RealField,
	P: SimdPairs<T> + Inspect<T>,
{
	fn index_lanes<S: Simd>(v: Self::IndexVec<S>) -> Vec<usize> {
		P::index_lanes::<S>(v)
	}
	fn index_one() -> Self::Index {
		P::index_one()
	}
	fn index_zero() -> Self::Index {
		P::index_zero()
	}
}

/// A check that runs once per instruction set.
pub(super) trait Check {
	fn run<T: TestValue, S: Simd>(simd: S)
	where
		T::Simd: Inspect<T>;
}

struct WithSimd<C, T>(core::marker::PhantomData<(C, T)>);
impl<C: Check, T: TestValue> pulp::WithSimd for WithSimd<C, T>
where
	T::Simd: Inspect<T>,
{
	type Output = ();
	fn with_simd<S: Simd>(self, simd: S) {
		C::run::<T, S>(simd)
	}
}

/// Runs `C` for `T` under the detected instruction set and under
/// `pulp::Scalar`.
pub(super) fn run_check<C: Check, T: TestValue>()
where
	T::Simd: Inspect<T>,
{
	pulp::Arch::new().dispatch(WithSimd::<C, T>(core::marker::PhantomData));
	pulp::WithSimd::with_simd(
		WithSimd::<C, T>(core::marker::PhantomData),
		pulp::Scalar::new(),
	);
}

/// Runs `C` for every scalar under test.
pub(super) fn run_all<C: Check>() {
	run_check::<C, f64>();
	run_check::<C, f32>();
	run_check::<C, Complex<f64>>();
	run_check::<C, Complex<f32>>();
	run_check::<C, Sf>();
	run_check::<C, Complex<Sf>>();
}

/// The provider's lane count is part of its layout, not of the type's size.
#[test]
fn providers_report_lanes() {
	struct Lanes;
	impl Check for Lanes {
		fn run<T: TestValue, S: Simd>(_: S)
		where
			T::Simd: Inspect<T>,
		{
			let lanes = <T::Simd as SimdLayout<T>>::lanes::<S>();
			assert!(lanes >= 1);
			assert_eq!(
				lanes == 1 && !<T::Simd as SimdLayout<T>>::IS_SIMD,
				!<T::Simd as SimdLayout<T>>::IS_SIMD,
			);
		}
	}
	run_all::<Lanes>();
	const { assert!(!<NoSimd as SimdLayout<Sf>>::IS_SIMD) };
	const { assert!(<NativeSimd as SimdLayout<f64>>::IS_SIMD) };
}
