//! Provider tests, run for every scalar of the crate and for the test-only
//! scalar-only type, under the detected instruction set and under
//! `pulp::Scalar`.

mod arith;
mod generic_kernel;
mod layout;
mod props;

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

/// A check that runs once per instruction set, with no input.
pub(super) trait Check {
	fn run<T: TestValue, S: Simd>(simd: S)
	where
		T::Simd: Inspect<T>;
}

/// A check that runs once per instruction set on generated input.
pub(super) trait DataCheck {
	fn run<T: TestValue, S: Simd>(&self, simd: S)
	where
		T::Simd: Inspect<T>;
}

/// A [`Check`] as a [`DataCheck`] without data.
struct NoData<C>(core::marker::PhantomData<C>);
impl<C: Check> DataCheck for NoData<C> {
	fn run<T: TestValue, S: Simd>(&self, simd: S)
	where
		T::Simd: Inspect<T>,
	{
		C::run::<T, S>(simd)
	}
}

struct WithSimd<'a, C, T>(&'a C, core::marker::PhantomData<T>);
impl<C: DataCheck, T: TestValue> pulp::WithSimd for WithSimd<'_, C, T>
where
	T::Simd: Inspect<T>,
{
	type Output = ();
	fn with_simd<S: Simd>(self, simd: S) {
		self.0.run::<T, S>(simd)
	}
}

/// Runs `check` for `T` under the detected instruction set, under
/// `pulp::Scalar`, and on x86-64 also explicitly under AVX2/FMA (`V3`) when
/// the host has it, so that the V3 shuffles are exercised whatever the
/// detection picks.
pub(super) fn for_each_isa<C: DataCheck, T: TestValue>(check: &C)
where
	T::Simd: Inspect<T>,
{
	let with = || WithSimd::<C, T>(check, core::marker::PhantomData);
	pulp::Arch::new().dispatch(with());
	pulp::WithSimd::with_simd(with(), pulp::Scalar::new());
	#[cfg(target_arch = "x86_64")]
	if let Some(v3) = pulp::x86::V3::try_new() {
		pulp::WithSimd::with_simd(with(), v3);
	}
}

/// Runs `check` for every scalar under test.
pub(super) fn for_all_types<C: DataCheck>(check: &C) {
	for_each_isa::<C, f64>(check);
	for_each_isa::<C, f32>(check);
	for_each_isa::<C, Complex<f64>>(check);
	for_each_isa::<C, Complex<f32>>(check);
	for_each_isa::<C, Sf>(check);
	for_each_isa::<C, Complex<Sf>>(check);
}

/// Runs `C` for every scalar under test.
pub(super) fn run_all<C: Check>() {
	for_all_types(&NoData::<C>(core::marker::PhantomData));
}

/// On an x86-64 host with AVX2 and FMA, the explicit `V3` pass of
/// [`for_each_isa`] runs. Fails if pulp stops offering it there, which would
/// silently drop the AVX2 paths from every check.
#[cfg(target_arch = "x86_64")]
#[cfg_attr(miri, ignore)]
#[test]
fn x86_v3_runs_where_the_host_has_it() {
	if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
		assert!(pulp::x86::V3::try_new().is_some());
	}
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

/// The one-lane index operations and the provider's `Debug` output.
#[test]
fn no_simd_index_ops_and_debug() {
	type L = NoSimd;
	let s = pulp::Scalar::new();
	assert_eq!(<L as SimdLayout<f64>>::index_add(&s, 2, 3), 5);
	assert_eq!(<L as SimdLayout<f64>>::index_add(&s, usize::MAX, 2), 1);
	assert_eq!(<L as SimdLayout<f64>>::first_true_mask(&s, true), 0);
	assert_eq!(<L as SimdLayout<f64>>::first_true_mask(&s, false), 1);
	assert_eq!(
		format!("{:?}", ComplexSimd::<NoSimd>::default()),
		"ComplexSimd"
	);
}
