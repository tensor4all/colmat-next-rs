//! The provider for `f32` and `f64`, on pulp's vector types.
//!
//! Ported from the SIMD items of colmat-rs's `impl ComplexField for f32/f64`
//! (`faer-traits/src/lib.rs` at `71337cdc`, lines 1648-2579). Element loads
//! go through element pointers and slices instead of register pointers, and
//! pair loads (for `Complex<f32/f64>`) replace the `Unit::IS_NATIVE_*`
//! branches of the default load hooks (lines 1515-1623). faer's MIT notice:
//! `LICENSE-faer`.

use super::{SimdField, SimdLayout, SimdPairs};
use pulp::{MemMask, Simd};

/// The provider for `f32` and `f64`: one pulp vector per register.
#[derive(Copy, Clone, Debug, Default)]
pub struct NativeSimd;

/// `0, 1, 2, ...`: lane indices of an unsplit register.
const IOTA_64: [u64; 64] = {
	let mut out = [0u64; 64];
	let mut i = 0;
	while i < 64 {
		out[i] = i as u64;
		i += 1;
	}
	out
};
/// `0, 0, 1, 1, 2, 2, ...`: the element index of each word of a pair array.
const IOTA2_64: [u64; 64] = {
	let mut out = [0u64; 64];
	let mut i = 0;
	while i < 64 {
		out[i] = (i / 2) as u64;
		i += 1;
	}
	out
};
const IOTA_32: [u32; 64] = {
	let mut out = [0u32; 64];
	let mut i = 0;
	while i < 64 {
		out[i] = i as u32;
		i += 1;
	}
	out
};
const IOTA2_32: [u32; 64] = {
	let mut out = [0u32; 64];
	let mut i = 0;
	while i < 64 {
		out[i] = (i / 2) as u32;
		i += 1;
	}
	out
};

/// Word ranges of the two registers that hold pairs `start..end` of a pair
/// array with `lanes` pairs: the pairs occupy words `2 start..2 end` of
/// `2 lanes` words, split at `lanes`. An empty range is normalised to `0..0`.
#[inline(always)]
fn split_pair_range(
	start: usize,
	end: usize,
	lanes: usize,
) -> ((usize, usize), (usize, usize)) {
	let s = start.min(lanes);
	let e = end.min(lanes).max(s);
	let (ws, we) = (2 * s, 2 * e);
	let a = (ws.min(lanes), we.min(lanes));
	let b = (ws.max(lanes) - lanes, we.max(lanes) - lanes);
	let norm = |(s, e): (usize, usize)| if s < e { (s, e) } else { (0, 0) };
	(norm(a), norm(b))
}

macro_rules! impl_native {
	(
		$ty:ident, $index:ident, $lanes:ident,
		$v:ident, $m:ident, $u:ident,
		$iota:ident, $iota2:ident
	) => {
		paste::paste! {
		// SAFETY: one register is one pulp vector of `$lanes` values of `$ty`,
		// in memory order (`load`/`store` copy the bytes, so lane `i` holds
		// element `i`, which is what `iota` returns). Masked loads and stores
		// use pulp's masked memory operations, which access exactly the
		// enabled lanes; `mask_between` and `mem_mask_between` both enable the
		// lanes `start..end` clamped to `$lanes`.
		unsafe impl SimdLayout<$ty> for NativeSimd {
			const IS_SIMD: bool = true;
			type Arch = pulp::Arch;
			type Ctx<S: Simd> = S;
			type Vec<S: Simd> = S::[<$v>];
			type Mask<S: Simd> = S::[<$m>];
			type MemMask<S: Simd> = MemMask<S::[<$m>]>;
			type Index = $index;
			type IndexVec<S: Simd> = S::[<$u>];

			#[inline(always)]
			fn ctx<S: Simd>(simd: S) -> S {
				simd
			}

			#[inline(always)]
			fn simd<S: Simd>(ctx: &S) -> S {
				*ctx
			}

			#[inline(always)]
			fn lanes<S: Simd>() -> usize {
				S::$lanes
			}

			#[inline(always)]
			fn mask_between<S: Simd>(
				ctx: &S,
				start: usize,
				end: usize,
			) -> S::[<$m>] {
				<Self as SimdLayout<$ty>>::mem_mask_between(ctx, start, end).mask()
			}

			#[inline(always)]
			fn mem_mask_between<S: Simd>(
				ctx: &S,
				start: usize,
				end: usize,
			) -> MemMask<S::[<$m>]> {
				let lanes = S::$lanes;
				let start = start.min(lanes) as $index;
				let end = end.min(lanes) as $index;
				ctx.[<mask_between_ $m>](start, end)
			}

			#[inline(always)]
			unsafe fn mask_load<S: Simd>(
				ctx: &S,
				mask: MemMask<S::[<$m>]>,
				ptr: *const $ty,
			) -> S::[<$v>] {
				// SAFETY: forwarded from the caller: every enabled lane's
				// address is valid for reads.
				unsafe { ctx.[<mask_load_ptr_ $v>](mask, ptr) }
			}

			#[inline(always)]
			unsafe fn mask_store<S: Simd>(
				ctx: &S,
				mask: MemMask<S::[<$m>]>,
				ptr: *mut $ty,
				value: S::[<$v>],
			) {
				// SAFETY: forwarded from the caller: every enabled lane's
				// address is valid for writes.
				unsafe { ctx.[<mask_store_ptr_ $v>](mask, ptr, value) }
			}

			#[inline(always)]
			fn load<S: Simd>(_: &S, src: &[$ty]) -> S::[<$v>] {
				assert_eq!(src.len(), S::$lanes, "one register of elements");
				bytemuck::pod_read_unaligned(bytemuck::cast_slice(src))
			}

			#[inline(always)]
			fn store<S: Simd>(_: &S, dst: &mut [$ty], value: S::[<$v>]) {
				assert_eq!(dst.len(), S::$lanes, "one register of elements");
				bytemuck::cast_slice_mut(dst)
					.copy_from_slice(bytemuck::bytes_of(&value));
			}

			#[inline(always)]
			fn splat<S: Simd>(ctx: &S, value: &$ty) -> S::[<$v>] {
				ctx.[<splat_ $v>](*value)
			}

			#[inline(always)]
			fn splat_real<S: Simd>(ctx: &S, value: &$ty) -> S::[<$v>] {
				ctx.[<splat_ $v>](*value)
			}

			#[inline(always)]
			fn select<S: Simd>(
				ctx: &S,
				mask: S::[<$m>],
				lhs: S::[<$v>],
				rhs: S::[<$v>],
			) -> S::[<$v>] {
				ctx.[<select_ $v>](mask, lhs, rhs)
			}

			#[inline(always)]
			fn iota<S: Simd>(ctx: &S) -> S::[<$u>] {
				ctx.[<partial_load_ $u>](&$iota[..S::$lanes])
			}

			#[inline(always)]
			fn index_select<S: Simd>(
				ctx: &S,
				mask: S::[<$m>],
				lhs: S::[<$u>],
				rhs: S::[<$u>],
			) -> S::[<$u>] {
				ctx.[<select_ $u>](mask, lhs, rhs)
			}

			#[inline(always)]
			fn index_splat<S: Simd>(ctx: &S, value: $index) -> S::[<$u>] {
				ctx.[<splat_ $u>](value)
			}

			#[inline(always)]
			fn index_add<S: Simd>(
				ctx: &S,
				lhs: S::[<$u>],
				rhs: S::[<$u>],
			) -> S::[<$u>] {
				ctx.[<add_ $u>](lhs, rhs)
			}

			#[inline(always)]
			fn index_less_than<S: Simd>(
				ctx: &S,
				lhs: S::[<$u>],
				rhs: S::[<$u>],
			) -> S::[<$m>] {
				ctx.[<less_than_ $u>](lhs, rhs)
			}

			#[inline(always)]
			fn and_mask<S: Simd>(
				ctx: &S,
				lhs: S::[<$m>],
				rhs: S::[<$m>],
			) -> S::[<$m>] {
				ctx.[<and_ $m>](lhs, rhs)
			}

			#[inline(always)]
			fn or_mask<S: Simd>(
				ctx: &S,
				lhs: S::[<$m>],
				rhs: S::[<$m>],
			) -> S::[<$m>] {
				ctx.[<or_ $m>](lhs, rhs)
			}

			#[inline(always)]
			fn not_mask<S: Simd>(ctx: &S, mask: S::[<$m>]) -> S::[<$m>] {
				ctx.[<not_ $m>](mask)
			}

			#[inline(always)]
			fn first_true_mask<S: Simd>(ctx: &S, mask: S::[<$m>]) -> usize {
				ctx.[<first_true_ $m>](mask)
			}
		}

		impl SimdField<$ty> for NativeSimd {
			#[inline(always)]
			fn add<S: Simd>(
				ctx: &S,
				lhs: S::[<$v>],
				rhs: S::[<$v>],
			) -> S::[<$v>] {
				ctx.[<add_ $v>](lhs, rhs)
			}

			#[inline(always)]
			fn sub<S: Simd>(
				ctx: &S,
				lhs: S::[<$v>],
				rhs: S::[<$v>],
			) -> S::[<$v>] {
				ctx.[<sub_ $v>](lhs, rhs)
			}

			#[inline(always)]
			fn neg<S: Simd>(ctx: &S, value: S::[<$v>]) -> S::[<$v>] {
				ctx.[<neg_ $v>](value)
			}

			#[inline(always)]
			fn conj<S: Simd>(_: &S, value: S::[<$v>]) -> S::[<$v>] {
				value
			}

			#[inline(always)]
			fn abs1<S: Simd>(ctx: &S, value: S::[<$v>]) -> S::[<$v>] {
				ctx.[<abs_ $v>](value)
			}

			#[inline(always)]
			fn abs_max<S: Simd>(ctx: &S, value: S::[<$v>]) -> S::[<$v>] {
				ctx.[<abs_ $v>](value)
			}

			#[inline(always)]
			fn mul_real<S: Simd>(
				ctx: &S,
				lhs: S::[<$v>],
				real_rhs: S::[<$v>],
			) -> S::[<$v>] {
				ctx.[<mul_ $v>](lhs, real_rhs)
			}

			#[inline(always)]
			fn mul_pow2<S: Simd>(
				ctx: &S,
				lhs: S::[<$v>],
				real_rhs: S::[<$v>],
			) -> S::[<$v>] {
				ctx.[<mul_ $v>](lhs, real_rhs)
			}

			#[inline(always)]
			fn mul<S: Simd>(
				ctx: &S,
				lhs: S::[<$v>],
				rhs: S::[<$v>],
			) -> S::[<$v>] {
				ctx.[<mul_ $v>](lhs, rhs)
			}

			#[inline(always)]
			fn conj_mul<S: Simd>(
				ctx: &S,
				lhs: S::[<$v>],
				rhs: S::[<$v>],
			) -> S::[<$v>] {
				ctx.[<mul_ $v>](lhs, rhs)
			}

			#[inline(always)]
			fn mul_add<S: Simd>(
				ctx: &S,
				lhs: S::[<$v>],
				rhs: S::[<$v>],
				acc: S::[<$v>],
			) -> S::[<$v>] {
				ctx.[<mul_add_e_ $v>](lhs, rhs, acc)
			}

			#[inline(always)]
			fn conj_mul_add<S: Simd>(
				ctx: &S,
				lhs: S::[<$v>],
				rhs: S::[<$v>],
				acc: S::[<$v>],
			) -> S::[<$v>] {
				ctx.[<mul_add_e_ $v>](lhs, rhs, acc)
			}

			#[inline(always)]
			fn abs2<S: Simd>(ctx: &S, value: S::[<$v>]) -> S::[<$v>] {
				ctx.[<mul_ $v>](value, value)
			}

			#[inline(always)]
			fn abs2_add<S: Simd>(
				ctx: &S,
				value: S::[<$v>],
				acc: S::[<$v>],
			) -> S::[<$v>] {
				ctx.[<mul_add_e_ $v>](value, value, acc)
			}

			#[inline(always)]
			fn reduce_sum<S: Simd>(ctx: &S, value: S::[<$v>]) -> $ty {
				ctx.[<reduce_sum_ $v>](value)
			}

			#[inline(always)]
			fn reduce_max<S: Simd>(ctx: &S, value: S::[<$v>]) -> $ty {
				ctx.[<reduce_max_ $v>](value)
			}

			#[inline(always)]
			fn equal<S: Simd>(
				ctx: &S,
				lhs: S::[<$v>],
				rhs: S::[<$v>],
			) -> S::[<$m>] {
				ctx.[<equal_ $v>](lhs, rhs)
			}

			#[inline(always)]
			fn less_than<S: Simd>(
				ctx: &S,
				lhs: S::[<$v>],
				rhs: S::[<$v>],
			) -> S::[<$m>] {
				ctx.[<less_than_ $v>](lhs, rhs)
			}

			#[inline(always)]
			fn less_than_or_equal<S: Simd>(
				ctx: &S,
				lhs: S::[<$v>],
				rhs: S::[<$v>],
			) -> S::[<$m>] {
				ctx.[<less_than_or_equal_ $v>](lhs, rhs)
			}

			#[inline(always)]
			fn greater_than<S: Simd>(
				ctx: &S,
				lhs: S::[<$v>],
				rhs: S::[<$v>],
			) -> S::[<$m>] {
				ctx.[<greater_than_ $v>](lhs, rhs)
			}

			#[inline(always)]
			fn greater_than_or_equal<S: Simd>(
				ctx: &S,
				lhs: S::[<$v>],
				rhs: S::[<$v>],
			) -> S::[<$m>] {
				ctx.[<greater_than_or_equal_ $v>](lhs, rhs)
			}
		}

		// SAFETY: a pair register is the two vectors read from `2 lanes`
		// consecutive values and deinterleaved, so lane `l` of both registers
		// comes from the same pair; `iota2` applies the same deinterleave to
		// the pair index of every word, so it names that pair whatever lane
		// order the instruction set's shuffle produces. Masked pair loads and
		// stores enable exactly the words of the enabled pairs
		// (`split_pair_range`) and access them with pulp's masked operations.
		unsafe impl SimdPairs<$ty> for NativeSimd {
			type MemMask2<S: Simd> = (MemMask<S::[<$m>]>, MemMask<S::[<$m>]>);

			#[inline(always)]
			fn mem_mask2_between<S: Simd>(
				ctx: &S,
				start: usize,
				end: usize,
			) -> Self::MemMask2<S> {
				let ((sa, ea), (sb, eb)) =
					split_pair_range(start, end, S::$lanes);
				(
					ctx.[<mask_between_ $m>](sa as $index, ea as $index),
					ctx.[<mask_between_ $m>](sb as $index, eb as $index),
				)
			}

			#[inline(always)]
			unsafe fn mask_load2<S: Simd>(
				ctx: &S,
				mask: Self::MemMask2<S>,
				ptr: *const [$ty; 2],
			) -> (S::[<$v>], S::[<$v>]) {
				let ptr = ptr as *const $ty;
				// The second half may lie past the allocation when only a few
				// pairs are enabled, so its address is formed with
				// `wrapping_add` and dereferenced only at enabled lanes.
				// SAFETY: the enabled words of the first register are words
				// `0..lanes` of the enabled pairs, those of the second are
				// words `lanes..2 lanes`; the caller guarantees those pairs
				// are readable.
				let words: [S::[<$v>]; 2] = unsafe {
					[
						ctx.[<mask_load_ptr_ $v>](mask.0, ptr),
						ctx.[<mask_load_ptr_ $v>](mask.1, ptr.wrapping_add(S::$lanes)),
					]
				};
				let [first, second] = ctx.[<deinterleave_shfl_ $v>](words);
				(first, second)
			}

			#[inline(always)]
			unsafe fn mask_store2<S: Simd>(
				ctx: &S,
				mask: Self::MemMask2<S>,
				ptr: *mut [$ty; 2],
				first: S::[<$v>],
				second: S::[<$v>],
			) {
				let ptr = ptr as *mut $ty;
				let [a, b] = ctx.[<interleave_shfl_ $v>]([first, second]);
				// SAFETY: as in `mask_load2`, for writes (and the same
				// `wrapping_add` for the second half).
				unsafe {
					ctx.[<mask_store_ptr_ $v>](mask.0, ptr, a);
					ctx.[<mask_store_ptr_ $v>](mask.1, ptr.wrapping_add(S::$lanes), b);
				}
			}

			#[inline(always)]
			fn load2<S: Simd>(
				ctx: &S,
				src: &[[$ty; 2]],
			) -> (S::[<$v>], S::[<$v>]) {
				assert_eq!(src.len(), S::$lanes, "one register of pairs");
				let words: [S::[<$v>]; 2] = bytemuck::pod_read_unaligned(
					bytemuck::cast_slice(src),
				);
				let [first, second] = ctx.[<deinterleave_shfl_ $v>](words);
				(first, second)
			}

			#[inline(always)]
			fn store2<S: Simd>(
				ctx: &S,
				dst: &mut [[$ty; 2]],
				first: S::[<$v>],
				second: S::[<$v>],
			) {
				assert_eq!(dst.len(), S::$lanes, "one register of pairs");
				let words: [S::[<$v>]; 2] =
					ctx.[<interleave_shfl_ $v>]([first, second]);
				bytemuck::cast_slice_mut(dst)
					.copy_from_slice(bytemuck::bytes_of(&words));
			}

			#[inline(always)]
			fn iota2<S: Simd>(ctx: &S) -> S::[<$u>] {
				let n = S::$lanes;
				let words: [S::[<$u>]; 2] = [
					ctx.[<partial_load_ $u>](&$iota2[..n]),
					ctx.[<partial_load_ $u>](&$iota2[n..2 * n]),
				];
				// The same shuffle as `load2`, so each lane names its pair.
				let [first, _] = ctx.[<deinterleave_shfl_ $v>](words);
				first
			}
		}
		}
	};
}

impl_native!(f64, u64, F64_LANES, f64s, m64s, u64s, IOTA_64, IOTA2_64);
impl_native!(f32, u32, F32_LANES, f32s, m32s, u32s, IOTA_32, IOTA2_32);
