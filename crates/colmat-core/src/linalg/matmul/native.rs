//! The native kernels of `f32`, `f64`, `c32` and `c64` ([`NativeKernels`]).
//!
//! - A product of at most `16^3` multiply-adds runs `nano-gemm`'s unrolled kernels, which take
//!   any strides and allocate nothing.
//! - A larger one runs a packed kernel, one call per participant with no parallelism of its own
//!   (a team splits the output first, [`split_product`]): `t4a-private-gemm-x86` on x86-64 with
//!   AVX2 and FMA or AVX-512F, `t4a-gemm` everywhere else (NEON on aarch64, and SME on Apple M4
//!   and later with the `apple-sme` feature, both selected by `t4a-gemm` at run time).
//! - Under Miri, which cannot run their assembly and intrinsics, the generic kernels stand in.
//!
//! # Allocation
//!
//! Both packed kernels keep their packing memory in thread-locals: `t4a-gemm` allocates an
//! L2-sized buffer on a thread's first call and a pack buffer for every product above that size;
//! `t4a-private-gemm-x86` allocates L3-sized buffers on a thread's first call. Neither takes
//! scratch from the caller yet (tensor4all/t4a-gemm#21, tensor4all/t4a-private-gemm-x86#14), so
//! [`NativeKernels`] is the exception the [`Gemm`] contract allows, and [`Gemm::gemm_scratch`]
//! is empty.
//!
//! Each impl names its concrete type, so a pointer is only ever handed to the kernel of the type
//! it points to: nothing is reinterpreted (debts K1, K2 of the design record; colmat-rs chose
//! these kernels through `IS_NATIVE_*` and `transmute_copy`, `faer/src/linalg/matmul/mod.rs` at
//! `71337cdc`, lines 1396-1595, whose calls are ported here; MIT, faer's notice in
//! `LICENSE-faer`).

use super::generic::{gemm_generic, split_product};
use super::{Accum, Gemm};
use crate::mat::{MatMut, MatRef};
use crate::{Conj, Par, ParShape, c32, c64};
use colmat_traits::NativeKernels;
use dyn_stack::{MemStack, StackReq};

/// Products of at most this many multiply-adds run `nano-gemm` (colmat-rs's threshold).
const NANO_GEMM_THRESHOLD: usize = 16 * 16 * 16;

/// The x86-64 instruction set of `t4a-private-gemm-x86` on this host, if it has one.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
fn x86_instr() -> Option<private_gemm_x86::InstrSet> {
	use private_gemm_x86::InstrSet;
	// `is_x86_feature_detected!` caches its answer; this runs once per product, at its entry.
	if std::arch::is_x86_feature_detected!("avx512f") {
		Some(InstrSet::Avx512)
	} else if std::arch::is_x86_feature_detected!("avx2")
		&& std::arch::is_x86_feature_detected!("fma")
	{
		Some(InstrSet::Avx256)
	} else {
		None
	}
}

macro_rules! native {
	($($ty:ty => $nano:ident, $dtype:ident, $zero:expr, $one:expr;)*) => {$(
		impl Gemm<$ty> for NativeKernels {
			#[inline]
			fn ns_per_mac() -> f64 {
				crate::utils::cost::rate::gemm_mac().of::<$ty>()
			}

			/// Empty: the packed kernels take their memory from thread-locals (module
			/// documentation, "Allocation").
			#[inline]
			fn gemm_scratch(_: usize, _: usize, _: usize, _: ParShape) -> StackReq {
				StackReq::EMPTY
			}

			fn gemm(
				dst: MatMut<'_, $ty, usize, usize>,
				accum: Accum,
				lhs: MatRef<'_, $ty, usize, usize>,
				conj_lhs: Conj,
				rhs: MatRef<'_, $ty, usize, usize>,
				conj_rhs: Conj,
				alpha: &$ty,
				par: Par<'_>,
				_: &mut MemStack,
			) {
				if cfg!(miri) {
					return gemm_generic(dst, accum, lhs, conj_lhs, rhs, conj_rhs, alpha, par);
				}
				let (m, n) = dst.shape();
				let k = lhs.ncols();
				if m.saturating_mul(n).saturating_mul(k) <= NANO_GEMM_THRESHOLD {
					return nano(dst, accum, lhs, conj_lhs, rhs, conj_rhs, alpha);
				}
				let alpha = *alpha;
				#[cfg(all(target_arch = "x86_64", feature = "std"))]
				if let Some(instr) = x86_instr() {
					let ns = <Self as Gemm<$ty>>::ns_per_mac();
					return split_product(dst, lhs, rhs, par, ns, |dst, lhs, rhs| {
						use private_gemm_x86::{gemm, Accum as XAccum, DType, DstKind, IType};
						let (m, n) = dst.shape();
						// SAFETY: the views give `m x n`, `m x k` and `k x n` matrices of `$ty`,
						// valid at every in-bounds index with their strides (view invariant;
						// strides are in elements, as the kernel takes them); `dst` is a unique
						// view, so it overlaps neither operand; the index and diagonal pointers
						// are null, which selects the full product without them; `instr` was
						// detected on this host; `alpha` points to a `$ty`, the type `DType`
						// names; one thread.
						unsafe {
							gemm(
								DType::$dtype,
								IType::U64,
								instr,
								m,
								n,
								lhs.ncols(),
								dst.as_ptr_mut() as *mut (),
								dst.row_stride(),
								dst.col_stride(),
								core::ptr::null(),
								core::ptr::null(),
								DstKind::Full,
								match accum {
									Accum::Replace => XAccum::Replace,
									Accum::Add => XAccum::Add,
								},
								lhs.as_ptr() as *const (),
								lhs.row_stride(),
								lhs.col_stride(),
								conj_lhs.is_conj(),
								core::ptr::null(),
								0,
								rhs.as_ptr() as *const (),
								rhs.row_stride(),
								rhs.col_stride(),
								conj_rhs.is_conj(),
								&raw const alpha as *const (),
								1,
							)
						}
					});
				}
				let ns = <Self as Gemm<$ty>>::ns_per_mac();
				split_product(dst, lhs, rhs, par, ns, |dst, lhs, rhs| {
					let (m, n) = dst.shape();
					// SAFETY: as for `nano-gemm` below; `gemm` takes the column stride before
					// the row stride, computes `dst ← a·dst + b·lhs·rhs` and does not read `dst`
					// when `read_dst` is false; no parallelism of its own.
					unsafe {
						gemm::gemm(
							m,
							n,
							lhs.ncols(),
							dst.as_ptr_mut(),
							dst.col_stride(),
							dst.row_stride(),
							accum == Accum::Add,
							lhs.as_ptr(),
							lhs.col_stride(),
							lhs.row_stride(),
							rhs.as_ptr(),
							rhs.col_stride(),
							rhs.row_stride(),
							$one,
							alpha,
							false,
							conj_lhs.is_conj(),
							conj_rhs.is_conj(),
							gemm::Parallelism::None,
						)
					}
				})
			}
		}

		impl NanoGemm for $ty {
			#[inline]
			fn nano(
				dst: MatMut<'_, $ty, usize, usize>,
				accum: Accum,
				lhs: MatRef<'_, $ty, usize, usize>,
				conj_lhs: Conj,
				rhs: MatRef<'_, $ty, usize, usize>,
				conj_rhs: Conj,
				alpha: &$ty,
			) {
				let (m, n) = dst.shape();
				// `nano-gemm` computes `dst ← a·dst + b·lhs·rhs`; `a = 0` for `Replace`, which
				// it does not read then (the reference tests start `dst` as NaN).
				let a: $ty = match accum {
					Accum::Replace => $zero,
					Accum::Add => $one,
				};
				// SAFETY: the views give `m x n`, `m x k` and `k x n` matrices of `$ty`, each valid
				// at every in-bounds index with its strides (view invariant), and `dst` is a
				// unique view, so it overlaps neither operand. `execute_*` reads and writes
				// exactly those elements (`nano-gemm` 0.2.2, `planless`).
				unsafe {
					nano_gemm::planless::$nano(
						m,
						n,
						lhs.ncols(),
						dst.as_ptr_mut(),
						dst.row_stride(),
						dst.col_stride(),
						lhs.as_ptr(),
						lhs.row_stride(),
						lhs.col_stride(),
						rhs.as_ptr(),
						rhs.row_stride(),
						rhs.col_stride(),
						a,
						*alpha,
						conj_lhs.is_conj(),
						conj_rhs.is_conj(),
					)
				}
			}
		}
	)*};
}

/// The `nano-gemm` call of a native type.
trait NanoGemm: Sized {
	#[allow(clippy::too_many_arguments)]
	fn nano(
		dst: MatMut<'_, Self, usize, usize>,
		accum: Accum,
		lhs: MatRef<'_, Self, usize, usize>,
		conj_lhs: Conj,
		rhs: MatRef<'_, Self, usize, usize>,
		conj_rhs: Conj,
		alpha: &Self,
	);
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn nano<T: NanoGemm>(
	dst: MatMut<'_, T, usize, usize>,
	accum: Accum,
	lhs: MatRef<'_, T, usize, usize>,
	conj_lhs: Conj,
	rhs: MatRef<'_, T, usize, usize>,
	conj_rhs: Conj,
	alpha: &T,
) {
	T::nano(dst, accum, lhs, conj_lhs, rhs, conj_rhs, alpha)
}

native! {
	f64 => execute_f64, F64, 0.0, 1.0;
	f32 => execute_f32, F32, 0.0, 1.0;
	c64 => execute_c64, C64, c64::new(0.0, 0.0), c64::new(1.0, 0.0);
	c32 => execute_c32, C32, c32::new(0.0, 0.0), c32::new(1.0, 0.0);
}
