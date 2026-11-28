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
//! Both packed kernels have two entries (#34), and each [`Gemm`] method takes one:
//!
//! - [`Gemm::gemm`] calls their `gemm_with_stack` entries, with the packing memory taken from
//!   the caller's stack: one part per tile of the team split, each as large as the largest
//!   request among the grid's tile shapes ([`split_product_scratch`]). The largest tile alone
//!   is not enough, since `t4a-gemm`'s request is not monotone in the shape for `f32` (at
//!   `k = 4096`, `4 x 156` asks 5.50 MiB and `4 x 176` asks 5.25 MiB).
//!   [`Gemm::gemm_scratch`] covers every width the product may run at. On x86-64,
//!   `t4a-private-gemm-x86` asks for its L3-sized buffers whatever the shape, about twice the
//!   L3 size per tile (`gemm_stack_bytes`).
//! - [`Gemm::gemm_alloc`] calls their allocating entries, which keep the packing memory in
//!   thread-locals: `t4a-gemm` allocates an L2-sized buffer on a thread's first call and a pack
//!   buffer for every product above that size; `t4a-private-gemm-x86` allocates L3-sized
//!   buffers on a thread's first call. The allocating entries of the product
//!   ([`matmul`](super::matmul), [`matmul_with_conj`](super::matmul_with_conj)) take this one,
//!   so a small product does not pay for sizing and allocating a stack on every call, and the
//!   x86-64 request above never reaches a per-call allocation.
//!
//! Each impl names its concrete type, so a pointer is only ever handed to the kernel of the type
//! it points to: nothing is reinterpreted (debts K1, K2 of the design record; colmat-rs chose
//! these kernels through `IS_NATIVE_*` and `transmute_copy`, `faer/src/linalg/matmul/mod.rs` at
//! `71337cdc`, lines 1396-1595, whose calls are ported here; MIT, faer's notice in
//! `LICENSE-faer`).

use super::generic::{gemm_generic, split_product, split_product_scratch};
use super::{Accum, Gemm};
use crate::mat::{MatMut, MatRef};
use crate::{Conj, Par, ParShape, c32, c64};
use colmat_traits::{ComplexField, NativeKernels};
use dyn_stack::{MemStack, StackReq};

/// Products of at most this many multiply-adds run `nano-gemm` (colmat-rs's threshold).
const NANO_GEMM_THRESHOLD: usize = 16 * 16 * 16;

/// The x86-64 instruction set of `t4a-private-gemm-x86` on this host, if it has one: read from
/// [`crate::host::info`], decided once per process (#55).
#[cfg(all(target_arch = "x86_64", feature = "std"))]
fn x86_instr() -> Option<private_gemm_x86::InstrSet> {
	use crate::host::X86Kernel;
	use private_gemm_x86::InstrSet;
	match crate::host::info().x86_kernel() {
		X86Kernel::Avx512 => Some(InstrSet::Avx512),
		X86Kernel::Avx2 => Some(InstrSet::Avx256),
		X86Kernel::None => None,
	}
}

/// The scratch of one `t4a-private-gemm-x86` call on an `r x c` tile of depth `k`: its
/// [`gemm_stack_bytes`](private_gemm_x86::gemm_stack_bytes), which already counts the bytes
/// the kernel skips to align its pages, so any start will do.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
fn x86_req(
	dtype: private_gemm_x86::DType,
	instr: private_gemm_x86::InstrSet,
	r: usize,
	c: usize,
	k: usize,
) -> StackReq {
	StackReq::new::<u8>(private_gemm_x86::gemm_stack_bytes(
		dtype, instr, r, c, k,
	))
}

macro_rules! native {
	($($ty:ty => $nano:ident, $dtype:ident, $zero:expr, $one:expr;)*) => {$(
		impl Gemm<$ty> for NativeKernels {
			#[inline]
			fn ns_per_mac() -> f64 {
				crate::utils::cost::rate::gemm_mac().of::<$ty>()
			}

			/// The packing memory of one packed call per tile of the team split; empty for
			/// the `nano-gemm` range and under Miri.
			fn gemm_scratch(m: usize, n: usize, k: usize, par: ParShape) -> StackReq {
				native_scratch::<$ty>(m, n, k, par)
			}

			/// The packed kernels' `*_with_stack` entries, with their memory from `stack`.
			fn gemm(
				dst: MatMut<'_, $ty, usize, usize>,
				accum: Accum,
				lhs: MatRef<'_, $ty, usize, usize>,
				conj_lhs: Conj,
				rhs: MatRef<'_, $ty, usize, usize>,
				conj_rhs: Conj,
				alpha: &$ty,
				par: Par<'_>,
				stack: &mut MemStack,
			) {
				native_gemm(dst, accum, lhs, conj_lhs, rhs, conj_rhs, alpha, par, Some(stack))
			}

			/// The packed kernels' allocating entries, with their memory in thread-locals
			/// (module documentation, "Allocation").
			fn gemm_alloc(
				dst: MatMut<'_, $ty, usize, usize>,
				accum: Accum,
				lhs: MatRef<'_, $ty, usize, usize>,
				conj_lhs: Conj,
				rhs: MatRef<'_, $ty, usize, usize>,
				conj_rhs: Conj,
				alpha: &$ty,
				par: Par<'_>,
			) {
				native_gemm(dst, accum, lhs, conj_lhs, rhs, conj_rhs, alpha, par, None)
			}
		}

		impl Native for $ty {
			const ONE: Self = $one;
			#[cfg(all(target_arch = "x86_64", feature = "std"))]
			const DTYPE: private_gemm_x86::DType = private_gemm_x86::DType::$dtype;

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

/// [`Gemm::gemm_scratch`] of a native type.
fn native_scratch<T: Native>(
	m: usize,
	n: usize,
	k: usize,
	par: ParShape,
) -> StackReq
where
	NativeKernels: Gemm<T>,
{
	if cfg!(miri)
		|| m.saturating_mul(n).saturating_mul(k) <= NANO_GEMM_THRESHOLD
	{
		return StackReq::EMPTY;
	}
	let ns = <NativeKernels as Gemm<T>>::ns_per_mac();
	// The host probe (#55) runs once per process. Its SME and cache probes call `sysctl` on
	// macOS, which allocates: run it here, in the query, so that the product that follows
	// allocates nothing even when it is the first of the process. `t4a-gemm` reads the same
	// cached answers. After the first call this is one atomic load.
	let _ = crate::host::info();
	#[cfg(all(target_arch = "x86_64", feature = "std"))]
	if let Some(instr) = x86_instr() {
		return split_product_scratch(m, n, k, par, ns, |r, c| {
			x86_req(T::DTYPE, instr, r, c, k)
		});
	}
	split_product_scratch(m, n, k, par, ns, |r, c| {
		gemm::gemm_scratch::<T>(r, c, k, gemm::Parallelism::None)
	})
}

/// [`Gemm::gemm`] (with `Some(stack)`) and [`Gemm::gemm_alloc`] (with `None`) of a native
/// type: `nano-gemm`, or one packed call per tile of the team split.
#[allow(clippy::too_many_arguments)]
fn native_gemm<T: Native>(
	dst: MatMut<'_, T, usize, usize>,
	accum: Accum,
	lhs: MatRef<'_, T, usize, usize>,
	conj_lhs: Conj,
	rhs: MatRef<'_, T, usize, usize>,
	conj_rhs: Conj,
	alpha: &T,
	par: Par<'_>,
	stack: Option<&mut MemStack>,
) where
	NativeKernels: Gemm<T>,
{
	if cfg!(miri) {
		return gemm_generic(
			dst, accum, lhs, conj_lhs, rhs, conj_rhs, alpha, par,
		);
	}
	let (m, n) = dst.shape();
	let k = lhs.ncols();
	if m.saturating_mul(n).saturating_mul(k) <= NANO_GEMM_THRESHOLD {
		return T::nano(dst, accum, lhs, conj_lhs, rhs, conj_rhs, alpha);
	}
	let alpha = *alpha;
	let ns = <NativeKernels as Gemm<T>>::ns_per_mac();
	// the allocating entries split an empty stack: no tile takes a part of it
	let from_stack = stack.is_some();
	let mut empty = [];
	let stack = match stack {
		Some(stack) => stack,
		None => MemStack::new(&mut empty),
	};
	#[cfg(all(target_arch = "x86_64", feature = "std"))]
	if let Some(instr) = x86_instr() {
		let tile_req = |r, c| {
			if from_stack {
				x86_req(T::DTYPE, instr, r, c, k)
			} else {
				StackReq::EMPTY
			}
		};
		return split_product(
			dst,
			lhs,
			rhs,
			par,
			ns,
			tile_req,
			stack,
			|dst, lhs, rhs, stack| {
				x86_tile(
					instr,
					dst,
					accum,
					lhs,
					conj_lhs,
					rhs,
					conj_rhs,
					&alpha,
					from_stack.then_some(stack),
				)
			},
		);
	}
	let tile_req = |r, c| {
		if from_stack {
			gemm::gemm_scratch::<T>(r, c, k, gemm::Parallelism::None)
		} else {
			StackReq::EMPTY
		}
	};
	split_product(
		dst,
		lhs,
		rhs,
		par,
		ns,
		tile_req,
		stack,
		|dst, lhs, rhs, stack| {
			t4a_tile(
				dst,
				accum,
				lhs,
				conj_lhs,
				rhs,
				conj_rhs,
				alpha,
				from_stack.then_some(stack),
			)
		},
	)
}

/// One `t4a-gemm` call on one thread: `gemm_with_stack` with `Some(stack)`, `gemm` (its
/// thread-locals) with `None`.
#[allow(clippy::too_many_arguments)]
fn t4a_tile<T: Native>(
	dst: MatMut<'_, T, usize, usize>,
	accum: Accum,
	lhs: MatRef<'_, T, usize, usize>,
	conj_lhs: Conj,
	rhs: MatRef<'_, T, usize, usize>,
	conj_rhs: Conj,
	alpha: T,
	stack: Option<&mut MemStack>,
) {
	let (m, n) = dst.shape();
	let k = lhs.ncols();
	let (dst_ptr, dst_cs, dst_rs) =
		(dst.as_ptr_mut(), dst.col_stride(), dst.row_stride());
	let (lhs_ptr, lhs_cs, lhs_rs) =
		(lhs.as_ptr(), lhs.col_stride(), lhs.row_stride());
	let (rhs_ptr, rhs_cs, rhs_rs) =
		(rhs.as_ptr(), rhs.col_stride(), rhs.row_stride());
	let read_dst = accum == Accum::Add;
	let (conj_lhs, conj_rhs) = (conj_lhs.is_conj(), conj_rhs.is_conj());
	// SAFETY: the views give `m x n`, `m x k` and `k x n` matrices of `T`, each valid at every
	// in-bounds index with its strides (view invariant), and `dst` is a unique view, so it
	// overlaps neither operand. `t4a-gemm` dispatches on `T`, one of the four types it
	// implements; it takes the column stride before the row stride, computes
	// `dst ← a·dst + b·lhs·rhs` and does not read `dst` when `read_dst` is false; no
	// parallelism of its own. `gemm_with_stack` takes its packing memory from `stack` (and
	// panics, before writing `dst`, if that is short).
	unsafe {
		match stack {
			Some(stack) => gemm::gemm_with_stack(
				m,
				n,
				k,
				dst_ptr,
				dst_cs,
				dst_rs,
				read_dst,
				lhs_ptr,
				lhs_cs,
				lhs_rs,
				rhs_ptr,
				rhs_cs,
				rhs_rs,
				T::ONE,
				alpha,
				false,
				conj_lhs,
				conj_rhs,
				gemm::Parallelism::None,
				stack,
			),
			None => gemm::gemm(
				m,
				n,
				k,
				dst_ptr,
				dst_cs,
				dst_rs,
				read_dst,
				lhs_ptr,
				lhs_cs,
				lhs_rs,
				rhs_ptr,
				rhs_cs,
				rhs_rs,
				T::ONE,
				alpha,
				false,
				conj_lhs,
				conj_rhs,
				gemm::Parallelism::None,
			),
		}
	}
}

/// One `t4a-private-gemm-x86` call on one thread: `gemm_with_stack` with `Some(stack)`,
/// `gemm` (its thread-locals) with `None`.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[allow(clippy::too_many_arguments)]
fn x86_tile<T: Native>(
	instr: private_gemm_x86::InstrSet,
	dst: MatMut<'_, T, usize, usize>,
	accum: Accum,
	lhs: MatRef<'_, T, usize, usize>,
	conj_lhs: Conj,
	rhs: MatRef<'_, T, usize, usize>,
	conj_rhs: Conj,
	alpha: &T,
	stack: Option<&mut MemStack>,
) {
	use private_gemm_x86::{
		Accum as XAccum, DstKind, IType, gemm, gemm_stack_bytes,
		gemm_with_stack,
	};
	let (m, n) = dst.shape();
	let k = lhs.ncols();
	let accum = match accum {
		Accum::Replace => XAccum::Replace,
		Accum::Add => XAccum::Add,
	};
	let (dst_ptr, dst_rs, dst_cs) = (
		dst.as_ptr_mut() as *mut (),
		dst.row_stride(),
		dst.col_stride(),
	);
	let (lhs_ptr, lhs_rs, lhs_cs) = (
		lhs.as_ptr() as *const (),
		lhs.row_stride(),
		lhs.col_stride(),
	);
	let (rhs_ptr, rhs_rs, rhs_cs) = (
		rhs.as_ptr() as *const (),
		rhs.row_stride(),
		rhs.col_stride(),
	);
	let (conj_lhs, conj_rhs) = (conj_lhs.is_conj(), conj_rhs.is_conj());
	let alpha = alpha as *const T as *const ();
	// SAFETY: the views give `m x n`, `m x k` and `k x n` matrices of `T`, valid at every
	// in-bounds index with their strides (view invariant; strides are in elements, as the
	// kernel takes them); `dst` is a unique view, so it overlaps neither operand; the index and
	// diagonal pointers are core::ptr::null(), which selects the full product without them; `instr` was
	// detected on this host; `alpha` points to a `T`, the type `T::DTYPE` names; one thread.
	// `gemm_with_stack` takes `gemm_stack_bytes` bytes of `stack`, which already count the
	// bytes it skips to align its pages (so any start will do).
	unsafe {
		match stack {
			Some(stack) => {
				let bytes = gemm_stack_bytes(T::DTYPE, instr, m, n, k);
				let (stack, _) = stack.make_aligned_uninit::<u8>(bytes, 1);
				gemm_with_stack(
					T::DTYPE,
					IType::U64,
					instr,
					m,
					n,
					k,
					dst_ptr,
					dst_rs,
					dst_cs,
					core::ptr::null(),
					core::ptr::null(),
					DstKind::Full,
					accum,
					lhs_ptr,
					lhs_rs,
					lhs_cs,
					conj_lhs,
					core::ptr::null(),
					0,
					rhs_ptr,
					rhs_rs,
					rhs_cs,
					conj_rhs,
					alpha,
					stack,
				)
			},
			None => gemm(
				T::DTYPE,
				IType::U64,
				instr,
				m,
				n,
				k,
				dst_ptr,
				dst_rs,
				dst_cs,
				core::ptr::null(),
				core::ptr::null(),
				DstKind::Full,
				accum,
				lhs_ptr,
				lhs_rs,
				lhs_cs,
				conj_lhs,
				core::ptr::null(),
				0,
				rhs_ptr,
				rhs_rs,
				rhs_cs,
				conj_rhs,
				alpha,
				1,
			),
		}
	}
}

/// A native type: its `nano-gemm` call and the constants its packed kernels take.
trait Native: ComplexField + Copy + 'static {
	/// The factor `1` of `dst ← 1·dst + α·lhs·rhs`.
	const ONE: Self;
	/// The type's tag for `t4a-private-gemm-x86`.
	#[cfg(all(target_arch = "x86_64", feature = "std"))]
	const DTYPE: private_gemm_x86::DType;

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

native! {
	f64 => execute_f64, F64, 0.0, 1.0;
	f32 => execute_f32, F32, 0.0, 1.0;
	c64 => execute_c64, C64, c64::new(0.0, 0.0), c64::new(1.0, 0.0);
	c32 => execute_c32, C32, c32::new(0.0, 0.0), c32::new(1.0, 0.0);
}
