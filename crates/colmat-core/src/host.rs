//! What this host can do, decided once per process and read by the kernels and the cost model
//! (#55).
//!
//! Before this module, the kernel entry (`linalg::matmul::native`) and the cost model
//! ([`HostClass::detect`]) each ran their own feature ladder. Nothing made them agree, and a
//! caller could not ask what the library had decided. Now both read [`info`], which a benchmark
//! or a tuning harness can also print (`{:?}`).
//!
//! The SME answers come from `t4a-gemm`'s own probe (`HasAppleSme`), and the cache sizes from
//! `t4a-gemm`'s own cache probe, so this module adds no probe of its own beyond the standard
//! library's feature detection. `t4a-private-gemm-x86` probes its caches privately and does not
//! expose them: its sizes are not recorded here.
//!
//! ```
//! let host = colmat_core::host::info();
//! println!("{host:?}");
//! assert_eq!(host.class, colmat_core::utils::cost::HostClass::detect());
//! ```

use crate::utils::cost::HostClass;

/// The instruction set that the x86-64 packed products give `t4a-private-gemm-x86`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum X86Kernel {
	/// AVX-512F (`InstrSet::Avx512`).
	Avx512,
	/// AVX2 with FMA, without AVX-512F (`InstrSet::Avx256`).
	Avx2,
	/// Neither, or not an x86-64 host: the packed products take `t4a-gemm` instead.
	None,
}

/// What this host can do, as the kernels and the cost model of this process see it.
///
/// Read it with [`info`]. Every field is decided once, at the first call. With `std`, the
/// features are detected at run time. Without `std`, they come from the target features of the
/// build.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct HostInfo {
	/// x86-64: AVX2 and FMA are both available.
	pub avx2_fma: bool,
	/// x86-64: AVX-512F is available.
	pub avx512f: bool,
	/// aarch64: NEON is available.
	pub neon: bool,
	/// aarch64: `FEAT_FCMA` (the complex multiply-add that `t4a-gemm`'s complex NEON kernels
	/// use) is available.
	pub fcma: bool,
	/// Apple aarch64 with the `apple-sme` feature: `t4a-gemm` may take its SME kernels for
	/// `f64` and `c64` (`HasAppleSme::get`). False elsewhere and under Miri.
	pub sme_f64: bool,
	/// As [`sme_f64`](Self::sme_f64), for `f32` and `c32` (`HasAppleSme::get_f32`).
	pub sme_f32: bool,
	/// The L1, L2 and L3 sizes in bytes that `t4a-gemm` blocks for (its `CACHE_INFO`, which
	/// falls back to built-in sizes when the probe fails). A level the host does not report is 0
	/// (no L3 on an Apple M1 Pro). `None` without `std` and under Miri,
	/// where the probe is not run. `t4a-private-gemm-x86` probes its own sizes, which are not
	/// exposed and may differ.
	pub gemm_caches: Option<[usize; 3]>,
	/// The class of the cost model's rate table, derived from the fields above.
	pub class: HostClass,
}

impl HostInfo {
	/// The instruction set that the x86-64 packed products give `t4a-private-gemm-x86`.
	pub const fn x86_kernel(&self) -> X86Kernel {
		if !cfg!(target_arch = "x86_64") {
			X86Kernel::None
		} else if self.avx512f {
			X86Kernel::Avx512
		} else if self.avx2_fma {
			X86Kernel::Avx2
		} else {
			X86Kernel::None
		}
	}

	/// The fields that the target features of the build fix, with no cache sizes. This is
	/// [`info`] without `std`.
	const fn from_build() -> Self {
		Self::with_class(HostInfo {
			avx2_fma: cfg!(all(
				target_feature = "avx2",
				target_feature = "fma"
			)),
			avx512f: cfg!(target_feature = "avx512f"),
			neon: cfg!(target_feature = "neon"),
			fcma: cfg!(target_feature = "fcma"),
			sme_f64: false,
			sme_f32: false,
			gemm_caches: None,
			class: HostClass::Generic,
		})
	}

	/// `info` with [`class`](Self::class) derived from its features.
	const fn with_class(mut self) -> Self {
		self.class = if cfg!(target_arch = "x86_64") {
			if self.avx512f {
				HostClass::X86_64Avx512
			} else if self.avx2_fma {
				HostClass::X86_64Avx2
			} else {
				HostClass::X86_64Scalar
			}
		} else if cfg!(all(target_arch = "aarch64", target_os = "macos")) {
			HostClass::Aarch64Apple
		} else if cfg!(target_arch = "aarch64") {
			HostClass::Aarch64
		} else {
			HostClass::Generic
		};
		self
	}

	/// The run time probe: the standard library's feature detection, `t4a-gemm`'s SME probe and
	/// its cache probe. The last two may allocate (`sysctl` on macOS), once per process.
	#[cfg(feature = "std")]
	fn probe() -> Self {
		let mut h = Self::from_build();
		#[cfg(target_arch = "x86_64")]
		{
			h.avx2_fma = std::arch::is_x86_feature_detected!("avx2")
				&& std::arch::is_x86_feature_detected!("fma");
			h.avx512f = std::arch::is_x86_feature_detected!("avx512f");
		}
		#[cfg(target_arch = "aarch64")]
		{
			h.neon = std::arch::is_aarch64_feature_detected!("neon");
			h.fcma = std::arch::is_aarch64_feature_detected!("fcma");
		}
		#[cfg(all(
			target_arch = "aarch64",
			target_vendor = "apple",
			feature = "apple-sme",
			not(miri)
		))]
		{
			use gemm_common::cache::HasAppleSme;
			h.sme_f64 = HasAppleSme::get();
			h.sme_f32 = HasAppleSme::get_f32();
		}
		#[cfg(not(miri))]
		{
			let c = &*gemm_common::cache::CACHE_INFO;
			h.gemm_caches =
				Some([c[0].cache_bytes, c[1].cache_bytes, c[2].cache_bytes]);
		}
		h.with_class()
	}
}

/// What this host can do: probed at the first call, then read from a static. Later calls
/// neither probe nor allocate (`tests/no_alloc.rs`).
pub fn info() -> &'static HostInfo {
	#[cfg(feature = "std")]
	{
		static INFO: std::sync::OnceLock<HostInfo> = std::sync::OnceLock::new();
		INFO.get_or_init(HostInfo::probe)
	}
	#[cfg(not(feature = "std"))]
	{
		static INFO: HostInfo = HostInfo::from_build();
		&INFO
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn the_answer_is_decided_once() {
		let first = info();
		for _ in 0..4 {
			assert!(core::ptr::eq(info(), first));
		}
		assert_eq!(*first, first.with_class());
	}

	#[cfg(all(target_arch = "x86_64", feature = "std"))]
	#[test]
	fn x86_features_are_the_standard_librarys() {
		let h = info();
		assert_eq!(h.avx512f, std::arch::is_x86_feature_detected!("avx512f"));
		assert_eq!(
			h.avx2_fma,
			std::arch::is_x86_feature_detected!("avx2")
				&& std::arch::is_x86_feature_detected!("fma")
		);
		assert!(!h.neon && !h.fcma && !h.sme_f64 && !h.sme_f32);
	}

	#[cfg(all(
		target_arch = "aarch64",
		target_vendor = "apple",
		feature = "apple-sme",
		not(miri)
	))]
	#[test]
	fn sme_is_t4a_gemms_answer() {
		use gemm_common::cache::HasAppleSme;
		assert_eq!(info().sme_f64, HasAppleSme::get());
		assert_eq!(info().sme_f32, HasAppleSme::get_f32());
	}

	#[cfg(all(feature = "std", not(miri)))]
	#[test]
	fn caches_are_t4a_gemms() {
		let c = &*gemm_common::cache::CACHE_INFO;
		assert_eq!(
			info().gemm_caches,
			Some([c[0].cache_bytes, c[1].cache_bytes, c[2].cache_bytes])
		);
	}

	#[cfg(feature = "threadteam")]
	#[test]
	fn the_class_agrees_with_threadteam() {
		assert_eq!(info().class.name(), threadteam::HostClass::detect().name());
	}
}
