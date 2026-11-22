//! The parallel rule: a region runs in parallel when its serial time is at least `PAYS` times
//! the dispatch cost of every section it opens (design record: `threadteam`
//! `docs/design/thresholds.md`).
//!
//! The dispatch cost belongs to the token ([`ParShape::dispatch_cost`]); the serial time of a
//! kernel family is its own throughput constant ([`Rates`]) for the scalar's kind (real or
//! complex). `PAYS` mirrors `threadteam::PAYS` so that
//! a build without the parallel route still decides the way the route does; a test compares the
//! two when `threadteam` is there.
//!
//! The width rules ([`par_width`], [`join_pays`], [`ParThreshold::region_width`], ...) are public,
//! so that a driver outside this crate gates its regions the same way (design record §5).
//!
//! # Provenance
//!
//! Ported from colmat-rs `faer/src/utils/cost.rs` at `71337cdc` (lines 1-512 and 615-1033, and
//! the tests that do not need a factorization). Changed:
//!
//! * [`Rate::of`] takes the `f64` rate for every real scalar and the `c64` rate for every
//!   complex one. colmat-rs halved it for the native `f32` and `c32` by testing
//!   `IS_NATIVE_F32`/`IS_NATIVE_C32`, which `colmat-traits` does not have (#7); until the scalar
//!   states its own weight (the next commit), `f32` and `c32` are priced as `f64` and `c64`.
//!   `Rates::gemm_single` and `ns_per_mac` (the measured `f32` product rate) wait for that
//!   weight too and are not ported here.
//! * `gemm_ns_per_mac` (the matrix product, core-3), `block_apply_estimate`, `PanelTeam` and
//!   `lookahead_width` (the factorizations) and the `par-trace` probes are not ported.
//! * The width functions are public.
//!
//! faer's MIT notice: `LICENSE-faer`.

use crate::ParShape;
use colmat_traits::ComplexField;
use core::num::NonZeroUsize;
use core::time::Duration;

#[cfg(feature = "threadteam")]
pub use threadteam::{PAYS, pays_at};

/// The multiple of the stack (mirrors `threadteam::PAYS`, for a build without the route).
#[cfg(not(feature = "threadteam"))]
pub const PAYS: f64 = 3.0;

/// `estimate >= PAYS * sections * dispatch_cost` (mirrors `threadteam::pays_at`).
#[cfg(not(feature = "threadteam"))]
pub fn pays_at(
	estimate: Duration,
	sections: u32,
	dispatch_cost: Duration,
) -> bool {
	if dispatch_cost == Duration::MAX {
		return false;
	}
	if sections == 0 {
		return true;
	}
	estimate.as_secs_f64()
		>= PAYS * f64::from(sections) * dispatch_cost.as_secs_f64()
}

/// The class of the machine, keyed by the kernel dispatch it will take, not by the CPU model.
/// The serial rate table has one row per class. The enum and the detection are kept in sync by
/// hand with `threadteam::HostClass`; a test compares the two when the parallel route is
/// compiled.
///
/// Rates and dispatch costs of one row must come from the same machine in the same clock state:
/// switching the boost on moves both by about 1.66 times with a stable ratio (within a few
/// percent), so a row that mixes boosted rates with unboosted costs, or the reverse, shifts every
/// floor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HostClass {
	/// `x86_64` with AVX-512F at run time (`x86_64-avx512`).
	X86_64Avx512,
	/// `x86_64` with AVX2 and FMA, without AVX-512F (`x86_64-avx2`).
	X86_64Avx2,
	/// Any other `x86_64` (`x86_64-scalar`).
	X86_64Scalar,
	/// `aarch64` on macOS (`aarch64-apple`).
	Aarch64Apple,
	/// Any other `aarch64` (`aarch64`).
	Aarch64,
	/// Everything else (`generic`).
	Generic,
}

impl HostClass {
	/// Every class, in declaration order.
	pub const ALL: [HostClass; 6] = [
		HostClass::X86_64Avx512,
		HostClass::X86_64Avx2,
		HostClass::X86_64Scalar,
		HostClass::Aarch64Apple,
		HostClass::Aarch64,
		HostClass::Generic,
	];

	/// The class of this host, detected at the first call and cached. With `std`, `x86_64` uses
	/// run time feature detection; without it the target features of the build decide.
	pub fn detect() -> Self {
		use core::sync::atomic::{AtomicU8, Ordering};
		static CLASS: AtomicU8 = AtomicU8::new(0);
		let c = CLASS.load(Ordering::Relaxed);
		if c != 0 {
			return Self::ALL[c as usize - 1];
		}
		let class = Self::detect_uncached();
		let idx = Self::ALL.iter().position(|&k| k == class).unwrap();
		CLASS.store(idx as u8 + 1, Ordering::Relaxed);
		class
	}

	#[cfg(all(target_arch = "x86_64", feature = "std"))]
	fn detect_uncached() -> Self {
		if std::arch::is_x86_feature_detected!("avx512f") {
			HostClass::X86_64Avx512
		} else if std::arch::is_x86_feature_detected!("avx2")
			&& std::arch::is_x86_feature_detected!("fma")
		{
			HostClass::X86_64Avx2
		} else {
			HostClass::X86_64Scalar
		}
	}

	#[cfg(all(target_arch = "x86_64", not(feature = "std")))]
	fn detect_uncached() -> Self {
		if cfg!(target_feature = "avx512f") {
			HostClass::X86_64Avx512
		} else if cfg!(all(target_feature = "avx2", target_feature = "fma")) {
			HostClass::X86_64Avx2
		} else {
			HostClass::X86_64Scalar
		}
	}

	#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
	fn detect_uncached() -> Self {
		HostClass::Aarch64Apple
	}

	#[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
	fn detect_uncached() -> Self {
		HostClass::Aarch64
	}

	#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
	fn detect_uncached() -> Self {
		HostClass::Generic
	}

	/// The class name used in the docs and the tables.
	pub const fn name(self) -> &'static str {
		match self {
			HostClass::X86_64Avx512 => "x86_64-avx512",
			HostClass::X86_64Avx2 => "x86_64-avx2",
			HostClass::X86_64Scalar => "x86_64-scalar",
			HostClass::Aarch64Apple => "aarch64-apple",
			HostClass::Aarch64 => "aarch64",
			HostClass::Generic => "generic",
		}
	}
}

/// Floor of the parallel regions of a factorization family: the one type of every
/// `par_threshold` field (LU, full-pivoting LU, Bunch-Kaufman, QR with column pivoting,
/// bidiagonalization, Hessenberg and tridiagonal reductions).
///
/// The default is [`Derived`](Self::Derived): each region is decided by [`ParShape::pays`], its
/// serial time (units of work times the family's serial throughput, [`rate`]) against the
/// dispatch cost of the parallelism it would use and the number of sections it opens.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum ParThreshold {
	/// Derive the floor of every region from its serial time and the dispatch cost.
	#[default]
	Derived,
	/// A fixed floor in the work unit of the family (documented where the family declares its
	/// gate: elements of the updated trailing block for the reductions, `f64` elements for LU), a
	/// region runs in parallel from that many units. `Fixed(0)` and `Fixed(1)` make every region
	/// parallel that has any work.
	Fixed(usize),
	/// Never run a region in parallel.
	Never,
}

/// The integer form used by the C API: `0` is [`Derived`](ParThreshold::Derived),
/// `usize::MAX` is [`Never`](ParThreshold::Never), any other value is
/// [`Fixed`](ParThreshold::Fixed).
impl From<usize> for ParThreshold {
	fn from(value: usize) -> Self {
		match value {
			0 => ParThreshold::Derived,
			usize::MAX => ParThreshold::Never,
			n => ParThreshold::Fixed(n),
		}
	}
}

/// See [`From<usize>`](ParThreshold#impl-From<usize>-for-ParThreshold); `Fixed(0)` is
/// written as `1`, the same behaviour, because `0` means derived.
impl From<ParThreshold> for usize {
	fn from(value: ParThreshold) -> Self {
		match value {
			ParThreshold::Derived => 0,
			ParThreshold::Fixed(n) => n.max(1),
			ParThreshold::Never => usize::MAX,
		}
	}
}

impl ParThreshold {
	/// The width a region of `units` of work in the unit of `rate` that opens `sections`
	/// sections runs at on `par` (1: sequential); the caller runs it on a sub-team of that width
	/// ([`with_width`](crate::utils::thread::with_width)).
	///
	/// [`Derived`](Self::Derived) is [`par_width`] of `units * rate`, the widest width that pays;
	/// an explicit [`Fixed`](Self::Fixed) floor keeps its rule, the whole degree when
	/// `units >= floor`; [`Never`](Self::Never) is 1.
	pub fn region_width<T: ComplexField>(
		self,
		par: &crate::Par,
		units: usize,
		rate: Rate,
		sections: u32,
	) -> usize {
		match self {
			Self::Never => 1,
			Self::Fixed(floor) => {
				if units >= floor {
					par.degree()
				} else {
					1
				}
			},
			Self::Derived => par_width::<T>(par, units, rate, sections),
		}
	}
}

/// Serial throughput of a kernel family, nanoseconds per unit of work for `f64` and `c64`, per
/// host class ([`Rates`]). Measured with one thread, L2 resident shapes. Another real scalar
/// takes the `f64` value, another complex scalar the `c64` value.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Rate {
	/// `f64`.
	pub real: f64,
	/// `c64`.
	pub cplx: f64,
}

impl Rate {
	/// Nanoseconds per unit for the scalar `T`: the `f64` or `c64` rate.
	#[inline]
	pub fn of<T: ComplexField>(self) -> f64 {
		if T::IS_REAL { self.real } else { self.cplx }
	}
}

/// The serial throughput constants of the kernel families of one host class, one per family
/// (design record: `threadteam` `docs/design/thresholds.md`, section 2.3). The unit is what the
/// gate of the family counts. A step of a factorization is timed inside the whole serial
/// factorization, so the constant includes the serial parts of a step (pivot search, Householder
/// generation) and slightly overestimates the time of the part that is split.
///
/// The row of a class is [`Rates::for_class`]; [`rates`] is the row of the host. Sources:
///
/// * `x86_64-avx512`: one thread on an AMD Ryzen AI 9 HX 470 (Zen 5) at 2.0 GHz, boost off,
///   median over shapes and rounds (`faer-benchmark`
///   `data/probes/20261026-awake-team-linux/tables-consts-exec3.md`); the divide and conquer and
///   sparse product from the region constants of the same file (`raw-exec3/rconsts-*.csv`).
/// * `aarch64-apple`: one thread on an Apple M5 Max, QoS `UserInitiated`, L2 resident shapes,
///   median over the shapes 96 to 768 and two processes
///   (`data/probes/20261026-threshold-recalibration`).
/// * every other class is unmeasured and takes the `generic` row: per entry the smaller of the
///   measured rows, so that an unknown host keeps work serial a little longer than it must
///   rather than dispatching too early.
///
/// A row's rates and the dispatch costs of the same class must come from the same machine and
/// clock state ([`HostClass`]).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Rates {
	/// GEMM and every kernel counted in multiply-adds (triangular products and solves), per
	/// multiply-add.
	pub gemm_mac: Rate,
	/// The multiply-adds of the block Householder apply, per multiply-add.
	pub block_apply_mac: Rate,
	/// Full-pivoting LU, one step: rank-one update fused with the pivot search, per element of
	/// the trailing block `(m-k-1)*(n-k-1)`.
	pub lu_full_pivot: Rate,
	/// Bunch-Kaufman with full pivoting, one step: rank-1 or rank-2 update fused with the pivot
	/// search, per element of the trailing block `rem*rem`.
	pub bunch_kaufman: Rate,
	/// QR with column pivoting, one step: fused update of the trailing block and the norms, per
	/// element `(m-k-1)*(n-k-1)`.
	pub qr_col_pivot: Rate,
	/// Bidiagonalization, one step: the fused two-sided update and the matrix-vector product, per
	/// element `(m-k-1)*(n-k-1)`.
	pub bidiag: Rate,
	/// Hessenberg reduction (unblocked), one step: two matrix-vector products and a rank-one
	/// update, per element `n*n` of the trailing block.
	pub hessenberg: Rate,
	/// Tridiagonalization, one step: the symmetric fused update, per element `rem*rem/2` of the
	/// triangle.
	pub tridiag: Rate,
	/// Matrix-vector product, per matrix element (the mean of the column major and row major
	/// products).
	pub matvec: Rate,
	/// Divide and conquer bidiagonal SVD (`bidiag_svd`, both singular vectors, non deflating
	/// data), the whole serial solve of size `m` per `m^2`; the join of size `n` has the two
	/// halves as work, `2 (n/2)^2`. The `m^2` term (secular equation, deflation, permutations)
	/// is larger than the merge products below `m` about 1900, so the unit is `m^2`; the
	/// constant is the median over `m` 64 to 512 (it is size dependent: 48 to 62 on the M5 fit,
	/// 145 to 178 on the Zen 5). Real types only (the complex value repeats the real one).
	pub bidiag_svd_dc: Rate,
	/// Divide and conquer symmetric tridiagonal EVD (`tridiag_evd`, eigenvectors on), the whole
	/// serial solve of size `m` per `m^2`, used as `bidiag_svd_dc` (88 to 102 on the Zen 5 over
	/// `m` 64 to 512). Real types only.
	pub tridiag_evd_dc: Rate,
	/// The secular-equation roots of one divide and conquer merge (`tridiag_evd`'s
	/// `compute_eigenvalues`): all `k` roots of a merge with `k` non deflated poles, per `k^2`
	/// (each root costs a fixed number of `O(k)` passes, 12 to 13 on a random spectrum). The
	/// Zen 5 value is the median of the six top level cells of
	/// `docs/worklogs/2026-11-11-eigh-dc-merge-phases.md` table A (f64 and c64, `k` 64, 96 and
	/// 128, one thread: 34.4 to 37.1). The M5 value is **not measured**: it is the Zen 5 value
	/// scaled by the ratio of the two rows' `tridiag_evd_dc` (31.0 / 97.4), of which this loop is
	/// the larger part. Real types only.
	pub secular_roots: Rate,
	/// Triangular inverse (`invert_*_triangular`), the whole serial inverse of size `m` per
	/// `m^3`; the join of size `n` has the two halves as work, `2 (n/2)^3`. The constant is the
	/// median over `m` 64 to 512 and falls with the size (on the Zen 5, `f64` 0.047 per `m^3`
	/// at 64 to 0.0145 at 512), so a join near the floor is a little overestimated.
	pub triangular_inverse: Rate,
	/// Sparse by sparse numeric product, per multiply-add of the symbolic flop count
	/// (`SparseMatMulInfo`, free to read); the median over the measured densities and sizes.
	pub sparse_matmul: Rate,
	/// Rank-one Householder apply (`N == 1`), per element of the target `M*K`.
	pub householder_rank_one: Rate,
	/// One row swap of one column in the LU swap stage (random pivot rows, columns of 64 to 1024
	/// elements, cache resident, one thread; memory bound, so the `c64` value is the larger
	/// element width, not four times the `f64` one).
	pub lu_swap: Rate,
}

const fn fmin(a: f64, b: f64) -> f64 {
	if b < a { b } else { a }
}

const fn rmin(a: Rate, b: Rate) -> Rate {
	Rate {
		real: fmin(a.real, b.real),
		cplx: fmin(a.cplx, b.cplx),
	}
}

const fn r(real: f64, cplx: f64) -> Rate {
	Rate { real, cplx }
}

impl Rates {
	/// Per entry the smaller value of the two rows.
	const fn min_of(a: &Rates, b: &Rates) -> Rates {
		Rates {
			gemm_mac: rmin(a.gemm_mac, b.gemm_mac),
			block_apply_mac: rmin(a.block_apply_mac, b.block_apply_mac),
			lu_full_pivot: rmin(a.lu_full_pivot, b.lu_full_pivot),
			bunch_kaufman: rmin(a.bunch_kaufman, b.bunch_kaufman),
			qr_col_pivot: rmin(a.qr_col_pivot, b.qr_col_pivot),
			bidiag: rmin(a.bidiag, b.bidiag),
			hessenberg: rmin(a.hessenberg, b.hessenberg),
			tridiag: rmin(a.tridiag, b.tridiag),
			matvec: rmin(a.matvec, b.matvec),
			bidiag_svd_dc: rmin(a.bidiag_svd_dc, b.bidiag_svd_dc),
			tridiag_evd_dc: rmin(a.tridiag_evd_dc, b.tridiag_evd_dc),
			secular_roots: rmin(a.secular_roots, b.secular_roots),
			triangular_inverse: rmin(
				a.triangular_inverse,
				b.triangular_inverse,
			),
			sparse_matmul: rmin(a.sparse_matmul, b.sparse_matmul),
			householder_rank_one: rmin(
				a.householder_rank_one,
				b.householder_rank_one,
			),
			lu_swap: rmin(a.lu_swap, b.lu_swap),
		}
	}

	/// `x86_64-avx512`: Zen 5 at 2.0 GHz, boost off (see the type docs).
	pub const X86_64_AVX512: Rates = Rates {
		gemm_mac: r(0.0641, 0.2565),
		block_apply_mac: r(0.0676, 0.2685),
		lu_full_pivot: r(0.4011, 0.8679),
		bunch_kaufman: r(0.8295, 1.3904),
		qr_col_pivot: r(0.3645, 1.0320),
		bidiag: r(1.0427, 2.1997),
		hessenberg: r(0.8576, 2.2212),
		tridiag: r(0.7695, 1.9252),
		// mean of matvec_colmajor (0.1743, 0.3373) and matvec_rowmajor (0.1565, 0.5057)
		matvec: r(0.1654, 0.4215),
		bidiag_svd_dc: r(152.9, 152.9),
		tridiag_evd_dc: r(97.4, 97.4),
		secular_roots: r(35.2, 35.2),
		triangular_inverse: r(0.01936, 0.07228),
		// median over spmm_{f64,c64}_n{2000,4000,8000,20000}_d{4,6,8,12,16,40}
		sparse_matmul: r(4.1315, 5.2342),
		householder_rank_one: r(0.4783, 1.0303),
		// Zen 5 at 2.0 GHz, one thread, a scratch probe of the swap loop: swaps of 64 to 256 rows over 768 to 960 columns, one
		// thread: 0.98 (`f64`) and 1.30 (`c64`) ns, 1.0 at 256 rows; the 0.55 of the M5 was
		// used for every class and made the swap stage of a 64 x 1024 LU 1.7 times too cheap
		lu_swap: r(0.98, 1.30),
	};

	/// `aarch64-apple`: Apple M5 Max (the constants of the first calibration).
	pub const AARCH64_APPLE: Rates = Rates {
		gemm_mac: r(0.037, 0.165),
		block_apply_mac: r(0.037, 0.165),
		lu_full_pivot: r(0.216, 0.6),
		bunch_kaufman: r(0.238, 0.38),
		qr_col_pivot: r(0.202, 0.713),
		bidiag: r(0.48, 1.36),
		hessenberg: r(0.451, 1.57),
		tridiag: r(0.339, 1.16),
		matvec: r(0.1, 0.34),
		bidiag_svd_dc: r(53.0, 53.0),
		tridiag_evd_dc: r(31.0, 31.0),
		// derived, not measured: 35.2 * 31.0 / 97.4 (see the field)
		secular_roots: r(11.2, 11.2),
		triangular_inverse: r(0.0096, 0.0344),
		sparse_matmul: r(1.5, 1.8),
		householder_rank_one: r(0.212, 0.725),
		// one row swap of one column, random pivot rows, columns of 64 to 1024 `f64`, cache
		// resident, median of 0.31 to 1.0 over `m` 128 to 1024 and 16 to 128 swaps per column
		// (colmat-rs `NS_PER_SWAP`)
		lu_swap: r(0.55, 0.55),
	};

	/// `generic` (every unmeasured class): per entry the smaller of the measured rows.
	pub const GENERIC: Rates =
		Rates::min_of(&Rates::X86_64_AVX512, &Rates::AARCH64_APPLE);

	/// The row of a host class.
	pub fn for_class(class: HostClass) -> &'static Rates {
		static AVX512: Rates = Rates::X86_64_AVX512;
		static APPLE: Rates = Rates::AARCH64_APPLE;
		static GENERIC: Rates = Rates::GENERIC;
		match class {
			HostClass::X86_64Avx512 => &AVX512,
			HostClass::Aarch64Apple => &APPLE,
			HostClass::X86_64Avx2
			| HostClass::X86_64Scalar
			| HostClass::Aarch64
			| HostClass::Generic => &GENERIC,
		}
	}
}

/// The rates of the host class of this process ([`HostClass::detect`]).
pub fn rates() -> &'static Rates {
	Rates::for_class(HostClass::detect())
}

/// The serial throughput constants of the host, one function per family ([`Rates`] has the
/// units and the sources).
pub mod rate {
	use super::{Rate, rates};
	macro_rules! families {
		($($name:ident),* $(,)?) => {$(
			#[doc = core::concat!("`Rates::", core::stringify!($name), "` of the host class.")]
			pub fn $name() -> Rate { rates().$name }
		)*};
	}
	families!(
		gemm_mac,
		block_apply_mac,
		lu_full_pivot,
		bunch_kaufman,
		qr_col_pivot,
		bidiag,
		hessenberg,
		tridiag,
		matvec,
		bidiag_svd_dc,
		tridiag_evd_dc,
		secular_roots,
		triangular_inverse,
		sparse_matmul,
		householder_rank_one,
		lu_swap,
	);
}

/// [`ParShape::pays`] of `par` at `width` participants, without building the shape. A gate is asked
/// hundreds of times by a small factorization, and the shape of a team token copies its whole cost
/// table on every call (`Par::shape`: 15 ns, the decision itself 40 ns, against 12 ns here, where the
/// token answers from its own table). `width` is clamped to the degree, below 2 never pays.
pub fn par_pays_time(
	par: &crate::Par,
	estimate: Duration,
	sections: u32,
	width: usize,
) -> bool {
	let width = width.min(par.degree());
	let Some(width) = NonZeroUsize::new(width).filter(|w| w.get() >= 2) else {
		return false;
	};
	match par {
		crate::Par::Seq => false,
		#[cfg(feature = "threadteam")]
		crate::Par::Team(team) => {
			// the same cost the shape reports (threadteam's table, extrapolated by threadteam above
			// its largest anchor)
			pays_at(estimate, sections, team.dispatch_cost(width))
		},
		// a build without the route has no team token to ask
		#[cfg(not(feature = "threadteam"))]
		_ => {
			let _ = (estimate, sections, width);
			false
		},
	}
}

/// Whether a region of serial time `estimate` that opens `sections` sections can pay on `par` at
/// *some* width: `estimate >= PAYS * sections * min over widths of the cost` ([`ParShape::min_dispatch_cost`]).
/// "Does not pay at the cheapest width" is the rejection that is right for any shape of table (a
/// rising one included); it says nothing about the width a region then chooses.
pub fn par_can_pay_time(
	par: &crate::Par,
	estimate: Duration,
	sections: u32,
) -> bool {
	if par.degree() < 2 {
		return false;
	}
	match par {
		crate::Par::Seq => false,
		#[cfg(feature = "threadteam")]
		crate::Par::Team(team) => par_can_pay_time_team(team, estimate, sections),
		#[cfg(not(feature = "threadteam"))]
		_ => {
			let _ = (estimate, sections);
			false
		},
	}
}

/// [`par_can_pay_time`] for a team token.
#[cfg(feature = "threadteam")]
pub fn par_can_pay_time_team(
	team: &threadteam::TeamExec<'_>,
	estimate: Duration,
	sections: u32,
) -> bool {
	team.size().get() >= 2
		&& pays_at(
			estimate,
			sections,
			team.dispatch_costs()
				.min_cost(threadteam::DispatchKind::Team),
		)
}

/// The widest width in `2..=degree` at which a region of serial time `estimate` that opens
/// `sections` sections pays on `par`, or 1 (sequential) when none does.
///
/// Every width is considered: the search runs from the full width down and stops at the first
/// that pays, which is the widest. A narrower width that does not pay says nothing about a wider
/// one, because a measured row can fall with the width (`aarch64-apple`'s team row is 3.0, 2.3,
/// 1.4 us at widths 2, 4, 6).
pub fn par_width_time(
	par: &crate::Par,
	estimate: Duration,
	sections: u32,
) -> usize {
	par_width_time_upto(par, estimate, sections, usize::MAX)
}

/// The widest width a region of a factorization family may run at: twice the largest width the
/// host's dispatch row measures, and never more than the degree.
///
/// The row is measured at widths 2 to 6 on `aarch64-apple` (3.0, 2.3, 1.4 us) and 2 to 8 on
/// `x86_64-avx512`; above its last anchor the cost is the linear extrapolation of that anchor
/// (2.8 us at width 12 and 4.2 us at 18 on `aarch64-apple`). Those values price a *steady
/// state* section -- and they are right about the median: an empty section is 2.5 to 3.0 us at
/// every width up to 18 on the same host. What they cannot see is the tail, and a factorization
/// pays the tail hundreds of times: every recursion level opens a swap stage and a trailing
/// update, `partial_piv_lu` at n = 1024 opens 63 of each. Measured on an Apple M5 Max (18
/// logical: 6 performance + 12 efficiency cores, macOS 26.7.2, 1 to 6 busy cores of background
/// load), the worst of 200 empty sections is 64 us at width 16 and 2.1 ms at width 18, and the
/// two regions at width 18 read 24.8, 31.5 and 38.2 GFLOP/s at n = 512, 1024 and 2048 against
/// 66.9, 87.3 and 116.9 at width 14: 2.7 to 3.1 times slower than a narrower width.
///
/// With the two LU regions capped at 12 (twice the row's last anchor) and everything else
/// unchanged, the same cases read 36.7, 75.2 and 127.3 GFLOP/s (1.5, 2.4 and 3.3 times the
/// uncapped run), and every width at or below the cap is unchanged. GEMM is not capped: it
/// opens one region per call, so the median is what it pays, and it reads 396 GFLOP/s at width
/// 18 on the same host.
pub fn max_region_width(degree: usize) -> usize {
	/// The largest measured width of the host class's dispatch row, as
	/// `threadteam::DispatchCosts::for_class` states it: `x86_64-avx512` is measured to 8,
	/// `x86_64-avx2` to 64, every other class to 6.
	fn largest_measured_width() -> usize {
		match HostClass::detect() {
			HostClass::X86_64Avx512 => 8,
			HostClass::X86_64Avx2 => 64,
			HostClass::X86_64Scalar
			| HostClass::Aarch64Apple
			| HostClass::Aarch64
			| HostClass::Generic => 6,
		}
	}
	// one doubling past the measured range: the extrapolation of an extrapolation is not
	// defended by any measurement
	degree.min(largest_measured_width() * 2)
}

/// [`par_width_time`] among the widths up to `max` (a region that cannot use more participants,
/// such as a split into at most `max` tasks).
pub fn par_width_time_upto(
	par: &crate::Par,
	estimate: Duration,
	sections: u32,
	max: usize,
) -> usize {
	match par {
		#[cfg(feature = "threadteam")]
		crate::Par::Team(team) => par_width_team(team, estimate, sections, max),
		_ => {
			let _ = (estimate, sections, max);
			1
		},
	}
}

/// [`par_width_time_upto`] for a team token.
#[cfg(feature = "threadteam")]
pub fn par_width_team(
	team: &threadteam::TeamExec<'_>,
	estimate: Duration,
	sections: u32,
	max: usize,
) -> usize {
	// the cheap rejection first: a region that does not pay at the cheapest width of the table
	// pays at none (a QR 128 asks hundreds of times)
	if max < 2 || !par_can_pay_time_team(team, estimate, sections) {
		return 1;
	}
	(2..=team.size().get().min(max))
		.rev()
		.find(|&w| {
			pays_at(
				estimate,
				sections,
				team.dispatch_cost(NonZeroUsize::new(w).unwrap()),
			)
		})
		.unwrap_or(1)
}

/// [`par_width_time`] of `units` of work at `rate`: the width the region runs at (1:
/// sequential).
pub fn par_width<T: ComplexField>(
	par: &crate::Par,
	units: usize,
	rate: Rate,
	sections: u32,
) -> usize {
	par_width_upto::<T>(par, units, rate, sections, usize::MAX)
}

/// [`par_width`] among the widths up to `max`.
pub fn par_width_upto<T: ComplexField>(
	par: &crate::Par,
	units: usize,
	rate: Rate,
	sections: u32,
	max: usize,
) -> usize {
	if par.degree() < 2 || max < 2 {
		return 1;
	}
	par_width_time_upto(par, serial_time(units, rate.of::<T>()), sections, max)
}

/// Whether a join of two halves pays: one section of width 2 (the two halves are the
/// participants; the budget is split between them, it is not a flat section over all of it), so
/// the dispatch cost is that of the shape at width 2 (`Team` 3.0 us, `Rayon` from outside 14 us
/// on the M5 table, against 1.4 and 19 at width 6). `units` of work at `rate`, `sections` joins.
pub fn join_pays<T: ComplexField>(
	par: &crate::Par,
	units: usize,
	rate: Rate,
	sections: u32,
) -> bool {
	if par.degree() < 2 {
		return false;
	}
	par_pays_time(par, serial_time(units, rate.of::<T>()), sections, 2)
}

/// Whether the join of a recursion that halves a problem of size `n` pays: its work is the two
/// halves, `2 (n/2)^power` units of `rate`, one join ([`join_pays`]). The rate is that of a whole serial
/// solve of size `m` per `m^power`, so no recurrence is assumed beyond the power: the work of
/// the subproblems at this level is what the two calls below it cost, and the merge after the
/// join is not counted (it is not parallel through this join). `power` is 3 for the triangular
/// inverse and 2 for the divide and conquer solves, whose measured time is `a m^2 + b m^3`
/// with the first term dominating up to `m` about 1900.
pub fn halves_pays<T: ComplexField>(
	par: &crate::Par,
	n: usize,
	rate: Rate,
	power: u32,
) -> bool {
	let h = n / 2;
	join_pays::<T>(par, h.saturating_pow(power).saturating_mul(2), rate, 1)
}

/// Serial time of `count` units at `ns` nanoseconds each.
pub fn serial_time(count: usize, ns: f64) -> Duration {
	Duration::from_secs_f64(count as f64 * ns * 1e-9)
}

impl ParShape {
	/// The cost of one parallel section of `width` participants on this shape, in the steady
	/// state (sections back to back); [`Duration::MAX`] for a sequential shape or `width < 2`.
	/// `width` is clamped to the degree of the shape.
	///
	/// `Team` is the cost of the team token the shape was taken from (threadteam's
	/// `DispatchKind::Team`, with the costs of that token's pool). Above the largest measured
	/// anchor of the row threadteam grows the cost linearly with the width (threadteam 0.1.1);
	/// faer keeps no second model.
	pub fn dispatch_cost(self, width: NonZeroUsize) -> Duration {
		let width = width.min(NonZeroUsize::new(self.degree().max(1)).unwrap());
		if width.get() < 2 {
			return Duration::MAX;
		}
		match self {
			Self::Seq => Duration::MAX,
			#[cfg(feature = "threadteam")]
			Self::Team(_, costs) => costs.cost(threadteam::DispatchKind::Team, width),
		}
	}

	/// The cheapest cost of one section over all widths of this shape (a lower bound of
	/// [`dispatch_cost`](Self::dispatch_cost) at every width), [`Duration::MAX`] for a shape of
	/// degree below 2.
	pub fn min_dispatch_cost(self) -> Duration {
		if self.degree() < 2 {
			return Duration::MAX;
		}
		match self {
			Self::Seq => Duration::MAX,
			#[cfg(feature = "threadteam")]
			Self::Team(_, costs) => costs.min_cost(threadteam::DispatchKind::Team),
		}
	}

	/// Whether a region of serial time `estimate` that opens `sections` sections of `width`
	/// participants pays on this shape: `estimate >= PAYS * sections * dispatch_cost(width)`.
	/// `sections` is the number of dispatches the region really performs.
	pub fn pays(
		self,
		estimate: Duration,
		sections: u32,
		width: NonZeroUsize,
	) -> bool {
		pays_at(estimate, sections, self.dispatch_cost(width))
	}

	/// [`pays`](Self::pays) at the full degree of the shape.
	pub fn pays_full(self, estimate: Duration, sections: u32) -> bool {
		match NonZeroUsize::new(self.degree()) {
			Some(width) => self.pays(estimate, sections, width),
			None => false,
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::c64;

	fn nz(n: usize) -> NonZeroUsize {
		NonZeroUsize::new(n).unwrap()
	}

	/// The M5 row: the tests that state numbers of the Apple team table use it with that table
	/// (rates and costs of one row).
	#[cfg(feature = "threadteam")]
	const M5: &Rates = &Rates::AARCH64_APPLE;

	#[test]
	fn seq_never_pays() {
		assert_eq!(ParShape::Seq.dispatch_cost(nz(4)), Duration::MAX);
		assert!(!ParShape::Seq.pays(Duration::from_secs(10), 1, nz(4)));
		assert!(!ParShape::Seq.pays_full(Duration::from_secs(10), 1));
	}

	#[cfg(feature = "threadteam")]
	#[test]
	fn min_dispatch_cost_is_the_cheapest_width() {
		use threadteam::{DispatchCosts, DispatchKind, WidthCosts};
		let us = Duration::from_micros;
		// falling, flat and rising tables
		for t in [
			[us(9), us(5), us(3)],
			[us(4), us(4), us(4)],
			[us(2), us(7), us(12)],
		] {
			let costs =
				DispatchCosts::for_class(threadteam::HostClass::Generic)
					.with(DispatchKind::Team, WidthCosts::new(t));
			let s = ParShape::Team(nz(6), costs);
			let want = (2..=6).map(|w| s.dispatch_cost(nz(w))).min().unwrap();
			assert_eq!(s.min_dispatch_cost(), want, "{t:?}");
		}
		assert_eq!(ParShape::Seq.min_dispatch_cost(), Duration::MAX);
		assert_eq!(
			ParShape::Team(
				nz(1),
				DispatchCosts::for_class(threadteam::HostClass::Generic)
			)
			.min_dispatch_cost(),
			Duration::MAX
		);
	}

	#[test]
	fn pays_is_the_multiple_of_the_cost() {
		let d = Duration::from_micros(2);
		assert!(pays_at(Duration::from_micros(6), 1, d));
		assert!(!pays_at(Duration::from_nanos(5999), 1, d));
		assert!(!pays_at(Duration::from_micros(11), 2, d));
		assert!(pays_at(Duration::from_micros(12), 2, d));
		assert!(!pays_at(Duration::from_secs(1), 1, Duration::MAX));
	}

	#[cfg(feature = "threadteam")]
	#[test]
	fn team_shape_carries_the_costs_of_its_token() {
		use threadteam::{DispatchCosts, DispatchKind, WidthCosts};
		let costs = DispatchCosts::for_class(threadteam::HostClass::Generic)
			.with(
				DispatchKind::Team,
				WidthCosts::flat(Duration::from_micros(5)),
			);
		let s = ParShape::Team(nz(4), costs);
		assert_eq!(s.dispatch_cost(nz(4)), Duration::from_micros(5));
		assert_eq!(s.dispatch_cost(nz(1)), Duration::MAX);
		// away from the exact boundary: `pays` compares `Duration`s as floats, and
		// `3 * 5 us` and `15 us` are not the same double
		assert!(s.pays(Duration::from_micros(16), 1, nz(4)));
		assert!(!s.pays(Duration::from_micros(14), 1, nz(4)));
	}

	#[cfg(feature = "threadteam")]
	/// Smallest `units` (doubling, then bisection) for which the region pays on `shape`.
	fn floor_units<T: crate::ComplexField>(
		shape: ParShape,
		r: Rate,
		sections: u32,
	) -> usize {
		let pays =
			|u: usize| shape.pays_full(serial_time(u, r.of::<T>()), sections);
		let mut hi = 1usize;
		while !pays(hi) {
			hi *= 2;
			assert!(hi < 1 << 40, "never pays");
		}
		let mut lo = hi / 2;
		while hi - lo > 1 {
			let mid = (lo + hi) / 2;
			if pays(mid) {
				hi = mid;
			} else {
				lo = mid;
			}
		}
		hi
	}

	/// (name, rate, sections as counted in the code of the gate), on the row `rt`
	fn gates(rt: &Rates) -> [(&'static str, Rate, u32); 10] {
		[
			("lu_full_pivot", rt.lu_full_pivot, 1),
			("bunch_kaufman", rt.bunch_kaufman, 1),
			("qr_col_pivot", rt.qr_col_pivot, 1),
			("bidiag", rt.bidiag, 2),
			("hessenberg", rt.hessenberg, 2),
			("tridiag", rt.tridiag, 1),
			("matvec", rt.matvec, 1),
			("householder_rank_one", rt.householder_rank_one, 1),
			("gemm_mac", rt.gemm_mac, 1),
			("sparse_matmul", rt.sparse_matmul, 1),
		]
	}

	/// Every entry of a row, as `(name, Rate)`.
	fn entries(rt: &Rates) -> [(&'static str, Rate); 16] {
		[
			("gemm_mac", rt.gemm_mac),
			("block_apply_mac", rt.block_apply_mac),
			("lu_full_pivot", rt.lu_full_pivot),
			("bunch_kaufman", rt.bunch_kaufman),
			("qr_col_pivot", rt.qr_col_pivot),
			("bidiag", rt.bidiag),
			("hessenberg", rt.hessenberg),
			("tridiag", rt.tridiag),
			("matvec", rt.matvec),
			("bidiag_svd_dc", rt.bidiag_svd_dc),
			("tridiag_evd_dc", rt.tridiag_evd_dc),
			("secular_roots", rt.secular_roots),
			("triangular_inverse", rt.triangular_inverse),
			("sparse_matmul", rt.sparse_matmul),
			("householder_rank_one", rt.householder_rank_one),
			("lu_swap", rt.lu_swap),
		]
	}

	#[test]
	fn every_class_resolves_to_its_table() {
		assert_eq!(
			Rates::for_class(HostClass::X86_64Avx512),
			&Rates::X86_64_AVX512
		);
		assert_eq!(
			Rates::for_class(HostClass::Aarch64Apple),
			&Rates::AARCH64_APPLE
		);
		for c in [
			HostClass::X86_64Avx2,
			HostClass::X86_64Scalar,
			HostClass::Aarch64,
			HostClass::Generic,
		] {
			assert_eq!(Rates::for_class(c), &Rates::GENERIC, "{}", c.name());
		}
		// the rows are the measured ones
		assert_eq!(Rates::X86_64_AVX512.gemm_mac, r(0.0641, 0.2565));
		assert_eq!(Rates::AARCH64_APPLE.gemm_mac, r(0.037, 0.165));
		assert_eq!(Rates::AARCH64_APPLE.bidiag_svd_dc, r(53.0, 53.0));
	}

	#[test]
	fn generic_is_the_per_entry_minimum_of_the_measured_rows() {
		let (a, b, g) = (
			entries(&Rates::X86_64_AVX512),
			entries(&Rates::AARCH64_APPLE),
			entries(&Rates::GENERIC),
		);
		for i in 0..a.len() {
			let (name, ga) = (a[i].0, a[i].1);
			assert_eq!(g[i].1.real, ga.real.min(b[i].1.real), "{name} real");
			assert_eq!(g[i].1.cplx, ga.cplx.min(b[i].1.cplx), "{name} cplx");
			assert!(
				g[i].1.real <= ga.real && g[i].1.real <= b[i].1.real,
				"{name}"
			);
		}
	}

	#[test]
	fn rows_are_positive() {
		for c in HostClass::ALL {
			for (name, r) in entries(Rates::for_class(c)) {
				assert!(r.real > 0.0 && r.cplx > 0.0, "{} {name}", c.name());
			}
		}
	}

	#[test]
	fn detection_is_stable_and_the_host_row_is_its_class() {
		let c = HostClass::detect();
		for _ in 0..4 {
			assert_eq!(HostClass::detect(), c);
		}
		assert_eq!(HostClass::detect_uncached(), c);
		assert_eq!(rates(), Rates::for_class(c));
		assert_eq!(rate::bidiag(), rates().bidiag);
		#[cfg(all(target_arch = "x86_64", feature = "std"))]
		if std::arch::is_x86_feature_detected!("avx512f") {
			assert_eq!(c, HostClass::X86_64Avx512);
		}
		#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
		assert_eq!(c, HostClass::Aarch64Apple);
	}

	#[cfg(feature = "threadteam")]
	/// faer's `HostClass` is kept in sync by hand with `threadteam`'s; they must agree.
	#[test]
	fn host_class_agrees_with_threadteam() {
		let tt = |c: HostClass| match c {
			HostClass::X86_64Avx512 => threadteam::HostClass::X86_64Avx512,
			HostClass::X86_64Avx2 => threadteam::HostClass::X86_64Avx2,
			HostClass::X86_64Scalar => threadteam::HostClass::X86_64Scalar,
			HostClass::Aarch64Apple => threadteam::HostClass::Aarch64Apple,
			HostClass::Aarch64 => threadteam::HostClass::Aarch64,
			HostClass::Generic => threadteam::HostClass::Generic,
		};
		assert_eq!(tt(HostClass::detect()), threadteam::HostClass::detect());
		for c in HostClass::ALL {
			assert_eq!(c.name(), tt(c).name());
		}
	}

	#[cfg(feature = "threadteam")]
	/// The team cost is threadteam's table, nothing else: faer keeps no second extrapolation
	/// (threadteam-rs#6). Above a row's largest anchor threadteam grows the cost in proportion to
	/// the width, so a region that does not pay at eight members does not pay at 64.
	#[test]
	fn the_team_cost_is_threadteams_table_at_every_width() {
		use threadteam::{DispatchCosts, DispatchKind, HostClass, WidthCosts};
		let us = Duration::from_micros;
		let rows = [
			DispatchCosts::for_class(HostClass::Generic),
			DispatchCosts::for_class(HostClass::X86_64Avx512),
			DispatchCosts::for_class(HostClass::X86_64Avx2),
			DispatchCosts::for_class(HostClass::Aarch64Apple),
			DispatchCosts::for_class(HostClass::Generic)
				.with(DispatchKind::Team, WidthCosts::flat(us(1))),
		];
		for costs in rows {
			let s = ParShape::Team(nz(128), costs);
			for w in 2..=128 {
				assert_eq!(
					s.dispatch_cost(nz(w)),
					costs.cost(DispatchKind::Team, nz(w)),
					"{costs:?} width {w}"
				);
			}
		}
		// above the largest anchor: `cost(w_m) * w / w_m` in integer nanoseconds. The flat row ends
		// at 6, the avx512 team row at 8, the Zen2 avx2 row at 64.
		let flat = ParShape::Team(nz(128), rows[4]);
		for (w, ns) in [
			(8, 1333),
			(12, 2000),
			(16, 2666),
			(32, 5333),
			(64, 10666),
			(128, 21333),
		] {
			assert_eq!(
				flat.dispatch_cost(nz(w)),
				Duration::from_nanos(ns),
				"flat width {w}"
			);
		}
		let avx512 = ParShape::Team(nz(128), rows[1]);
		assert_eq!(avx512.dispatch_cost(nz(8)), Duration::from_nanos(980));
		assert_eq!(avx512.dispatch_cost(nz(64)), Duration::from_nanos(980 * 8));
		let zen2 = ParShape::Team(nz(128), rows[2]);
		assert_eq!(zen2.dispatch_cost(nz(64)), Duration::from_nanos(9333));
		assert_eq!(zen2.dispatch_cost(nz(128)), Duration::from_nanos(18666));
		// a region that does not pay at 8 members does not pay at 64 either
		let at_8 = flat.dispatch_cost(nz(8)) * 3;
		assert!(flat.pays(at_8, 1, nz(8)));
		assert!(!flat.pays(at_8, 1, nz(64)));
	}

	#[cfg(feature = "threadteam")]
	#[cfg_attr(
		miri,
		ignore = "threadteam's pool calls foreign functions Miri does not support"
	)]
	#[test]
	fn fixed_and_never_keep_their_meaning() {
		let r = rate::lu_full_pivot();
		// a fixed floor takes the whole degree at the floor, nothing below it; never is serial
		assert_eq!(
			ParThreshold::Fixed(100).region_width::<f64>(
				&crate::Par::Seq,
				100,
				r,
				1
			),
			1
		);
		crate::utils::team::with_team(6, |par| {
			assert_eq!(
				ParThreshold::Never.region_width::<f64>(&par, usize::MAX, r, 1),
				1
			);
			assert_eq!(
				ParThreshold::Fixed(100).region_width::<f64>(&par, 99, r, 1),
				1
			);
			assert_eq!(
				ParThreshold::Fixed(100).region_width::<f64>(&par, 100, r, 1),
				6
			);
		});
		// the integer form of the C API
		assert_eq!(ParThreshold::from(0), ParThreshold::Derived);
		assert_eq!(ParThreshold::from(usize::MAX), ParThreshold::Never);
		assert_eq!(ParThreshold::from(7), ParThreshold::Fixed(7));
		assert_eq!(usize::from(ParThreshold::Fixed(0)), 1);
	}

	#[test]
	fn derived_gates_never_pay_on_a_sequential_shape() {
		let seq = crate::Par::Seq;
		for (_, r, sections) in gates(rates()) {
			assert_eq!(
				ParThreshold::Derived.region_width::<f64>(
					&seq,
					usize::MAX / 2,
					r,
					sections
				),
				1
			);
			assert_eq!(
				ParThreshold::Derived.region_width::<c64>(
					&seq,
					usize::MAX / 2,
					r,
					sections
				),
				1
			);
			assert!(!ParShape::Seq.pays_full(
				serial_time(usize::MAX / 2, r.of::<f64>()),
				sections
			));
		}
	}

	#[cfg(feature = "threadteam")]
	#[test]
	fn derived_floors_on_a_team_follow_the_rule() {
		use threadteam::DispatchCosts;
		let team = ParShape::Team(
			nz(6),
			DispatchCosts::for_class(threadteam::HostClass::Aarch64Apple),
		);
		let d = team.dispatch_cost(nz(6)).as_secs_f64() * 1e9;
		for (name, r, sections) in gates(M5) {
			for (ty, ns) in [("f64", r.real), ("c64", r.cplx)] {
				let floor = if ty == "f64" {
					floor_units::<f64>(team, r, sections)
				} else {
					floor_units::<c64>(team, r, sections)
				};
				// the floor is PAYS * sections * D / rate, within the rounding of a `Duration` (nanoseconds)
				let want = PAYS * f64::from(sections) * d / ns;
				assert!(
					(floor as f64 - want).abs() <= 4.0 + want * 2e-3,
					"{name} {ty}: floor {floor}, rule {want}"
				);
			}
		}
		// representative sizes at width 6 on the M5 table (D = 1.4 us, 4.2 us per section)
		let pays = |u: usize, r: Rate, f: u32| {
			team.pays_full(serial_time(u, r.of::<f64>()), f)
		};
		// full pivoting LU: a 100 x 100 trailing block stays serial, 200 x 200 is parallel
		assert!(!pays(100 * 100, M5.lu_full_pivot, 1));
		assert!(pays(200 * 200, M5.lu_full_pivot, 1));
		// bidiagonalization counts two sections: it needs twice the work of a one-section step
		assert!(
			floor_units::<f64>(team, M5.bidiag, 2)
				>= 2 * floor_units::<f64>(team, M5.bidiag, 1) - 2
		);
		// a 256 x 256 matrix-vector product (65536 elements) pays on a team, not below 100 x 100
		assert!(pays(256 * 256, M5.matvec, 1));
		assert!(!pays(64 * 64, M5.matvec, 1));
		// the old fixed floors were 192*256 = 49152 elements for every one of these steps; the
		// derived floors differ per family (the faster the kernel, the larger the floor)
		let lu = floor_units::<f64>(team, M5.lu_full_pivot, 1);
		let bidiag = floor_units::<f64>(team, M5.bidiag, 1);
		assert!(lu > bidiag);
	}

	#[cfg(feature = "threadteam")]
	#[cfg_attr(
		miri,
		ignore = "threadteam's pool calls foreign functions Miri does not support"
	)]
	#[test]
	fn par_width_reads_the_degree_of_the_token() {
		let seq = crate::Par::Seq;
		assert_eq!(
			par_width::<f64>(&seq, usize::MAX / 2, rate::matvec(), 1),
			1
		);
		crate::utils::team::with_team(6, |par| {
			assert_eq!(
				par_width::<f64>(&par, 10_000_000, rate::matvec(), 1),
				6
			);
			assert_eq!(par_width::<f64>(&par, 1_000, rate::matvec(), 1), 1);
			// a cap below the degree bounds the search
			assert_eq!(
				par_width_upto::<f64>(&par, 10_000_000, rate::matvec(), 1, 3),
				3
			);
		});
		crate::utils::team::with_team(1, |par| {
			assert_eq!(
				par_width::<f64>(&par, 10_000_000, rate::matvec(), 1),
				1
			);
		});
	}

	#[test]
	fn rates_are_per_dtype() {
		let r = rate::bidiag();
		assert_eq!(r.of::<f64>(), r.real);
		assert_eq!(r.of::<c64>(), r.cplx);
		// until the scalar states its weight, `f32` and `c32` take the `f64` and `c64` rate
		assert_eq!(r.of::<f32>(), r.real);
		assert_eq!(r.of::<crate::c32>(), r.cplx);
		// the complex rate of the product is 4.5 times the real one, not 1: the floor bug of the
		// colmat-rs team route
		let g = rate::gemm_mac();
		assert!(g.of::<c64>() > 4.0 * g.of::<f64>());
	}

	#[cfg(feature = "threadteam")]
	/// the join gates: the work of a join of size `n` is two halves, `2 (n/2)^3` units
	#[cfg_attr(
		miri,
		ignore = "threadteam's pool calls foreign functions Miri does not support"
	)]
	#[test]
	fn halves_join_gates_follow_the_rule() {
		let seq = crate::Par::Seq;
		for r in [
			rate::bidiag_svd_dc(),
			rate::tridiag_evd_dc(),
			rate::triangular_inverse(),
		] {
			assert!(!halves_pays::<f64>(&seq, 1 << 20, r, 2));
		}
		crate::utils::team::with_team(6, |par| {
			for r in [
				rate::bidiag_svd_dc(),
				rate::tridiag_evd_dc(),
				rate::triangular_inverse(),
			] {
				// the smallest size whose join pays is the one the rule gives, and it is monotone
				let power = if r == rate::triangular_inverse() {
					3
				} else {
					2
				};
				let first = (2..4096)
					.find(|&n| halves_pays::<f64>(&par, n, r, power))
					.expect("never pays");
				// a team section costs 0.9 us at width 6 instead of the 18 us a plain token from
				// outside the pool paid, so the first paying join is small (was above 8)
				assert!(first > 1, "{first}");
				assert!(!halves_pays::<f64>(&par, first - 1, r, power));
				assert!(halves_pays::<f64>(&par, first, r, power));
				assert!(halves_pays::<f64>(&par, 2 * first, r, power));
				// work is the power of half the size, two halves
				let h = first / 2;
				let work_ns = 2.0 * (h as f64).powi(power as i32) * r.real;
				let want =
					PAYS * par.shape().dispatch_cost(nz(2)).as_secs_f64() * 1e9;
				assert!(work_ns >= want * 0.999, "{work_ns} {want}");
			}
		});
	}

	#[cfg(feature = "threadteam")]
	#[test]
	fn join_gates_on_a_team_follow_the_width_two_cost() {
		use threadteam::DispatchCosts;
		let team = ParShape::Team(
			nz(6),
			DispatchCosts::for_class(threadteam::HostClass::Aarch64Apple),
		);
		for (name, r, sections) in gates(M5) {
			assert!(floor_units::<f64>(team, r, sections) > 0, "{name}");
		}
		// a join is a section of width 2: 3.0 us on the team table, so 9 us of work
		let d2 = team.dispatch_cost(nz(2));
		assert_eq!(d2, Duration::from_micros(3));
		let par = |ns: f64| pays_at(Duration::from_secs_f64(ns * 1e-9), 1, d2);
		assert!(par(9000.0) && !par(8900.0));
		// representative joins on the M5 team table: the divide and conquer joins pay from a size
		// of about 20 (far below their recursion thresholds, 128 and 64), the triangular inverse
		// joins from about 150 (`2 (n/2)^3` at 0.0096 ns against 9 us)
		for (r, power, yes, no) in [
			(M5.bidiag_svd_dc, 2u32, 128usize, 8usize),
			(M5.tridiag_evd_dc, 2, 64, 8),
			(M5.triangular_inverse, 3, 256, 64),
		] {
			let pays = |n: usize| {
				let h = n / 2;
				pays_at(serial_time(2 * h.pow(power), r.real), 1, d2)
			};
			assert!(pays(yes));
			assert!(!pays(no));
		}
		// a sparse product of 1e6 multiply-adds is above the floor, 1e3 below
		assert!(team.pays_full(
			serial_time(1_000_000, M5.sparse_matmul.of::<f64>()),
			1
		));
		assert!(
			!team
				.pays_full(serial_time(1_000, M5.sparse_matmul.of::<f64>()), 1)
		);
	}
}
