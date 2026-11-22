//! The cost weight of a scalar (S3, optional).
//!
//! New in this trial (#20); replaces colmat-rs's `Rate::of<T>`
//! (`faer/src/utils/cost.rs` at `71337cdc`, lines 259-273), which chose a
//! rate by `IS_NATIVE_F32`/`IS_NATIVE_C32`. The scalar now states its own
//! weight against `f64`, so the cost model never asks which type it has.

/// The serial time of a scalar's operations as a multiple of `f64`'s (of
/// `c64`'s for a complex type), for the two kinds of kernel family.
///
/// The cost model of `colmat-core` measures each family's rate for `f64`
/// and `c64`, and multiplies it by the weight of the scalar. A family is
/// either compute-bound (the time follows the arithmetic: products,
/// triangular solves, block reflectors) or memory-bound (the time follows
/// the bytes moved: copies, swaps, matrix-vector products, rank-one
/// updates); a scalar that costs much more arithmetic per byte than `f64`
/// has different ratios for the two (design document §8: double-double is
/// about 2–3 times `f64` on level-1 work and 12–16 on a product).
///
/// The default of [`crate::ComplexField::COST`] is [`CostWeight::F64`]: an
/// unmeasured scalar is priced as `f64`, a lower bound for any extended
/// precision type, so its regions stay serial a little longer than they
/// need to rather than dispatching too early.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CostWeight {
	/// Weight of the compute-bound families.
	pub compute: f64,
	/// Weight of the memory-bound families.
	pub memory: f64,
}

impl CostWeight {
	/// The weight of `f64` (and of `c64` against the complex rates): one.
	pub const F64: Self = Self {
		compute: 1.0,
		memory: 1.0,
	};

	/// The weight of `f32` (and of `c32`): half of `f64` for both kinds,
	/// twice the lanes per register and half the bytes. colmat-rs used the
	/// same ratio for every family but the GEMM, where it had a separate
	/// `f32` measurement that agreed with it to 0.2 percent.
	pub const F32: Self = Self {
		compute: 0.5,
		memory: 0.5,
	};
}
