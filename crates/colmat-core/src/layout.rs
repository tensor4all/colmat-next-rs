//! Copies between a column-major and a row-major matrix of the same shape.
//!
//! An element-wise copy walks the destination in its own order, so when the
//! source has the other layout every read steps by the source's major stride.
//! These copies move [`COPY_TILE`](crate::layout::COPY_TILE)-sided tiles instead: each tile is read
//! along the row-major side and written along the column-major side.
//!
//! # Provenance
//!
//! Ported from colmat-rs `faer/src/linalg/mod.rs` at `71337cdc` (lines
//! 93-217 and 225-600: `COPY_TILE`, `TILE_COPY_CACHE_BYTES`,
//! `TILE_COPY_POWER_OF_TWO_STRIDE_BYTES`, `tile_copy_pays`,
//! `copy_across_layouts`, `copy_across_layouts_block`,
//! `copy_lower_across_layouts`). Changed: the serial path only, and
//! `copy_across_layouts` is crate-private until it takes `Par`. The parallel
//! split of `copy_across_layouts` (and its `Par` argument) depends on the
//! execution layer and is left for the next step of the design record (§7);
//! `column_major_copy` is left with it. The thresholds and the measurements
//! they cite are colmat-rs's (the `tensor4all/faer` issue numbers are
//! colmat-rs's former repository) and have not been re-measured here. faer's
//! MIT notice: `LICENSE-faer`.

use crate::mat::{MatMut, MatRef};
use reborrow::*;

/// The side of the tiles the layout copies move: 16 elements is a cache line
/// or more for every scalar the copies take (`f32` is 64 bytes).
pub const COPY_TILE: usize = 16;

/// The working set, in bytes over both buffers, from which staging a layout
/// copy in [`COPY_TILE`] tiles pays.
///
/// It is the pair of buffers of a `512 x 512` f64 copy: the first cell of
/// colmat-rs's measurement (`tensor4all/faer#323`, `Llt::new` on a row-major
/// input) where the tiled path wins, while `511 x 511` still loses by 3-5%.
/// The boundary follows the bytes copied, so an f32 copy of the same shape
/// crosses at `725 x 725` instead.
const TILE_COPY_CACHE_BYTES: usize = 2 * 512 * 512 * 8;

/// The destination column stride, in bytes, above which a power-of-two stride
/// makes the tiles pay below [`TILE_COPY_CACHE_BYTES`], together with a
/// power-of-two source row stride.
///
/// colmat-rs measured (`tensor4all/faer#323`, `#352`) that below the working
/// set the element-wise walk is faster unless the source's row stride is a
/// power of two (L1 set aliasing on the strided reads); the destination side
/// alone moves nothing. The clause therefore needs both strides.
const TILE_COPY_POWER_OF_TWO_STRIDE_BYTES: usize = 2048;

#[cfg(all(test, feature = "std"))]
std::thread_local! {
	static TILED_COPIES: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}

/// Test-only count of the copies [`copy_lower_across_layouts`] took in tiles
/// on the calling thread (per thread, because tests run concurrently).
#[cfg(all(test, feature = "std"))]
pub(crate) fn tiled_copies() -> usize {
	TILED_COPIES.with(|c| c.get())
}

/// Whether staging a `nrows x ncols` layout copy in [`COPY_TILE`] tiles pays,
/// for elements of `size_of::<T>()` bytes, a destination column stride of
/// `col_stride` elements and a source row stride of `row_stride` elements.
///
/// Two boundaries: the working set (`TILE_COPY_CACHE_BYTES`, counting both
/// buffers whole), and a pair of power-of-two strides with the destination's
/// at or above `TILE_COPY_POWER_OF_TWO_STRIDE_BYTES`.
#[inline]
pub fn tile_copy_pays<T>(
	nrows: usize,
	ncols: usize,
	col_stride: isize,
	row_stride: isize,
) -> bool {
	if nrows
		.saturating_mul(ncols)
		.saturating_mul(2 * core::mem::size_of::<T>())
		>= TILE_COPY_CACHE_BYTES
	{
		return true;
	}
	let size = core::mem::size_of::<T>();
	let col_stride_bytes = col_stride.unsigned_abs().saturating_mul(size);
	let row_stride_bytes = row_stride.unsigned_abs().saturating_mul(size);
	col_stride_bytes.is_power_of_two()
		&& col_stride_bytes >= TILE_COPY_POWER_OF_TWO_STRIDE_BYTES
		&& row_stride_bytes.is_power_of_two()
}

/// Copies `src` into `dst` elementwise, where the two have transposed
/// strides: one is column-major (row stride `±1`), the other is not, over the
/// same shape, in tiles.
///
/// # Panics
///
/// Panics if the shapes differ.
///
/// Crate-private while it is serial only: the public entry takes a `Par`
/// argument and is added with the execution layer (design record §7, step 2),
/// so that no serial-only public name has to be broken then.
#[track_caller]
pub(crate) fn copy_across_layouts<T: Clone>(
	dst: MatMut<'_, T>,
	src: MatRef<'_, T>,
) {
	equator::assert!(all(
		dst.nrows() == src.nrows(),
		dst.ncols() == src.ncols(),
	));
	copy_across_layouts_block(dst, src);
}

/// The tiled copy, on the calling thread: read each tile along whichever side
/// is row-major and write it along whichever is column-major, so neither side
/// steps by the other's row stride. Each source value is cloned into an
/// `Option` tile and moved into the destination, without another clone or a
/// seed value. Every slot the second loop reads was written by the first.
fn copy_across_layouts_block<T: Clone>(
	mut dst: MatMut<'_, T>,
	src: MatRef<'_, T>,
) {
	const B: usize = COPY_TILE;
	let (m, n) = dst.shape();
	// `src` row-major means `dst` is the column-major one, and the other way
	// round. When both or neither are, either order is correct.
	let src_row_major = src.row_stride().unsigned_abs() != 1;
	let mut tile: [Option<T>; B * B] = core::array::from_fn(|_| None);
	let mut jb = 0;
	while jb < n {
		let j1 = Ord::min(jb + B, n);
		let mut ib = 0;
		while ib < m {
			let i1 = Ord::min(ib + B, m);
			if src_row_major {
				for i in ib..i1 {
					for j in jb..j1 {
						tile[(i - ib) * B + (j - jb)] =
							Some(src[(i, j)].clone());
					}
				}
				for j in jb..j1 {
					for i in ib..i1 {
						dst[(i, j)] =
							tile[(i - ib) * B + (j - jb)].take().unwrap();
					}
				}
			} else {
				for j in jb..j1 {
					for i in ib..i1 {
						tile[(i - ib) * B + (j - jb)] =
							Some(src[(i, j)].clone());
					}
				}
				for i in ib..i1 {
					for j in jb..j1 {
						dst[(i, j)] =
							tile[(i - ib) * B + (j - jb)].take().unwrap();
					}
				}
			}
			ib += B;
		}
		jb += B;
	}
}

/// Copies the lower triangle of `src`, diagonal included, into `dst` in
/// tiles, when `dst` is column-major, `src` is not, and [`tile_copy_pays`]
/// holds for the working set and the strides; returns `false` and touches
/// nothing otherwise.
///
/// The elements written are exactly those
/// `MatMut::copy_from_triangular_lower` writes (column `j < min(m, n)`, rows
/// `j..m`), so the upper triangle of `dst` is left as it was. Each
/// [`COPY_TILE`]-column band is the diagonal tile, copied elementwise below
/// its diagonal, and the rectangle under it, copied in tiles.
pub(crate) fn copy_lower_across_layouts<T: Clone>(
	mut dst: MatMut<'_, T>,
	src: MatRef<'_, T>,
) -> bool {
	let (m, n) = dst.shape();
	if dst.row_stride() != 1
		|| src.row_stride().unsigned_abs() == 1
		|| !tile_copy_pays::<T>(m, n, dst.col_stride(), src.row_stride())
	{
		return false;
	}
	let k = Ord::min(m, n);
	let mut jb = 0;
	while jb < k {
		let j1 = Ord::min(jb + COPY_TILE, k);
		for j in jb..j1 {
			for i in j..j1 {
				dst[(i, j)] = src[(i, j)].clone();
			}
		}
		if j1 < m {
			copy_across_layouts_block(
				dst.rb_mut().submatrix_mut(j1, jb, m - j1, j1 - jb),
				src.submatrix(j1, jb, m - j1, j1 - jb),
			);
		}
		jb = j1;
	}
	#[cfg(all(test, feature = "std"))]
	TILED_COPIES.with(|c| c.set(c.get() + 1));
	true
}
