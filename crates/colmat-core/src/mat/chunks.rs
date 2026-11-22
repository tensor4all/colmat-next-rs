//! Chunks of a matrix view, dealt out to a team ([`Producer`]).
//!
//! The replacement of colmat-rs's rayon iterators of the views (`par_col_chunks`,
//! `par_row_chunks` and their `_mut` forms, `faer/src/mat/matref.rs` and `matmut.rs` at
//! `71337cdc`): the same names and chunk boundaries, but the value is a [`Producer`] of this
//! crate, which `utils::thread::for_each` splits on the leader of a team
//! (debt K7). The `par_*_iter` and `par_*_partition` forms are not ported yet. faer's MIT
//! notice: `LICENSE-faer`.

use super::{MatMut, MatRef};
use crate::utils::thread::Producer;
use crate::{Shape, Stride};

/// The columns of a matrix view in chunks of at most `size` columns
/// ([`MatRef::par_col_chunks`]).
pub struct ColChunks<'a, T, Rows, RStride, CStride> {
	mat: MatRef<'a, T, Rows, usize, RStride, CStride>,
	size: usize,
}

/// The rows of a matrix view in chunks of at most `size` rows
/// ([`MatRef::par_row_chunks`]).
pub struct RowChunks<'a, T, Cols, RStride, CStride> {
	mat: MatRef<'a, T, usize, Cols, RStride, CStride>,
	size: usize,
}

/// The columns of a mutable matrix view in chunks of at most `size` columns
/// ([`MatMut::par_col_chunks_mut`]).
pub struct ColChunksMut<'a, T, Rows, RStride, CStride> {
	mat: Option<MatMut<'a, T, Rows, usize, RStride, CStride>>,
	size: usize,
}

/// The rows of a mutable matrix view in chunks of at most `size` rows
/// ([`MatMut::par_row_chunks_mut`]).
pub struct RowChunksMut<'a, T, Cols, RStride, CStride> {
	mat: Option<MatMut<'a, T, usize, Cols, RStride, CStride>>,
	size: usize,
}

impl<'a, T, Rows: Shape, Cols: Shape, RStride: Stride, CStride: Stride>
	MatRef<'a, T, Rows, Cols, RStride, CStride>
{
	/// The columns of this matrix in chunks of `chunk_size` columns, the last one possibly
	/// narrower, as a [`Producer`] for `utils::thread::for_each`.
	///
	/// # Panics
	///
	/// Panics if `chunk_size == 0`.
	#[track_caller]
	#[inline]
	pub fn par_col_chunks(
		self,
		chunk_size: usize,
	) -> ColChunks<'a, T, Rows, RStride, CStride> {
		equator::assert!(chunk_size > 0);
		ColChunks {
			mat: self.as_dyn_cols(),
			size: chunk_size,
		}
	}

	/// The rows of this matrix in chunks of `chunk_size` rows, the last one possibly
	/// shorter, as a [`Producer`] for `utils::thread::for_each`.
	///
	/// # Panics
	///
	/// Panics if `chunk_size == 0`.
	#[track_caller]
	#[inline]
	pub fn par_row_chunks(
		self,
		chunk_size: usize,
	) -> RowChunks<'a, T, Cols, RStride, CStride> {
		equator::assert!(chunk_size > 0);
		RowChunks {
			mat: self.as_dyn_rows(),
			size: chunk_size,
		}
	}
}

impl<'a, T, Rows: Shape, Cols: Shape, RStride: Stride, CStride: Stride>
	MatMut<'a, T, Rows, Cols, RStride, CStride>
{
	/// see [`MatRef::par_col_chunks`]
	///
	/// # Panics
	///
	/// Panics if `chunk_size == 0`.
	#[track_caller]
	#[inline]
	pub fn par_col_chunks_mut(
		self,
		chunk_size: usize,
	) -> ColChunksMut<'a, T, Rows, RStride, CStride> {
		equator::assert!(chunk_size > 0);
		ColChunksMut {
			mat: Some(self.as_dyn_cols_mut()),
			size: chunk_size,
		}
	}

	/// see [`MatRef::par_row_chunks`]
	///
	/// # Panics
	///
	/// Panics if `chunk_size == 0`.
	#[track_caller]
	#[inline]
	pub fn par_row_chunks_mut(
		self,
		chunk_size: usize,
	) -> RowChunksMut<'a, T, Cols, RStride, CStride> {
		equator::assert!(chunk_size > 0);
		RowChunksMut {
			mat: Some(self.as_dyn_rows_mut()),
			size: chunk_size,
		}
	}
}

impl<'a, T: Sync, Rows: Shape, RStride: Stride, CStride: Stride> Iterator
	for ColChunks<'a, T, Rows, RStride, CStride>
{
	type Item = MatRef<'a, T, Rows, usize, RStride, CStride>;

	#[inline]
	fn next(&mut self) -> Option<Self::Item> {
		let n = self.mat.ncols();
		if n == 0 {
			return None;
		}
		let (head, tail) = self.mat.split_at_col(Ord::min(self.size, n));
		self.mat = tail;
		Some(head)
	}
}

impl<'a, T: Sync, Rows: Shape, RStride: Stride, CStride: Stride> Producer
	for ColChunks<'a, T, Rows, RStride, CStride>
{
	type IntoIter = Self;
	type Item = MatRef<'a, T, Rows, usize, RStride, CStride>;

	#[inline]
	fn len(&self) -> usize {
		self.mat.ncols().div_ceil(self.size)
	}

	#[inline]
	#[track_caller]
	fn split_at(self, index: usize) -> (Self, Self) {
		equator::assert!(index <= self.len());
		let col = Ord::min(index * self.size, self.mat.ncols());
		let (a, b) = self.mat.split_at_col(col);
		(
			Self {
				mat: a,
				size: self.size,
			},
			Self {
				mat: b,
				size: self.size,
			},
		)
	}

	#[inline]
	fn into_items(self) -> Self {
		self
	}
}

impl<'a, T: Sync, Cols: Shape, RStride: Stride, CStride: Stride> Iterator
	for RowChunks<'a, T, Cols, RStride, CStride>
{
	type Item = MatRef<'a, T, usize, Cols, RStride, CStride>;

	#[inline]
	fn next(&mut self) -> Option<Self::Item> {
		let m = self.mat.nrows();
		if m == 0 {
			return None;
		}
		let (head, tail) = self.mat.split_at_row(Ord::min(self.size, m));
		self.mat = tail;
		Some(head)
	}
}

impl<'a, T: Sync, Cols: Shape, RStride: Stride, CStride: Stride> Producer
	for RowChunks<'a, T, Cols, RStride, CStride>
{
	type IntoIter = Self;
	type Item = MatRef<'a, T, usize, Cols, RStride, CStride>;

	#[inline]
	fn len(&self) -> usize {
		self.mat.nrows().div_ceil(self.size)
	}

	#[inline]
	#[track_caller]
	fn split_at(self, index: usize) -> (Self, Self) {
		equator::assert!(index <= self.len());
		let row = Ord::min(index * self.size, self.mat.nrows());
		let (a, b) = self.mat.split_at_row(row);
		(
			Self {
				mat: a,
				size: self.size,
			},
			Self {
				mat: b,
				size: self.size,
			},
		)
	}

	#[inline]
	fn into_items(self) -> Self {
		self
	}
}

impl<'a, T: Send, Rows: Shape, RStride: Stride, CStride: Stride> Iterator
	for ColChunksMut<'a, T, Rows, RStride, CStride>
{
	type Item = MatMut<'a, T, Rows, usize, RStride, CStride>;

	#[inline]
	fn next(&mut self) -> Option<Self::Item> {
		let mat = self.mat.take()?;
		let n = mat.ncols();
		if n == 0 {
			return None;
		}
		let (head, tail) = mat.split_at_col_mut(Ord::min(self.size, n));
		self.mat = Some(tail);
		Some(head)
	}
}

impl<'a, T: Send, Rows: Shape, RStride: Stride, CStride: Stride> Producer
	for ColChunksMut<'a, T, Rows, RStride, CStride>
{
	type IntoIter = Self;
	type Item = MatMut<'a, T, Rows, usize, RStride, CStride>;

	#[inline]
	fn len(&self) -> usize {
		self.mat
			.as_ref()
			.map_or(0, |m| m.ncols().div_ceil(self.size))
	}

	#[inline]
	#[track_caller]
	fn split_at(self, index: usize) -> (Self, Self) {
		equator::assert!(index <= self.len());
		let size = self.size;
		match self.mat {
			None => (Self { mat: None, size }, Self { mat: None, size }),
			Some(mat) => {
				let col = Ord::min(index * size, mat.ncols());
				let (a, b) = mat.split_at_col_mut(col);
				(Self { mat: Some(a), size }, Self { mat: Some(b), size })
			},
		}
	}

	#[inline]
	fn into_items(self) -> Self {
		self
	}
}

impl<'a, T: Send, Cols: Shape, RStride: Stride, CStride: Stride> Iterator
	for RowChunksMut<'a, T, Cols, RStride, CStride>
{
	type Item = MatMut<'a, T, usize, Cols, RStride, CStride>;

	#[inline]
	fn next(&mut self) -> Option<Self::Item> {
		let mat = self.mat.take()?;
		let m = mat.nrows();
		if m == 0 {
			return None;
		}
		let (head, tail) = mat.split_at_row_mut(Ord::min(self.size, m));
		self.mat = Some(tail);
		Some(head)
	}
}

impl<'a, T: Send, Cols: Shape, RStride: Stride, CStride: Stride> Producer
	for RowChunksMut<'a, T, Cols, RStride, CStride>
{
	type IntoIter = Self;
	type Item = MatMut<'a, T, usize, Cols, RStride, CStride>;

	#[inline]
	fn len(&self) -> usize {
		self.mat
			.as_ref()
			.map_or(0, |m| m.nrows().div_ceil(self.size))
	}

	#[inline]
	#[track_caller]
	fn split_at(self, index: usize) -> (Self, Self) {
		equator::assert!(index <= self.len());
		let size = self.size;
		match self.mat {
			None => (Self { mat: None, size }, Self { mat: None, size }),
			Some(mat) => {
				let row = Ord::min(index * size, mat.nrows());
				let (a, b) = mat.split_at_row_mut(row);
				(Self { mat: Some(a), size }, Self { mat: Some(b), size })
			},
		}
	}

	#[inline]
	fn into_items(self) -> Self {
		self
	}
}
