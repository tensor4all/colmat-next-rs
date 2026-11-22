//! Tests of the execution layer (#20): the producers.

use crate::internal_prelude::*;
use crate::utils::thread::{Producer, Zip};
use std::assert;

#[test]
fn ranges_split_anywhere_and_walk_in_order() {
	let r = 3..10usize;
	assert!(Producer::len(&r) == 7);
	for k in 0..=7 {
		let (a, b) = (3..10usize).split_at(k);
		assert!(a == (3..3 + k));
		assert!(b == (3 + k..10));
	}
	let items: Vec<_> = (2..5usize).zip_eq(10..13usize).into_items().collect();
	assert!(items == [(2, 10), (3, 11), (4, 12)]);
}

#[test]
#[should_panic]
fn zip_eq_rejects_different_lengths() {
	let _: Zip<_, _> = (0..3usize).zip_eq(0..4usize);
}

/// The chunks of a view: the same boundaries as colmat-rs's `par_*_chunks`, whatever the split.
#[test]
fn chunks_split_on_chunk_boundaries() {
	let a = Mat::from_fn(7, 10, |i, j| (10 * i + j) as f64);
	for size in 1..=11 {
		let p = a.as_ref().par_col_chunks(size);
		assert!(p.len() == 10usize.div_ceil(size));
		for k in 0..=p.len() {
			let (l, r) = a.as_ref().par_col_chunks(size).split_at(k);
			let mut col = 0;
			for c in l.into_items().chain(r.into_items()) {
				assert!(c.nrows() == 7 && c.ncols() >= 1 && c.ncols() <= size);
				for j in 0..c.ncols() {
					for i in 0..7 {
						assert!(c[(i, j)] == a[(i, col + j)]);
					}
				}
				col += c.ncols();
			}
			assert!(col == 10);
		}
		let p = a.as_ref().par_row_chunks(size);
		assert!(p.len() == 7usize.div_ceil(size));
		let rows: usize = p.into_items().map(|c| c.nrows()).sum();
		assert!(rows == 7);
	}
	// mutable chunks write disjoint parts that cover the matrix
	let mut b = Mat::<f64>::zeros(7, 10);
	let (l, r) = b.as_mut().par_col_chunks_mut(3).split_at(2);
	for (k, mut c) in l.into_items().chain(r.into_items()).enumerate() {
		for j in 0..c.ncols() {
			for i in 0..c.nrows() {
				c[(i, j)] += (k + 1) as f64;
			}
		}
	}
	for j in 0..10 {
		for i in 0..7 {
			assert!(b[(i, j)] == (j / 3 + 1) as f64);
		}
	}
	let mut b = Mat::<f64>::zeros(7, 10);
	for (k, mut c) in b.as_mut().par_row_chunks_mut(2).into_items().enumerate()
	{
		for j in 0..c.ncols() {
			for i in 0..c.nrows() {
				c[(i, j)] += (k + 1) as f64;
			}
		}
	}
	for j in 0..10 {
		for i in 0..7 {
			assert!(b[(i, j)] == (i / 2 + 1) as f64);
		}
	}
	// empty views give no chunk
	let e = Mat::<f64>::zeros(0, 5);
	assert!(e.as_ref().par_row_chunks(2).len() == 0);
	assert!(e.as_ref().par_col_chunks(2).len() == 3);
}
