//! The ported index helpers.

use crate::{Index, IndexCore, SignedIndex};

#[test]
fn unsigned_truncate_and_extend() {
	assert_eq!(<u8 as IndexCore>::truncate(0x1_23), 0x23);
	assert_eq!(<u16 as IndexCore>::truncate(0x1_2345), 0x2345);
	assert_eq!(<u32 as IndexCore>::truncate(7), 7u32);
	assert_eq!(<u64 as IndexCore>::truncate(7), 7u64);
	assert_eq!(<u128 as IndexCore>::truncate(7), 7u128);
	assert_eq!(<usize as IndexCore>::truncate(7), 7usize);
	assert_eq!(0xABu8.zx(), 0xAB);
	assert_eq!(0xABCDu16.zx(), 0xABCD);
	assert_eq!(u32::MAX.zx(), u32::MAX as usize);
	assert_eq!(5u64.zx(), 5);
	assert_eq!(5u128.zx(), 5);
	assert_eq!(5usize.zx(), 5);
}

#[test]
fn signed_truncate_and_extend() {
	assert_eq!(<i32 as SignedIndex>::truncate(5), 5i32);
	assert_eq!(<i64 as SignedIndex>::truncate(5), 5i64);
	assert_eq!(<isize as SignedIndex>::truncate(5), 5isize);
	// Zero extension keeps the bits of the narrow type; sign extension
	// replicates the sign bit.
	assert_eq!((-1i32).zx(), u32::MAX as usize);
	assert_eq!((-1i32).sx(), usize::MAX);
	assert_eq!(3i32.sx(), 3);
	assert_eq!((-1i64).zx(), u64::MAX as usize);
	assert_eq!((-1i64).sx(), usize::MAX);
	assert_eq!(3i64.zx(), 3);
	assert_eq!((-1isize).zx(), usize::MAX);
	assert_eq!((-2isize).sx(), usize::MAX - 1);
	assert_eq!(3isize.zx(), 3);
}

#[test]
fn sums_detect_overflow() {
	assert_eq!(i32::sum_nonnegative(&[1, 2, 3]), Some(6));
	assert_eq!(i32::sum_nonnegative(&[i32::MAX, 0]), Some(i32::MAX));
	assert_eq!(i32::sum_nonnegative(&[i32::MAX, 1]), None);
	assert_eq!(u32::sum_nonnegative(&[1, 2]), Some(3));
	assert_eq!(u32::sum_nonnegative(&[i32::MAX as u32, 1]), None);
}

#[test]
fn canonical_views_and_signed_casts() {
	let v = [1usize, 2, 3];
	assert_eq!(
		usize::canonicalize(&v)
			.iter()
			.map(|&x| x as usize)
			.collect::<Vec<_>>(),
		v
	);
	let mut w = [1usize, 2];
	usize::canonicalize_mut(&mut w)[1] = 9;
	assert_eq!(w, [1, 9]);
	assert_eq!(u32::from_signed(-1i32), u32::MAX);
	assert_eq!(u32::MAX.to_signed(), -1i32);
	assert_eq!(<u64 as Index>::BITS, 64);
}
