//! Helper utilities: bound-checked indices, and the execution layer (the team skeletons, the
//! cost model and the static partitions of a region).
#![allow(missing_docs)]
/// compile-time bound-checked indexing types
pub mod bound;
pub mod cost;
pub mod partition;
#[cfg(feature = "threadteam")]
pub mod team;
pub mod thread;
