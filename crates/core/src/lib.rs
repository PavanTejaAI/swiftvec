mod bm25;
mod brute;
pub mod dataset;
mod distance;
mod hit;
mod hnsw;
mod int8;
mod rng;
mod visited;

pub use bm25::{rrf_fuse, Bm25Index};
pub use brute::top_k;
pub use dataset::{clustered, uniform, Dataset};
pub use distance::{dot, l2sq, Metric};
pub use hit::Hit;
#[cfg(feature = "mmap")]
pub use hnsw::{MappedIndex, Mapping};
pub use hnsw::{Hnsw, HnswConfig, Query, Storage};
pub use int8::{calibrate_range, dot_i8, l2sq_i8, quantize_into};
pub use rng::Xoshiro256;
pub use visited::Visited;

pub struct SearchCtx {
    pub(crate) visited: Visited,
    pub(crate) q8: Vec<i8>,
}

impl SearchCtx {
    pub fn with_capacity(n: usize) -> Self {
        Self {
            visited: Visited::with_capacity(n),
            q8: Vec::new(),
        }
    }
}
