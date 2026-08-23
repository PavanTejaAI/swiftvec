use crate::distance::Metric;
use crate::hit::Hit;
use std::collections::BinaryHeap;

pub fn top_k(
    vectors: &[f32],
    dim: usize,
    query: &[f32],
    k: usize,
    metric: Metric,
    filter: Option<&dyn Fn(u32) -> bool>,
) -> Vec<Hit> {
    let mut heap: BinaryHeap<Hit> = BinaryHeap::with_capacity(k + 1);
    for id in 0..vectors.len() / dim {
        if let Some(f) = filter {
            if !f(id as u32) {
                continue;
            }
        }
        let off = id * dim;
        let h = Hit {
            dist: metric.distance(query, &vectors[off..off + dim]),
            id: id as u32,
        };
        if heap.len() < k {
            heap.push(h);
        } else if let Some(&top) = heap.peek() {
            if h < top {
                *heap.peek_mut().unwrap() = h;
            }
        }
    }
    heap.into_sorted_vec()
}
