use crate::hit::Hit;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

const K1: f32 = 1.2;
const B: f32 = 0.75;
const RRF_K: f32 = 60.0;

fn tokenize(text: &str, out: &mut Vec<String>) {
    let lower = text.to_ascii_lowercase();
    let mut cur = String::new();
    for c in lower.chars() {
        if c.is_ascii_alphanumeric() {
            cur.push(c);
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
}

pub struct Bm25Index {
    postings: HashMap<String, Vec<(u32, u32)>>,
    doc_len: Vec<u32>,
    total_len: u64,
    n: u32,
}

impl Default for Bm25Index {
    fn default() -> Self {
        Self::new()
    }
}

impl Bm25Index {
    pub fn new() -> Self {
        Self {
            postings: HashMap::new(),
            doc_len: Vec::new(),
            total_len: 0,
            n: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.n as usize
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    pub fn add(&mut self, text: &str) -> u32 {
        let id = self.n;
        let mut toks = Vec::new();
        tokenize(text, &mut toks);
        self.total_len += toks.len() as u64;
        self.doc_len.push(toks.len() as u32);
        let mut tf: HashMap<String, u32> = HashMap::new();
        for t in toks {
            *tf.entry(t).or_insert(0) += 1;
        }
        for (t, f) in tf {
            self.postings.entry(t).or_default().push((id, f));
        }
        self.n += 1;
        id
    }

    pub fn search(&self, query: &str, k: usize) -> Vec<(f32, u32)> {
        if self.n == 0 || k == 0 {
            return Vec::new();
        }
        let mut toks = Vec::new();
        tokenize(query, &mut toks);
        toks.sort_unstable();
        toks.dedup();
        let avgdl = (self.total_len as f32 / self.n as f32).max(1.0);
        let mut scores: HashMap<u32, f32> = HashMap::new();
        for t in &toks {
            let post = match self.postings.get(t) {
                Some(p) => p,
                None => continue,
            };
            let df = post.len() as f32;
            let idf = (1.0 + (self.n as f32 - df + 0.5) / (df + 0.5)).ln();
            for &(doc, tf) in post {
                let dl = self.doc_len[doc as usize] as f32;
                let norm = tf as f32 + K1 * (1.0 - B + B * dl / avgdl);
                *scores.entry(doc).or_insert(0.0) += idf * tf as f32 * (K1 + 1.0) / norm;
            }
        }
        let mut heap: BinaryHeap<Reverse<Hit>> = BinaryHeap::with_capacity(k + 1);
        for (doc, s) in scores {
            let h = Hit { dist: s, id: doc };
            if heap.len() < k {
                heap.push(Reverse(h));
            } else if let Some(&Reverse(worst)) = heap.peek() {
                if h.dist > worst.dist {
                    *heap.peek_mut().unwrap() = Reverse(h);
                }
            }
        }
        let out: Vec<(f32, u32)> = heap
            .into_sorted_vec()
            .into_iter()
            .map(|Reverse(h)| (h.dist, h.id))
            .collect();
        out
    }
}

pub fn rrf_fuse(
    vector: &[Hit],
    text: &[(f32, u32)],
    k: usize,
    w_vector: f32,
    w_text: f32,
) -> Vec<(u32, f32)> {
    let mut acc: HashMap<u32, f32> = HashMap::new();
    for (rank, h) in vector.iter().enumerate() {
        *acc.entry(h.id).or_insert(0.0) += w_vector / (RRF_K + rank as f32 + 1.0);
    }
    for (rank, (_, id)) in text.iter().enumerate() {
        *acc.entry(*id).or_insert(0.0) += w_text / (RRF_K + rank as f32 + 1.0);
    }
    let mut v: Vec<(u32, f32)> = acc.into_iter().collect();
    v.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    v.truncate(k);
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bm25_ranks_relevant_doc_first() {
        let mut ix = Bm25Index::new();
        ix.add("The quick brown fox jumps over the lazy dog");
        ix.add("Machine learning models train on large datasets");
        ix.add("Deep learning uses neural networks with many layers");
        let hits = ix.search("neural networks deep", 2);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].1, 2);
        let none = ix.search("quantum entanglement", 3);
        assert!(none.is_empty());
    }

    #[test]
    fn rrf_prefers_docs_in_both_lists() {
        let vector = vec![
            Hit { dist: 0.1, id: 0 },
            Hit { dist: 0.2, id: 1 },
        ];
        let text = vec![(2.0f32, 1u32), (1.0, 5)];
        let fused = rrf_fuse(&vector, &text, 2, 1.0, 1.0);
        assert_eq!(fused.len(), 2);
        assert_eq!(fused[0].0, 1);
    }
}
