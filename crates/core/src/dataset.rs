use crate::rng::Xoshiro256;

pub struct Dataset {
    pub dim: usize,
    pub vectors: Vec<f32>,
    pub queries: Vec<f32>,
}

impl Dataset {
    pub fn n(&self) -> usize {
        self.vectors.len() / self.dim
    }
}

pub fn uniform(n: usize, dim: usize, queries: usize, seed: u64) -> Dataset {
    let mut rng = Xoshiro256::new(seed);
    let scale = 1.0 / (dim as f32).sqrt();
    let mut vectors = Vec::with_capacity(n * dim);
    for _ in 0..n * dim {
        vectors.push(rng.next_gauss() * scale);
    }
    let mut qs = Vec::with_capacity(queries * dim);
    for _ in 0..queries * dim {
        qs.push(rng.next_gauss() * scale);
    }
    Dataset {
        dim,
        vectors,
        queries: qs,
    }
}

pub fn clustered(n: usize, dim: usize, clusters: usize, queries: usize, seed: u64) -> Dataset {
    let clusters = clusters.max(1);
    let mut rng = Xoshiro256::new(seed);
    let mut centers = Vec::with_capacity(clusters * dim);
    for _ in 0..clusters * dim {
        centers.push(rng.next_gauss());
    }
    let scale = 1.0 / (dim as f32).sqrt();
    let mut vectors = Vec::with_capacity(n * dim);
    for i in 0..n {
        let c = (i % clusters) * dim;
        for d in 0..dim {
            let v = centers[c + d] + rng.next_gauss() * 0.35;
            vectors.push(v * scale);
        }
    }
    let mut qs = Vec::with_capacity(queries * dim);
    for i in 0..queries {
        let c = (i % clusters) * dim;
        for d in 0..dim {
            let v = centers[c + d] + rng.next_gauss() * 0.35;
            qs.push(v * scale);
        }
    }
    Dataset {
        dim,
        vectors,
        queries: qs,
    }
}
