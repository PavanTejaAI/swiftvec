use swiftvec_core::dataset::{clustered, uniform};
use swiftvec_core::{calibrate_range, top_k, Hnsw, HnswConfig, Metric, Storage};
use std::collections::HashSet;

fn recall(hits: &[swiftvec_core::Hit], truth: &[swiftvec_core::Hit]) -> f32 {
    let t: HashSet<u32> = truth.iter().map(|h| h.id).collect();
    hits.iter().filter(|h| t.contains(&h.id)).count() as f32 / truth.len().max(1) as f32
}

#[test]
fn hnsw_recall_dot() {
    let n = 20000;
    let dim = 64;
    let qs = 100;
    let ds = clustered(n, dim, 128, qs, 7);
    let mut ix = Hnsw::with_capacity(HnswConfig::new(dim, Metric::Dot), n);
    for i in 0..n {
        ix.add(&ds.vectors[i * dim..(i + 1) * dim]);
    }
    let k = 10;
    let mut total = 0.0f32;
    for qi in 0..qs {
        let q = &ds.queries[qi * dim..(qi + 1) * dim];
        let truth = top_k(&ds.vectors, dim, q, k, Metric::Dot, None);
        let hits = ix.search(q, k, 128, None);
        assert_eq!(hits.len(), k);
        total += recall(&hits, &truth);
    }
    let avg = total / qs as f32;
    assert!(avg >= 0.95, "recall {avg}");
}

#[test]
fn hnsw_recall_l2() {
    let n = 5000;
    let dim = 32;
    let qs = 50;
    let ds = clustered(n, dim, 32, qs, 11);
    let mut ix = Hnsw::with_capacity(HnswConfig::new(dim, Metric::L2), n);
    for i in 0..n {
        ix.add(&ds.vectors[i * dim..(i + 1) * dim]);
    }
    let k = 10;
    let mut total = 0.0f32;
    for qi in 0..qs {
        let q = &ds.queries[qi * dim..(qi + 1) * dim];
        let truth = top_k(&ds.vectors, dim, q, k, Metric::L2, None);
        let hits = ix.search(q, k, 64, None);
        assert_eq!(hits.len(), k);
        total += recall(&hits, &truth);
    }
    let avg = total / qs as f32;
    assert!(avg >= 0.95, "recall {avg}");
}

#[test]
fn hnsw_filtered_recall() {
    let n = 10000;
    let dim = 32;
    let qs = 50;
    let ds = clustered(n, dim, 64, qs, 13);
    let mut ix = Hnsw::with_capacity(HnswConfig::new(dim, Metric::Dot), n);
    for i in 0..n {
        ix.add(&ds.vectors[i * dim..(i + 1) * dim]);
    }
    let k = 10;
    let mut total = 0.0f32;
    for qi in 0..qs {
        let q = &ds.queries[qi * dim..(qi + 1) * dim];
        let f = |id: u32| id % 2 == 0;
        let truth = top_k(&ds.vectors, dim, q, k, Metric::Dot, Some(&f));
        let hits = ix.search(q, k, 256, Some(&f));
        assert!(hits.iter().all(|h| h.id % 2 == 0));
        total += recall(&hits, &truth);
    }
    let avg = total / qs as f32;
    assert!(avg >= 0.9, "recall {avg}");
}

#[test]
fn hnsw_recall_int8() {
    let n = 20000;
    let dim = 64;
    let qs = 100;
    let ds = clustered(n, dim, 128, qs, 17);
    let mut cfg = HnswConfig::new(dim, Metric::Dot);
    cfg.storage = Storage::Int8;
    cfg.qrange = calibrate_range(&ds.vectors);
    let mut ix = Hnsw::with_capacity(cfg, n);
    for i in 0..n {
        ix.add(&ds.vectors[i * dim..(i + 1) * dim]);
    }
    let k = 10;
    let mut total = 0.0f32;
    for qi in 0..qs {
        let q = &ds.queries[qi * dim..(qi + 1) * dim];
        let truth = top_k(&ds.vectors, dim, q, k, Metric::Dot, None);
        let hits = ix.search(q, k, 128, None);
        assert_eq!(hits.len(), k);
        total += recall(&hits, &truth);
    }
    let avg = total / qs as f32;
    assert!(avg >= 0.9, "recall {avg}");
}

#[test]
fn hnsw_recall_uniform() {
    let n = 20000;
    let dim = 64;
    let qs = 100;
    let ds = uniform(n, dim, qs, 23);
    let mut ix = Hnsw::with_capacity(HnswConfig::new(dim, Metric::Dot), n);
    for i in 0..n {
        ix.add(&ds.vectors[i * dim..(i + 1) * dim]);
    }
    let k = 10;
    let mut total = 0.0f32;
    for qi in 0..qs {
        let q = &ds.queries[qi * dim..(qi + 1) * dim];
        let truth = top_k(&ds.vectors, dim, q, k, Metric::Dot, None);
        let hits = ix.search(q, k, 256, None);
        assert_eq!(hits.len(), k);
        total += recall(&hits, &truth);
    }
    let avg = total / qs as f32;
    assert!(avg >= 0.9, "recall {avg}");
}

#[test]
fn hnsw_recall_int8_rerank() {
    let n = 20000;
    let dim = 64;
    let qs = 100;
    let ds = uniform(n, dim, qs, 29);
    let mut cfg = HnswConfig::new(dim, Metric::Dot);
    cfg.storage = Storage::Int8;
    cfg.qrange = calibrate_range(&ds.vectors);
    cfg.rerank_f32 = true;
    let mut ix = Hnsw::with_capacity(cfg, n);
    for i in 0..n {
        ix.add(&ds.vectors[i * dim..(i + 1) * dim]);
    }
    let k = 10;
    let mut total = 0.0f32;
    for qi in 0..qs {
        let q = &ds.queries[qi * dim..(qi + 1) * dim];
        let truth = top_k(&ds.vectors, dim, q, k, Metric::Dot, None);
        let hits = ix.search(q, k, 256, None);
        assert_eq!(hits.len(), k);
        total += recall(&hits, &truth);
    }
    let avg = total / qs as f32;
    assert!(avg >= 0.93, "recall {avg}");
}

#[test]
fn packed_matches_unpacked() {
    let n = 5000;
    let dim = 32;
    let ds = clustered(n, dim, 32, 20, 31);
    let mut ix = Hnsw::with_capacity(HnswConfig::new(dim, Metric::Dot), n);
    for i in 0..n {
        ix.add(&ds.vectors[i * dim..(i + 1) * dim]);
    }
    let pre: Vec<_> = (0..20)
        .map(|qi| ix.search(&ds.queries[qi * dim..(qi + 1) * dim], 10, 128, None))
        .collect();
    ix.pack();
    for (qi, want) in pre.iter().enumerate() {
        let got = ix.search(&ds.queries[qi * dim..(qi + 1) * dim], 10, 128, None);
        assert_eq!(&got, want);
    }
}

#[test]
fn snapshot_round_trip() {
    let n = 5000;
    let dim = 32;
    let ds = clustered(n, dim, 32, 20, 37);
    let mut cfg = HnswConfig::new(dim, Metric::Dot);
    cfg.storage = Storage::Int8;
    cfg.qrange = calibrate_range(&ds.vectors);
    cfg.rerank_f32 = true;
    let mut ix = Hnsw::with_capacity(cfg, n);
    for i in 0..n {
        ix.add(&ds.vectors[i * dim..(i + 1) * dim]);
    }
    let mut buf = Vec::new();
    ix.save(&mut buf).unwrap();
    let mut loaded = Hnsw::load(&mut buf.as_slice()).unwrap();
    assert_eq!(loaded.len(), n);
    let extra = vec![0.01f32; dim];
    loaded.add(&extra);
    assert_eq!(loaded.len(), n + 1);
    for qi in 0..20 {
        let q = &ds.queries[qi * dim..(qi + 1) * dim];
        assert_eq!(ix.search(q, 10, 128, None), loaded.search(q, 10, 128, None));
    }
    let mut packed_buf = Vec::new();
    let mut packed_ix = Hnsw::load(&mut buf.as_slice()).unwrap();
    packed_ix.pack();
    packed_ix.save(&mut packed_buf).unwrap();
    let reloaded = Hnsw::load(&mut packed_buf.as_slice()).unwrap();
    for qi in 0..20 {
        let q = &ds.queries[qi * dim..(qi + 1) * dim];
        assert_eq!(ix.search(q, 10, 128, None), reloaded.search(q, 10, 128, None));
    }
}

#[test]
fn deterministic_build() {
    let n = 3000;
    let dim = 32;
    let ds = clustered(n, dim, 16, 5, 99);
    let mut a = Hnsw::with_capacity(HnswConfig::new(dim, Metric::Dot), n);
    let mut b = Hnsw::with_capacity(HnswConfig::new(dim, Metric::Dot), n);
    for i in 0..n {
        let v = &ds.vectors[i * dim..(i + 1) * dim];
        a.add(v);
        b.add(v);
    }
    for qi in 0..5 {
        let q = &ds.queries[qi * dim..(qi + 1) * dim];
        assert_eq!(a.search(q, 10, 128, None), b.search(q, 10, 128, None));
    }
}
