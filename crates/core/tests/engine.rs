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

#[test]
fn cascade_recall_clustered() {
    let n = 20000;
    let dim = 64;
    let qs = 100;
    let ds = clustered(n, dim, 128, qs, 41);
    let mut cfg = HnswConfig::new(dim, Metric::Dot);
    cfg.storage = Storage::Int8;
    cfg.qrange = calibrate_range(&ds.vectors);
    cfg.rerank_f32 = true;
    cfg.cascade = true;
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
        total += recall(&hits, &truth);
    }
    let avg = total / qs as f32;
    assert!(avg >= 0.93, "cascade clustered recall {avg}");
}

#[test]
fn cascade_recall_uniform() {
    let n = 20000;
    let dim = 64;
    let qs = 100;
    let ds = uniform(n, dim, qs, 43);
    let mut cfg = HnswConfig::new(dim, Metric::Dot);
    cfg.storage = Storage::Int8;
    cfg.qrange = calibrate_range(&ds.vectors);
    cfg.rerank_f32 = true;
    cfg.cascade = true;
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
        total += recall(&hits, &truth);
    }
    let avg = total / qs as f32;
    assert!(avg >= 0.90, "cascade uniform recall {avg}");
}

#[test]
fn cascade_snapshot_round_trip() {
    let n = 5000;
    let dim = 32;
    let ds = clustered(n, dim, 32, 20, 47);
    let mut cfg = HnswConfig::new(dim, Metric::Dot);
    cfg.storage = Storage::Int8;
    cfg.qrange = calibrate_range(&ds.vectors);
    cfg.rerank_f32 = true;
    cfg.cascade = true;
    let mut ix = Hnsw::with_capacity(cfg, n);
    for i in 0..n {
        ix.add(&ds.vectors[i * dim..(i + 1) * dim]);
    }
    ix.pack();
    let mut buf = Vec::new();
    ix.save(&mut buf).unwrap();
    let loaded = Hnsw::load(&mut buf.as_slice()).unwrap();
    assert_eq!(loaded.len(), n);
    for qi in 0..20 {
        let q = &ds.queries[qi * dim..(qi + 1) * dim];
        assert_eq!(
            ix.search(q, 10, 128, None),
            loaded.search(q, 10, 128, None)
        );
    }
}

#[test]
fn v1_reader_rejects_v2_and_v2_reads_v2() {
    let n = 2000;
    let dim = 16;
    let ds = clustered(n, dim, 8, 10, 53);
    let mut ix = Hnsw::with_capacity(HnswConfig::new(dim, Metric::Dot), n);
    for i in 0..n {
        ix.add(&ds.vectors[i * dim..(i + 1) * dim]);
    }
    let mut buf = Vec::new();
    ix.save(&mut buf).unwrap();
    let magic_ok = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    assert_eq!(magic_ok, 0x5357_5643);
    let ver = u32::from_le_bytes([
        buf[4],
        buf[5],
        buf[6],
        buf[7],
    ]);
    assert_eq!(ver, 2);
    Hnsw::load(&mut buf.as_slice()).unwrap();
}

#[cfg(feature = "mmap")]
mod mmap_tests {
    use swiftvec_core::{calibrate_range, top_k, Hnsw, HnswConfig, Mapping, MappedIndex, Metric, Storage};
    use swiftvec_core::dataset::clustered;

    #[test]
    fn mapped_matches_owned() {
        let n = 5000;
        let dim = 32;
        let ds = clustered(n, dim, 32, 20, 59);
        let mut cfg = HnswConfig::new(dim, Metric::Dot);
        cfg.storage = Storage::Int8;
        cfg.qrange = calibrate_range(&ds.vectors);
        cfg.rerank_f32 = true;
        let mut ix = Hnsw::with_capacity(cfg, n);
        for i in 0..n {
            ix.add(&ds.vectors[i * dim..(i + 1) * dim]);
        }
        ix.pack();
        let dir = std::env::temp_dir().join(format!("swiftvec-mmap-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ix.swiftvec");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            use std::io::Write;
            ix.save(&mut f).unwrap();
        }
        let mapping = Mapping::open(&path).unwrap();
        let view = MappedIndex::decode(mapping.data()).unwrap();
        assert_eq!(view.len(), n);
        for qi in 0..20 {
            let q = &ds.queries[qi * dim..(qi + 1) * dim];
            let truth = top_k(&ds.vectors, dim, q, 10, Metric::Dot, None);
            let owned = ix.search(q, 10, 128, None);
            let mapped = view.search(q, 10, 128, None);
            assert_eq!(owned, mapped);
            let t: std::collections::HashSet<u32> = truth.iter().map(|h| h.id).collect();
            let r = owned.iter().filter(|h| t.contains(&h.id)).count() as f32 / 10.0;
            assert!(r >= 0.9);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn mapped_rejects_unpackable() {
        let path = std::env::temp_dir().join(format!("swiftvec-mmap-neg-{}.bin", std::process::id()));
        std::fs::write(&path, [0u8; 64]).unwrap();
        let mapping = Mapping::open(&path).unwrap();
        assert!(MappedIndex::decode(mapping.data()).is_err());
        std::fs::remove_file(&path).ok();
    }
}
