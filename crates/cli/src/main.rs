use clap::{Parser, Subcommand, ValueEnum};
use swiftvec_core::dataset::{clustered, uniform};
use swiftvec_core::{calibrate_range, top_k, Hnsw, HnswConfig, Hit, Metric, SearchCtx, Storage};
use swiftvec_embed::{EmbedOptions, Embedder};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

const MOSS_QUERIES: [&str; 15] = [
    "neural network training data",
    "anomaly detection patterns",
    "computer vision image processing",
    "natural language processing",
    "reinforcement learning rewards",
    "transfer learning pretrained models",
    "distributed computing systems",
    "cryptographic data encryption",
    "database indexing performance",
    "knowledge graph entities",
    "generative adversarial networks",
    "attention mechanism transformers",
    "dimensionality reduction compression",
    "federated learning privacy",
    "stream processing pipelines",
];

#[derive(Parser)]
#[command(name = "swiftvec", about = "fully on-device vector search engine")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Bench(BenchArgs),
    Embed(EmbedArgs),
    Live(LiveArgs),
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum StorageArg {
    F32,
    Int8,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, ValueEnum)]
enum DataArg {
    Clustered,
    Uniform,
}

#[derive(clap::Args)]
struct BenchArgs {
    #[arg(long, default_value_t = 100_000)]
    n: usize,
    #[arg(long, default_value_t = 128)]
    dim: usize,
    #[arg(long, default_value_t = 128)]
    clusters: usize,
    #[arg(long, default_value_t = 200)]
    queries: usize,
    #[arg(long, default_value_t = 10)]
    k: usize,
    #[arg(long, value_delimiter = ',', default_value = "64,128,256")]
    ef: Vec<usize>,
    #[arg(long, default_value = "dot")]
    metric: String,
    #[arg(long)]
    filter: Option<u32>,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 16)]
    m: usize,
    #[arg(long, default_value_t = 200)]
    ef_construction: usize,
    #[arg(long, value_enum, default_value = "f32")]
    storage: StorageArg,
    #[arg(long)]
    qrange: Option<f32>,
    #[arg(long, value_enum, default_value = "clustered")]
    data: DataArg,
    #[arg(long)]
    rerank: bool,
}

#[derive(clap::Args)]
struct EmbedArgs {
    #[arg(long, default_value = "models/leaf-ir")]
    model: PathBuf,
    #[arg(long, required = true)]
    text: Vec<String>,
    #[arg(long)]
    query: bool,
    #[arg(long)]
    dim: Option<usize>,
    #[arg(long)]
    threads: Option<usize>,
    #[arg(long, default_value = "model_quantized.onnx")]
    onnx: String,
}

#[derive(clap::Args)]
struct LiveArgs {
    #[arg(long, default_value = "models/leaf-ir")]
    model: PathBuf,
    #[arg(long, default_value = "benchmarks/data/bench_100k_docs.json")]
    corpus: PathBuf,
    #[arg(long, default_value = "benchmarks/data/corpus-embeddings.bin")]
    cache: PathBuf,
    #[arg(long, default_value_t = 100_000)]
    docs: usize,
    #[arg(long, default_value_t = 50)]
    rounds: usize,
    #[arg(long, default_value_t = 3)]
    warmup: usize,
    #[arg(long, value_enum, default_value = "f32")]
    storage: StorageArg,
    #[arg(long, default_value_t = 0)]
    dim: usize,
    #[arg(long, default_value_t = 5)]
    k: usize,
    #[arg(long, value_delimiter = ',', default_value = "16,32,64,128")]
    ef: Vec<usize>,
    #[arg(long, default_value_t = 16)]
    m: usize,
    #[arg(long, default_value_t = 200)]
    ef_construction: usize,
    #[arg(long, default_value_t = 4)]
    embed_threads: usize,
    #[arg(long)]
    rerank: bool,
    #[arg(long)]
    rebuild: bool,
}

fn pct_f(sorted: &[f64], p: f64) -> f64 {
    let idx = (p * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn pct_u(sorted: &[u128], p: f64) -> u128 {
    let idx = (p * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn recall_of(hits: &[Hit], truth: &[Hit]) -> f32 {
    let t: HashSet<u32> = truth.iter().map(|h| h.id).collect();
    hits.iter().filter(|h| t.contains(&h.id)).count() as f32 / truth.len().max(1) as f32
}

fn main() {
    match Cli::parse().cmd {
        Cmd::Bench(a) => bench(a),
        Cmd::Embed(a) => embed(a),
        Cmd::Live(a) => live(a),
    }
}

fn embed(a: EmbedArgs) {
    let t0 = Instant::now();
    let threads = a
        .threads
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(2, |n| n.get().min(4)));
    let mut emb =
        Embedder::load_with_config(&a.model, threads, &a.onnx).unwrap_or_else(|e| panic!("{e}"));
    let load = t0.elapsed();
    let texts: Vec<&str> = a.text.iter().map(|s| s.as_str()).collect();
    let t1 = Instant::now();
    let out = emb
        .embed(
            &texts,
            a.query,
            &EmbedOptions {
                truncate_dim: a.dim,
            },
        )
        .unwrap_or_else(|e| panic!("{e}"));
    let infer = t1.elapsed();
    println!(
        "model={} onnx={} dim={} max_seq={} threads={} load={:.0}ms infer={:.2}ms/text",
        a.model.display(),
        a.onnx,
        out.first().map_or(0, Vec::len),
        emb.max_seq_length(),
        threads,
        load.as_secs_f64() * 1000.0,
        infer.as_secs_f64() * 1000.0 / texts.len() as f64
    );
    for (i, v) in out.iter().enumerate() {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!(
            "[{i}] dim={} norm={norm:.4} head={:?}",
            v.len(),
            &v[..v.len().min(4)]
        );
    }
}

fn load_cache_vecs(path: &Path) -> (usize, Vec<f32>) {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let magic = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    assert_eq!(magic, 0x4843_5232, "bad cache file");
    let n = u32::from_le_bytes([b[4], b[5], b[6], b[7]]) as usize;
    let dim = u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize;
    assert!(dim > 0, "bad cache dim");
    let mut vectors = vec![0f32; n * dim];
    for (i, v) in vectors.iter_mut().enumerate() {
        let c = &b[12 + i * 4..12 + i * 4 + 4];
        *v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
    }
    (dim, vectors)
}

fn build_cache(a: &LiveArgs) -> (usize, Vec<f32>) {
    let t0 = Instant::now();
    let raw = std::fs::read_to_string(&a.corpus)
        .unwrap_or_else(|e| panic!("corpus {}: {e}", a.corpus.display()));
    let parsed: serde_json::Value = serde_json::from_str(&raw).unwrap_or_else(|e| panic!("{e}"));
    let arr = parsed.as_array().unwrap_or_else(|| panic!("corpus json"));
    let mut docs: Vec<&str> = Vec::with_capacity(arr.len().min(a.docs));
    for d in arr {
        if let Some(t) = d["text"].as_str() {
            docs.push(t);
        }
        if docs.len() >= a.docs {
            break;
        }
    }
    println!(
        "corpus: {} docs={} ({:.1}s)",
        a.corpus.display(),
        docs.len(),
        t0.elapsed().as_secs_f64()
    );
    let mut emb = Embedder::load_with_config(&a.model, a.embed_threads, "model_quantized.onnx")
        .unwrap_or_else(|e| panic!("{e}"));
    let dim = emb.dim();
    let mut vectors = Vec::with_capacity(docs.len() * dim);
    let batch = 64usize;
    let t1 = Instant::now();
    let mut done = 0usize;
    while done < docs.len() {
        let end = (done + batch).min(docs.len());
        let mut out = emb
            .embed(&docs[done..end], false, &EmbedOptions::default())
            .unwrap_or_else(|e| panic!("{e}"));
        for v in out.drain(..) {
            vectors.extend_from_slice(&v);
        }
        done = end;
        if done % 6400 == 0 || done == docs.len() {
            println!(
                "embedded {}/{} ({:.0} docs/s)",
                done,
                docs.len(),
                done as f64 / t1.elapsed().as_secs_f64()
            );
        }
    }
    let mut b = Vec::with_capacity(12 + vectors.len() * 4);
    b.extend_from_slice(&0x4843_5232u32.to_le_bytes());
    b.extend_from_slice(&((vectors.len() / dim) as u32).to_le_bytes());
    b.extend_from_slice(&(dim as u32).to_le_bytes());
    for v in &vectors {
        b.extend_from_slice(&v.to_le_bytes());
    }
    if let Some(parent) = a.cache.parent() {
        std::fs::create_dir_all(parent).unwrap_or_else(|e| panic!("{e}"));
    }
    std::fs::write(&a.cache, b).unwrap_or_else(|e| panic!("{e}"));
    println!("cache stored: {}", a.cache.display());
    (dim, vectors)
}

fn live(a: LiveArgs) {
    let (cache_dim, cache_vecs) = if a.cache.exists() && !a.rebuild {
        load_cache_vecs(&a.cache)
    } else {
        build_cache(&a)
    };
    let dim = if a.dim > 0 { a.dim } else { cache_dim };
    let n = cache_vecs.len() / cache_dim;
    let mut truncated = Vec::with_capacity(n * dim);
    for i in 0..n {
        truncated.extend_from_slice(&cache_vecs[i * cache_dim..i * cache_dim + dim]);
    }
    drop(cache_vecs);
    let mut cfg = HnswConfig::new(dim, Metric::Dot);
    cfg.m = a.m;
    cfg.ef_construction = a.ef_construction;
    let storage = match a.storage {
        StorageArg::F32 => Storage::F32,
        StorageArg::Int8 => Storage::Int8,
    };
    cfg.storage = storage;
    if a.rerank && storage == Storage::Int8 {
        cfg.rerank_f32 = true;
    }
    if a.rerank && storage == Storage::F32 {
        panic!("--rerank requires --storage int8");
    }
    if storage == Storage::Int8 {
        cfg.qrange = 0.3;
    }
    let rerank_on = a.rerank && storage == Storage::Int8;
    let t0 = Instant::now();
    let mut ix = Hnsw::with_capacity(cfg, n);
    for i in 0..n {
        ix.add(&truncated[i * dim..(i + 1) * dim]);
    }
    let build = t0.elapsed();
    let tp = Instant::now();
    ix.pack();
    let pack = tp.elapsed();
    let mut emb = Embedder::load_with_config(&a.model, a.embed_threads, "model_quantized.onnx")
        .unwrap_or_else(|e| panic!("{e}"));
    let opts = EmbedOptions {
        truncate_dim: if a.dim > 0 { Some(dim) } else { None },
    };
    let mut qvecs = Vec::with_capacity(MOSS_QUERIES.len() * dim);
    for q in MOSS_QUERIES {
        let mut v = emb
            .embed(&[q], true, &opts)
            .unwrap_or_else(|e| panic!("{e}"));
        qvecs.extend(v.remove(0));
    }
    let mut truths = Vec::with_capacity(MOSS_QUERIES.len());
    for qi in 0..MOSS_QUERIES.len() {
        truths.push(top_k(
            &truncated,
            dim,
            &qvecs[qi * dim..(qi + 1) * dim],
            a.k,
            Metric::Dot,
            None,
        ));
    }
    let vmb = ix.vector_bytes() as f64 / 1e6;
    let gmb = ix.link_count() as f64 * 4.0 / 1e6;
    println!(
        "swiftvec live | corpus=usemoss/moss benchmarks/bench_100k_docs.json docs={n} queries={} rounds={} warmup={} top_k={}",
        MOSS_QUERIES.len(), a.rounds, a.warmup, a.k
    );
    println!(
        "model=mdbr-leaf-ir (quantized onnx, in-process) storage={storage:?} rerank={rerank_on} dim={dim} m={} efc={} embed_threads={}",
        a.m, a.ef_construction, a.embed_threads
    );
    println!(
        "build: {:.1}s ({:.0} docs/s) | pack: {:.2}s | mem: vectors={:.1}MB graph={:.1}MB",
        build.as_secs_f64(),
        n as f64 / build.as_secs_f64(),
        pack.as_secs_f64(),
        vmb,
        gmb
    );
    let mut ctx = SearchCtx::with_capacity(n);
    let cq = Instant::now();
    let mut cold = emb
        .embed(&[MOSS_QUERIES[0]], true, &opts)
        .unwrap_or_else(|e| panic!("{e}"));
    ix.search_with(&mut ctx, &cold.remove(0), a.k, a.ef[0], None);
    println!("cold query: {:.2} ms", cq.elapsed().as_secs_f64() * 1000.0);
    println!(
        "reference (moss.dev published, apple m4 pro): p50=3.1ms p95=4.3ms p99=5.4ms (built-in moss-minilm, embedding included)"
    );
    println!(
        "{:>5} {:>9} {:>13} {:>13} {:>10} {:>10} {:>10} {:>10}",
        "ef", "recall", "search_p50_us", "search_p99_us", "p50_ms", "p95_ms", "p99_ms", "mean_ms"
    );
    for &ef in &a.ef {
        let mut e2e: Vec<f64> = Vec::with_capacity(a.rounds * MOSS_QUERIES.len());
        let mut search_us: Vec<f64> = Vec::with_capacity(a.rounds * MOSS_QUERIES.len());
        let mut recalls = 0.0f32;
        for round in 0..(a.warmup + a.rounds) {
            for (qi, q) in MOSS_QUERIES.iter().enumerate() {
                let t0 = Instant::now();
                let mut v = emb.embed(&[q], true, &opts).unwrap_or_else(|e| panic!("{e}"));
                let t1 = Instant::now();
                let hits = ix.search_with(&mut ctx, &v.remove(0), a.k, ef, None);
                let t2 = Instant::now();
                if round >= a.warmup {
                    e2e.push((t2 - t0).as_secs_f64() * 1000.0);
                    search_us.push((t2 - t1).as_secs_f64() * 1e6);
                    if round == a.warmup {
                        recalls += recall_of(&hits, &truths[qi]);
                    }
                }
            }
        }
        e2e.sort_by(|x, y| x.partial_cmp(y).unwrap());
        search_us.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let mean: f64 = e2e.iter().sum::<f64>() / e2e.len() as f64;
        println!(
            "{:>5} {:>9.4} {:>13.0} {:>13.0} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
            ef,
            recalls / MOSS_QUERIES.len() as f32,
            pct_f(&search_us, 0.5),
            pct_f(&search_us, 0.99),
            pct_f(&e2e, 0.5),
            pct_f(&e2e, 0.95),
            pct_f(&e2e, 0.99),
            mean
        );
    }
}

fn bench(a: BenchArgs) {
    let metric = Metric::parse(&a.metric).unwrap_or_else(|| panic!("metric must be dot|l2"));
    let ds = match a.data {
        DataArg::Clustered => clustered(a.n, a.dim, a.clusters, a.queries, a.seed),
        DataArg::Uniform => uniform(a.n, a.dim, a.queries, a.seed),
    };
    let mut cfg = HnswConfig::new(a.dim, metric);
    cfg.m = a.m;
    cfg.ef_construction = a.ef_construction;
    let storage = match a.storage {
        StorageArg::F32 => Storage::F32,
        StorageArg::Int8 => Storage::Int8,
    };
    cfg.storage = storage;
    let qrange = match (a.qrange, storage) {
        (Some(r), Storage::Int8) => r,
        (None, Storage::Int8) => calibrate_range(&ds.vectors),
        _ => 0.0,
    };
    cfg.qrange = qrange;
    if a.rerank && storage == Storage::Int8 {
        cfg.rerank_f32 = true;
    }
    let t0 = Instant::now();
    let mut ix = Hnsw::with_capacity(cfg, a.n);
    for i in 0..a.n {
        ix.add(&ds.vectors[i * a.dim..(i + 1) * a.dim]);
    }
    let build = t0.elapsed();
    let owned = a.filter.map(|p| move |id: u32| id % 100 < p);
    let flt: Option<&dyn Fn(u32) -> bool> = owned.as_ref().map(|f| f as &dyn Fn(u32) -> bool);
    let mut truths = Vec::with_capacity(a.queries);
    for qi in 0..a.queries {
        let q = &ds.queries[qi * a.dim..(qi + 1) * a.dim];
        truths.push(top_k(&ds.vectors, a.dim, q, a.k, metric, flt));
    }
    let vmb = ix.vector_bytes() as f64 / 1e6;
    let gmb = ix.link_count() as f64 * 4.0 / 1e6;
    println!(
        "swiftvec bench | data={:?}-synthetic n={} dim={} clusters={} metric={} m={} efc={} storage={:?} qrange={:.3} filter={:?} seed={}",
        a.data,
        a.n,
        a.dim,
        a.clusters,
        a.metric,
        a.m,
        a.ef_construction,
        storage,
        qrange,
        a.filter,
        a.seed
    );
    println!(
        "build: {:.2}s ({:.0} docs/s) | layers={} | mem: vectors={:.1}MB graph={:.1}MB total={:.1}MB",
        build.as_secs_f64(),
        a.n as f64 / build.as_secs_f64(),
        ix.layers(),
        vmb,
        gmb,
        vmb + gmb
    );
    println!(
        "{:>6} {:>10} {:>10} {:>10} {:>12} {:>8}",
        "ef", "p50_us", "p95_us", "p99_us", "qps", "recall"
    );
    for &ef in &a.ef {
        let mut ctx = SearchCtx::with_capacity(a.n);
        let mut times: Vec<u128> = Vec::with_capacity(a.queries);
        let mut recalls = 0.0f32;
        for qi in 0..a.queries {
            let q = &ds.queries[qi * a.dim..(qi + 1) * a.dim];
            let t = Instant::now();
            let hits = ix.search_with(&mut ctx, q, a.k, ef, flt);
            times.push(t.elapsed().as_micros());
            recalls += recall_of(&hits, &truths[qi]);
        }
        times.sort_unstable();
        let total: u128 = times.iter().sum();
        let qps = 1_000_000.0 * a.queries as f64 / total as f64;
        println!(
            "{:>6} {:>10} {:>10} {:>10} {:>12.0} {:>8.4}",
            ef,
            pct_u(&times, 0.5),
            pct_u(&times, 0.95),
            pct_u(&times, 0.99),
            qps,
            recalls / a.queries as f32
        );
    }
}
