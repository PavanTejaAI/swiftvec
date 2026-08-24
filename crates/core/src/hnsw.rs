use crate::distance::Metric;
use crate::hit::Hit;
use crate::int8::{dot_i8, l2sq_i8, quantize_into};
use crate::rng::Xoshiro256;
use crate::visited::Visited;
use crate::SearchCtx;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::io::{Read, Write};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Storage {
    F32,
    Int8,
}

pub struct HnswConfig {
    pub dim: usize,
    pub m: usize,
    pub ef_construction: usize,
    pub seed: u64,
    pub metric: Metric,
    pub storage: Storage,
    pub qrange: f32,
    pub rerank_f32: bool,
    pub cascade: bool,
}

impl HnswConfig {
    pub fn new(dim: usize, metric: Metric) -> Self {
        Self {
            dim,
            m: 16,
            ef_construction: 200,
            seed: 0x5EED_5EED,
            metric,
            storage: Storage::F32,
            qrange: 0.3,
            rerank_f32: false,
            cascade: false,
        }
    }
}

pub struct Query<'a> {
    v: &'a [f32],
    q8: &'a [i8],
    qs: Option<[u64; 4]>,
}

impl<'a> Query<'a> {
    fn new_f32(v: &'a [f32]) -> Self {
        Self {
            v,
            q8: &[],
            qs: None,
        }
    }

    fn new_int8(v: &'a [f32], q8: &'a [i8]) -> Self {
        Self { v, q8, qs: None }
    }

    fn new_int8_cascade(v: &'a [f32], q8: &'a [i8], qs: [u64; 4]) -> Self {
        Self {
            v,
            q8,
            qs: Some(qs),
        }
    }
}

struct PackedLevel {
    offsets: Vec<u32>,
    targets: Vec<u32>,
}

#[cfg(feature = "mmap")]
struct PackedLevelRef<'a> {
    offsets: &'a [u32],
    targets: &'a [u32],
}

const SNAPSHOT_MAGIC: u32 = 0x5357_5643;
const SNAPSHOT_VERSION_V1: u32 = 1;
const SNAPSHOT_VERSION: u32 = 2;

pub(crate) const CASCADE_LANES: usize = 4;
pub(crate) const CASCADE_BITS: u32 = 256;
pub(crate) const CASCADE_TAU: u32 = 128;
const HYPERPLANE_SALT: u64 = 0x5167_45ED_51A7_0001;

pub(crate) trait GraphAccess {
    fn ga_dim(&self) -> usize;
    fn ga_metric(&self) -> Metric;
    fn ga_storage(&self) -> Storage;
    fn ga_rerank(&self) -> bool;
    fn ga_qrange(&self) -> f32;
    fn ga_qscale(&self) -> f32;
    fn ga_entry(&self) -> u32;
    fn ga_max_level(&self) -> u32;
    fn ga_levels(&self) -> &[u32];
    fn ga_vectors(&self) -> &[f32];
    fn ga_codes(&self) -> &[i8];
    fn ga_neighbors(&self, level: usize, id: u32) -> &[u32];
    fn ga_cascade(&self) -> bool;
    fn ga_sigs(&self) -> &[u64];
    fn ga_hyperplanes(&self) -> &[f32];
}

fn hyperplanes(seed: u64, dim: usize) -> Vec<f32> {
    let mut rng = Xoshiro256::new(seed ^ HYPERPLANE_SALT);
    let mut hp = Vec::with_capacity(CASCADE_BITS as usize * dim);
    for _ in 0..(CASCADE_BITS as usize * dim) {
        hp.push(rng.next_gauss());
    }
    hp
}

fn signature(hp: &[f32], v: &[f32]) -> [u64; CASCADE_LANES] {
    let dim = v.len();
    let mut out = [0u64; CASCADE_LANES];
    for i in 0..CASCADE_BITS as usize {
        if crate::distance::dot(v, &hp[i * dim..(i + 1) * dim]) >= 0.0 {
            out[i / 64] |= 1u64 << (i % 64);
        }
    }
    out
}

#[inline]
fn hamming(a: &[u64; CASCADE_LANES], b: &[u64; CASCADE_LANES]) -> u32 {
    let mut h = 0u32;
    for i in 0..CASCADE_LANES {
        h += (a[i] ^ b[i]).count_ones();
    }
    h
}

fn g_row_f32<'a, G: GraphAccess>(g: &'a G, id: u32) -> &'a [f32] {
    let off = id as usize * g.ga_dim();
    &g.ga_vectors()[off..off + g.ga_dim()]
}

fn g_row_i8<'a, G: GraphAccess>(g: &'a G, id: u32) -> &'a [i8] {
    let off = id as usize * g.ga_dim();
    &g.ga_codes()[off..off + g.ga_dim()]
}

#[cfg(target_arch = "x86_64")]
fn g_prefetch<G: GraphAccess>(g: &G, id: u32) {
    let p = unsafe {
        match g.ga_storage() {
            Storage::F32 => g.ga_vectors().as_ptr().add(id as usize * g.ga_dim()) as *const u8,
            Storage::Int8 => g.ga_codes().as_ptr().add(id as usize * g.ga_dim()) as *const u8,
        }
    };
    unsafe {
        std::arch::x86_64::_mm_prefetch(p as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn g_prefetch<G: GraphAccess>(_g: &G, _id: u32) {}

fn g_node_sig<G: GraphAccess>(g: &G, id: u32) -> Option<[u64; CASCADE_LANES]> {
    if !g.ga_cascade() {
        return None;
    }
    let s = &g.ga_sigs()[id as usize * CASCADE_LANES..(id + 1) as usize * CASCADE_LANES];
    Some([s[0], s[1], s[2], s[3]])
}

fn g_dist_to<G: GraphAccess>(g: &G, q: &Query, id: u32) -> f32 {
    if q.q8.is_empty() {
        g.ga_metric().distance(q.v, g_row_f32(g, id))
    } else {
        match g.ga_metric() {
            Metric::Dot => 1.0 - dot_i8(q.q8, g_row_i8(g, id)) as f32 * g.ga_qscale(),
            Metric::L2 => l2sq_i8(q.q8, g_row_i8(g, id)) as f32 * g.ga_qscale(),
        }
    }
}

fn g_dist_between<G: GraphAccess>(g: &G, a: u32, b: u32) -> f32 {
    match g.ga_storage() {
        Storage::F32 => g.ga_metric().distance(g_row_f32(g, a), g_row_f32(g, b)),
        Storage::Int8 => match g.ga_metric() {
            Metric::Dot => 1.0 - dot_i8(g_row_i8(g, a), g_row_i8(g, b)) as f32 * g.ga_qscale(),
            Metric::L2 => l2sq_i8(g_row_i8(g, a), g_row_i8(g, b)) as f32 * g.ga_qscale(),
        },
    }
}

fn g_greedy<G: GraphAccess>(g: &G, q: &Query, mut cur: u32, level: usize) -> u32 {
    let mut best = g_dist_to(g, q, cur);
    loop {
        let mut improved = false;
        let nbrs = g.ga_neighbors(level, cur);
        for (i, &n) in nbrs.iter().enumerate() {
            #[cfg(target_arch = "x86_64")]
            if i + 1 < nbrs.len() {
                g_prefetch(g, nbrs[i + 1]);
            }
            if let Some(qs) = &q.qs {
                if let Some(ns) = g_node_sig(g, n) {
                    if hamming(qs, &ns) > CASCADE_TAU {
                        continue;
                    }
                }
            }
            let d = g_dist_to(g, q, n);
            if d < best {
                best = d;
                cur = n;
                improved = true;
            }
        }
        if !improved {
            return cur;
        }
    }
}

fn g_search_layer<G: GraphAccess>(
    g: &G,
    q: &Query,
    eps: &[u32],
    ef: usize,
    level: usize,
    filter: Option<&dyn Fn(u32) -> bool>,
    visited: &mut Visited,
) -> Vec<Hit> {
    let ef = ef.max(1);
    visited.reset();
    let mut frontier: BinaryHeap<Reverse<Hit>> = BinaryHeap::with_capacity(ef + 8);
    let mut best: BinaryHeap<Hit> = BinaryHeap::with_capacity(ef + 1);
    for &e in eps {
        if visited.mark(e as usize) {
            let h = Hit {
                dist: g_dist_to(g, q, e),
                id: e,
            };
            frontier.push(Reverse(h));
            if filter.map_or(true, |f| f(e)) && best.len() < ef {
                best.push(h);
            }
        }
    }
    while let Some(Reverse(c)) = frontier.pop() {
        if best.len() >= ef {
            if let Some(&worst) = best.peek() {
                if c.dist > worst.dist {
                    break;
                }
            }
        }
        let nbrs = g.ga_neighbors(level, c.id);
        for (i, &n) in nbrs.iter().enumerate() {
            #[cfg(target_arch = "x86_64")]
            if i + 1 < nbrs.len() {
                g_prefetch(g, nbrs[i + 1]);
            }
            if visited.mark(n as usize) {
                if let Some(qs) = &q.qs {
                    if let Some(ns) = g_node_sig(g, n) {
                        if hamming(qs, &ns) > CASCADE_TAU {
                            continue;
                        }
                    }
                }
                let h = Hit {
                    dist: g_dist_to(g, q, n),
                    id: n,
                };
                let pass = filter.map_or(true, |f| f(n));
                if pass && best.len() < ef {
                    best.push(h);
                    frontier.push(Reverse(h));
                } else if pass {
                    if let Some(&worst) = best.peek() {
                        if h.dist < worst.dist {
                            best.pop();
                            best.push(h);
                            frontier.push(Reverse(h));
                        }
                    }
                } else {
                    frontier.push(Reverse(h));
                }
            }
        }
    }
    best.into_sorted_vec()
}

pub struct Hnsw {
    dim: usize,
    m: usize,
    m0: usize,
    ef_construction: usize,
    metric: Metric,
    storage: Storage,
    rerank: bool,
    qrange: f32,
    qscale: f32,
    mult: f32,
    rng: Xoshiro256,
    vectors: Vec<f32>,
    codes: Vec<i8>,
    levels: Vec<u32>,
    layers: Vec<Vec<Vec<u32>>>,
    packed: Option<Vec<PackedLevel>>,
    link_cache: Option<usize>,
    entry: u32,
    max_level: u32,
    visited: Visited,
    cascade: bool,
    hp_seed: u64,
    hp: Vec<f32>,
    sigs: Vec<u64>,
}

impl Hnsw {
    pub fn with_capacity(cfg: HnswConfig, n_hint: usize) -> Self {
        let m = cfg.m.max(2);
        let m0 = m * 2;
        Self {
            dim: cfg.dim,
            m,
            m0,
            ef_construction: cfg.ef_construction.max(m),
            metric: cfg.metric,
            storage: cfg.storage,
            rerank: cfg.storage == Storage::Int8 && cfg.rerank_f32,
            qrange: cfg.qrange.max(1e-9),
            qscale: (cfg.qrange.max(1e-9) / 127.0).powi(2),
            mult: 1.0 / (m as f32).ln(),
            rng: Xoshiro256::new(cfg.seed),
            vectors: if cfg.storage == Storage::F32 || (cfg.storage == Storage::Int8 && cfg.rerank_f32)
            {
                Vec::with_capacity(n_hint * cfg.dim)
            } else {
                Vec::new()
            },
            codes: if cfg.storage == Storage::Int8 {
                Vec::with_capacity(n_hint * cfg.dim)
            } else {
                Vec::new()
            },
            levels: Vec::with_capacity(n_hint),
            layers: Vec::new(),
            packed: None,
            link_cache: None,
            entry: 0,
            max_level: 0,
            visited: Visited::with_capacity(n_hint + 1),
            cascade: cfg.cascade,
            hp_seed: cfg.seed,
            hp: if cfg.cascade {
                hyperplanes(cfg.seed, cfg.dim)
            } else {
                Vec::new()
            },
            sigs: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.levels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.levels.is_empty()
    }

    pub fn metric(&self) -> Metric {
        self.metric
    }

    pub fn storage(&self) -> Storage {
        self.storage
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn layers(&self) -> usize {
        self.max_level as usize + 1
    }

    pub fn link_count(&self) -> usize {
        match self.link_cache {
            Some(c) => c,
            None => self
                .layers
                .iter()
                .map(|l| l.iter().map(|v| v.len()).sum::<usize>())
                .sum(),
        }
    }

    pub fn vector_bytes(&self) -> usize {
        match self.storage {
            Storage::F32 => self.vectors.len() * 4,
            Storage::Int8 => {
                self.codes.len()
                    + if self.rerank {
                        self.vectors.len() * 4
                    } else {
                        0
                    }
            }
        }
    }

    pub fn pack(&mut self) {
        assert!(self.packed.is_none(), "pack() is terminal");
        let links = self.link_count();
        let mut packed = Vec::with_capacity(self.layers.len());
        for l in self.layers.drain(..) {
            let total: usize = l.iter().map(|v| v.len()).sum();
            let mut offsets = Vec::with_capacity(l.len() + 1);
            let mut targets = Vec::with_capacity(total);
            offsets.push(0u32);
            let mut acc = 0u32;
            for v in l {
                targets.extend_from_slice(&v);
                acc += v.len() as u32;
                offsets.push(acc);
            }
            packed.push(PackedLevel { offsets, targets });
        }
        self.link_cache = Some(links);
        self.packed = Some(packed);
    }

    pub fn save(&self, w: &mut impl Write) -> std::io::Result<()> {
        let mut b = Vec::new();
        put_u32(&mut b, SNAPSHOT_MAGIC);
        put_u32(&mut b, SNAPSHOT_VERSION);
        put_u32(&mut b, self.dim as u32);
        put_u32(&mut b, self.m as u32);
        put_u32(&mut b, self.m0 as u32);
        put_u32(&mut b, self.ef_construction as u32);
        put_u8(&mut b, match self.metric {
            Metric::Dot => 0,
            Metric::L2 => 1,
        });
        put_u8(&mut b, match self.storage {
            Storage::F32 => 0,
            Storage::Int8 => 1,
        });
        put_u8(&mut b, self.rerank as u8);
        put_u8(&mut b, self.packed.is_some() as u8);
        put_f32(&mut b, self.qrange);
        put_f32(&mut b, self.qscale);
        put_f32(&mut b, self.mult);
        put_u32(&mut b, self.entry);
        put_u32(&mut b, self.max_level);
        put_u32(&mut b, self.levels.len() as u32);
        put_pad(&mut b);
        for &l in &self.levels {
            put_u32(&mut b, l);
        }
        put_pad(&mut b);
        put_u64(&mut b, self.vectors.len() as u64);
        for v in &self.vectors {
            put_f32(&mut b, *v);
        }
        put_pad(&mut b);
        put_u64(&mut b, self.codes.len() as u64);
        put_bytes(
            &mut b,
            unsafe { std::slice::from_raw_parts(self.codes.as_ptr() as *const u8, self.codes.len()) },
        );
        put_u8(&mut b, self.cascade as u8);
        if self.cascade {
            put_u64(&mut b, self.hp_seed);
            put_pad(&mut b);
            put_u64(&mut b, self.sigs.len() as u64);
            put_bytes(
                &mut b,
                unsafe {
                    std::slice::from_raw_parts(self.sigs.as_ptr() as *const u8, self.sigs.len() * 8)
                },
            );
        }
        match &self.packed {
            Some(packed) => {
                put_u32(&mut b, packed.len() as u32);
                for level in packed {
                    put_pad(&mut b);
                    put_u32(&mut b, level.offsets.len() as u32);
                    for &o in &level.offsets {
                        put_u32(&mut b, o);
                    }
                    put_pad(&mut b);
                    put_u32(&mut b, level.targets.len() as u32);
                    for &t in &level.targets {
                        put_u32(&mut b, t);
                    }
                }
            }
            None => {
                put_u32(&mut b, self.layers.len() as u32);
                for layer in &self.layers {
                    put_u32(&mut b, layer.len() as u32);
                    for node in layer {
                        put_u32(&mut b, node.len() as u32);
                        for &t in node {
                            put_u32(&mut b, t);
                        }
                    }
                }
            }
        }
        w.write_all(&b)
    }

    pub fn load(r: &mut impl Read) -> std::io::Result<Self> {
        let mut b = Vec::new();
        r.read_to_end(&mut b)?;
        let mut i = 0usize;
        let magic = get_u32(&b, &mut i)?;
        if magic != SNAPSHOT_MAGIC {
            return Err(std::io::Error::other("bad snapshot magic"));
        }
        let version = get_u32(&b, &mut i)?;
        if version != SNAPSHOT_VERSION_V1 && version != SNAPSHOT_VERSION {
            return Err(std::io::Error::other("unsupported snapshot version"));
        }
        let v2 = version == SNAPSHOT_VERSION;
        let dim = get_u32(&b, &mut i)? as usize;
        let m = get_u32(&b, &mut i)? as usize;
        let m0 = get_u32(&b, &mut i)? as usize;
        let ef_construction = get_u32(&b, &mut i)? as usize;
        let metric = match get_u8(&b, &mut i)? {
            0 => Metric::Dot,
            1 => Metric::L2,
            _ => return Err(std::io::Error::other("bad metric")),
        };
        let storage = match get_u8(&b, &mut i)? {
            0 => Storage::F32,
            1 => Storage::Int8,
            _ => return Err(std::io::Error::other("bad storage")),
        };
        let rerank = get_u8(&b, &mut i)? != 0;
        let was_packed = get_u8(&b, &mut i)? != 0;
        let qrange = get_f32(&b, &mut i)?;
        let qscale = get_f32(&b, &mut i)?;
        let mult = get_f32(&b, &mut i)?;
        let entry = get_u32(&b, &mut i)?;
        let max_level = get_u32(&b, &mut i)?;
        let n = get_u32(&b, &mut i)? as usize;
        if dim == 0 || n == 0 {
            return Err(std::io::Error::other("bad snapshot dims"));
        }
        if v2 {
            get_pad(&b, &mut i)?;
        }
        let mut levels = Vec::with_capacity(n);
        for _ in 0..n {
            levels.push(get_u32(&b, &mut i)?);
        }
        if v2 {
            get_pad(&b, &mut i)?;
        }
        let nv = get_u64(&b, &mut i)? as usize;
        let mut vectors = Vec::with_capacity(nv);
        for _ in 0..nv {
            vectors.push(get_f32(&b, &mut i)?);
        }
        if v2 {
            get_pad(&b, &mut i)?;
        }
        let nc = get_u64(&b, &mut i)? as usize;
        if i + nc > b.len() {
            return Err(std::io::Error::other("snapshot truncated"));
        }
        let codes = unsafe {
            let p = b.as_ptr().add(i) as *const i8;
            i += nc;
            Vec::from(std::slice::from_raw_parts(p, nc))
        };
        let (cascade, hp_seed, sigs) = if v2 {
            if get_u8(&b, &mut i)? != 0 {
                let seed = get_u64(&b, &mut i)?;
                get_pad(&b, &mut i)?;
                let ns = get_u64(&b, &mut i)? as usize;
                if ns / CASCADE_LANES != n || i + ns * 8 > b.len() {
                    return Err(std::io::Error::other("snapshot truncated"));
                }
                let mut s = Vec::with_capacity(ns);
                for _ in 0..ns {
                    s.push(get_u64(&b, &mut i)?);
                }
                (true, seed, s)
            } else {
                (false, 0x5EED_5EED, Vec::new())
            }
        } else {
            (false, 0x5EED_5EED, Vec::new())
        };
        let layer_count = get_u32(&b, &mut i)? as usize;
        let mut packed = None;
        let mut layers = Vec::new();
        let mut link_total = 0usize;
        if was_packed {
            let mut p = Vec::with_capacity(layer_count);
            for _ in 0..layer_count {
                if v2 {
                    get_pad(&b, &mut i)?;
                }
                let no = get_u32(&b, &mut i)? as usize;
                let mut offsets = Vec::with_capacity(no);
                for _ in 0..no {
                    offsets.push(get_u32(&b, &mut i)?);
                }
                if v2 {
                    get_pad(&b, &mut i)?;
                }
                let nt = get_u32(&b, &mut i)? as usize;
                let mut targets = Vec::with_capacity(nt);
                for _ in 0..nt {
                    targets.push(get_u32(&b, &mut i)?);
                }
                link_total += nt;
                p.push(PackedLevel { offsets, targets });
            }
            packed = Some(p);
        } else {
            for _ in 0..layer_count {
                let ns = get_u32(&b, &mut i)? as usize;
                let mut layer = Vec::with_capacity(ns);
                for _ in 0..ns {
                    let nd = get_u32(&b, &mut i)? as usize;
                    let mut node = Vec::with_capacity(nd);
                    for _ in 0..nd {
                        node.push(get_u32(&b, &mut i)?);
                    }
                    link_total += nd;
                    layer.push(node);
                }
                layers.push(layer);
            }
        }
        Ok(Self {
            dim,
            m,
            m0,
            ef_construction,
            metric,
            storage,
            rerank,
            qrange,
            qscale,
            mult,
            rng: Xoshiro256::new(0x5EED_5EED),
            vectors,
            codes,
            levels,
            layers,
            packed,
            link_cache: if was_packed { Some(link_total / 2) } else { None },
            entry,
            max_level,
            visited: Visited::with_capacity(n + 1),
            cascade,
            hp_seed,
            hp: if cascade {
                hyperplanes(hp_seed, dim)
            } else {
                Vec::new()
            },
            sigs,
        })
    }

    pub fn add(&mut self, v: &[f32]) -> u32 {
        assert!(self.packed.is_none(), "pack() is terminal");
        debug_assert_eq!(v.len(), self.dim);
        match self.storage {
            Storage::F32 => self.vectors.extend_from_slice(v),
            Storage::Int8 => {
                let mut c = vec![0i8; self.dim];
                quantize_into(&mut c, v, self.qrange);
                self.codes.extend_from_slice(&c);
                if self.rerank {
                    self.vectors.extend_from_slice(v);
                }
            }
        }
        if self.cascade {
            let s = signature(&self.hp, v);
            self.sigs.extend_from_slice(&s);
        }
        let id = self.levels.len() as u32;
        let level = self.sample_level();
        self.levels.push(level);
        self.visited.grow(id as usize + 1);
        while self.layers.len() <= level as usize {
            self.layers.push(Vec::new());
        }
        for l in 0..=level as usize {
            self.layers[l].resize(id as usize + 1, Vec::new());
        }
        if id == 0 {
            self.entry = 0;
            self.max_level = level;
            return 0;
        }
        let top = self.max_level;
        let mut cur = self.entry;
        let mut q8buf: Vec<i8> = Vec::new();
        if self.storage == Storage::Int8 {
            q8buf.resize(self.dim, 0);
            quantize_into(&mut q8buf, v, self.qrange);
        }
        let qs = if self.cascade {
            Some(signature(&self.hp, v))
        } else {
            None
        };
        let q = match qs {
            Some(s) => Query::new_int8_cascade(v, &q8buf, s),
            None => {
                if q8buf.is_empty() {
                    Query::new_f32(v)
                } else {
                    Query::new_int8(v, &q8buf)
                }
            }
        };
        if top > level {
            for l in ((level + 1)..=top).rev() {
                cur = g_greedy(self, &q, cur, l as usize);
            }
        }
        let mut eps = vec![cur];
        let up = level.min(top);
        let mut visited = std::mem::take(&mut self.visited);
        for l in (0..=up).rev() {
            let cands =
                g_search_layer(self, &q, &eps, self.ef_construction, l as usize, None, &mut visited);
            eps = cands.iter().map(|h| h.id).collect();
            let selected = self.select_neighbors(cands, self.m_for(l as u32));
            for s in &selected {
                self.link(l as usize, id, *s);
            }
        }
        self.visited = visited;
        if level > top {
            self.entry = id;
            self.max_level = level;
        }
        id
    }

    pub fn search(
        &self,
        q: &[f32],
        k: usize,
        ef: usize,
        filter: Option<&dyn Fn(u32) -> bool>,
    ) -> Vec<Hit> {
        let mut ctx = SearchCtx::with_capacity(self.len());
        self.search_with(&mut ctx, q, k, ef, filter)
    }

    pub fn search_with(
        &self,
        ctx: &mut SearchCtx,
        q: &[f32],
        k: usize,
        ef: usize,
        filter: Option<&dyn Fn(u32) -> bool>,
    ) -> Vec<Hit> {
        if self.is_empty() || k == 0 {
            return Vec::new();
        }
        let ef = ef.max(k);
        let SearchCtx { visited, q8 } = ctx;
        if self.storage == Storage::Int8 {
            q8.clear();
            q8.resize(self.dim, 0);
            quantize_into(q8, q, self.qrange);
        }
        let qs = if self.cascade {
            Some(signature(&self.hp, q))
        } else {
            None
        };
        let qr = match qs {
            Some(s) => Query::new_int8_cascade(q, q8, s),
            None => {
                if q8.is_empty() {
                    Query::new_f32(q)
                } else {
                    Query::new_int8(q, q8)
                }
            }
        };
        let mut cur = self.entry;
        for l in (1..=self.max_level).rev() {
            cur = g_greedy(self, &qr, cur, l as usize);
        }
        let mut hits = g_search_layer(self, &qr, &[cur], ef, 0, filter, visited);
        if self.rerank {
            for h in hits.iter_mut() {
                h.dist = self.metric.distance(q, g_row_f32(self, h.id));
            }
            hits.sort_unstable();
        }
        hits.truncate(k);
        hits
    }

    fn m_for(&self, level: u32) -> usize {
        if level == 0 {
            self.m0
        } else {
            self.m
        }
    }

    fn sample_level(&mut self) -> u32 {
        let r = self.rng.next_f32().max(1e-38);
        ((-r.ln()) * self.mult).floor() as u32
    }

    fn select_neighbors(&self, cands: Vec<Hit>, m: usize) -> Vec<u32> {
        let mut sorted = cands;
        sorted.sort_unstable();
        let mut res: Vec<u32> = Vec::with_capacity(m);
        let mut pruned: Vec<u32> = Vec::new();
        for h in &sorted {
            if res.len() == m {
                break;
            }
            let closer = res.iter().all(|&r| g_dist_between(self, h.id, r) > h.dist);
            if closer {
                res.push(h.id);
            } else {
                pruned.push(h.id);
            }
        }
        for &e in &pruned {
            if res.len() == m {
                break;
            }
            res.push(e);
        }
        res
    }

    fn link(&mut self, level: usize, a: u32, b: u32) {
        let mmax = self.m_for(level as u32);
        {
            let slot = &mut self.layers[level][a as usize];
            if !slot.contains(&b) {
                slot.push(b);
            }
        }
        let shrink = {
            let slot = &mut self.layers[level][b as usize];
            if slot.contains(&a) {
                false
            } else {
                slot.push(a);
                slot.len() > mmax
            }
        };
        if shrink {
            let ids = std::mem::take(&mut self.layers[level][b as usize]);
            let cands: Vec<Hit> = ids
                .iter()
                .map(|&e| Hit {
                    dist: g_dist_between(self, b, e),
                    id: e,
                })
                .collect();
            let sel = self.select_neighbors(cands, mmax);
            self.layers[level][b as usize] = sel;
        }
    }
}

impl GraphAccess for Hnsw {
    fn ga_dim(&self) -> usize {
        self.dim
    }
    fn ga_metric(&self) -> Metric {
        self.metric
    }
    fn ga_storage(&self) -> Storage {
        self.storage
    }
    fn ga_rerank(&self) -> bool {
        self.rerank
    }
    fn ga_qrange(&self) -> f32 {
        self.qrange
    }
    fn ga_qscale(&self) -> f32 {
        self.qscale
    }
    fn ga_entry(&self) -> u32 {
        self.entry
    }
    fn ga_max_level(&self) -> u32 {
        self.max_level
    }
    fn ga_levels(&self) -> &[u32] {
        &self.levels
    }
    fn ga_vectors(&self) -> &[f32] {
        &self.vectors
    }
    fn ga_codes(&self) -> &[i8] {
        &self.codes
    }
    fn ga_neighbors(&self, level: usize, id: u32) -> &[u32] {
        if let Some(p) = &self.packed {
            let l = &p[level];
            let s = l.offsets[id as usize] as usize;
            let e = l.offsets[id as usize + 1] as usize;
            &l.targets[s..e]
        } else {
            self.layers[level]
                .get(id as usize)
                .map_or(&[], Vec::as_slice)
        }
    }
    fn ga_cascade(&self) -> bool {
        self.cascade
    }
    fn ga_sigs(&self) -> &[u64] {
        &self.sigs
    }
    fn ga_hyperplanes(&self) -> &[f32] {
        &self.hp
    }
}

#[cfg(feature = "mmap")]
pub struct Mapping {
    map: memmap2::Mmap,
}

#[cfg(feature = "mmap")]
impl Mapping {
    pub fn open(path: &std::path::Path) -> std::io::Result<Self> {
        let f = std::fs::File::open(path)?;
        let map = unsafe { memmap2::Mmap::map(&f)? };
        Ok(Self { map })
    }

    pub fn data(&self) -> &[u8] {
        self.map.as_ref()
    }
}

#[cfg(feature = "mmap")]
pub struct MappedIndex<'a> {
    dim: usize,
    metric: Metric,
    storage: Storage,
    rerank: bool,
    qrange: f32,
    qscale: f32,
    entry: u32,
    max_level: u32,
    n: usize,
    levels: &'a [u32],
    vectors: &'a [f32],
    codes: &'a [i8],
    packed: Vec<PackedLevelRef<'a>>,
    cascade: bool,
    hp: Vec<f32>,
    sigs: &'a [u64],
}

#[cfg(feature = "mmap")]
impl<'a> MappedIndex<'a> {
    pub fn decode(b: &'a [u8]) -> std::io::Result<Self> {
        let mut i = 0usize;
        let magic = get_u32(b, &mut i)?;
        if magic != SNAPSHOT_MAGIC {
            return Err(std::io::Error::other("bad snapshot magic"));
        }
        let version = get_u32(b, &mut i)?;
        if version != SNAPSHOT_VERSION {
            return Err(std::io::Error::other("mmap requires a v2 packed snapshot"));
        }
        let dim = get_u32(b, &mut i)? as usize;
        let _m = get_u32(b, &mut i)? as usize;
        let _m0 = get_u32(b, &mut i)? as usize;
        let _efc = get_u32(b, &mut i)? as usize;
        let metric = match get_u8(b, &mut i)? {
            0 => Metric::Dot,
            1 => Metric::L2,
            _ => return Err(std::io::Error::other("bad metric")),
        };
        let storage = match get_u8(b, &mut i)? {
            0 => Storage::F32,
            1 => Storage::Int8,
            _ => return Err(std::io::Error::other("bad storage")),
        };
        let rerank = get_u8(b, &mut i)? != 0;
        let packed_flag = get_u8(b, &mut i)? != 0;
        if !packed_flag {
            return Err(std::io::Error::other("mmap requires pack() before save()"));
        }
        let qrange = get_f32(b, &mut i)?;
        let qscale = get_f32(b, &mut i)?;
        let _mult = get_f32(b, &mut i)?;
        let entry = get_u32(b, &mut i)?;
        let max_level = get_u32(b, &mut i)?;
        let n = get_u32(b, &mut i)? as usize;
        if dim == 0 || n == 0 {
            return Err(std::io::Error::other("bad snapshot dims"));
        }
        get_pad(b, &mut i)?;
        let levels = le_u32_slice(b, &mut i, n)?;
        get_pad(b, &mut i)?;
        let nv = get_u64(b, &mut i)? as usize;
        if nv % dim != 0 {
            return Err(std::io::Error::other("bad vector count"));
        }
        let vectors = le_f32_slice(b, &mut i, nv)?;
        get_pad(b, &mut i)?;
        let nc = get_u64(b, &mut i)? as usize;
        let codes = le_i8_slice(b, &mut i, nc)?;
        let (cascade, hp, sigs) = if get_u8(b, &mut i)? != 0 {
            let hp_seed = get_u64(b, &mut i)?;
            get_pad(b, &mut i)?;
            let ns = get_u64(b, &mut i)? as usize;
            if ns / CASCADE_LANES != n {
                return Err(std::io::Error::other("bad signature count"));
            }
            let sigs = le_u64_slice(b, &mut i, ns)?;
            (true, hyperplanes(hp_seed, dim), sigs)
        } else {
            (false, Vec::new(), &[][..])
        };
        let layer_count = get_u32(b, &mut i)? as usize;
        let mut packed = Vec::with_capacity(layer_count);
        for _ in 0..layer_count {
            get_pad(b, &mut i)?;
            let no = get_u32(b, &mut i)? as usize;
            let offsets = le_u32_slice(b, &mut i, no)?;
            get_pad(b, &mut i)?;
            let nt = get_u32(b, &mut i)? as usize;
            let targets = le_u32_slice(b, &mut i, nt)?;
            if no < 1 || offsets[no - 1] as usize != nt {
                return Err(std::io::Error::other("bad csr section"));
            }
            for w in 1..no {
                if offsets[w] < offsets[w - 1] {
                    return Err(std::io::Error::other("bad csr section"));
                }
            }
            packed.push(PackedLevelRef { offsets, targets });
        }
        if i != b.len() {
            return Err(std::io::Error::other("trailing snapshot bytes"));
        }
        Ok(Self {
            dim,
            metric,
            storage,
            rerank,
            qrange,
            qscale,
            entry,
            max_level,
            n,
            levels,
            vectors,
            codes,
            packed,
            cascade,
            hp,
            sigs,
        })
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    pub fn search(
        &self,
        q: &[f32],
        k: usize,
        ef: usize,
        filter: Option<&dyn Fn(u32) -> bool>,
    ) -> Vec<Hit> {
        let mut ctx = SearchCtx::with_capacity(self.n);
        self.search_with(&mut ctx, q, k, ef, filter)
    }

    pub fn search_with(
        &self,
        ctx: &mut SearchCtx,
        q: &[f32],
        k: usize,
        ef: usize,
        filter: Option<&dyn Fn(u32) -> bool>,
    ) -> Vec<Hit> {
        if self.is_empty() || k == 0 {
            return Vec::new();
        }
        let ef = ef.max(k);
        let SearchCtx { visited, q8 } = ctx;
        if self.storage == Storage::Int8 {
            q8.clear();
            q8.resize(self.dim, 0);
            quantize_into(q8, q, self.qrange);
        }
        let qs = if self.cascade {
            Some(signature(&self.hp, q))
        } else {
            None
        };
        let qr = match qs {
            Some(s) => Query::new_int8_cascade(q, q8, s),
            None => {
                if q8.is_empty() {
                    Query::new_f32(q)
                } else {
                    Query::new_int8(q, q8)
                }
            }
        };
        let mut cur = self.entry;
        for l in (1..=self.max_level).rev() {
            cur = g_greedy(self, &qr, cur, l as usize);
        }
        let mut hits = g_search_layer(self, &qr, &[cur], ef, 0, filter, visited);
        if self.rerank {
            for h in hits.iter_mut() {
                h.dist = self.metric.distance(q, g_row_f32(self, h.id));
            }
            hits.sort_unstable();
        }
        hits.truncate(k);
        hits
    }
}

#[cfg(feature = "mmap")]
impl<'a> GraphAccess for MappedIndex<'a> {
    fn ga_dim(&self) -> usize {
        self.dim
    }
    fn ga_metric(&self) -> Metric {
        self.metric
    }
    fn ga_storage(&self) -> Storage {
        self.storage
    }
    fn ga_rerank(&self) -> bool {
        self.rerank
    }
    fn ga_qrange(&self) -> f32 {
        self.qrange
    }
    fn ga_qscale(&self) -> f32 {
        self.qscale
    }
    fn ga_entry(&self) -> u32 {
        self.entry
    }
    fn ga_max_level(&self) -> u32 {
        self.max_level
    }
    fn ga_levels(&self) -> &[u32] {
        self.levels
    }
    fn ga_vectors(&self) -> &[f32] {
        self.vectors
    }
    fn ga_codes(&self) -> &[i8] {
        self.codes
    }
    fn ga_neighbors(&self, level: usize, id: u32) -> &[u32] {
        let l = &self.packed[level];
        let idx = id as usize;
        if idx + 1 >= l.offsets.len() {
            return &[];
        }
        let s = l.offsets[idx] as usize;
        let e = l.offsets[idx + 1] as usize;
        &l.targets[s..e]
    }
    fn ga_cascade(&self) -> bool {
        self.cascade
    }
    fn ga_sigs(&self) -> &[u64] {
        self.sigs
    }
    fn ga_hyperplanes(&self) -> &[f32] {
        &self.hp
    }
}

fn put_pad(b: &mut Vec<u8>) {
    let rem = b.len() % 8;
    let n = if rem == 0 { 0 } else { 8 - rem };
    put_u32(b, n as u32);
    for _ in 0..n {
        put_u8(b, 0);
    }
}

fn get_pad(b: &[u8], i: &mut usize) -> std::io::Result<()> {
    let n = get_u32(b, i)? as usize;
    need(b, i, n)?;
    *i += n;
    Ok(())
}

#[cfg(feature = "mmap")]
fn le_u32_slice<'a>(b: &'a [u8], i: &mut usize, n: usize) -> std::io::Result<&'a [u32]> {
    need(b, i, n * 4)?;
    if *i % 4 != 0 {
        return Err(std::io::Error::other("unaligned snapshot section"));
    }
    #[cfg(target_endian = "little")]
    {
        let s = unsafe { std::slice::from_raw_parts(b.as_ptr().add(*i) as *const u32, n) };
        *i += n * 4;
        Ok(s)
    }
    #[cfg(target_endian = "big")]
    {
        let _ = n;
        Err(std::io::Error::other("mmap unsupported on big-endian hosts"))
    }
}

#[cfg(feature = "mmap")]
fn le_f32_slice<'a>(b: &'a [u8], i: &mut usize, n: usize) -> std::io::Result<&'a [f32]> {
    need(b, i, n * 4)?;
    if *i % 4 != 0 {
        return Err(std::io::Error::other("unaligned snapshot section"));
    }
    #[cfg(target_endian = "little")]
    {
        let s = unsafe { std::slice::from_raw_parts(b.as_ptr().add(*i) as *const f32, n) };
        *i += n * 4;
        Ok(s)
    }
    #[cfg(target_endian = "big")]
    {
        let _ = n;
        Err(std::io::Error::other("mmap unsupported on big-endian hosts"))
    }
}

#[cfg(feature = "mmap")]
fn le_i8_slice<'a>(b: &'a [u8], i: &mut usize, n: usize) -> std::io::Result<&'a [i8]> {
    need(b, i, n)?;
    #[cfg(target_endian = "little")]
    {
        let s = unsafe { std::slice::from_raw_parts(b.as_ptr().add(*i) as *const i8, n) };
        *i += n;
        Ok(s)
    }
    #[cfg(target_endian = "big")]
    {
        let _ = n;
        Err(std::io::Error::other("mmap unsupported on big-endian hosts"))
    }
}

#[cfg(feature = "mmap")]
fn le_u64_slice<'a>(b: &'a [u8], i: &mut usize, n: usize) -> std::io::Result<&'a [u64]> {
    need(b, i, n * 8)?;
    if *i % 8 != 0 {
        return Err(std::io::Error::other("unaligned snapshot section"));
    }
    #[cfg(target_endian = "little")]
    {
        let s = unsafe { std::slice::from_raw_parts(b.as_ptr().add(*i) as *const u64, n) };
        *i += n * 8;
        Ok(s)
    }
    #[cfg(target_endian = "big")]
    {
        let _ = n;
        Err(std::io::Error::other("mmap unsupported on big-endian hosts"))
    }
}

fn put_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}

fn put_u64(b: &mut Vec<u8>, v: u64) {
    b.extend_from_slice(&v.to_le_bytes());
}

fn put_u8(b: &mut Vec<u8>, v: u8) {
    b.push(v);
}

fn put_f32(b: &mut Vec<u8>, v: f32) {
    b.extend_from_slice(&v.to_le_bytes());
}

fn put_bytes(b: &mut Vec<u8>, v: &[u8]) {
    b.extend_from_slice(v);
}

fn need(b: &[u8], i: &mut usize, n: usize) -> std::io::Result<()> {
    if *i + n > b.len() {
        return Err(std::io::Error::other("snapshot truncated"));
    }
    Ok(())
}

fn get_u32(b: &[u8], i: &mut usize) -> std::io::Result<u32> {
    need(b, i, 4)?;
    let v = u32::from_le_bytes([b[*i], b[*i + 1], b[*i + 2], b[*i + 3]]);
    *i += 4;
    Ok(v)
}

fn get_u64(b: &[u8], i: &mut usize) -> std::io::Result<u64> {
    need(b, i, 8)?;
    let mut t = [0u8; 8];
    t.copy_from_slice(&b[*i..*i + 8]);
    *i += 8;
    Ok(u64::from_le_bytes(t))
}

fn get_u8(b: &[u8], i: &mut usize) -> std::io::Result<u8> {
    need(b, i, 1)?;
    let v = b[*i];
    *i += 1;
    Ok(v)
}

fn get_f32(b: &[u8], i: &mut usize) -> std::io::Result<f32> {
    need(b, i, 4)?;
    let v = f32::from_le_bytes([b[*i], b[*i + 1], b[*i + 2], b[*i + 3]]);
    *i += 4;
    Ok(v)
}
