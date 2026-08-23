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
        }
    }
}

pub struct Query<'a> {
    v: &'a [f32],
    q8: &'a [i8],
}

impl<'a> Query<'a> {
    fn new_f32(v: &'a [f32]) -> Self {
        Self { v, q8: &[] }
    }

    fn new_int8(v: &'a [f32], q8: &'a [i8]) -> Self {
        Self { v, q8 }
    }
}

struct PackedLevel {
    offsets: Vec<u32>,
    targets: Vec<u32>,
}

const SNAPSHOT_MAGIC: u32 = 0x5357_5643;
const SNAPSHOT_VERSION: u32 = 1;

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
        for &l in &self.levels {
            put_u32(&mut b, l);
        }
        put_u64(&mut b, self.vectors.len() as u64);
        for v in &self.vectors {
            put_f32(&mut b, *v);
        }
        put_u64(&mut b, self.codes.len() as u64);
        put_bytes(
            &mut b,
            unsafe { std::slice::from_raw_parts(self.codes.as_ptr() as *const u8, self.codes.len()) },
        );
        match &self.packed {
            Some(packed) => {
                put_u32(&mut b, packed.len() as u32);
                for level in packed {
                    put_u32(&mut b, level.offsets.len() as u32);
                    for &o in &level.offsets {
                        put_u32(&mut b, o);
                    }
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
        if version != SNAPSHOT_VERSION {
            return Err(std::io::Error::other("unsupported snapshot version"));
        }
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
        let mut levels = Vec::with_capacity(n);
        for _ in 0..n {
            levels.push(get_u32(&b, &mut i)?);
        }
        let nv = get_u64(&b, &mut i)? as usize;
        let mut vectors = Vec::with_capacity(nv);
        for _ in 0..nv {
            vectors.push(get_f32(&b, &mut i)?);
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
        let layer_count = get_u32(&b, &mut i)? as usize;
        let mut packed = None;
        let mut layers = Vec::new();
        let mut link_total = 0usize;
        if was_packed {
            let mut p = Vec::with_capacity(layer_count);
            for _ in 0..layer_count {
                let no = get_u32(&b, &mut i)? as usize;
                let mut offsets = Vec::with_capacity(no);
                for _ in 0..no {
                    offsets.push(get_u32(&b, &mut i)?);
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
        let q = if q8buf.is_empty() {
            Query::new_f32(v)
        } else {
            Query::new_int8(v, &q8buf)
        };
        if top > level {
            for l in ((level + 1)..=top).rev() {
                cur = self.greedy(&q, cur, l as usize);
            }
        }
        let mut eps = vec![cur];
        let up = level.min(top);
        let mut visited = std::mem::take(&mut self.visited);
        for l in (0..=up).rev() {
            let cands =
                self.search_layer(&q, &eps, self.ef_construction, l as usize, None, &mut visited);
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
        let qr = if q8.is_empty() {
            Query::new_f32(q)
        } else {
            Query::new_int8(q, q8)
        };
        let mut cur = self.entry;
        for l in (1..=self.max_level).rev() {
            cur = self.greedy(&qr, cur, l as usize);
        }
        let mut hits = self.search_layer(&qr, &[cur], ef, 0, filter, visited);
        if self.rerank {
            for h in hits.iter_mut() {
                h.dist = self.metric.distance(q, self.row_f32(h.id));
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

    #[inline]
    fn row_f32(&self, id: u32) -> &[f32] {
        let off = id as usize * self.dim;
        &self.vectors[off..off + self.dim]
    }

    #[inline]
    fn row_i8(&self, id: u32) -> &[i8] {
        let off = id as usize * self.dim;
        &self.codes[off..off + self.dim]
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    unsafe fn prefetch_row(&self, id: u32) {
        let p = match self.storage {
            Storage::F32 => {
                self.vectors.as_ptr().add(id as usize * self.dim) as *const u8
            }
            Storage::Int8 => self.codes.as_ptr().add(id as usize * self.dim) as *const u8,
        };
        std::arch::x86_64::_mm_prefetch(p as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }

    #[inline]
    fn dist_to(&self, q: &Query, id: u32) -> f32 {
        if q.q8.is_empty() {
            self.metric.distance(q.v, self.row_f32(id))
        } else {
            match self.metric {
                Metric::Dot => 1.0 - dot_i8(q.q8, self.row_i8(id)) as f32 * self.qscale,
                Metric::L2 => l2sq_i8(q.q8, self.row_i8(id)) as f32 * self.qscale,
            }
        }
    }

    #[inline]
    fn dist_between(&self, a: u32, b: u32) -> f32 {
        match self.storage {
            Storage::F32 => self.metric.distance(self.row_f32(a), self.row_f32(b)),
            Storage::Int8 => match self.metric {
                Metric::Dot => 1.0 - dot_i8(self.row_i8(a), self.row_i8(b)) as f32 * self.qscale,
                Metric::L2 => l2sq_i8(self.row_i8(a), self.row_i8(b)) as f32 * self.qscale,
            },
        }
    }

    fn neighbors(&self, level: usize, id: u32) -> &[u32] {
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

    fn greedy(&self, q: &Query, mut cur: u32, level: usize) -> u32 {
        let mut best = self.dist_to(q, cur);
        loop {
            let mut improved = false;
            let nbrs = self.neighbors(level, cur);
            for (i, &n) in nbrs.iter().enumerate() {
                #[cfg(target_arch = "x86_64")]
                if i + 1 < nbrs.len() {
                    unsafe { self.prefetch_row(nbrs[i + 1]) };
                }
                let d = self.dist_to(q, n);
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

    fn search_layer(
        &self,
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
                    dist: self.dist_to(q, e),
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
            let nbrs = self.neighbors(level, c.id);
            for (i, &n) in nbrs.iter().enumerate() {
                #[cfg(target_arch = "x86_64")]
                if i + 1 < nbrs.len() {
                    unsafe { self.prefetch_row(nbrs[i + 1]) };
                }
                if visited.mark(n as usize) {
                    let h = Hit {
                        dist: self.dist_to(q, n),
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

    fn select_neighbors(&self, cands: Vec<Hit>, m: usize) -> Vec<u32> {
        let mut sorted = cands;
        sorted.sort_unstable();
        let mut res: Vec<u32> = Vec::with_capacity(m);
        let mut pruned: Vec<u32> = Vec::new();
        for h in &sorted {
            if res.len() == m {
                break;
            }
            let closer = res.iter().all(|&r| self.dist_between(h.id, r) > h.dist);
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
                    dist: self.dist_between(b, e),
                    id: e,
                })
                .collect();
            let sel = self.select_neighbors(cands, mmax);
            self.layers[level][b as usize] = sel;
        }
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
