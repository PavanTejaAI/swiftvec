use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::ToPyObject;
use std::io::Cursor;
use std::path::PathBuf;
use std::sync::Mutex;
use swiftvec_core::{rrf_fuse, Bm25Index, Hnsw, HnswConfig, Metric, SearchCtx, Storage};
use swiftvec_embed::{EmbedOptions, Embedder};

fn take<'a>(b: &'a [u8], off: &mut usize, n: usize) -> PyResult<&'a [u8]> {
    if *off + n > b.len() {
        return Err(PyRuntimeError::new_err("corrupt snapshot"));
    }
    let s = &b[*off..*off + n];
    *off += n;
    Ok(s)
}

fn err(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

fn json_from_py(v: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    if v.is_none() {
        return Ok(serde_json::Value::Null);
    }
    if let Ok(b) = v.extract::<bool>() {
        return Ok(serde_json::Value::Bool(b));
    }
    if let Ok(s) = v.extract::<String>() {
        return Ok(serde_json::Value::String(s));
    }
    if let Ok(i) = v.extract::<i64>() {
        return Ok(serde_json::Value::Number(i.into()));
    }
    if let Ok(f) = v.extract::<f64>() {
        if let Some(n) = serde_json::Number::from_f64(f) {
            return Ok(serde_json::Value::Number(n));
        }
    }
    if let Ok(d) = v.downcast::<pyo3::types::PyDict>() {
        let mut m = serde_json::Map::new();
        for (k, val) in d.iter() {
            let key: String = k.extract()?;
            m.insert(key, json_from_py(&val)?);
        }
        return Ok(serde_json::Value::Object(m));
    }
    if let Ok(l) = v.downcast::<pyo3::types::PyList>() {
        let mut a = Vec::with_capacity(l.len());
        for item in l.iter() {
            a.push(json_from_py(&item)?);
        }
        return Ok(serde_json::Value::Array(a));
    }
    Err(PyRuntimeError::new_err(
        "metadata values must be str, int, float, bool, None, dict or list",
    ))
}

fn json_to_py(py: Python<'_>, v: &serde_json::Value) -> PyObject {
    use serde_json::Value;
    match v {
        Value::Null => py.None(),
        Value::Bool(b) => b.to_object(py),
        Value::String(s) => s.to_object(py),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_object(py)
            } else {
                n.as_f64().unwrap_or(0.0).to_object(py)
            }
        }
        Value::Array(a) => {
            let items: Vec<PyObject> = a.iter().map(|x| json_to_py(py, x)).collect();
            items.to_object(py)
        }
        Value::Object(o) => {
            let d = pyo3::types::PyDict::new(py);
            for (k, val) in o {
                d.set_item(k, json_to_py(py, val)).unwrap();
            }
            d.into_any().unbind()
        }
    }
}

#[pyclass]
struct SearchResult {
    #[pyo3(get)]
    id: String,
    #[pyo3(get)]
    score: f32,
    #[pyo3(get)]
    metadata: PyObject,
}

#[pymethods]
impl SearchResult {
    fn __repr__(&self) -> String {
        format!(
            "<SearchResult id={} score={:.4}>",
            self.id, self.score
        )
    }
}

#[pyclass]
struct SwiftVec {
    embedder: Mutex<Embedder>,
    index: Option<Hnsw>,
    ctx: SearchCtx,
    bm25: Bm25Index,
    ids: Vec<String>,
    metas: Vec<serde_json::Value>,
    texts: Vec<String>,
    dim: usize,
    model: PathBuf,
}

#[pymethods]
impl SwiftVec {
    #[new]
    #[pyo3(signature = (model_dir="models/leaf-ir", dim=None))]
    fn new(model_dir: &str, dim: Option<usize>) -> PyResult<Self> {
        let model = PathBuf::from(model_dir);
        let embedder = Embedder::load(&model).map_err(err)?;
        let full = embedder.dim();
        let dim = dim.unwrap_or(full);
        if dim == 0 || dim > full {
            return Err(PyRuntimeError::new_err(format!(
                "dim must be in 1..={full}"
            )));
        }
        Ok(Self {
            embedder: Mutex::new(embedder),
            index: None,
            ctx: SearchCtx::with_capacity(0),
            bm25: Bm25Index::new(),
            ids: Vec::new(),
            metas: Vec::new(),
            texts: Vec::new(),
            dim,
            model,
        })
    }

    #[pyo3(signature = (ids, texts, metadatas=None))]
    fn add_batch(
        &mut self,
        py: Python<'_>,
        ids: Vec<String>,
        texts: Vec<String>,
        metadatas: Option<Vec<Option<Py<PyAny>>>>,
    ) -> PyResult<usize> {
        if ids.len() != texts.len() {
            return Err(PyRuntimeError::new_err("ids and texts must have equal length"));
        }
        if let Some(m) = &metadatas {
            if m.len() != ids.len() {
                return Err(PyRuntimeError::new_err(
                    "metadatas must have the same length as ids",
                ));
            }
        }
        if texts.is_empty() {
            return Ok(self.ids.len());
        }
        let mut metas_converted = Vec::with_capacity(texts.len());
        if let Some(m) = &metadatas {
            for item in m {
                match item {
                    Some(obj) => metas_converted.push(json_from_py(obj.bind(py))?),
                    None => metas_converted.push(serde_json::Value::Null),
                }
            }
        } else {
            for _ in &texts {
                metas_converted.push(serde_json::Value::Null);
            }
        }
        let opts = self.opts();
        let mut embedder = self.embedder.lock().unwrap();
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let emb: &mut Embedder = &mut embedder;
        let vecs = py
            .allow_threads(|| emb.embed(&refs, false, &opts))
            .map_err(err)?;
        drop(embedder);
        let index = self.index.get_or_insert_with(|| {
            let mut cfg = HnswConfig::new(self.dim, Metric::Dot);
            cfg.storage = Storage::Int8;
            cfg.rerank_f32 = true;
            cfg.qrange = 0.3;
            Hnsw::with_capacity(cfg, 1024)
        });
        for (((id, text), meta), v) in ids.into_iter().zip(texts).zip(metas_converted).zip(vecs) {
            self.ids.push(id);
            self.texts.push(text.clone());
            self.metas.push(meta);
            self.bm25.add(&text);
            index.add(&v);
        }
        self.ctx = SearchCtx::with_capacity(index.len());
        Ok(self.ids.len())
    }

    #[pyo3(signature = (id, text, metadata=None))]
    fn add(
        &mut self,
        py: Python<'_>,
        id: String,
        text: String,
        metadata: Option<Py<PyAny>>,
    ) -> PyResult<usize> {
        self.add_batch(py, vec![id], vec![text], metadata.map(|m| vec![Some(m)]))
    }

    #[pyo3(signature = (query, top_k=5, ef=None, alpha=None))]
    fn search(
        &mut self,
        py: Python<'_>,
        query: &str,
        top_k: usize,
        ef: Option<usize>,
        alpha: Option<f32>,
    ) -> PyResult<Vec<SearchResult>> {
        if self.index.is_none() {
            return Err(PyRuntimeError::new_err(
                "index is empty, call add() first",
            ));
        }
        if top_k == 0 || top_k > self.ids.len() {
            return Err(PyRuntimeError::new_err(format!(
                "top_k must be in 1..={}",
                self.ids.len()
            )));
        }
        let opts = self.opts();
        let mut embedder = self.embedder.lock().unwrap();
        let emb: &mut Embedder = &mut embedder;
        let qv = py
            .allow_threads(|| emb.embed(&[query], true, &opts))
            .map_err(err)?
            .remove(0);
        drop(embedder);
        let SwiftVec {
            index,
            ctx,
            ids,
            bm25,
            metas,
            ..
        } = self;
        let index = index.as_ref().unwrap();
        match alpha {
            None => {
                let ef = ef.unwrap_or(top_k * 8).max(top_k);
                let hits = py.allow_threads(|| index.search_with(ctx, &qv, top_k, ef, None));
                Ok(hits
                    .into_iter()
                    .map(|h| SearchResult {
                        id: ids[h.id as usize].clone(),
                        score: 1.0 - h.dist,
                        metadata: json_to_py(py, &metas[h.id as usize]),
                    })
                    .collect())
            }
            Some(a) => {
                if !(0.0..=1.0).contains(&a) {
                    return Err(PyRuntimeError::new_err("alpha must be in [0, 1]"));
                }
                let fetch = (top_k * 4).max(16).min(ids.len());
                let ef = ef.unwrap_or(fetch * 2).max(fetch);
                let hits = py.allow_threads(|| index.search_with(ctx, &qv, fetch, ef, None));
                let kw = bm25.search(query, fetch);
                let fused = rrf_fuse(&hits, &kw, top_k, a, 1.0 - a);
                Ok(fused
                    .into_iter()
                    .map(|(id, score)| SearchResult {
                        id: ids[id as usize].clone(),
                        score,
                        metadata: json_to_py(py, &metas[id as usize]),
                    })
                    .collect())
            }
        }
    }

    fn save(&self, path: &str) -> PyResult<()> {
        let index = self
            .index
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("index is empty, call add() first"))?;
        let mut core = Vec::new();
        index.save(&mut core).map_err(err)?;
        let ids = serde_json::to_vec(&self.ids).map_err(err)?;
        let metas = serde_json::to_vec(&self.metas).map_err(err)?;
        let texts = serde_json::to_vec(&self.texts).map_err(err)?;
        let mut out = Vec::with_capacity(8 + core.len() + ids.len() + metas.len() + texts.len());
        out.extend_from_slice(&(core.len() as u64).to_le_bytes());
        out.extend_from_slice(&core);
        out.extend_from_slice(&(ids.len() as u64).to_le_bytes());
        out.extend_from_slice(&ids);
        out.extend_from_slice(&(metas.len() as u64).to_le_bytes());
        out.extend_from_slice(&metas);
        out.extend_from_slice(&texts);
        std::fs::write(path, out).map_err(err)
    }

    #[staticmethod]
    #[pyo3(signature = (path, model_dir="models/leaf-ir"))]
    fn load(path: &str, model_dir: &str) -> PyResult<Self> {
        let b = std::fs::read(path).map_err(err)?;
        if b.len() < 8 {
            return Err(PyRuntimeError::new_err("not a swiftvec snapshot"));
        }
        let mut off = 0usize;
        let core_len = u64::from_le_bytes(take(&b, &mut off, 8)?.try_into().unwrap()) as usize;
        let index = Hnsw::load(&mut Cursor::new(take(&b, &mut off, core_len)?)).map_err(err)?;
        let ids_len = u64::from_le_bytes(take(&b, &mut off, 8)?.try_into().unwrap()) as usize;
        let ids: Vec<String> = serde_json::from_slice(take(&b, &mut off, ids_len)?).map_err(err)?;
        let metas_len = u64::from_le_bytes(take(&b, &mut off, 8)?.try_into().unwrap()) as usize;
        let metas: Vec<serde_json::Value> =
            serde_json::from_slice(take(&b, &mut off, metas_len)?).map_err(err)?;
        let texts_len = b.len() - off;
        let texts: Vec<String> =
            serde_json::from_slice(take(&b, &mut off, texts_len)?).map_err(err)?;
        let mut bm25 = Bm25Index::new();
        for t in &texts {
            bm25.add(t);
        }
        let model = PathBuf::from(model_dir);
        let embedder = Embedder::load(&model).map_err(err)?;
        let n = index.len();
        let dim = index.dim();
        Ok(Self {
            embedder: Mutex::new(embedder),
            index: Some(index),
            ctx: SearchCtx::with_capacity(n),
            bm25,
            ids,
            metas,
            texts,
            dim,
            model,
        })
    }

    fn info(&self, py: Python<'_>) -> PyObject {
        let d = pyo3::types::PyDict::new(py);
        d.set_item("docs", self.ids.len()).unwrap();
        d.set_item("dim", self.dim).unwrap();
        d.set_item("model_dir", self.model.to_str().unwrap_or("")).unwrap();
        d.set_item("hybrid", true).unwrap();
        d.into_any().unbind()
    }

    #[getter]
    fn dim(&self) -> usize {
        self.dim
    }

    fn __len__(&self) -> usize {
        self.ids.len()
    }

    fn __repr__(&self) -> String {
        format!("<SwiftVec docs={} dim={}>", self.ids.len(), self.dim)
    }
}

impl SwiftVec {
    fn opts(&self) -> EmbedOptions {
        EmbedOptions {
            truncate_dim: if self.dim > 0 && self.dim < 768 {
                Some(self.dim)
            } else {
                None
            },
        }
    }
}

#[pymodule]
fn swiftvec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SwiftVec>()?;
    m.add_class::<SearchResult>()?;
    Ok(())
}
