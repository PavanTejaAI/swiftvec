use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use safetensors::SafeTensors;
use std::fmt;
use std::path::{Path, PathBuf};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

#[derive(Debug)]
pub enum EmbedError {
    Io(PathBuf, std::io::Error),
    Ort(String),
    Tok(String),
    St(String),
    Json(String),
    Shape(String),
}

impl fmt::Display for EmbedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EmbedError::Io(p, e) => write!(f, "{}: {e}", p.display()),
            EmbedError::Ort(e) => write!(f, "ort: {e}"),
            EmbedError::Tok(e) => write!(f, "tokenizer: {e}"),
            EmbedError::St(e) => write!(f, "safetensors: {e}"),
            EmbedError::Json(e) => write!(f, "json: {e}"),
            EmbedError::Shape(e) => write!(f, "shape: {e}"),
        }
    }
}

impl std::error::Error for EmbedError {}

fn ort_e(e: impl fmt::Display) -> EmbedError {
    EmbedError::Ort(e.to_string())
}

pub struct EmbedOptions {
    pub truncate_dim: Option<usize>,
}

impl Default for EmbedOptions {
    fn default() -> Self {
        Self { truncate_dim: None }
    }
}

pub struct Embedder {
    session: Session,
    tokenizer: Tokenizer,
    dense_w: Vec<f32>,
    dense_b: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
    query_prompt: String,
    max_len: usize,
    output_name: String,
}

impl Embedder {
    pub fn load(dir: &Path) -> Result<Self, EmbedError> {
        Self::load_with_config(
            dir,
            std::thread::available_parallelism().map_or(2, |n| n.get().min(4)),
            "model_quantized.onnx",
        )
    }

    pub fn load_with_threads(dir: &Path, threads: usize) -> Result<Self, EmbedError> {
        Self::load_with_config(dir, threads, "model_quantized.onnx")
    }

    pub fn load_with_config(dir: &Path, threads: usize, onnx_file: &str) -> Result<Self, EmbedError> {
        if std::env::var_os("ORT_DYLIB_PATH").is_none() {
            for c in ["onnxruntime.dll", "vendor/onnxruntime/onnxruntime.dll"] {
                if Path::new(c).exists() {
                    std::env::set_var("ORT_DYLIB_PATH", c);
                    break;
                }
            }
        }
        let read = |p: &str| {
            std::fs::read_to_string(dir.join(p)).map_err(|e| EmbedError::Io(dir.join(p), e))
        };
        let parse = |p: &str| -> Result<serde_json::Value, EmbedError> {
            serde_json::from_str(&read(p)?).map_err(|e| EmbedError::Json(e.to_string()))
        };
        let cfg = parse("config.json")?;
        let hidden = cfg["hidden_size"].as_u64().unwrap_or(0) as usize;
        let st_cfg = parse("config_sentence_transformers.json")?;
        let query_prompt = st_cfg["prompts"]["query"]
            .as_str()
            .unwrap_or("Represent this sentence for searching relevant passages: ")
            .to_string();
        let sb_cfg = parse("sentence_bert_config.json")?;
        let max_len = sb_cfg["max_seq_length"].as_u64().unwrap_or(512) as usize;
        let dense_cfg = parse("2_Dense/config.json")?;
        let in_dim = dense_cfg["in_features"].as_u64().unwrap_or(0) as usize;
        let out_dim = dense_cfg["out_features"].as_u64().unwrap_or(0) as usize;
        if hidden != in_dim || hidden == 0 {
            return Err(EmbedError::Shape(format!("hidden {hidden} vs dense in {in_dim}")));
        }
        let st_path = dir.join("2_Dense/model.safetensors");
        let st_bytes = std::fs::read(&st_path).map_err(|e| EmbedError::Io(st_path, e))?;
        let st = SafeTensors::deserialize(&st_bytes).map_err(|e| EmbedError::St(e.to_string()))?;
        let mut dense_w = Vec::new();
        let mut dense_b = Vec::new();
        for name in st.names() {
            let t = st.tensor(name).map_err(|e| EmbedError::St(e.to_string()))?;
            match t.shape() {
                [o, i] if *o == out_dim && *i == in_dim => {
                    dense_w = t
                        .data()
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                }
                [o] if *o == out_dim => {
                    dense_b = t
                        .data()
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                }
                _ => {}
            }
        }
        if dense_w.is_empty() || dense_b.is_empty() {
            return Err(EmbedError::Shape("dense tensors missing".into()));
        }
        let mut tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| EmbedError::Tok(e.to_string()))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: max_len,
                ..Default::default()
            }))
            .map_err(|e| EmbedError::Tok(e.to_string()))?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            pad_id: 0,
            pad_type_id: 0,
            pad_token: "[PAD]".into(),
            direction: tokenizers::PaddingDirection::Right,
            pad_to_multiple_of: None,
        }));
        let onnx = dir.join("onnx").join(onnx_file);
        if !onnx.exists() {
            return Err(EmbedError::Io(
                onnx,
                std::io::Error::new(std::io::ErrorKind::NotFound, "run tools/fetch_model.sh"),
            ));
        }
        let builder = Session::builder().map_err(ort_e)?;
        let builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort_e)?;
        let mut builder = builder.with_intra_threads(threads).map_err(ort_e)?;
        let mut session = builder.commit_from_file(&onnx).map_err(ort_e)?;
        let output_name = session
            .outputs()
            .first()
            .map(|o| o.name().to_string())
            .unwrap_or_default();
        let warm_ids = Tensor::from_array(([1usize, 2usize], vec![101i64, 102i64])).map_err(ort_e)?;
        let warm_mask = Tensor::from_array(([1usize, 2usize], vec![1i64, 1i64])).map_err(ort_e)?;
        let warm_tt = Tensor::from_array(([1usize, 2usize], vec![0i64, 0i64])).map_err(ort_e)?;
        session
            .run(ort::inputs![
                "input_ids" => warm_ids,
                "attention_mask" => warm_mask,
                "token_type_ids" => warm_tt,
            ])
            .map_err(ort_e)?;
        Ok(Self {
            session,
            tokenizer,
            dense_w,
            dense_b,
            in_dim,
            out_dim,
            query_prompt,
            max_len,
            output_name,
        })
    }

    pub fn dim(&self) -> usize {
        self.out_dim
    }

    pub fn max_seq_length(&self) -> usize {
        self.max_len
    }

    pub fn embed(
        &mut self,
        texts: &[&str],
        query: bool,
        opts: &EmbedOptions,
    ) -> Result<Vec<Vec<f32>>, EmbedError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let batch: Vec<String> = texts
            .iter()
            .map(|t| {
                if query {
                    format!("{}{}", self.query_prompt, t)
                } else {
                    (*t).to_string()
                }
            })
            .collect();
        let refs: Vec<&str> = batch.iter().map(|s| s.as_str()).collect();
        let encodings = self
            .tokenizer
            .encode_batch(refs, true)
            .map_err(|e| EmbedError::Tok(e.to_string()))?;
        let seq = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(1);
        let n = encodings.len();
        let mut ids = Vec::with_capacity(n * seq);
        let mut mask = Vec::with_capacity(n * seq);
        let mut tt = Vec::with_capacity(n * seq);
        for e in &encodings {
            ids.extend(e.get_ids().iter().map(|&t| t as i64));
            mask.extend(e.get_attention_mask().iter().map(|&t| t as i64));
            tt.extend(e.get_type_ids().iter().map(|&t| t as i64));
        }
        let ids_t = Tensor::from_array(([n, seq], ids)).map_err(ort_e)?;
        let mask_t = Tensor::from_array(([n, seq], mask.clone())).map_err(ort_e)?;
        let tt_t = Tensor::from_array(([n, seq], tt)).map_err(ort_e)?;
        let outputs = self
            .session
            .run(ort::inputs![
                "input_ids" => ids_t,
                "attention_mask" => mask_t,
                "token_type_ids" => tt_t,
            ])
            .map_err(ort_e)?;
        let (_shape, raw) = outputs[self.output_name.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(ort_e)?;
        if raw.len() != n * seq * self.in_dim {
            return Err(EmbedError::Shape(format!(
                "hidden len {} != {}",
                raw.len(),
                n * seq * self.in_dim
            )));
        }
        let mut out = Vec::with_capacity(n);
        for r in 0..n {
            let mut pooled = vec![0.0f32; self.in_dim];
            let mut count = 0.0f32;
            for t in 0..seq {
                if mask[r * seq + t] > 0 {
                    let base = (r * seq + t) * self.in_dim;
                    for d in 0..self.in_dim {
                        pooled[d] += raw[base + d];
                    }
                    count += 1.0;
                }
            }
            let inv = 1.0 / count.max(1.0);
            let dim = opts.truncate_dim.unwrap_or(self.out_dim).min(self.out_dim);
            let mut v = vec![0.0f32; dim];
            for o in 0..dim {
                let wrow = &self.dense_w[o * self.in_dim..(o + 1) * self.in_dim];
                let mut acc = self.dense_b[o];
                for i in 0..self.in_dim {
                    acc += pooled[i] * inv * wrow[i];
                }
                v[o] = acc;
            }
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for x in &mut v {
                *x /= norm;
            }
            out.push(v);
        }
        Ok(out)
    }
}
