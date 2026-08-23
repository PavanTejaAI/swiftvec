#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Metric {
    Dot,
    L2,
}

impl Metric {
    #[inline]
    pub fn distance(self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Metric::Dot => 1.0 - dot(a, b),
            Metric::L2 => l2sq(a, b),
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "dot" => Some(Metric::Dot),
            "l2" => Some(Metric::L2),
            _ => None,
        }
    }
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
            return unsafe { dot_avx2(a, b) };
        }
    }
    dot_scalar(a, b)
}

fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let n8 = n & !7;
    let mut s = [0.0f32; 8];
    let mut i = 0;
    while i < n8 {
        for j in 0..8 {
            s[j] += a[i + j] * b[i + j];
        }
        i += 8;
    }
    let mut acc = s[0] + s[1] + s[2] + s[3] + s[4] + s[5] + s[6] + s[7];
    while i < n {
        acc += a[i] * b[i];
        i += 1;
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let n32 = n & !31;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0;
    while i < n32 {
        acc0 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(i)),
            _mm256_loadu_ps(b.as_ptr().add(i)),
            acc0,
        );
        acc1 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(i + 8)),
            _mm256_loadu_ps(b.as_ptr().add(i + 8)),
            acc1,
        );
        acc0 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(i + 16)),
            _mm256_loadu_ps(b.as_ptr().add(i + 16)),
            acc0,
        );
        acc1 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(i + 24)),
            _mm256_loadu_ps(b.as_ptr().add(i + 24)),
            acc1,
        );
        i += 32;
    }
    let n8 = n & !7;
    while i < n8 {
        acc0 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(i)),
            _mm256_loadu_ps(b.as_ptr().add(i)),
            acc0,
        );
        i += 8;
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    let mut t = [0.0f32; 8];
    _mm256_storeu_ps(t.as_mut_ptr(), acc0);
    let mut acc = t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
    while i < n {
        acc += a[i] * b[i];
        i += 1;
    }
    acc
}

pub fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
            return unsafe { l2sq_avx2(a, b) };
        }
    }
    l2sq_scalar(a, b)
}

fn l2sq_scalar(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let n8 = n & !7;
    let mut s = [0.0f32; 8];
    let mut i = 0;
    while i < n8 {
        for j in 0..8 {
            let d = a[i + j] - b[i + j];
            s[j] += d * d;
        }
        i += 8;
    }
    let mut acc = s[0] + s[1] + s[2] + s[3] + s[4] + s[5] + s[6] + s[7];
    while i < n {
        let d = a[i] - b[i];
        acc += d * d;
        i += 1;
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn l2sq_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let n32 = n & !31;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0;
    while i < n32 {
        let d0 = _mm256_sub_ps(_mm256_loadu_ps(a.as_ptr().add(i)), _mm256_loadu_ps(b.as_ptr().add(i)));
        let d1 = _mm256_sub_ps(
            _mm256_loadu_ps(a.as_ptr().add(i + 8)),
            _mm256_loadu_ps(b.as_ptr().add(i + 8)),
        );
        let d2 = _mm256_sub_ps(
            _mm256_loadu_ps(a.as_ptr().add(i + 16)),
            _mm256_loadu_ps(b.as_ptr().add(i + 16)),
        );
        let d3 = _mm256_sub_ps(
            _mm256_loadu_ps(a.as_ptr().add(i + 24)),
            _mm256_loadu_ps(b.as_ptr().add(i + 24)),
        );
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);
        acc0 = _mm256_fmadd_ps(d2, d2, acc0);
        acc1 = _mm256_fmadd_ps(d3, d3, acc1);
        i += 32;
    }
    let n8 = n & !7;
    while i < n8 {
        let d = _mm256_sub_ps(_mm256_loadu_ps(a.as_ptr().add(i)), _mm256_loadu_ps(b.as_ptr().add(i)));
        acc0 = _mm256_fmadd_ps(d, d, acc0);
        i += 8;
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    let mut t = [0.0f32; 8];
    _mm256_storeu_ps(t.as_mut_ptr(), acc0);
    let mut acc = t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
    while i < n {
        let d = a[i] - b[i];
        acc += d * d;
        i += 1;
    }
    acc
}
