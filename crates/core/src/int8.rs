pub fn quantize_into(dst: &mut [i8], src: &[f32], range: f32) {
    let s = 127.0 / range;
    for (d, &x) in dst.iter_mut().zip(src.iter()) {
        *d = (x * s).round().clamp(-127.0, 127.0) as i8;
    }
}

pub fn calibrate_range(vectors: &[f32]) -> f32 {
    let m = vectors
        .iter()
        .fold(0.0f32, |m, &x| if x.abs() > m { x.abs() } else { m });
    m.max(1e-9)
}

pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return unsafe { dot_i8_avx2(a, b) };
        }
    }
    dot_i8_scalar(a, b)
}

fn dot_i8_scalar(a: &[i8], b: &[i8]) -> i32 {
    let mut acc = 0i32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        acc += x as i32 * y as i32;
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_avx2(a: &[i8], b: &[i8]) -> i32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_si256();
    let n16 = a.len() & !15;
    let mut i = 0;
    while i < n16 {
        let av = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
        let bv = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
        let a16 = _mm256_cvtepi8_epi16(av);
        let b16 = _mm256_cvtepi8_epi16(bv);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a16, b16));
        i += 16;
    }
    let mut t = [0i32; 8];
    _mm256_storeu_si256(t.as_mut_ptr() as *mut __m256i, acc);
    let mut acc_s = t.iter().sum::<i32>();
    while i < a.len() {
        acc_s += a[i] as i32 * b[i] as i32;
        i += 1;
    }
    acc_s
}

pub fn l2sq_i8(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let n8 = n & !7;
    let mut s = [0i32; 8];
    let mut i = 0;
    while i < n8 {
        for j in 0..8 {
            let d = a[i + j] as i32 - b[i + j] as i32;
            s[j] += d * d;
        }
        i += 8;
    }
    let mut acc = s.iter().sum::<i32>();
    while i < n {
        let d = a[i] as i32 - b[i] as i32;
        acc += d * d;
        i += 1;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> i8 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as i32 % 251 - 125) as i8
    }

    #[test]
    fn dot_i8_kernels_agree() {
        let mut seed = 42u64;
        for len in [1usize, 7, 16, 17, 31, 64, 255, 256, 768] {
            let a: Vec<i8> = (0..len).map(|_| lcg(&mut seed)).collect();
            let b: Vec<i8> = (0..len).map(|_| lcg(&mut seed)).collect();
            let want = dot_i8_scalar(&a, &b);
            assert_eq!(dot_i8(&a, &b), want);
            #[cfg(target_arch = "x86_64")]
            if std::arch::is_x86_feature_detected!("avx2") {
                assert_eq!(unsafe { dot_i8_avx2(&a, &b) }, want);
            }
        }
    }

    #[test]
    fn quantize_round_trip() {
        let src = [-0.3f32, 0.0, 0.29, -0.15, 0.3];
        let mut dst = [0i8; 5];
        quantize_into(&mut dst, &src, 0.3);
        for (d, s) in dst.iter().zip(src.iter()) {
            let back = *d as f32 * 0.3 / 127.0;
            assert!((back - s).abs() < 0.01);
        }
        assert!(calibrate_range(&src) - 0.3 < 1e-6);
    }
}
