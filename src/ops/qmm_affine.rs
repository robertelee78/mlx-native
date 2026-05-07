//! Fused affine quantized matmul (`Y = X @ dequant(W)^T`) — ADR-020
//! iter-15 DWQ inference primitive.
//!
//! Given a DWQ-trained Linear weight stored as
//! `(q_int: u8[N, K], scales: f32[N, K/group_size], biases:
//! f32[N, K/group_size])` and an FP32 activation tensor
//! `x: f32[M, K]`, computes
//!
//! ```text
//! y[m, n] = Σ_k x[m, k] · (q_int[n, k] · scales[n, g(k)] + biases[n, g(k)])
//! ```
//!
//! where `g(k) = k / group_size`.  This is mathematically equivalent
//! to the two-step composition `w_dq = qdq_affine_forward(q_int,
//! scales, biases) → y = matmul(x, w_dq^T)`, but fused into a
//! single kernel pass to avoid materializing the dequantized
//! weight tensor in GPU memory (relevant for large Linears where
//! `N · K · 4 bytes` is multi-hundred-MB).
//!
//! Layout matches iter-13b's `qdq_affine` kernel family: UNPACKED
//! uint8 codes (one byte per nibble; supports up to 8-bit
//! quantization without bit-packing).  A packed-byte variant
//! matching mlx's on-disk convention (2 nibbles per byte for bits=4)
//! is deferred to iter-15b.
//!
//! Performance: iter-15 ships a correctness-first per-element kernel
//! (one thread per `(m, n)` output element).  iter-15b will land a
//! tiled + simdgroup-MMA variant matching mlx's `affine_qmm_t`
//! (BM=BK=BN=32, WM=WN=2 — 4 simdgroups per TG, 128 threads).

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

pub static QMM_AFFINE_SHADER_SOURCE: &str = include_str!("../shaders/qmm_affine.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("qmm_affine_t_f32", QMM_AFFINE_SHADER_SOURCE);
}

/// Dispatch the fused affine quantized matmul.
///
/// `meta` must be a u32 buffer of length 4: `[M, N, K, group_size]`.
///
/// Returns `Err(MlxError::InvalidArgument)` for any of:
///   - `M`, `N`, `K`, `group_size` not all > 0.
///   - `K % group_size != 0`.
///   - `group_size` not a power of two in `[2, 1024]`.
///   - Buffer dtype or element-count mismatches.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_qmm_affine_t_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    x: &MlxBuffer,
    q_int: &MlxBuffer,
    scales: &MlxBuffer,
    biases: &MlxBuffer,
    y: &MlxBuffer,
    meta: &MlxBuffer,
    m: u32,
    n: u32,
    k: u32,
    group_size: u32,
) -> Result<()> {
    const OP: &str = "qmm_affine_t_f32";
    if m == 0 || n == 0 || k == 0 {
        return Err(MlxError::InvalidArgument(format!(
            "{OP}: M, N, K must all be > 0; got ({m}, {n}, {k})"
        )));
    }
    if !(2..=1024).contains(&group_size) || !group_size.is_power_of_two() {
        return Err(MlxError::InvalidArgument(format!(
            "{OP}: group_size must be a power of two in [2, 1024]; got {group_size}"
        )));
    }
    if k % group_size != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "{OP}: K ({k}) must be divisible by group_size ({group_size})"
        )));
    }
    if x.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(format!(
            "{OP}: x dtype {} not f32",
            x.dtype()
        )));
    }
    if q_int.dtype() != DType::U8 {
        return Err(MlxError::InvalidArgument(format!(
            "{OP}: q_int dtype {} not u8",
            q_int.dtype()
        )));
    }
    if scales.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(format!(
            "{OP}: scales dtype {} not f32",
            scales.dtype()
        )));
    }
    if biases.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(format!(
            "{OP}: biases dtype {} not f32",
            biases.dtype()
        )));
    }
    if y.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(format!(
            "{OP}: y dtype {} not f32",
            y.dtype()
        )));
    }
    let m_us = m as usize;
    let n_us = n as usize;
    let k_us = k as usize;
    let gs_us = group_size as usize;
    if x.element_count() != m_us * k_us {
        return Err(MlxError::InvalidArgument(format!(
            "{OP}: x element_count {} != M*K = {}",
            x.element_count(),
            m_us * k_us
        )));
    }
    if q_int.element_count() != n_us * k_us {
        return Err(MlxError::InvalidArgument(format!(
            "{OP}: q_int element_count {} != N*K = {}",
            q_int.element_count(),
            n_us * k_us
        )));
    }
    let n_groups = n_us * (k_us / gs_us);
    if scales.element_count() != n_groups {
        return Err(MlxError::InvalidArgument(format!(
            "{OP}: scales element_count {} != N * K/group_size = {}",
            scales.element_count(),
            n_groups
        )));
    }
    if biases.element_count() != n_groups {
        return Err(MlxError::InvalidArgument(format!(
            "{OP}: biases element_count {} != N * K/group_size = {}",
            biases.element_count(),
            n_groups
        )));
    }
    if y.element_count() != m_us * n_us {
        return Err(MlxError::InvalidArgument(format!(
            "{OP}: y element_count {} != M*N = {}",
            y.element_count(),
            m_us * n_us
        )));
    }
    if meta.byte_len() < 16 {
        return Err(MlxError::InvalidArgument(format!(
            "{OP}: meta must be >= 16 bytes ([M,N,K,group_size] u32); got {}",
            meta.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline(OP, device)?;
    // Threadgroup: 16x16 (256 threads) — fits well within Apple's
    // max-threads-per-tg.  Grid: ceil(M/16) x ceil(N/16) x 1
    // threadgroups; each TG covers a 16x16 output tile, one thread
    // per output element.
    let tg_x: u64 = std::cmp::min(16, m as u64);
    let tg_y: u64 = std::cmp::min(16, n as u64);
    let tg_count_x = (m as u64).div_ceil(tg_x);
    let tg_count_y = (n as u64).div_ceil(tg_y);
    encoder.encode_threadgroups(
        pipeline,
        &[(0, x), (1, q_int), (2, scales), (3, biases), (4, y), (5, meta)],
        MTLSize::new(tg_count_x, tg_count_y, 1),
        MTLSize::new(tg_x, tg_y, 1),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MlxDevice;

    /// CPU oracle: `y = x @ dequant(q_int, scales, biases)^T`.
    fn qmm_affine_t_cpu(
        x: &[f32],
        q_int: &[u8],
        scales: &[f32],
        biases: &[f32],
        m: usize,
        n: usize,
        k: usize,
        group_size: usize,
    ) -> Vec<f32> {
        let groups_per_row = k / group_size;
        let mut y = vec![0.0f32; m * n];
        for r in 0..m {
            for col in 0..n {
                let mut acc = 0.0f64;
                for g in 0..groups_per_row {
                    let s = scales[col * groups_per_row + g] as f64;
                    let b = biases[col * groups_per_row + g] as f64;
                    for i in 0..group_size {
                        let kk = g * group_size + i;
                        let q = q_int[col * k + kk] as f64;
                        let w_dq = q * s + b;
                        acc += (x[r * k + kk] as f64) * w_dq;
                    }
                }
                y[r * n + col] = acc as f32;
            }
        }
        y
    }

    fn alloc_f32(device: &MlxDevice, n: usize, shape: Vec<usize>) -> MlxBuffer {
        device
            .alloc_buffer(n * 4, DType::F32, shape)
            .expect("alloc f32")
    }

    fn alloc_u8(device: &MlxDevice, n: usize, shape: Vec<usize>) -> MlxBuffer {
        device.alloc_buffer(n, DType::U8, shape).expect("alloc u8")
    }

    fn make_meta(device: &MlxDevice, m: u32, n: u32, k: u32, gs: u32) -> MlxBuffer {
        let mut buf = device.alloc_buffer(16, DType::U32, vec![4]).unwrap();
        let dst = buf.as_mut_slice::<u32>().unwrap();
        dst.copy_from_slice(&[m, n, k, gs]);
        buf
    }

    #[test]
    fn qmm_affine_t_matches_cpu_oracle_4bit_g32() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        let m = 8usize;
        let n = 16usize;
        let k = 64usize;
        let gs = 32usize;
        let groups_per_row = k / gs;

        // Synthetic deterministic fixture.
        let x: Vec<f32> = (0..(m * k))
            .map(|i| ((i as f32) * 0.013 - 0.4).sin() * 0.6)
            .collect();
        let q_int: Vec<u8> = (0..(n * k)).map(|i| ((i * 11 + 3) % 16) as u8).collect();
        let scales: Vec<f32> = (0..(n * groups_per_row))
            .map(|i| 0.05 + (i as f32) * 0.003)
            .collect();
        let biases: Vec<f32> = (0..(n * groups_per_row))
            .map(|i| -0.2 + (i as f32) * 0.011)
            .collect();

        let mut x_buf = alloc_f32(&device, m * k, vec![m, k]);
        x_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&x);
        let mut q_buf = alloc_u8(&device, n * k, vec![n, k]);
        q_buf.as_mut_slice::<u8>().unwrap().copy_from_slice(&q_int);
        let mut s_buf = alloc_f32(&device, n * groups_per_row, vec![n, groups_per_row]);
        s_buf
            .as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(&scales);
        let mut b_buf = alloc_f32(&device, n * groups_per_row, vec![n, groups_per_row]);
        b_buf
            .as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(&biases);
        let y_buf = alloc_f32(&device, m * n, vec![m, n]);
        let meta = make_meta(&device, m as u32, n as u32, k as u32, gs as u32);

        let mut encoder = device.command_encoder().unwrap();
        dispatch_qmm_affine_t_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &x_buf,
            &q_buf,
            &s_buf,
            &b_buf,
            &y_buf,
            &meta,
            m as u32,
            n as u32,
            k as u32,
            gs as u32,
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();

        let gpu = y_buf.as_slice::<f32>().unwrap();
        let cpu = qmm_affine_t_cpu(&x, &q_int, &scales, &biases, m, n, k, gs);
        for i in 0..(m * n) {
            assert!(
                (gpu[i] - cpu[i]).abs() < 1e-3 * cpu[i].abs().max(1.0),
                "y[{i}]: gpu={} cpu={}",
                gpu[i],
                cpu[i]
            );
        }
    }

    #[test]
    fn qmm_affine_t_handles_unaligned_m_n() {
        // M, N not divisible by 16 (the threadgroup size).
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        let m = 7usize;
        let n = 13usize;
        let k = 64usize;
        let gs = 32usize;
        let groups_per_row = k / gs;

        let x: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.011 - 0.5).collect();
        let q_int: Vec<u8> = (0..(n * k)).map(|i| ((i * 7) % 16) as u8).collect();
        let scales: Vec<f32> = (0..(n * groups_per_row))
            .map(|_| 0.07)
            .collect();
        let biases: Vec<f32> = (0..(n * groups_per_row))
            .map(|_| -0.1)
            .collect();

        let mut x_buf = alloc_f32(&device, m * k, vec![m, k]);
        x_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&x);
        let mut q_buf = alloc_u8(&device, n * k, vec![n, k]);
        q_buf.as_mut_slice::<u8>().unwrap().copy_from_slice(&q_int);
        let mut s_buf = alloc_f32(&device, n * groups_per_row, vec![n, groups_per_row]);
        s_buf
            .as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(&scales);
        let mut b_buf = alloc_f32(&device, n * groups_per_row, vec![n, groups_per_row]);
        b_buf
            .as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(&biases);
        let y_buf = alloc_f32(&device, m * n, vec![m, n]);
        let meta = make_meta(&device, m as u32, n as u32, k as u32, gs as u32);

        let mut encoder = device.command_encoder().unwrap();
        dispatch_qmm_affine_t_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &x_buf,
            &q_buf,
            &s_buf,
            &b_buf,
            &y_buf,
            &meta,
            m as u32,
            n as u32,
            k as u32,
            gs as u32,
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();

        let gpu = y_buf.as_slice::<f32>().unwrap();
        let cpu = qmm_affine_t_cpu(&x, &q_int, &scales, &biases, m, n, k, gs);
        for i in 0..(m * n) {
            assert!(
                (gpu[i] - cpu[i]).abs() < 1e-3 * cpu[i].abs().max(1.0),
                "unaligned y[{i}]: gpu={} cpu={}",
                gpu[i],
                cpu[i]
            );
        }
    }

    /// Cross-validate against composing iter-13b's `qdq_affine_forward`
    /// + a host-side standard matmul: the two paths must agree
    /// byte-for-byte (or within FP rounding noise) since the fused
    /// kernel is mathematically the same operation.
    #[test]
    fn qmm_affine_t_equals_qdq_then_matmul_composition() {
        use crate::ops::qdq_affine::dispatch_qdq_affine_forward_f32;

        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        let m = 5usize;
        let n = 9usize;
        let k = 96usize;
        let gs = 32usize;
        let groups_per_row = k / gs;

        let x: Vec<f32> = (0..(m * k)).map(|i| ((i as f32) * 0.017).cos() * 0.5).collect();
        let q_int: Vec<u8> = (0..(n * k)).map(|i| ((i * 13 + 5) % 16) as u8).collect();
        let scales: Vec<f32> = (0..(n * groups_per_row))
            .map(|i| 0.04 + (i as f32) * 0.005)
            .collect();
        let biases: Vec<f32> = (0..(n * groups_per_row))
            .map(|i| -0.05 + (i as f32) * 0.013)
            .collect();

        // Path A: fused kernel.
        let mut x_buf_a = alloc_f32(&device, m * k, vec![m, k]);
        x_buf_a.as_mut_slice::<f32>().unwrap().copy_from_slice(&x);
        let mut q_buf_a = alloc_u8(&device, n * k, vec![n, k]);
        q_buf_a.as_mut_slice::<u8>().unwrap().copy_from_slice(&q_int);
        let mut s_buf_a = alloc_f32(&device, n * groups_per_row, vec![n, groups_per_row]);
        s_buf_a.as_mut_slice::<f32>().unwrap().copy_from_slice(&scales);
        let mut b_buf_a = alloc_f32(&device, n * groups_per_row, vec![n, groups_per_row]);
        b_buf_a.as_mut_slice::<f32>().unwrap().copy_from_slice(&biases);
        let y_a = alloc_f32(&device, m * n, vec![m, n]);
        let meta = make_meta(&device, m as u32, n as u32, k as u32, gs as u32);

        let mut encoder = device.command_encoder().unwrap();
        dispatch_qmm_affine_t_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &x_buf_a,
            &q_buf_a,
            &s_buf_a,
            &b_buf_a,
            &y_a,
            &meta,
            m as u32,
            n as u32,
            k as u32,
            gs as u32,
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();

        // Path B: dequant via iter-13b's qdq_affine_forward, then host-side
        // matmul oracle (since dense_mm_f32 has its own size constraints
        // that we'd have to satisfy separately).  This compares the
        // FUSED kernel against the SAME math executed via the
        // explicit-dequant path.
        let n_total = n * k;
        let mut q_buf_b = alloc_u8(&device, n_total, vec![n_total]);
        q_buf_b
            .as_mut_slice::<u8>()
            .unwrap()
            .copy_from_slice(&q_int);
        let mut s_buf_b = alloc_f32(&device, n * groups_per_row, vec![n * groups_per_row]);
        s_buf_b.as_mut_slice::<f32>().unwrap().copy_from_slice(&scales);
        let mut b_buf_b = alloc_f32(&device, n * groups_per_row, vec![n * groups_per_row]);
        b_buf_b.as_mut_slice::<f32>().unwrap().copy_from_slice(&biases);
        let w_dq = alloc_f32(&device, n_total, vec![n_total]);
        let mut fwd_meta = device.alloc_buffer(8, DType::U32, vec![2]).unwrap();
        fwd_meta
            .as_mut_slice::<u32>()
            .unwrap()
            .copy_from_slice(&[n_total as u32, gs as u32]);

        let mut encoder = device.command_encoder().unwrap();
        dispatch_qdq_affine_forward_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &q_buf_b,
            &s_buf_b,
            &b_buf_b,
            &w_dq,
            &fwd_meta,
            gs as u32,
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();

        // Host-side standard matmul on the dequantized w (treating
        // it as [n, k] per the layout convention) — y[m, n_idx] =
        // Σ_k x[m, k] * w_dq[n_idx, k].
        let w_dq_host = w_dq.as_slice::<f32>().unwrap();
        let mut y_b = vec![0.0f32; m * n];
        for r in 0..m {
            for col in 0..n {
                let mut acc = 0.0f64;
                for kk in 0..k {
                    acc += (x[r * k + kk] as f64) * (w_dq_host[col * k + kk] as f64);
                }
                y_b[r * n + col] = acc as f32;
            }
        }

        let y_a_host = y_a.as_slice::<f32>().unwrap();
        for i in 0..(m * n) {
            assert!(
                (y_a_host[i] - y_b[i]).abs() < 1e-3 * y_b[i].abs().max(1.0),
                "fused vs composed at i={i}: fused={} composed={}",
                y_a_host[i],
                y_b[i]
            );
        }
    }

    #[test]
    fn rejects_k_not_divisible_by_group_size() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        let x_buf = alloc_f32(&device, 32, vec![1, 32]);
        let q_buf = alloc_u8(&device, 32, vec![1, 32]);
        let s_buf = alloc_f32(&device, 1, vec![1]);
        let b_buf = alloc_f32(&device, 1, vec![1]);
        let y_buf = alloc_f32(&device, 1, vec![1, 1]);
        let meta = make_meta(&device, 1, 1, 32, 5);
        let mut encoder = device.command_encoder().unwrap();
        let res = dispatch_qmm_affine_t_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &x_buf,
            &q_buf,
            &s_buf,
            &b_buf,
            &y_buf,
            &meta,
            1,
            1,
            32,
            5, // not power of 2
        );
        assert!(res.is_err());
    }

    #[test]
    fn rejects_dtype_mismatch() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        // Pass an f32 buffer where q_int (u8) is expected.
        let x_buf = alloc_f32(&device, 32, vec![1, 32]);
        let wrong_q = alloc_f32(&device, 32, vec![1, 32]);
        let s_buf = alloc_f32(&device, 1, vec![1]);
        let b_buf = alloc_f32(&device, 1, vec![1]);
        let y_buf = alloc_f32(&device, 1, vec![1, 1]);
        let meta = make_meta(&device, 1, 1, 32, 32);
        let mut encoder = device.command_encoder().unwrap();
        let res = dispatch_qmm_affine_t_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &x_buf,
            &wrong_q,
            &s_buf,
            &b_buf,
            &y_buf,
            &meta,
            1,
            1,
            32,
            32,
        );
        assert!(res.is_err());
    }
}
