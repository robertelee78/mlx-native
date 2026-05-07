//! Differentiable affine quantize-dequantize primitives — ADR-020 iter-13b
//! Track 2 DWQ-proper training loop.
//!
//! Per mlx-lm `dwq.py` + `mx.QuantizedLinear` `unfreeze(keys=
//! ["scales","biases"])` semantics: the integer codes `q_int` are
//! pre-quantized ONCE and then FROZEN; only the per-group `scales`
//! and `biases` flow gradients during distillation.  Thus the
//! mathematically correct, mlx-equivalent forward + backward is:
//!
//! ```text
//! forward:    qdq[i] = q_int[i] · scales[g(i)] + biases[g(i)]
//! backward:   d/d(scales[g]) = Σ_{i ∈ g} q_int[i] · dy[i]
//!             d/d(biases[g]) = Σ_{i ∈ g} dy[i]
//! ```
//!
//! Four kernels ship together:
//!
//!   1. [`dispatch_qdq_affine_init_f32`]      — one-shot init from a
//!      frozen FP32 weight: per-group min/max → s = (max−min)/(n_bins−1),
//!      b = min, q_int = clip(round((w − b) / s), 0, n_bins−1).
//!   2. [`dispatch_qdq_affine_forward_f32`]   — qdq[i] = q_int[i]·s_g + b_g.
//!   3. [`dispatch_qdq_affine_backward_scales_f32`]  — per-group reduction
//!      d_scales[g] = Σ q_int[i]·dy[i].
//!   4. [`dispatch_qdq_affine_backward_biases_f32`]  — per-group reduction
//!      d_biases[g] = Σ dy[i].
//!
//! All four use FP32 for scales/biases/dy/qdq and `uchar` for `q_int`
//! (one byte per element; supports up to 8-bit quantization without
//! packing).
//!
//! `group_size` must be a power of two between 2 and 1024 (kernel
//! tree-reduction constraint) and must divide `n_total` evenly.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

pub static QDQ_AFFINE_SHADER_SOURCE: &str =
    include_str!("../shaders/qdq_affine.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("qdq_affine_init_f32", QDQ_AFFINE_SHADER_SOURCE);
    registry.register_source("qdq_affine_forward_f32", QDQ_AFFINE_SHADER_SOURCE);
    registry.register_source(
        "qdq_affine_backward_scales_f32",
        QDQ_AFFINE_SHADER_SOURCE,
    );
    registry.register_source(
        "qdq_affine_backward_biases_f32",
        QDQ_AFFINE_SHADER_SOURCE,
    );
}

fn validate_common(
    op: &str,
    n_total: usize,
    group_size: u32,
    n_groups: usize,
) -> Result<()> {
    if n_total == 0 {
        return Err(MlxError::InvalidArgument(format!(
            "{op}: n_total must be > 0"
        )));
    }
    if group_size < 2 || group_size > 1024 || !group_size.is_power_of_two() {
        return Err(MlxError::InvalidArgument(format!(
            "{op}: group_size must be a power of two in [2, 1024]; got {group_size}"
        )));
    }
    if n_total % (group_size as usize) != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "{op}: n_total ({n_total}) must be divisible by group_size ({group_size})"
        )));
    }
    if n_groups != n_total / (group_size as usize) {
        return Err(MlxError::InvalidArgument(format!(
            "{op}: n_groups ({n_groups}) must equal n_total/group_size ({})",
            n_total / (group_size as usize)
        )));
    }
    Ok(())
}

fn check_dtype(buf: &MlxBuffer, want: DType, label: &str, op: &str) -> Result<()> {
    if buf.dtype() != want {
        return Err(MlxError::InvalidArgument(format!(
            "{op}: {label} dtype {} != expected {}",
            buf.dtype(),
            want
        )));
    }
    Ok(())
}

fn check_count(buf: &MlxBuffer, want: usize, label: &str, op: &str) -> Result<()> {
    if buf.element_count() != want {
        return Err(MlxError::InvalidArgument(format!(
            "{op}: {label} element_count {} != expected {want}",
            buf.element_count()
        )));
    }
    Ok(())
}

/// One-shot per-group affine init.  Reads `w` (frozen FP32 weight),
/// writes `scales[n_groups]`, `biases[n_groups]`, and
/// `q_int[n_total]`.  `meta` must contain `[group_size, n_bins]` as
/// u32 (8 bytes).
///
/// `n_bins = 2^bits` and must satisfy `n_bins <= 256` (q_int is u8).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_qdq_affine_init_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    w: &MlxBuffer,
    scales: &MlxBuffer,
    biases: &MlxBuffer,
    q_int: &MlxBuffer,
    meta: &MlxBuffer,
    group_size: u32,
    n_bins: u32,
) -> Result<()> {
    const OP: &str = "qdq_affine_init_f32";
    let n_total = w.element_count();
    let n_groups = scales.element_count();
    validate_common(OP, n_total, group_size, n_groups)?;
    if !(2..=256).contains(&n_bins) {
        return Err(MlxError::InvalidArgument(format!(
            "{OP}: n_bins must be in [2, 256]; got {n_bins}"
        )));
    }
    check_dtype(w, DType::F32, "w", OP)?;
    check_dtype(scales, DType::F32, "scales", OP)?;
    check_dtype(biases, DType::F32, "biases", OP)?;
    check_dtype(q_int, DType::U8, "q_int", OP)?;
    check_count(biases, n_groups, "biases", OP)?;
    check_count(q_int, n_total, "q_int", OP)?;
    if meta.byte_len() < 8 {
        return Err(MlxError::InvalidArgument(format!(
            "{OP}: meta must be ≥8 bytes ([group_size, n_bins] u32); got {}",
            meta.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline(OP, device)?;
    let shared_mem_bytes: u64 = 2 * u64::from(group_size) * 4;
    encoder.encode_threadgroups_with_shared(
        pipeline,
        &[(0, w), (1, scales), (2, biases), (3, q_int), (4, meta)],
        &[(0, shared_mem_bytes)],
        MTLSize::new(n_groups as u64, 1, 1),
        MTLSize::new(u64::from(group_size), 1, 1),
    );
    Ok(())
}

/// Forward dispatch: `qdq[i] = q_int[i] · scales[g(i)] + biases[g(i)]`.
///
/// `meta` must contain `[n_total, group_size]` as u32 (8 bytes).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_qdq_affine_forward_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    q_int: &MlxBuffer,
    scales: &MlxBuffer,
    biases: &MlxBuffer,
    qdq: &MlxBuffer,
    meta: &MlxBuffer,
    group_size: u32,
) -> Result<()> {
    const OP: &str = "qdq_affine_forward_f32";
    let n_total = q_int.element_count();
    let n_groups = scales.element_count();
    validate_common(OP, n_total, group_size, n_groups)?;
    check_dtype(q_int, DType::U8, "q_int", OP)?;
    check_dtype(scales, DType::F32, "scales", OP)?;
    check_dtype(biases, DType::F32, "biases", OP)?;
    check_dtype(qdq, DType::F32, "qdq", OP)?;
    check_count(biases, n_groups, "biases", OP)?;
    check_count(qdq, n_total, "qdq", OP)?;
    if meta.byte_len() < 8 {
        return Err(MlxError::InvalidArgument(format!(
            "{OP}: meta must be ≥8 bytes ([n_total, group_size] u32); got {}",
            meta.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline(OP, device)?;
    let tg_size = std::cmp::min(256u64, n_total as u64);
    let n_tgs = (n_total as u64).div_ceil(tg_size);
    encoder.encode_threadgroups(
        pipeline,
        &[(0, q_int), (1, scales), (2, biases), (3, qdq), (4, meta)],
        MTLSize::new(n_tgs, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );
    Ok(())
}

/// Backward w.r.t. scales: `d_scales[g] = Σ q_int[i] · dy[i]` over `i ∈ g`.
///
/// `meta` must contain `[group_size]` as u32 (4 bytes).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_qdq_affine_backward_scales_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    q_int: &MlxBuffer,
    dy: &MlxBuffer,
    d_scales: &MlxBuffer,
    meta: &MlxBuffer,
    group_size: u32,
) -> Result<()> {
    const OP: &str = "qdq_affine_backward_scales_f32";
    let n_total = q_int.element_count();
    let n_groups = d_scales.element_count();
    validate_common(OP, n_total, group_size, n_groups)?;
    check_dtype(q_int, DType::U8, "q_int", OP)?;
    check_dtype(dy, DType::F32, "dy", OP)?;
    check_dtype(d_scales, DType::F32, "d_scales", OP)?;
    check_count(dy, n_total, "dy", OP)?;
    if meta.byte_len() < 4 {
        return Err(MlxError::InvalidArgument(format!(
            "{OP}: meta must be ≥4 bytes ([group_size] u32); got {}",
            meta.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline(OP, device)?;
    let shared_mem_bytes: u64 = u64::from(group_size) * 4;
    encoder.encode_threadgroups_with_shared(
        pipeline,
        &[(0, q_int), (1, dy), (2, d_scales), (3, meta)],
        &[(0, shared_mem_bytes)],
        MTLSize::new(n_groups as u64, 1, 1),
        MTLSize::new(u64::from(group_size), 1, 1),
    );
    Ok(())
}

/// Backward w.r.t. biases: `d_biases[g] = Σ dy[i]` over `i ∈ g`.
///
/// `meta` must contain `[group_size]` as u32 (4 bytes).
pub fn dispatch_qdq_affine_backward_biases_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    dy: &MlxBuffer,
    d_biases: &MlxBuffer,
    meta: &MlxBuffer,
    group_size: u32,
) -> Result<()> {
    const OP: &str = "qdq_affine_backward_biases_f32";
    let n_groups = d_biases.element_count();
    let n_total = dy.element_count();
    validate_common(OP, n_total, group_size, n_groups)?;
    check_dtype(dy, DType::F32, "dy", OP)?;
    check_dtype(d_biases, DType::F32, "d_biases", OP)?;
    if meta.byte_len() < 4 {
        return Err(MlxError::InvalidArgument(format!(
            "{OP}: meta must be ≥4 bytes ([group_size] u32); got {}",
            meta.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline(OP, device)?;
    let shared_mem_bytes: u64 = u64::from(group_size) * 4;
    encoder.encode_threadgroups_with_shared(
        pipeline,
        &[(0, dy), (1, d_biases), (2, meta)],
        &[(0, shared_mem_bytes)],
        MTLSize::new(n_groups as u64, 1, 1),
        MTLSize::new(u64::from(group_size), 1, 1),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MlxDevice;

    /// CPU oracle for the affine init: per-group min/max → s, b, and
    /// per-element q_int via `clip(round((w−b)/s), 0, n_bins−1)`.
    fn affine_init_cpu(
        w: &[f32],
        group_size: usize,
        n_bins: u32,
    ) -> (Vec<f32>, Vec<f32>, Vec<u8>) {
        assert!(w.len() % group_size == 0);
        let n_groups = w.len() / group_size;
        let mut scales = vec![0.0f32; n_groups];
        let mut biases = vec![0.0f32; n_groups];
        let mut q_int = vec![0u8; w.len()];
        for g in 0..n_groups {
            let block = &w[g * group_size..(g + 1) * group_size];
            let mut mn = block[0];
            let mut mx = block[0];
            for &v in block {
                mn = mn.min(v);
                mx = mx.max(v);
            }
            let mut s = (mx - mn) / (n_bins - 1) as f32;
            if !(s > 0.0) {
                s = 1.0;
            }
            let b = mn;
            scales[g] = s;
            biases[g] = b;
            for (i, &v) in block.iter().enumerate() {
                let z = (v - b) / s;
                let q = z.round() as i32;
                q_int[g * group_size + i] = q.clamp(0, (n_bins - 1) as i32) as u8;
            }
        }
        (scales, biases, q_int)
    }

    /// CPU oracle for the forward.
    fn affine_forward_cpu(
        q_int: &[u8],
        scales: &[f32],
        biases: &[f32],
        group_size: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; q_int.len()];
        for (i, &q) in q_int.iter().enumerate() {
            let g = i / group_size;
            out[i] = q as f32 * scales[g] + biases[g];
        }
        out
    }

    fn make_meta_2(device: &MlxDevice, a: u32, b: u32) -> MlxBuffer {
        let mut buf = device
            .alloc_buffer(8, DType::U32, vec![2])
            .expect("alloc meta");
        let dst = buf.as_mut_slice::<u32>().expect("as_mut_slice u32");
        dst[0] = a;
        dst[1] = b;
        buf
    }

    fn make_meta_1(device: &MlxDevice, a: u32) -> MlxBuffer {
        let mut buf = device
            .alloc_buffer(4, DType::U32, vec![1])
            .expect("alloc meta");
        let dst = buf.as_mut_slice::<u32>().expect("as_mut_slice u32");
        dst[0] = a;
        buf
    }

    fn alloc_f32(device: &MlxDevice, n: usize, shape: Vec<usize>) -> MlxBuffer {
        device
            .alloc_buffer(n * 4, DType::F32, shape)
            .expect("alloc f32")
    }

    fn alloc_u8(device: &MlxDevice, n: usize, shape: Vec<usize>) -> MlxBuffer {
        device.alloc_buffer(n, DType::U8, shape).expect("alloc u8")
    }

    #[test]
    fn init_matches_cpu_oracle_4bit_g32() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();

        let group_size = 32usize;
        let n_groups = 4usize;
        let n_total = group_size * n_groups;
        let n_bins: u32 = 16; // 4-bit

        // Mix of positive, negative, near-zero values across 4 groups.
        let mut w = vec![0.0f32; n_total];
        for i in 0..n_total {
            w[i] = ((i as f32) * 0.137 - 17.5).sin() * 2.5;
        }

        let mut w_buf = alloc_f32(&device, n_total, vec![n_total]);
        w_buf
            .as_mut_slice::<f32>()
            .expect("as_mut_slice")
            .copy_from_slice(&w);

        let scales_buf = alloc_f32(&device, n_groups, vec![n_groups]);
        let biases_buf = alloc_f32(&device, n_groups, vec![n_groups]);
        let q_int_buf = alloc_u8(&device, n_total, vec![n_total]);
        let meta_buf = make_meta_2(&device, group_size as u32, n_bins);

        let mut encoder = device.command_encoder().expect("encoder");
        dispatch_qdq_affine_init_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &w_buf,
            &scales_buf,
            &biases_buf,
            &q_int_buf,
            &meta_buf,
            group_size as u32,
            n_bins,
        )
        .expect("dispatch");
        encoder.commit_and_wait().expect("wait");

        let gpu_s = scales_buf.as_slice::<f32>().unwrap().to_vec();
        let gpu_b = biases_buf.as_slice::<f32>().unwrap().to_vec();
        let gpu_q = q_int_buf.as_slice::<u8>().unwrap().to_vec();

        let (cpu_s, cpu_b, cpu_q) = affine_init_cpu(&w, group_size, n_bins);

        for g in 0..n_groups {
            assert!(
                (gpu_s[g] - cpu_s[g]).abs() < 1e-6,
                "scales[{g}]: gpu={} cpu={}",
                gpu_s[g],
                cpu_s[g]
            );
            assert!(
                (gpu_b[g] - cpu_b[g]).abs() < 1e-6,
                "biases[{g}]: gpu={} cpu={}",
                gpu_b[g],
                cpu_b[g]
            );
        }
        for i in 0..n_total {
            assert_eq!(gpu_q[i], cpu_q[i], "q_int[{i}]");
        }
    }

    #[test]
    fn init_handles_uniform_group_degenerate() {
        // All-equal group: max == min ⇒ kernel sets s := 1.0, b := w_min,
        // q_int := 0 for every element ⇒ qdq round-trip is exact.
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        let group_size = 32usize;
        let n_groups = 1usize;
        let n_total = group_size;

        let w = vec![3.14f32; n_total];
        let mut w_buf = alloc_f32(&device, n_total, vec![n_total]);
        w_buf
            .as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(&w);
        let scales_buf = alloc_f32(&device, n_groups, vec![n_groups]);
        let biases_buf = alloc_f32(&device, n_groups, vec![n_groups]);
        let q_int_buf = alloc_u8(&device, n_total, vec![n_total]);
        let meta_buf = make_meta_2(&device, group_size as u32, 16);

        let mut encoder = device.command_encoder().unwrap();
        dispatch_qdq_affine_init_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &w_buf,
            &scales_buf,
            &biases_buf,
            &q_int_buf,
            &meta_buf,
            group_size as u32,
            16,
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();

        let s = scales_buf.as_slice::<f32>().unwrap()[0];
        let b = biases_buf.as_slice::<f32>().unwrap()[0];
        let q = q_int_buf.as_slice::<u8>().unwrap();
        assert_eq!(s, 1.0);
        assert_eq!(b, 3.14);
        for &qi in q {
            assert_eq!(qi, 0);
        }
    }

    #[test]
    fn forward_matches_cpu_oracle() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        let group_size = 64usize;
        let n_groups = 7usize;
        let n_total = group_size * n_groups;

        // Hand-built q_int + scales + biases (no init).
        let mut q = vec![0u8; n_total];
        for i in 0..n_total {
            q[i] = ((i * 31) % 16) as u8;
        }
        let scales: Vec<f32> = (0..n_groups).map(|g| 0.05 + (g as f32) * 0.01).collect();
        let biases: Vec<f32> = (0..n_groups).map(|g| -0.4 + (g as f32) * 0.07).collect();

        let mut q_buf = alloc_u8(&device, n_total, vec![n_total]);
        q_buf
            .as_mut_slice::<u8>()
            .unwrap()
            .copy_from_slice(&q);
        let mut scales_buf = alloc_f32(&device, n_groups, vec![n_groups]);
        scales_buf
            .as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(&scales);
        let mut biases_buf = alloc_f32(&device, n_groups, vec![n_groups]);
        biases_buf
            .as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(&biases);
        let qdq_buf = alloc_f32(&device, n_total, vec![n_total]);
        let meta_buf = make_meta_2(&device, n_total as u32, group_size as u32);

        let mut encoder = device.command_encoder().unwrap();
        dispatch_qdq_affine_forward_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &q_buf,
            &scales_buf,
            &biases_buf,
            &qdq_buf,
            &meta_buf,
            group_size as u32,
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();

        let gpu = qdq_buf.as_slice::<f32>().unwrap().to_vec();
        let cpu = affine_forward_cpu(&q, &scales, &biases, group_size);
        for i in 0..n_total {
            assert!(
                (gpu[i] - cpu[i]).abs() < 1e-6,
                "qdq[{i}]: gpu={} cpu={}",
                gpu[i],
                cpu[i]
            );
        }
    }

    #[test]
    fn init_then_forward_recovers_w_within_quant_error() {
        // End-to-end sanity: init → forward should produce qdq with
        // |qdq − w| ≤ s/2 (the half-step quant error bound).
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        let group_size = 32usize;
        let n_groups = 5usize;
        let n_total = group_size * n_groups;
        let n_bins: u32 = 16;

        let mut w = vec![0.0f32; n_total];
        for i in 0..n_total {
            w[i] = ((i as f32) * 0.51).sin() + ((i as f32) * 0.123).cos() * 0.3;
        }
        let mut w_buf = alloc_f32(&device, n_total, vec![n_total]);
        w_buf
            .as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(&w);
        let scales_buf = alloc_f32(&device, n_groups, vec![n_groups]);
        let biases_buf = alloc_f32(&device, n_groups, vec![n_groups]);
        let q_int_buf = alloc_u8(&device, n_total, vec![n_total]);
        let meta_init = make_meta_2(&device, group_size as u32, n_bins);

        let mut encoder = device.command_encoder().unwrap();
        dispatch_qdq_affine_init_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &w_buf,
            &scales_buf,
            &biases_buf,
            &q_int_buf,
            &meta_init,
            group_size as u32,
            n_bins,
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();

        let qdq_buf = alloc_f32(&device, n_total, vec![n_total]);
        let meta_fwd = make_meta_2(&device, n_total as u32, group_size as u32);
        let mut encoder = device.command_encoder().unwrap();
        dispatch_qdq_affine_forward_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &q_int_buf,
            &scales_buf,
            &biases_buf,
            &qdq_buf,
            &meta_fwd,
            group_size as u32,
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();

        let qdq = qdq_buf.as_slice::<f32>().unwrap();
        let s = scales_buf.as_slice::<f32>().unwrap();
        for g in 0..n_groups {
            for i in 0..group_size {
                let idx = g * group_size + i;
                let bound = s[g] * 0.5 + 1e-6;
                assert!(
                    (qdq[idx] - w[idx]).abs() <= bound,
                    "qdq[{idx}]={} w[{idx}]={} diff={} bound={}",
                    qdq[idx],
                    w[idx],
                    (qdq[idx] - w[idx]).abs(),
                    bound
                );
            }
        }
    }

    #[test]
    fn backward_scales_matches_cpu_reduction() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        let group_size = 64usize;
        let n_groups = 3usize;
        let n_total = group_size * n_groups;

        let q: Vec<u8> = (0..n_total).map(|i| ((i * 7) % 16) as u8).collect();
        let dy: Vec<f32> = (0..n_total)
            .map(|i| (i as f32 * 0.137).sin() * 0.2 - 0.05)
            .collect();

        let mut q_buf = alloc_u8(&device, n_total, vec![n_total]);
        q_buf.as_mut_slice::<u8>().unwrap().copy_from_slice(&q);
        let mut dy_buf = alloc_f32(&device, n_total, vec![n_total]);
        dy_buf
            .as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(&dy);
        let d_scales_buf = alloc_f32(&device, n_groups, vec![n_groups]);
        let meta_buf = make_meta_1(&device, group_size as u32);

        let mut encoder = device.command_encoder().unwrap();
        dispatch_qdq_affine_backward_scales_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &q_buf,
            &dy_buf,
            &d_scales_buf,
            &meta_buf,
            group_size as u32,
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();

        let gpu = d_scales_buf.as_slice::<f32>().unwrap();
        for g in 0..n_groups {
            let mut acc = 0.0f64; // higher-precision oracle to avoid float-order drift
            for i in 0..group_size {
                let idx = g * group_size + i;
                acc += q[idx] as f64 * dy[idx] as f64;
            }
            let cpu = acc as f32;
            assert!(
                (gpu[g] - cpu).abs() < 1e-4 * cpu.abs().max(1.0),
                "d_scales[{g}]: gpu={} cpu={}",
                gpu[g],
                cpu
            );
        }
    }

    #[test]
    fn backward_biases_matches_cpu_reduction() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        let group_size = 32usize;
        let n_groups = 8usize;
        let n_total = group_size * n_groups;

        let dy: Vec<f32> = (0..n_total)
            .map(|i| ((i as f32) * 0.21).cos() * 0.5 + 0.1)
            .collect();

        let mut dy_buf = alloc_f32(&device, n_total, vec![n_total]);
        dy_buf
            .as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(&dy);
        let d_biases_buf = alloc_f32(&device, n_groups, vec![n_groups]);
        let meta_buf = make_meta_1(&device, group_size as u32);

        let mut encoder = device.command_encoder().unwrap();
        dispatch_qdq_affine_backward_biases_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &dy_buf,
            &d_biases_buf,
            &meta_buf,
            group_size as u32,
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();

        let gpu = d_biases_buf.as_slice::<f32>().unwrap();
        for g in 0..n_groups {
            let mut acc = 0.0f64;
            for i in 0..group_size {
                let idx = g * group_size + i;
                acc += dy[idx] as f64;
            }
            let cpu = acc as f32;
            assert!(
                (gpu[g] - cpu).abs() < 1e-4 * cpu.abs().max(1.0),
                "d_biases[{g}]: gpu={} cpu={}",
                gpu[g],
                cpu
            );
        }
    }

    /// Finite-difference falsifier for the analytical backward.
    ///
    /// `qdq[i] = q_int[i] · scales[g(i)] + biases[g(i)]`
    ///
    /// Verifies that perturbing scales/biases by ±h and observing the
    /// loss `L = Σ_i (qdq[i] · dy[i])` (so dL/d(qdq[i]) = dy[i]) yields:
    ///
    ///   dL/d(scales[g]) ≈ Σ_{i ∈ g} q_int[i] · dy[i]   (matches kernel)
    ///   dL/d(biases[g]) ≈ Σ_{i ∈ g} dy[i]               (matches kernel)
    #[test]
    fn finite_diff_falsifier_scales_and_biases() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        let group_size = 32usize;
        let n_groups = 3usize;
        let n_total = group_size * n_groups;

        let q: Vec<u8> = (0..n_total).map(|i| ((i * 11 + 3) % 16) as u8).collect();
        let scales: Vec<f32> = (0..n_groups).map(|g| 0.07 + (g as f32) * 0.013).collect();
        let biases: Vec<f32> =
            (0..n_groups).map(|g| -0.13 + (g as f32) * 0.029).collect();
        let dy: Vec<f32> = (0..n_total)
            .map(|i| ((i as f32) * 0.317).sin() * 0.4 - 0.1)
            .collect();

        // Analytical gradients via the kernels.
        let mut q_buf = alloc_u8(&device, n_total, vec![n_total]);
        q_buf.as_mut_slice::<u8>().unwrap().copy_from_slice(&q);
        let mut dy_buf = alloc_f32(&device, n_total, vec![n_total]);
        dy_buf
            .as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(&dy);
        let d_scales_buf = alloc_f32(&device, n_groups, vec![n_groups]);
        let d_biases_buf = alloc_f32(&device, n_groups, vec![n_groups]);
        let meta_buf = make_meta_1(&device, group_size as u32);

        let mut encoder = device.command_encoder().unwrap();
        dispatch_qdq_affine_backward_scales_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &q_buf,
            &dy_buf,
            &d_scales_buf,
            &meta_buf,
            group_size as u32,
        )
        .unwrap();
        dispatch_qdq_affine_backward_biases_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &dy_buf,
            &d_biases_buf,
            &meta_buf,
            group_size as u32,
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();
        let analytic_ds = d_scales_buf.as_slice::<f32>().unwrap().to_vec();
        let analytic_db = d_biases_buf.as_slice::<f32>().unwrap().to_vec();

        // Finite difference oracle.
        let h = 1e-3f32;
        let loss = |s: &[f32], b: &[f32]| -> f64 {
            let mut acc = 0.0f64;
            for i in 0..n_total {
                let g = i / group_size;
                let qdq_i = q[i] as f32 * s[g] + b[g];
                acc += (qdq_i * dy[i]) as f64;
            }
            acc
        };
        for g in 0..n_groups {
            let mut s_plus = scales.clone();
            let mut s_minus = scales.clone();
            s_plus[g] += h;
            s_minus[g] -= h;
            let fd = ((loss(&s_plus, &biases) - loss(&s_minus, &biases))
                / (2.0 * h as f64)) as f32;
            // Tolerance: q_int ≤ 15, dy O(0.5), 32 elements ⇒ |grad| ≤ ~240
            let tol = 1e-2 * fd.abs().max(1.0);
            assert!(
                (analytic_ds[g] - fd).abs() < tol,
                "FD scales[{g}]: analytic={} fd={} tol={}",
                analytic_ds[g],
                fd,
                tol
            );

            let mut b_plus = biases.clone();
            let mut b_minus = biases.clone();
            b_plus[g] += h;
            b_minus[g] -= h;
            let fd_b = ((loss(&scales, &b_plus) - loss(&scales, &b_minus))
                / (2.0 * h as f64)) as f32;
            let tol_b = 1e-2 * fd_b.abs().max(1.0);
            assert!(
                (analytic_db[g] - fd_b).abs() < tol_b,
                "FD biases[{g}]: analytic={} fd={} tol={}",
                analytic_db[g],
                fd_b,
                tol_b
            );
        }
    }

    #[test]
    fn rejects_non_power_of_two_group_size() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        let n_total = 30usize;
        let group_size = 30u32;
        let q_buf = alloc_u8(&device, n_total, vec![n_total]);
        let scales_buf = alloc_f32(&device, 1, vec![1]);
        let biases_buf = alloc_f32(&device, 1, vec![1]);
        let qdq_buf = alloc_f32(&device, n_total, vec![n_total]);
        let meta_buf = make_meta_2(&device, n_total as u32, group_size);

        let mut encoder = device.command_encoder().unwrap();
        let res = dispatch_qdq_affine_forward_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &q_buf,
            &scales_buf,
            &biases_buf,
            &qdq_buf,
            &meta_buf,
            group_size,
        );
        assert!(res.is_err(), "non-power-of-two group_size must be rejected");
    }

    #[test]
    fn rejects_dtype_mismatch() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        let n_total = 32usize;
        let group_size = 32u32;
        // Pass an f32 buffer where q_int (u8) is expected.
        let wrong_q = alloc_f32(&device, n_total, vec![n_total]);
        let scales_buf = alloc_f32(&device, 1, vec![1]);
        let biases_buf = alloc_f32(&device, 1, vec![1]);
        let qdq_buf = alloc_f32(&device, n_total, vec![n_total]);
        let meta_buf = make_meta_2(&device, n_total as u32, group_size);

        let mut encoder = device.command_encoder().unwrap();
        let res = dispatch_qdq_affine_forward_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &wrong_q,
            &scales_buf,
            &biases_buf,
            &qdq_buf,
            &meta_buf,
            group_size,
        );
        assert!(res.is_err());
    }
}
