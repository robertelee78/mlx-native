//! 2-D row-major slice + concat-by-column primitives.
//!
//! Used by hf2q's ADR-020 Track 1 multi-head SDPA on GpuTape:
//! Q/K/V tensors are sliced into per-head views, each head runs the
//! single-head SDPA chain, and per-head context outputs are
//! concatenated back into the full attention output.
//!
//! Two kernels:
//! - `slice_2d_cols_f32(input[rows, in_cols], output[rows, out_cols], (in_cols, out_cols, start_col))`
//!   produces `output[r, c] = input[r, start_col + c]`.
//! - `copy_2d_cols_into_f32(src[rows, src_cols], dst[rows, dst_cols], (src_cols, dst_cols, start))`
//!   writes `dst[r, start + c] = src[r, c]` for `c < src_cols`.  Caller
//!   pre-zeros (or pre-populates) `dst`; this kernel writes the slab only.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

pub static SLICE_CONCAT_2D_SHADER_SOURCE: &str =
    include_str!("../shaders/slice_concat_2d.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("slice_2d_cols_f32", SLICE_CONCAT_2D_SHADER_SOURCE);
    registry.register_source("copy_2d_cols_into_f32", SLICE_CONCAT_2D_SHADER_SOURCE);
}

/// Slice `output[r, c] = input[r, start_col + c]` for `c < out_cols`.
///
/// `params_buf` must be at least 12 bytes (3 × u32: in_cols, out_cols, start_col).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_slice_2d_cols_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
    params_buf: &MlxBuffer,
    rows: u32,
    in_cols: u32,
    out_cols: u32,
    start_col: u32,
) -> Result<()> {
    if rows == 0 || in_cols == 0 || out_cols == 0 {
        return Err(MlxError::InvalidArgument(
            "slice_2d_cols: rows/in_cols/out_cols must all be > 0".into(),
        ));
    }
    if start_col + out_cols > in_cols {
        return Err(MlxError::InvalidArgument(format!(
            "slice_2d_cols: start_col({start_col}) + out_cols({out_cols}) > in_cols({in_cols})"
        )));
    }
    if input.element_count() != (rows as usize) * (in_cols as usize) {
        return Err(MlxError::InvalidArgument(format!(
            "slice_2d_cols: input element count {} != rows({rows}) * in_cols({in_cols})",
            input.element_count(),
        )));
    }
    if output.element_count() != (rows as usize) * (out_cols as usize) {
        return Err(MlxError::InvalidArgument(format!(
            "slice_2d_cols: output element count {} != rows({rows}) * out_cols({out_cols})",
            output.element_count(),
        )));
    }
    for (label, buf) in [("input", input), ("output", output)] {
        if buf.dtype() != DType::F32 {
            return Err(MlxError::InvalidArgument(format!(
                "slice_2d_cols: {label} dtype {} not f32",
                buf.dtype()
            )));
        }
    }
    if params_buf.byte_len() < 12 {
        return Err(MlxError::InvalidArgument(format!(
            "slice_2d_cols: params_buf too small (need 12 bytes for 3×u32, got {})",
            params_buf.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("slice_2d_cols_f32", device)?;
    encoder.encode(
        pipeline,
        &[(0, input), (1, output), (2, params_buf)],
        MTLSize::new(out_cols as u64, rows as u64, 1),
        MTLSize::new(
            std::cmp::min(out_cols as u64, 32),
            std::cmp::min(rows as u64, 8),
            1,
        ),
    );
    Ok(())
}

/// Write `src[rows, src_cols]` into `dst[rows, dst_cols]` at column
/// offset `start_col`.  Does NOT touch dst columns outside the slab —
/// caller pre-zeros (or pre-populates) `dst`.
///
/// `params_buf` must be at least 12 bytes (3 × u32: src_cols, dst_cols, start_col).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_copy_2d_cols_into_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    src: &MlxBuffer,
    dst: &MlxBuffer,
    params_buf: &MlxBuffer,
    rows: u32,
    src_cols: u32,
    dst_cols: u32,
    start_col: u32,
) -> Result<()> {
    if rows == 0 || src_cols == 0 || dst_cols == 0 {
        return Err(MlxError::InvalidArgument(
            "copy_2d_cols_into: rows/src_cols/dst_cols must all be > 0".into(),
        ));
    }
    if start_col + src_cols > dst_cols {
        return Err(MlxError::InvalidArgument(format!(
            "copy_2d_cols_into: start_col({start_col}) + src_cols({src_cols}) > dst_cols({dst_cols})"
        )));
    }
    if src.element_count() != (rows as usize) * (src_cols as usize) {
        return Err(MlxError::InvalidArgument(format!(
            "copy_2d_cols_into: src element count {} != rows({rows}) * src_cols({src_cols})",
            src.element_count(),
        )));
    }
    if dst.element_count() != (rows as usize) * (dst_cols as usize) {
        return Err(MlxError::InvalidArgument(format!(
            "copy_2d_cols_into: dst element count {} != rows({rows}) * dst_cols({dst_cols})",
            dst.element_count(),
        )));
    }
    for (label, buf) in [("src", src), ("dst", dst)] {
        if buf.dtype() != DType::F32 {
            return Err(MlxError::InvalidArgument(format!(
                "copy_2d_cols_into: {label} dtype {} not f32",
                buf.dtype()
            )));
        }
    }
    if params_buf.byte_len() < 12 {
        return Err(MlxError::InvalidArgument(format!(
            "copy_2d_cols_into: params_buf too small (need 12 bytes for 3×u32, got {})",
            params_buf.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("copy_2d_cols_into_f32", device)?;
    encoder.encode(
        pipeline,
        &[(0, src), (1, dst), (2, params_buf)],
        MTLSize::new(src_cols as u64, rows as u64, 1),
        MTLSize::new(
            std::cmp::min(src_cols as u64, 32),
            std::cmp::min(rows as u64, 8),
            1,
        ),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MlxDevice;

    fn build_device_buf(device: &MlxDevice, data: &[f32], shape: Vec<usize>) -> MlxBuffer {
        let n_bytes = data.len() * 4;
        let mut buf = device
            .alloc_buffer(n_bytes, DType::F32, shape)
            .expect("alloc");
        buf.as_mut_slice::<f32>().expect("as_mut").copy_from_slice(data);
        buf
    }

    fn write_params_u32(buf: &mut MlxBuffer, vals: &[u32]) {
        let slice: &mut [u32] = buf.as_mut_slice().expect("params as_mut");
        slice[..vals.len()].copy_from_slice(vals);
    }

    #[test]
    fn slice_2d_cols_byte_identical_to_cpu() {
        let device = MlxDevice::new().expect("device");
        let rows = 4u32;
        let in_cols = 12u32;
        let out_cols = 4u32;
        let start_col = 5u32;
        let input: Vec<f32> = (0..rows * in_cols).map(|i| (i as f32) * 0.5 - 1.0).collect();
        let in_buf = build_device_buf(&device, &input, vec![rows as usize, in_cols as usize]);
        let out_buf = build_device_buf(
            &device,
            &vec![0.0_f32; (rows * out_cols) as usize],
            vec![rows as usize, out_cols as usize],
        );
        let mut params = device.alloc_buffer(12, DType::F32, vec![3]).expect("params");
        write_params_u32(&mut params, &[in_cols, out_cols, start_col]);

        let mut registry = KernelRegistry::new();
        register(&mut registry);
        let mut encoder = device.command_encoder().expect("encoder");
        dispatch_slice_2d_cols_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &in_buf,
            &out_buf,
            &params,
            rows,
            in_cols,
            out_cols,
            start_col,
        )
        .expect("slice dispatch");
        encoder.commit_and_wait().expect("commit");

        let gpu = out_buf.as_slice::<f32>().unwrap();
        for r in 0..rows as usize {
            for c in 0..out_cols as usize {
                let expected = input[r * in_cols as usize + start_col as usize + c];
                assert_eq!(
                    gpu[r * out_cols as usize + c].to_bits(),
                    expected.to_bits(),
                    "mismatch at ({r},{c})"
                );
            }
        }
    }

    #[test]
    fn copy_2d_cols_into_byte_identical_to_cpu() {
        // Pre-fill dst with sentinel 999.0; copy src into a slab;
        // verify slab matches src and surrounding cells are untouched.
        let device = MlxDevice::new().expect("device");
        let rows = 3u32;
        let src_cols = 4u32;
        let dst_cols = 12u32;
        let start_col = 5u32;
        let src: Vec<f32> = (0..rows * src_cols).map(|i| (i as f32) * 0.7 + 1.5).collect();
        let dst_init: Vec<f32> = vec![999.0; (rows * dst_cols) as usize];
        let src_buf = build_device_buf(&device, &src, vec![rows as usize, src_cols as usize]);
        let dst_buf = build_device_buf(
            &device,
            &dst_init,
            vec![rows as usize, dst_cols as usize],
        );
        let mut params = device.alloc_buffer(12, DType::F32, vec![3]).expect("params");
        write_params_u32(&mut params, &[src_cols, dst_cols, start_col]);

        let mut registry = KernelRegistry::new();
        register(&mut registry);
        let mut encoder = device.command_encoder().expect("encoder");
        dispatch_copy_2d_cols_into_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &src_buf,
            &dst_buf,
            &params,
            rows,
            src_cols,
            dst_cols,
            start_col,
        )
        .expect("copy dispatch");
        encoder.commit_and_wait().expect("commit");

        let gpu = dst_buf.as_slice::<f32>().unwrap();
        for r in 0..rows as usize {
            for c in 0..dst_cols as usize {
                let expected = if c >= start_col as usize
                    && c < (start_col + src_cols) as usize
                {
                    src[r * src_cols as usize + (c - start_col as usize)]
                } else {
                    999.0
                };
                assert_eq!(
                    gpu[r * dst_cols as usize + c].to_bits(),
                    expected.to_bits(),
                    "mismatch at ({r},{c})"
                );
            }
        }
    }

    #[test]
    fn slice_then_copy_back_round_trips() {
        // Sanity: slice every column from a tensor, copy each slice back
        // into a fresh dst at the same column offset; result must equal
        // the original tensor.
        let device = MlxDevice::new().expect("device");
        let rows = 5u32;
        let cols = 16u32;
        let chunk = 4u32;
        let n_chunks = cols / chunk;
        let input: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.13 - 2.5).collect();
        let in_buf = build_device_buf(&device, &input, vec![rows as usize, cols as usize]);

        // Build accumulator dst initialized to 0.
        let dst_buf = build_device_buf(
            &device,
            &vec![0.0_f32; (rows * cols) as usize],
            vec![rows as usize, cols as usize],
        );

        let mut registry = KernelRegistry::new();
        register(&mut registry);
        let mut encoder = device.command_encoder().expect("encoder");
        for h in 0..n_chunks {
            let start = h * chunk;
            // slice → temp
            let temp_buf = device
                .alloc_buffer(
                    (rows * chunk * 4) as usize,
                    DType::F32,
                    vec![rows as usize, chunk as usize],
                )
                .expect("temp");
            let mut p_slice = device.alloc_buffer(12, DType::F32, vec![3]).expect("p_slice");
            write_params_u32(&mut p_slice, &[cols, chunk, start]);
            dispatch_slice_2d_cols_f32(
                &mut encoder,
                &mut registry,
                device.metal_device(),
                &in_buf,
                &temp_buf,
                &p_slice,
                rows,
                cols,
                chunk,
                start,
            )
            .unwrap();
            encoder.memory_barrier();
            // copy temp → dst at start
            let mut p_copy = device.alloc_buffer(12, DType::F32, vec![3]).expect("p_copy");
            write_params_u32(&mut p_copy, &[chunk, cols, start]);
            dispatch_copy_2d_cols_into_f32(
                &mut encoder,
                &mut registry,
                device.metal_device(),
                &temp_buf,
                &dst_buf,
                &p_copy,
                rows,
                chunk,
                cols,
                start,
            )
            .unwrap();
            encoder.memory_barrier();
        }
        encoder.commit_and_wait().expect("commit");

        let gpu = dst_buf.as_slice::<f32>().unwrap();
        for (i, (g, c)) in gpu.iter().zip(input.iter()).enumerate() {
            assert_eq!(g.to_bits(), c.to_bits(), "round-trip mismatch at {i}");
        }
    }

    #[test]
    fn slice_rejects_out_of_range() {
        let device = MlxDevice::new().expect("device");
        let in_buf = device
            .alloc_buffer(4 * 12 * 4, DType::F32, vec![4, 12])
            .expect("in");
        let out_buf = device
            .alloc_buffer(4 * 4 * 4, DType::F32, vec![4, 4])
            .expect("out");
        let params = device.alloc_buffer(12, DType::F32, vec![3]).expect("params");
        let mut registry = KernelRegistry::new();
        register(&mut registry);
        let mut encoder = device.command_encoder().expect("encoder");
        let err = dispatch_slice_2d_cols_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &in_buf,
            &out_buf,
            &params,
            4,
            12,
            4,
            10, // 10 + 4 = 14 > 12
        )
        .expect_err("must reject");
        assert!(format!("{err}").contains("> in_cols"));
    }
}
