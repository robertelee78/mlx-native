//! Source-ordered weighted RMS: F32-carried BF16 operands must reproduce the
//! BF16-storage reference producer, including output rounding and view bounds.
#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used, clippy::unwrap_used)]

use half::bf16;
use metal::MTLSize;
use mlx_native::ops::dflash2_rms_norm::dispatch_dflash2_rms_norm_bf16;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

// Independent BF16-storage oracle for the scalar-column/eight-partial law.
// Association follows the reference implementation pinned in the dispatcher;
// it intentionally does not use the generic float4 norm as its own oracle.
const REFERENCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void test_dflash2_rms_bf16_storage(
    device const bfloat *x [[buffer(0)]], device const bfloat *w [[buffer(1)]],
    device bfloat *y [[buffer(2)]], device const float *p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
#pragma clang fp reassociate(off)
    const uint width = uint(p[1]);
    const ulong base = ulong(row) * width;
    float square_sum = 0.0f;
    for (uint column = tid; column < width; column += 256) {
        const float value = float(x[base + column]);
        square_sum += value * value;
    }
    threadgroup float groups[8];
    const float group_sum = simd_sum(square_sum);
    if (lane == 0) groups[sg] = group_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint group = 0; group < 8; ++group) total += groups[group];
        groups[0] = rsqrt(total / float(width) + p[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint column = tid; column < width; column += 256)
        y[base + column] = bfloat((float(x[base + column]) * groups[0]) * float(w[column]));
}
"#;

fn buffer(device: &MlxDevice, elements: usize, dtype: DType) -> MlxBuffer {
    device
        .alloc_buffer(elements * dtype.size_of(), dtype, vec![elements])
        .unwrap()
}

fn value(row: usize, column: usize, salt: u32) -> bf16 {
    let mut bits = (row as u32).wrapping_mul(0x9e37_79b1)
        ^ (column as u32).wrapping_mul(0x85eb_ca77)
        ^ salt;
    bits ^= bits >> 16;
    bits = bits.wrapping_mul(0x7feb_352d);
    bits ^= bits >> 15;
    let unit = (bits & 0xffff) as f32 / 32768.0 - 1.0;
    bf16::from_f32(unit * [0.01, 0.25, 3.0, 37.0][(bits >> 16) as usize % 4])
}

#[test]
fn source_ordered_bf16_norm_matches_reference_and_preserves_view_guards() {
    let Ok(device) = MlxDevice::new() else {
        eprintln!("SKIP: Metal device unavailable");
        return;
    };
    let mut registry = KernelRegistry::new();
    registry.register_source("test_dflash2_rms_bf16_storage", REFERENCE);
    const SENTINEL: f32 = -928.0;
    for (rows, dim) in [(1, 31), (8, 128), (8, 2_048), (8, 5_120), (64, 5_120)] {
        let elements = rows * dim;
        let mut input_root = buffer(&device, (rows + 2) * dim, DType::F32);
        let mut output_root = buffer(&device, (rows + 2) * dim, DType::F32);
        let mut input_bf16 = buffer(&device, elements, DType::BF16);
        let mut weight_f32 = buffer(&device, dim, DType::F32);
        let mut weight_bf16 = buffer(&device, dim, DType::BF16);
        let expected = buffer(&device, elements, DType::BF16);
        let mut params = buffer(&device, 2, DType::F32);
        input_root.as_mut_slice::<f32>().unwrap().fill(SENTINEL);
        output_root.as_mut_slice::<f32>().unwrap().fill(SENTINEL);
        {
            let input = input_root.as_mut_slice::<f32>().unwrap();
            let bits = input_bf16.as_mut_slice::<u16>().unwrap();
            for row in 0..rows {
                for column in 0..dim {
                    let v = value(row, column, 0x6384_92bd);
                    input[dim + row * dim + column] = v.to_f32();
                    bits[row * dim + column] = v.to_bits();
                }
            }
        }
        {
            let weight = weight_f32.as_mut_slice::<f32>().unwrap();
            let bits = weight_bf16.as_mut_slice::<u16>().unwrap();
            for column in 0..dim {
                let w = value(0, column, 0x1e37_02af);
                weight[column] = w.to_f32();
                bits[column] = w.to_bits();
            }
        }
        params.as_mut_slice::<f32>().unwrap().copy_from_slice(&[1e-6, dim as f32]);
        let input = input_root.slice_view((dim * 4) as u64, elements);
        // Include the trailing guard in the view: requested rows, not view
        // capacity, bound the write. The leading guard is outside the view.
        let output = output_root.slice_view((dim * 4) as u64, elements + dim);
        let mut encoder = device.command_encoder().unwrap();
        dispatch_dflash2_rms_norm_bf16(
            &mut encoder, &mut registry, device.metal_device(), &input,
            &weight_f32, &output, &params, rows as u32, dim as u32,
        )
        .unwrap();
        let reference = registry
            .get_pipeline("test_dflash2_rms_bf16_storage", device.metal_device())
            .unwrap();
        encoder.encode_threadgroups(
            reference,
            &[(0, &input_bf16), (1, &weight_bf16), (2, &expected), (3, &params)],
            MTLSize::new(rows as u64, 1, 1), MTLSize::new(256, 1, 1),
        );
        encoder.commit_and_wait().unwrap();
        let actual = output_root.as_slice::<f32>().unwrap();
        let want = expected.as_slice::<u16>().unwrap();
        for (index, (got, bits)) in actual[dim..dim + elements].iter().zip(want).enumerate() {
            assert_eq!(got.to_bits(), bf16::from_bits(*bits).to_f32().to_bits(),
                "rows={rows} dim={dim} element={index}: weighted BF16 producer differs");
        }
        assert!(actual[..dim].iter().all(|x| x.to_bits() == SENTINEL.to_bits()));
        assert!(actual[dim + elements..].iter().all(|x| x.to_bits() == SENTINEL.to_bits()));
        println!("source_ordered_rms rows={rows} dim={dim}: {elements} BF16 values and both guards exact");

        // A larger backing allocation must not hide a short logical window.
        let short = output_root.slice_view((dim * 4) as u64, elements - 1);
        let mut rejected = device.command_encoder().unwrap();
        assert!(dispatch_dflash2_rms_norm_bf16(
            &mut rejected, &mut registry, device.metal_device(), &input,
            &weight_f32, &short, &params, rows as u32, dim as u32,
        ).is_err(), "short output view must reject before encoding");
        assert!(dispatch_dflash2_rms_norm_bf16(
            &mut rejected, &mut registry, device.metal_device(), &input,
            &weight_bf16, &output, &params, rows as u32, dim as u32,
        ).is_err(), "BF16 storage must not be read at F32 stride");
        let misaligned = output_root.slice_view(1, elements);
        for (input, weight, output, params) in [
            (&misaligned, &weight_f32, &output, &params),
            (&input, &misaligned, &output, &params),
            (&input, &weight_f32, &misaligned, &params),
            (&input, &weight_f32, &output, &misaligned),
        ] {
            assert!(dispatch_dflash2_rms_norm_bf16(
                &mut rejected, &mut registry, device.metal_device(), input,
                weight, output, params, rows as u32, dim as u32,
            ).is_err(), "misaligned F32 storage must reject before encoding");
        }
    }
}
