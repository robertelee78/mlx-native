//! The stock RMS norm dispatches bind their threadgroup memory in multiples
//! of 16 bytes, as Metal's API contract requires (the debug layer aborts
//! otherwise), and binding more than the 4 or 8 bytes they bound before for
//! narrow F32 rows changes no output bit.
//!
//! For every row width from 4 to 260 columns:
//!
//! - `dispatch_rms_norm` and the baked `build_rms_norm_decode_record` give
//!   the same bits, and
//! - `dispatch_fused_post_ff_norm2_endlayer_f32` (scalar and per-channel
//!   layer scale) gives its bits,
//!
//! each compared, outside the Metal debug layer, with the same pipeline and
//! geometry encoded with the previous bound length. Under the debug layer
//! that reference cannot be encoded, so the test prints a digest of every
//! output; it must equal the digest of a plain run.

#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::too_many_arguments)]

use metal::MTLSize;
use mlx_native::ops::encode_helpers::{as_bytes, KernelArg};
use mlx_native::ops::rms_norm::{
    build_rms_norm_decode_record, dispatch_fused_post_ff_norm2_endlayer_f32, dispatch_rms_norm,
};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

const EPS: f32 = 1e-6;
const WIDTHS: std::ops::RangeInclusive<u32> = 4..=260;
const ROWS: [u32; 2] = [1, 5];

fn value(row: usize, col: usize, salt: u32) -> f32 {
    let mut h = (row as u32).wrapping_mul(0x9E37_79B1)
        ^ (col as u32).wrapping_mul(0x85EB_CA77)
        ^ salt.wrapping_mul(0xC2B2_AE35);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    let unit = (h & 0xFFFF) as f32 / 32768.0 - 1.0;
    unit * [0.02, 1.0, 41.0, 700.0][(h >> 16) as usize % 4]
}

fn rows_of(rows: u32, dim: u32, salt: u32) -> Vec<f32> {
    (0..rows as usize)
        .flat_map(|row| (0..dim as usize).map(move |col| value(row, col, salt)))
        .collect()
}

fn f32_buffer(device: &MlxDevice, values: &[f32]) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer((values.len() * 4).max(16), DType::F32, vec![values.len()])
        .unwrap();
    buffer.as_mut_slice::<f32>().unwrap()[..values.len()].copy_from_slice(values);
    buffer
}

fn bits(buffer: &MlxBuffer, elements: usize) -> Vec<u32> {
    buffer.as_slice::<u32>().unwrap()[..elements].to_vec()
}

fn debug_layer() -> bool {
    std::env::var("MTL_DEBUG_LAYER").is_ok_and(|value| value != "0")
}

fn fold(digest: &mut u64, words: &[u32]) {
    for &word in words {
        *digest = (*digest ^ u64::from(word)).wrapping_mul(0x0000_0100_0000_01b3);
    }
}

/// The bound length before the fix: one float per simdgroup for the float4
/// kernels, one per thread for the scalar ones.
fn previous_length(float4: bool, tg_size: u64) -> u64 {
    if float4 {
        (tg_size / 32).max(1) * 4
    } else {
        tg_size * 4
    }
}

#[test]
fn rms_norm_binds_16_byte_units_without_changing_bits() {
    let Ok(device) = MlxDevice::new() else { return };
    let mut registry = KernelRegistry::new();
    let mut digest = 0xcbf2_9ce4_8422_2325_u64;
    let mut narrow = 0;
    for dim in WIDTHS {
        for rows in ROWS {
            let elements = (rows * dim) as usize;
            let input = f32_buffer(&device, &rows_of(rows, dim, 1));
            let weight = f32_buffer(&device, &rows_of(1, dim, 2));
            let params = f32_buffer(&device, &[EPS, dim as f32]);
            let record = build_rms_norm_decode_record(
                &mut registry,
                device.metal_device(),
                DType::F32,
                rows,
                dim,
            )
            .unwrap()
            .expect("an F32 norm record");
            let bound = record.threadgroup_mem[0].1;
            assert!(
                bound > 0 && bound % 16 == 0,
                "dim {dim}: the norm binds {bound} bytes of threadgroup memory"
            );
            let float4 = dim % 4 == 0;
            let previous = previous_length(float4, record.threads_per_tg.width);
            assert!(previous <= bound);
            narrow += usize::from(previous < bound);

            let direct = f32_buffer(&device, &vec![0.0; elements]);
            let baked = f32_buffer(&device, &vec![0.0; elements]);
            let mut encoder = device.command_encoder().unwrap();
            dispatch_rms_norm(
                &mut encoder,
                &mut registry,
                device.metal_device(),
                &input,
                &weight,
                &direct,
                &params,
                rows,
                dim,
            )
            .unwrap();
            encoder.dispatch_record(&record, &[&input, &weight, &baked, &params]);
            encoder.commit_and_wait().unwrap();
            let got = bits(&direct, elements);
            assert!(
                got.iter().any(|&word| word != got[0]) || elements == 1,
                "dim {dim}: constant output"
            );
            assert!(
                bits(&baked, elements) == got,
                "dim {dim}, {rows} rows: the baked record differs from dispatch_rms_norm"
            );
            if !debug_layer() {
                let mut old = record.clone();
                old.threadgroup_mem = vec![(0, previous)];
                let reference = f32_buffer(&device, &vec![0.0; elements]);
                let mut encoder = device.command_encoder().unwrap();
                encoder.dispatch_record(&old, &[&input, &weight, &reference, &params]);
                encoder.commit_and_wait().unwrap();
                assert!(
                    bits(&reference, elements) == got,
                    "dim {dim}, {rows} rows: output differs from the {previous}-byte binding"
                );
            }
            fold(&mut digest, &got);
        }
    }
    // Every float4 width up to 64 columns bound 4 or 8 bytes before.
    assert_eq!(narrow, 16 * ROWS.len(), "widths whose bound length changed");
    println!("rms_norm threadgroup memory digest: {digest:016x}");
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct EndlayerParams {
    eps: f32,
    dim: u32,
    scalar_is_vector: u32,
}

#[test]
fn fused_post_ff_norm2_endlayer_binds_16_byte_units_without_changing_bits() {
    let Ok(device) = MlxDevice::new() else { return };
    let mut registry = KernelRegistry::new();
    let mut digest = 0xcbf2_9ce4_8422_2325_u64;
    for dim in WIDTHS {
        for (rows, scalar_is_vector) in [(1_u32, false), (5, true)] {
            let elements = (rows * dim) as usize;
            let attn_out = f32_buffer(&device, &rows_of(rows, dim, 3));
            let moe_accum = f32_buffer(&device, &rows_of(rows, dim, 4));
            let residual = f32_buffer(&device, &rows_of(rows, dim, 5));
            let w2 = f32_buffer(&device, &rows_of(1, dim, 6));
            let w3 = f32_buffer(&device, &rows_of(1, dim, 7));
            let scalar = if scalar_is_vector {
                f32_buffer(&device, &rows_of(1, dim, 8))
            } else {
                f32_buffer(&device, &[0.75])
            };
            let run = |registry: &mut KernelRegistry, previous: bool| {
                let mlp_down = f32_buffer(&device, &vec![0.0; elements]);
                let hidden = f32_buffer(&device, &vec![0.0; elements]);
                let mut encoder = device.command_encoder().unwrap();
                if previous {
                    // The dispatch as it was: same kernel choice and
                    // geometry, the unrounded bound length.
                    let float4 = dim % 4 == 0;
                    let kernel = if float4 {
                        "fused_post_ff_norm2_endlayer_f32_v2"
                    } else {
                        "fused_post_ff_norm2_endlayer_f32"
                    };
                    let pipeline = registry
                        .get_pipeline(kernel, device.metal_device())
                        .unwrap();
                    let tg_size = std::cmp::min(256, dim.next_power_of_two()) as u64;
                    let params = EndlayerParams {
                        eps: EPS,
                        dim,
                        scalar_is_vector: u32::from(scalar_is_vector),
                    };
                    encoder.encode_threadgroups_with_args_and_shared(
                        pipeline,
                        &[
                            (0, KernelArg::Buffer(&attn_out)),
                            (1, KernelArg::Buffer(&moe_accum)),
                            (2, KernelArg::Buffer(&residual)),
                            (3, KernelArg::Buffer(&w2)),
                            (4, KernelArg::Buffer(&w3)),
                            (5, KernelArg::Buffer(&scalar)),
                            (6, KernelArg::Buffer(&mlp_down)),
                            (7, KernelArg::Buffer(&hidden)),
                            (8, KernelArg::Bytes(as_bytes(&params))),
                        ],
                        &[(0, previous_length(float4, tg_size))],
                        MTLSize::new(u64::from(rows), 1, 1),
                        MTLSize::new(tg_size, 1, 1),
                    );
                } else {
                    dispatch_fused_post_ff_norm2_endlayer_f32(
                        &mut encoder,
                        registry,
                        device.metal_device(),
                        &attn_out,
                        &moe_accum,
                        &residual,
                        &w2,
                        &w3,
                        &scalar,
                        &mlp_down,
                        &hidden,
                        EPS,
                        rows,
                        dim,
                        scalar_is_vector,
                    )
                    .unwrap();
                }
                encoder.commit_and_wait().unwrap();
                let mut words = bits(&mlp_down, elements);
                words.extend(bits(&hidden, elements));
                words
            };
            let got = run(&mut registry, false);
            if !debug_layer() {
                assert!(
                    run(&mut registry, true) == got,
                    "dim {dim}, {rows} rows: output differs from the previous binding"
                );
            }
            fold(&mut digest, &got);
        }
    }
    println!("fused_post_ff_norm2_endlayer threadgroup memory digest: {digest:016x}");
}
