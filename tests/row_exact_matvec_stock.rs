//! Stock-order row-exact mat-vec (`ops::row_exact_matvec_stock`) against the
//! stock one-row kernels, on the DeepSeek-V4 decode shapes.
//!
//! For every shape, m = 1..=8 and every packing, each output row is bitwise
//! equal to the stock one-row kernel on that input row:
//! - F16: `dispatch_dense_matvec_f16w_f32io` (`dense_matvec_f16w_f32io`);
//! - Q8_0: `quantized_matmul_ggml` at m = 1 (`kernel_mul_mv_q8_0_f32_nr2`);
//! - grouped output-A: `quantized_matmul_ggml_batched_mv` at m = 1 per token
//!   (batch = 8 groups, same kernel).
//!
//! That implies batch ≡ serial (every row equals the same stock value at any
//! m). Also: repeated dispatches are bitwise equal, outputs are within the
//! F64 bound `gamma_K * sum |w x|` (`gamma_K = K u / (1 - K u)`,
//! `u = 2^-24`), and a sensitivity control shows the inputs expose a
//! different summation order (the Class-B `row_exact_matvec` kernels differ
//! from stock on them).
//!
//! Run under both shader paths (the env var is latched per process):
//! `cargo test --release --test row_exact_matvec_stock` and
//! `MLX_PRECOMPILED_METALLIB=0 cargo test --release --test row_exact_matvec_stock`.

#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used, clippy::panic, clippy::too_many_arguments)]

use mlx_native::ops::dense_gemm::{dispatch_dense_matvec_f16w_f32io, DenseGemmF16Params};
use mlx_native::ops::row_exact_matvec::{
    dispatch_row_exact_matvec, RowExactMatvecShape, RowExactWeight,
};
use mlx_native::ops::row_exact_matvec_stock::{
    dispatch_row_exact_matvec_stock, dispatch_row_exact_matvec_stock_packed,
    row_exact_matvec_stock_kernel_names, RowExactStockPacking,
};
use mlx_native::{
    quantized_matmul_ggml, quantized_matmul_ggml_batched_mv, DType,
    GgmlBatchedQuantizedMatmulParams, GgmlQuantizedMatmulParams, GgmlType, KernelPipelineOrigin,
    KernelRegistry, MlxBuffer, MlxDevice,
};

const MAX_M: usize = 8;
/// F64 reference MACs checked per shape (columns are subsampled above this).
const REFERENCE_MACS: usize = 32 << 20;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }

    /// Mixed magnitudes (2^-6..2^6), so a different summation order shows.
    fn mixed(&mut self) -> f32 {
        let exponent = (self.next() % 13) as i32 - 6;
        self.unit() * 2f32.powi(exponent)
    }
}

fn f16_weight(device: &MlxDevice, rows: usize, k: usize, seed: u64) -> MlxBuffer {
    let mut rng = Rng(seed | 1);
    let mut buffer = device
        .alloc_buffer(rows * k * 2, DType::F16, vec![rows, k])
        .expect("f16 weight");
    for v in buffer.as_mut_slice::<u16>().expect("f16 slice") {
        *v = half::f16::from_f32(rng.mixed() * 0.05).to_bits();
    }
    buffer
}

fn q8_0_weight(device: &MlxDevice, rows: usize, k: usize, seed: u64) -> MlxBuffer {
    let mut rng = Rng(seed | 1);
    let blocks = rows * k / 32;
    let mut buffer = device
        .alloc_buffer(blocks * 34, DType::U8, vec![rows, k])
        .expect("q8_0 weight");
    for block in buffer
        .as_mut_slice::<u8>()
        .expect("q8_0 slice")
        .chunks_exact_mut(34)
    {
        let exponent = (rng.next() % 9) as i32 - 12;
        let d = half::f16::from_f32((0.5 + 0.5 * rng.unit().abs()) * 2f32.powi(exponent));
        block[..2].copy_from_slice(&d.to_bits().to_le_bytes());
        for chunk in block[2..].chunks_mut(8) {
            let r = rng.next().to_le_bytes();
            chunk.copy_from_slice(&r[..chunk.len()]);
        }
    }
    buffer
}

fn weight(device: &MlxDevice, ty: RowExactWeight, rows: usize, k: usize, seed: u64) -> MlxBuffer {
    match ty {
        RowExactWeight::F16 => f16_weight(device, rows, k, seed),
        RowExactWeight::Q8_0 => q8_0_weight(device, rows, k, seed),
        other => unreachable!("the stock-order family has no {other:?} kernels"),
    }
}

fn input(device: &MlxDevice, len: usize, seed: u64) -> MlxBuffer {
    let mut rng = Rng(seed | 1);
    let mut buffer = device
        .alloc_buffer(len * 4, DType::F32, vec![len])
        .expect("input");
    for v in buffer.as_mut_slice::<f32>().expect("input slice") {
        *v = rng.mixed();
    }
    buffer
}

fn output(device: &MlxDevice, len: usize) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(len * 4, DType::F32, vec![len])
        .expect("output");
    buffer
        .as_mut_slice::<u32>()
        .expect("output slice")
        .fill(0x7fc0_dead);
    buffer
}

fn bits(buffer: &MlxBuffer) -> Vec<u32> {
    buffer.as_slice::<u32>().expect("bits").to_vec()
}

fn expected_origin() -> KernelPipelineOrigin {
    match std::env::var("MLX_PRECOMPILED_METALLIB").as_deref() {
        Ok("0") | Ok("false") | Ok("off") => KernelPipelineOrigin::RuntimeSource,
        _ => KernelPipelineOrigin::PrecompiledMetallib,
    }
}

/// Stock one-row outputs of input rows 0..8, `[8, groups * n]` token-major.
fn stock_rows(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    ty: RowExactWeight,
    w: &MlxBuffer,
    x: &MlxBuffer,
    groups: usize,
    n: usize,
    k: usize,
) -> Vec<u32> {
    let width = groups * k;
    let y = output(device, MAX_M * groups * n);
    let mut encoder = device.command_encoder().expect("encoder");
    for j in 0..MAX_M {
        let xj = x.slice_view((j * width * 4) as u64, width);
        let yj = y.slice_view((j * groups * n * 4) as u64, groups * n);
        match (ty, groups) {
            (RowExactWeight::F16, 1) => dispatch_dense_matvec_f16w_f32io(
                &mut encoder,
                registry,
                device.metal_device(),
                &xj,
                w,
                &yj,
                &DenseGemmF16Params {
                    m: 1,
                    n: n as u32,
                    k: k as u32,
                },
            ),
            (RowExactWeight::Q8_0, 1) => quantized_matmul_ggml(
                &mut encoder,
                registry,
                device,
                &xj,
                w,
                &yj,
                &GgmlQuantizedMatmulParams {
                    m: 1,
                    n: n as u32,
                    k: k as u32,
                    ggml_type: GgmlType::Q8_0,
                },
            ),
            (RowExactWeight::Q8_0, _) => quantized_matmul_ggml_batched_mv(
                &mut encoder,
                registry,
                device,
                &xj,
                w,
                &yj,
                &GgmlBatchedQuantizedMatmulParams {
                    batch: groups as u32,
                    m: 1,
                    n: n as u32,
                    k: k as u32,
                    ggml_type: GgmlType::Q8_0,
                },
            ),
            (RowExactWeight::F16, _) => panic!("no grouped F16 stock product"),
            (other, _) => unreachable!("the stock-order family has no {other:?} kernels"),
        }
        .expect("stock dispatch");
    }
    encoder.commit_and_wait().expect("stock commit");
    if ty == RowExactWeight::Q8_0 {
        let label = format!("kernel_mul_mv_q8_0_f32_nr2|700:i{groups}|701:i1|702:i1");
        assert!(
            registry.pipeline_identity(&label).is_ok(),
            "stock Q8_0 reference must be the nr2 matvec ({label})"
        );
    }
    bits(&y)
}

fn packings(ty: RowExactWeight) -> Vec<RowExactStockPacking> {
    match ty {
        RowExactWeight::F16 => {
            let mut out = Vec::new();
            for rows in [1, 4] {
                for unroll in [1, 4] {
                    for simdgroups in [1, 2] {
                        out.push(RowExactStockPacking::F16 {
                            rows,
                            simdgroups,
                            unroll,
                        });
                    }
                }
            }
            out
        }
        RowExactWeight::Q8_0 => {
            let mut out = Vec::new();
            for rows in [1, 2] {
                for virtual_simdgroups in [1, 2, 4] {
                    out.push(RowExactStockPacking::Q8_0 {
                        rows,
                        virtual_simdgroups,
                    });
                }
            }
            out
        }
        other => unreachable!("the stock-order family has no {other:?} kernels"),
    }
}

fn shape_for(m: usize, groups: usize, n: usize, k: usize) -> RowExactMatvecShape {
    if groups == 1 {
        RowExactMatvecShape::contiguous(m as u32, n as u32, k as u32)
    } else {
        RowExactMatvecShape::grouped_token_major(m as u32, groups as u32, n as u32, k as u32)
    }
}

fn first_difference(got: &[u32], want: &[u32]) -> Option<usize> {
    got.iter().zip(want).position(|(a, b)| a != b)
}

/// Every row of every m = 1..=8 dispatch, under every packing and the
/// default, equals the stock one-row output. Returns the stock bits.
fn check_shape(ty: RowExactWeight, groups: usize, n: usize, k: usize, seed: u64) -> Vec<u32> {
    let label = format!("{ty:?} {groups} x [{n} x {k}]");
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let w = weight(&device, ty, groups * n, k, seed);
    let x = input(&device, MAX_M * groups * k, seed ^ 0x9e37_79b9_7f4a_7c15);
    let stock = stock_rows(&device, &mut registry, ty, &w, &x, groups, n, k);
    let width = groups * n;

    let mut runs = 0;
    for m in 1..=MAX_M {
        let shape = shape_for(m, groups, n, k);
        let mut schedules: Vec<Option<RowExactStockPacking>> =
            packings(ty).into_iter().map(Some).collect();
        schedules.push(None);
        let outputs: Vec<MlxBuffer> = schedules
            .iter()
            .map(|_| output(&device, m * width))
            .collect();
        let mut encoder = device.command_encoder().expect("encoder");
        for (schedule, y) in schedules.iter().zip(&outputs) {
            match schedule {
                Some(packing) => dispatch_row_exact_matvec_stock_packed(
                    &mut encoder,
                    &mut registry,
                    device.metal_device(),
                    ty,
                    &w,
                    &x,
                    y,
                    &shape,
                    packing,
                ),
                None => dispatch_row_exact_matvec_stock(
                    &mut encoder,
                    &mut registry,
                    device.metal_device(),
                    ty,
                    &w,
                    &x,
                    y,
                    &shape,
                ),
            }
            .expect("stock-order dispatch");
        }
        encoder.commit_and_wait().expect("commit");
        for (schedule, y) in schedules.iter().zip(&outputs) {
            let got = bits(y);
            if let Some(i) = first_difference(&got, &stock[..m * width]) {
                panic!(
                    "{label}: m={m} {schedule:?}: row {} col {} = {:#010x}, stock m=1 {:#010x}",
                    i / width,
                    i % width,
                    got[i],
                    stock[i]
                );
            }
            runs += 1;
        }
    }

    // Determinism of the default dispatch at m = 8.
    let shape = shape_for(MAX_M, groups, n, k);
    for _ in 0..2 {
        let y = output(&device, MAX_M * width);
        let mut encoder = device.command_encoder().expect("encoder");
        dispatch_row_exact_matvec_stock(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            ty,
            &w,
            &x,
            &y,
            &shape,
        )
        .expect("dispatch");
        encoder.commit_and_wait().expect("commit");
        assert_eq!(bits(&y), stock, "{label}: repeated m = 8 run differs");
    }

    for name in row_exact_matvec_stock_kernel_names() {
        if let Ok(identity) = registry.pipeline_identity(name) {
            assert_eq!(identity.origin, expected_origin(), "{name} origin");
        }
    }
    let worst = check_f64_bound(&label, ty, &w, &x, &stock, groups, n, k);
    eprintln!(
        "{label}: {runs} dispatches (m = 1..8 x every packing + default) equal stock m=1 bitwise; deterministic; max |err|/bound {worst:.3e}"
    );
    stock
}

fn dequantize_row(ty: RowExactWeight, w: &[u8], row: usize, k: usize) -> Vec<f64> {
    match ty {
        RowExactWeight::F16 => w[row * k * 2..(row + 1) * k * 2]
            .chunks_exact(2)
            .map(|h| f64::from(half::f16::from_le_bytes([h[0], h[1]]).to_f32()))
            .collect(),
        RowExactWeight::Q8_0 => {
            let row_bytes = k / 32 * 34;
            w[row * row_bytes..(row + 1) * row_bytes]
                .chunks_exact(34)
                .flat_map(|block| {
                    let d = f64::from(half::f16::from_le_bytes([block[0], block[1]]).to_f32());
                    block[2..].iter().map(move |q| d * f64::from(*q as i8))
                })
                .collect()
        }
        other => unreachable!("the stock-order family has no {other:?} kernels"),
    }
}

/// Max of `|y - ref| / bound` over checked cells; panics past 1.
fn check_f64_bound(
    label: &str,
    ty: RowExactWeight,
    w: &MlxBuffer,
    x: &MlxBuffer,
    y_bits: &[u32],
    groups: usize,
    n: usize,
    k: usize,
) -> f64 {
    let u = 2f64.powi(-24);
    let gamma = k as f64 * u / (1.0 - k as f64 * u);
    let wb = w.as_slice::<u8>().expect("weight bytes");
    let xv = x.as_slice::<f32>().expect("input");
    let width = groups * n;
    let stride = (width * k * MAX_M).div_ceil(REFERENCE_MACS).max(1);
    let mut cols: Vec<usize> = (0..width).step_by(stride).collect();
    if cols.last() != Some(&(width - 1)) {
        cols.push(width - 1);
    }
    let mut worst = 0f64;
    for &col in &cols {
        let g = col / n;
        let wr = dequantize_row(ty, wb, col, k);
        for j in 0..MAX_M {
            let xr = &xv[j * groups * k + g * k..j * groups * k + (g + 1) * k];
            let (exact, magnitude) = wr.iter().zip(xr).fold((0f64, 0f64), |(e, m), (w, x)| {
                let p = w * f64::from(*x);
                (e + p, m + p.abs())
            });
            let got = f64::from(f32::from_bits(y_bits[j * width + col]));
            let bound = gamma * magnitude + 1e-30;
            let ratio = (got - exact).abs() / bound;
            assert!(
                ratio <= 1.0,
                "{label}: row {j} col {col}: got {got}, F64 {exact}, bound {bound}"
            );
            worst = worst.max(ratio);
        }
    }
    worst
}

const F16_SHAPES: [(usize, usize); 8] = [
    (24, 16384),  // hc_attn_fn / hc_ffn_fn
    (4, 16384),   // output_hc_fn
    (256, 4096),  // router, indexer compressor
    (1024, 4096), // CSA main compressor kv / gate
    (512, 4096),  // HCA main compressor kv / gate
    (8192, 1024), // indexer q_b
    (64, 4096),   // indexer proj
    (37, 4100),   // tail: K % 128 != 0 (uneven steps per lane), odd N
];

const Q8_0_SHAPES: [(usize, usize); 8] = [
    (1024, 4096),    // attn_q_a
    (512, 4096),     // attn_kv
    (32768, 1024),   // attn_q_b
    (4096, 8192),    // attn_output_b
    (2048, 4096),    // shared gate / up
    (4096, 2048),    // shared down
    (129_280, 4096), // output head
    (37, 4128),      // tail: K / 32 % 32 != 0, odd N
];

#[test]
fn row_exact_stock_f16_rows_equal_stock_one_row_kernel() {
    for (i, (n, k)) in F16_SHAPES.into_iter().enumerate() {
        check_shape(RowExactWeight::F16, 1, n, k, 0x5f16 + i as u64);
    }
}

#[test]
fn row_exact_stock_q8_0_rows_equal_stock_one_row_kernel() {
    for (i, (n, k)) in Q8_0_SHAPES.into_iter().enumerate() {
        check_shape(RowExactWeight::Q8_0, 1, n, k, 0x5080 + i as u64);
    }
}

/// DeepSeek-V4 attention output-A: 8 groups of Q8_0 [1024 x 4096] on
/// token-major [m, 8 * 4096] input, against the stock grouped matvec.
#[test]
fn row_exact_stock_grouped_output_a_rows_equal_stock_one_row_kernel() {
    check_shape(RowExactWeight::Q8_0, 8, 1024, 4096, 0x50a0);
}

/// The inputs above must expose summation-order changes: the Class-B
/// `row_exact_matvec` kernels differ from stock on them where their order
/// differs. (Class-B Q8_0 at its four-simdgroup reduction happens to be the
/// stock order, so the Q8_0 control uses a shape where Class B picks one
/// simdgroup: `row_exact_reduction_nsg(Q8_0, 4096, 2048) == 1`.)
#[test]
fn row_exact_stock_inputs_expose_a_different_summation_order() {
    for (ty, n, k, seed) in [
        (RowExactWeight::F16, 24, 16384, 0x5f16),
        (RowExactWeight::Q8_0, 4096, 2048, 0x5085),
    ] {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        let w = weight(&device, ty, n, k, seed);
        let x = input(&device, MAX_M * k, seed ^ 0x9e37_79b9_7f4a_7c15);
        let stock = stock_rows(&device, &mut registry, ty, &w, &x, 1, n, k);
        let y = output(&device, MAX_M * n);
        let mut encoder = device.command_encoder().expect("encoder");
        dispatch_row_exact_matvec(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            ty,
            &w,
            &x,
            &y,
            &RowExactMatvecShape::contiguous(MAX_M as u32, n as u32, k as u32),
        )
        .expect("class-B dispatch");
        encoder.commit_and_wait().expect("commit");
        let differing = bits(&y).iter().zip(&stock).filter(|(a, b)| a != b).count();
        assert!(
            differing > 0,
            "{ty:?} [{n} x {k}]: test inputs do not expose order changes"
        );
        eprintln!(
            "{ty:?} [{n} x {k}]: class-B row-exact differs from stock in {differing} of {} outputs",
            stock.len()
        );
    }
}

#[test]
fn row_exact_stock_rejects_bad_packings() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let w = f16_weight(&device, 8, 128, 1);
    let x = input(&device, 128, 2);
    let y = output(&device, 8);
    let shape = RowExactMatvecShape::contiguous(1, 8, 128);
    let mut encoder = device.command_encoder().expect("encoder");
    for packing in [
        RowExactStockPacking::Q8_0 {
            rows: 2,
            virtual_simdgroups: 1,
        },
        RowExactStockPacking::F16 {
            rows: 2,
            simdgroups: 1,
            unroll: 1,
        },
        RowExactStockPacking::F16 {
            rows: 1,
            simdgroups: 0,
            unroll: 1,
        },
        RowExactStockPacking::F16 {
            rows: 1,
            simdgroups: 1,
            unroll: 2,
        },
    ] {
        assert!(
            dispatch_row_exact_matvec_stock_packed(
                &mut encoder,
                &mut registry,
                device.metal_device(),
                RowExactWeight::F16,
                &w,
                &x,
                &y,
                &shape,
                &packing,
            )
            .is_err(),
            "{packing:?}"
        );
    }
}

/// A packing whose threadgroup exceeds the pipeline's thread limit is refused
/// before encoding; one within the limit dispatches and stays exact.
#[test]
fn row_exact_stock_refuses_threadgroups_above_the_pipeline_limit() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let (n, k, m) = (64usize, 4096usize, MAX_M);
    let w = f16_weight(&device, n, k, 3);
    let x = input(&device, m * k, 4);
    let shape = RowExactMatvecShape::contiguous(m as u32, n as u32, k as u32);
    // The heaviest F16 instantiation (4 rows, unroll 4, 8 inputs).
    let limit = registry
        .get_pipeline("row_exact_stock_mv_f16_f32_r4_u4_m8", device.metal_device())
        .expect("pipeline")
        .max_total_threads_per_threadgroup();
    let stock = stock_rows(&device, &mut registry, RowExactWeight::F16, &w, &x, 1, n, k);
    for simdgroups in [1u32, 2, 4, 8, 16, 32] {
        let packing = RowExactStockPacking::F16 {
            rows: 4,
            simdgroups,
            unroll: 4,
        };
        let y = output(&device, m * n);
        let mut encoder = device.command_encoder().expect("encoder");
        let result = dispatch_row_exact_matvec_stock_packed(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            RowExactWeight::F16,
            &w,
            &x,
            &y,
            &shape,
            &packing,
        );
        if u64::from(32 * simdgroups) > limit {
            assert!(
                result.is_err(),
                "{packing:?} exceeds limit {limit} but was encoded"
            );
        } else {
            result.expect("dispatch within the limit");
            encoder.commit_and_wait().expect("commit");
            assert_eq!(bits(&y), stock, "{packing:?}");
        }
    }
    // Packings above the limit only exist when the limit is below 1024.
    eprintln!("row_exact_stock_mv_f16_f32_r4_u4_m8 thread limit {limit}");
}
