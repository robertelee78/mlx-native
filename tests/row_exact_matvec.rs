//! Row-exact few-row mat-vec (`ops::row_exact_matvec`) correctness on the
//! DeepSeek-V4 decode shapes.
//!
//! For every shape and both weight formats:
//! - F64 reference: `|y - ref| <= gamma_K * sum_i |w_i x_i|` with
//!   `gamma_K = K u / (1 - K u)`, `u = 2^-24` (worst-case F32 summation of K
//!   products; every reduction path has at most K roundings).
//! - Row exactness: row `j` of an `m = B` dispatch is bitwise equal to the
//!   `m = 1` dispatch of row `j`, for B = 1..=8, under `nr0` 1 and 2, at the
//!   default reduction `nsg` and at nsg 1 and 4 on every shape.
//! - Determinism: repeated `m = 8` dispatches are bitwise equal.
//! - Grouped token-major dispatch equals the per-group dispatch (at the
//!   grouped weight's reduction nsg) bitwise.
//!
//! Run under both shader paths (the env var is latched per process):
//! `cargo test --release --test row_exact_matvec` and
//! `MLX_PRECOMPILED_METALLIB=0 cargo test --release --test row_exact_matvec`.

#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used, clippy::panic, clippy::too_many_arguments)]

use mlx_native::ops::row_exact_matvec::{
    dispatch_row_exact_matvec, dispatch_row_exact_matvec_scheduled, row_exact_matvec_kernel_names,
    row_exact_matvec_nsg_limit, row_exact_reduction_nsg, RowExactMatvecSchedule,
    RowExactMatvecShape, RowExactWeight,
};
use mlx_native::{DType, KernelPipelineOrigin, KernelRegistry, MlxBuffer, MlxDevice};

const MAX_M: usize = 8;
/// F64 reference MACs checked per shape (columns are subsampled above this).
const REFERENCE_MACS: usize = 64 << 20;

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
}

struct Weights {
    buffer: MlxBuffer,
    ty: RowExactWeight,
    k: usize,
}

impl Weights {
    /// Dequantized weight row `row` in F64.
    fn row(&self, row: usize) -> Vec<f64> {
        let bytes = self.buffer.as_slice::<u8>().expect("weight bytes");
        match self.ty {
            RowExactWeight::F16 => bytes[row * self.k * 2..(row + 1) * self.k * 2]
                .chunks_exact(2)
                .map(|h| f64::from(half::f16::from_le_bytes([h[0], h[1]]).to_f32()))
                .collect(),
            RowExactWeight::Q8_0 => {
                let row_bytes = self.k / 32 * 34;
                bytes[row * row_bytes..(row + 1) * row_bytes]
                    .chunks_exact(34)
                    .flat_map(|block| {
                        let d = f64::from(half::f16::from_le_bytes([block[0], block[1]]).to_f32());
                        block[2..].iter().map(move |q| d * f64::from(*q as i8))
                    })
                    .collect()
            }
        }
    }
}

fn f16_weights(device: &MlxDevice, rows: usize, k: usize, seed: u64) -> Weights {
    let mut rng = Rng(seed | 1);
    let mut buffer = device
        .alloc_buffer(rows * k * 2, DType::F16, vec![rows, k])
        .expect("f16 weight");
    for v in buffer.as_mut_slice::<u16>().expect("f16 slice") {
        *v = half::f16::from_f32(rng.unit() * 0.06).to_bits();
    }
    Weights {
        buffer,
        ty: RowExactWeight::F16,
        k,
    }
}

fn q8_0_weights(device: &MlxDevice, rows: usize, k: usize, seed: u64) -> Weights {
    let mut rng = Rng(seed | 1);
    let blocks = rows * k / 32;
    let mut buffer = device
        .alloc_buffer(blocks * 34, DType::U8, vec![blocks * 34])
        .expect("q8_0 weight");
    for block in buffer
        .as_mut_slice::<u8>()
        .expect("q8_0 slice")
        .chunks_exact_mut(34)
    {
        let d = half::f16::from_f32(0.0005 + (rng.next() % 1024) as f32 * 0.000_004);
        block[..2].copy_from_slice(&d.to_bits().to_le_bytes());
        for chunk in block[2..].chunks_mut(8) {
            let r = rng.next().to_le_bytes();
            chunk.copy_from_slice(&r[..chunk.len()]);
        }
    }
    Weights {
        buffer,
        ty: RowExactWeight::Q8_0,
        k,
    }
}

fn weights(device: &MlxDevice, ty: RowExactWeight, rows: usize, k: usize, seed: u64) -> Weights {
    match ty {
        RowExactWeight::F16 => f16_weights(device, rows, k, seed),
        RowExactWeight::Q8_0 => q8_0_weights(device, rows, k, seed),
    }
}

fn f32_buffer(device: &MlxDevice, values: &[f32]) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(values.len() * 4, DType::F32, vec![values.len()])
        .expect("f32 buffer");
    buffer
        .as_mut_slice::<f32>()
        .expect("f32 slice")
        .copy_from_slice(values);
    buffer
}

/// Output fill; a launched dispatch overwrites every element.
const SENTINEL: u32 = 0x7fc0_dead;

fn output(device: &MlxDevice, len: usize) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(len * 4, DType::F32, vec![len])
        .expect("output");
    buffer
        .as_mut_slice::<u32>()
        .expect("output slice")
        .fill(SENTINEL);
    buffer
}

fn bits(buffer: &MlxBuffer) -> Vec<u32> {
    buffer.as_slice::<u32>().expect("bits").to_vec()
}

/// Encode and wait. `None` when the dispatch refused `schedule` because its
/// threads exceed the pipeline's threadgroup limit on this device (asserted
/// from the error, and logged); any other refusal panics. A launched
/// dispatch must have overwritten every sentinel.
fn try_run(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    ty: RowExactWeight,
    w: &MlxBuffer,
    x: &MlxBuffer,
    y: &MlxBuffer,
    shape: &RowExactMatvecShape,
    schedule: Option<RowExactMatvecSchedule>,
) -> Option<()> {
    let mut encoder = device.command_encoder().expect("encoder");
    let encoded = match schedule {
        Some(schedule) => dispatch_row_exact_matvec_scheduled(
            &mut encoder,
            registry,
            device.metal_device(),
            ty,
            w,
            x,
            y,
            shape,
            &schedule,
        ),
        None => dispatch_row_exact_matvec(
            &mut encoder,
            registry,
            device.metal_device(),
            ty,
            w,
            x,
            y,
            shape,
        ),
    };
    if let Err(error) = encoded {
        let message = error.to_string();
        let schedule = schedule.expect("the default schedule must always launch");
        assert!(
            message.contains("threads per threadgroup on this device"),
            "{ty:?} m={} {schedule:?}: unexpected refusal: {message}",
            shape.m
        );
        let limit =
            row_exact_matvec_nsg_limit(registry, device.metal_device(), ty).expect("nsg limit");
        assert!(
            schedule.nsg > limit,
            "{ty:?} m={} {schedule:?} refused although the device nsg limit is {limit}",
            shape.m
        );
        eprintln!(
            "skip {ty:?} m={} {schedule:?}: pipeline threadgroup limit ({message})",
            shape.m
        );
        encoder.commit_and_wait().expect("commit");
        return None;
    }
    encoder.commit_and_wait().expect("commit");
    let written = bits(y);
    assert!(
        !written.contains(&SENTINEL),
        "{ty:?} m={} {schedule:?}: launched dispatch left output unwritten",
        shape.m
    );
    Some(())
}

fn run(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    ty: RowExactWeight,
    w: &MlxBuffer,
    x: &MlxBuffer,
    y: &MlxBuffer,
    shape: &RowExactMatvecShape,
    schedule: Option<RowExactMatvecSchedule>,
) {
    try_run(device, registry, ty, w, x, y, shape, schedule)
        .unwrap_or_else(|| panic!("{ty:?} m={} {schedule:?} must launch", shape.m));
}

/// Max of `|y - ref| / bound` over checked cells; panics past 1.
fn check_reference(
    label: &str,
    w: &Weights,
    first_row: usize,
    x: &[f32],
    y: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> f64 {
    let u = 2f64.powi(-24);
    let gamma = k as f64 * u / (1.0 - k as f64 * u);
    let stride = (n * k * m).div_ceil(REFERENCE_MACS).max(1);
    let mut worst = 0f64;
    let mut cols: Vec<usize> = (0..n).step_by(stride).collect();
    if cols.last() != Some(&(n - 1)) {
        cols.push(n - 1);
    }
    for &col in &cols {
        let wr = w.row(first_row + col);
        for j in 0..m {
            let xr = &x[j * k..(j + 1) * k];
            let (mut exact, mut magnitude) = (0f64, 0f64);
            for (wi, xi) in wr.iter().zip(xr) {
                let p = wi * f64::from(*xi);
                exact += p;
                magnitude += p.abs();
            }
            let got = f64::from(y[j * n + col]);
            let bound = gamma * magnitude + 1e-30;
            let ratio = (got - exact).abs() / bound;
            assert!(
                ratio <= 1.0,
                "{label}: row {j} col {col}: got {got}, F64 {exact}, |err| {} > bound {bound}",
                (got - exact).abs()
            );
            worst = worst.max(ratio);
        }
    }
    worst
}

fn expected_origin() -> KernelPipelineOrigin {
    match std::env::var("MLX_PRECOMPILED_METALLIB").as_deref() {
        Ok("0") | Ok("false") | Ok("off") => KernelPipelineOrigin::RuntimeSource,
        _ => KernelPipelineOrigin::PrecompiledMetallib,
    }
}

/// Every packing of a reduction `nsg`.
fn packings(nsg: u32) -> [RowExactMatvecSchedule; 2] {
    [1, 2].map(|nr0| RowExactMatvecSchedule { nsg, nr0 })
}

/// Row exactness at one reduction `nsg`: rows of every `m = 1..=8` dispatch,
/// under every packing, equal the `m = 1` dispatch of that row. Packings the
/// device cannot launch at this `nsg` are skipped (see [`try_run`]); every
/// launched one is compared. Returns the widest launched output as
/// `(m, bits)`, or `None` when no `m = 1` packing launches at this `nsg`.
fn check_reduction(
    label: &str,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    ty: RowExactWeight,
    w: &MlxBuffer,
    x: &MlxBuffer,
    n: usize,
    k: usize,
    nsg: u32,
) -> Option<(usize, Vec<u32>)> {
    let single_shape = RowExactMatvecShape::contiguous(1, n as u32, k as u32);
    // Packing is numerically inert: any m = 1 packing that launches is the
    // reference.
    let reference = packings(nsg).into_iter().find(|&schedule| {
        let y = output(device, n);
        try_run(
            device,
            registry,
            ty,
            w,
            x,
            &y,
            &single_shape,
            Some(schedule),
        )
        .is_some()
    })?;
    let single: Vec<Vec<u32>> = (0..MAX_M)
        .map(|j| {
            let xj = x.slice_view((j * k * 4) as u64, k);
            let y = output(device, n);
            run(
                device,
                registry,
                ty,
                w,
                &xj,
                &y,
                &single_shape,
                Some(reference),
            );
            bits(&y)
        })
        .collect();
    let mut widest = None;
    for m in 1..=MAX_M {
        let shape = RowExactMatvecShape::contiguous(m as u32, n as u32, k as u32);
        for schedule in packings(nsg) {
            let y = output(device, m * n);
            if try_run(device, registry, ty, w, x, &y, &shape, Some(schedule)).is_none() {
                continue;
            }
            let got = bits(&y);
            for (j, row) in single.iter().enumerate().take(m) {
                if got[j * n..(j + 1) * n] != row[..] {
                    let col = (0..n)
                        .find(|&c| got[j * n + c] != row[c])
                        .expect("differing column");
                    panic!(
                        "{label}: m={m} {schedule:?} row {j} col {col} differs from m=1: {:#010x} vs {:#010x}",
                        got[j * n + col], row[col]
                    );
                }
            }
            widest = Some((m, got));
        }
    }
    widest
}

fn check_shape(ty: RowExactWeight, n: usize, k: usize, seed: u64) {
    let label = format!("{ty:?} [{n} x {k}]");
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let w = weights(&device, ty, n, k, seed);
    let mut rng = Rng(seed ^ 0x9e37_79b9_7f4a_7c15);
    let xv: Vec<f32> = (0..MAX_M * k).map(|_| rng.unit()).collect();
    let x = f32_buffer(&device, &xv);

    let nsg = row_exact_reduction_nsg(ty, n as u32, k as u32);
    let limit =
        row_exact_matvec_nsg_limit(&mut registry, device.metal_device(), ty).expect("nsg limit");
    let device_nsg = nsg.min(limit);
    for m in 1..=MAX_M {
        let shape = RowExactMatvecShape::contiguous(m as u32, n as u32, k as u32);
        let auto = RowExactMatvecSchedule::auto(ty, &shape);
        assert_eq!(auto.nsg, nsg, "{label}: auto nsg must not depend on m");
        let on_device =
            RowExactMatvecSchedule::for_device(&mut registry, device.metal_device(), ty, &shape)
                .expect("device schedule");
        assert_eq!(
            on_device.nsg, device_nsg,
            "{label}: device nsg must not depend on m"
        );
    }

    // The default (device) reduction, the shape-derived one if the device
    // caps it, then the single-simdgroup and the threadgroup-memory
    // reductions on every shape. The default reduction must launch at every
    // m and packing.
    let mut reductions = vec![device_nsg];
    for other in [nsg, 1, 4] {
        if !reductions.contains(&other) {
            reductions.push(other);
        }
    }
    let mut checked = Vec::new();
    for r_nsg in reductions {
        let Some((m, full)) = check_reduction(
            &label,
            &device,
            &mut registry,
            ty,
            &w.buffer,
            &x,
            n,
            k,
            r_nsg,
        ) else {
            assert!(
                r_nsg > limit,
                "{label}: nsg {r_nsg} within the limit {limit} must launch"
            );
            checked.push(format!("nsg {r_nsg}: skipped (device limit {limit})"));
            continue;
        };
        if r_nsg <= limit {
            assert_eq!(
                m, MAX_M,
                "{label}: nsg {r_nsg} within the limit must launch at m = 8"
            );
        }
        let yv: Vec<f32> = full.iter().map(|b| f32::from_bits(*b)).collect();
        let worst = check_reference(&label, &w, 0, &xv, &yv, m, n, k);
        checked.push(format!("nsg {r_nsg}: {worst:.2e}"));
    }

    // Determinism of the default dispatch.
    let shape = RowExactMatvecShape::contiguous(MAX_M as u32, n as u32, k as u32);
    let runs: Vec<Vec<u32>> = (0..3)
        .map(|_| {
            let y = output(&device, MAX_M * n);
            run(&device, &mut registry, ty, &w.buffer, &x, &y, &shape, None);
            bits(&y)
        })
        .collect();
    assert!(
        runs.windows(2).all(|p| p[0] == p[1]),
        "{label}: repeated m = 8 run differs"
    );

    for name in row_exact_matvec_kernel_names() {
        if let Ok(identity) = registry.pipeline_identity(name) {
            assert_eq!(identity.origin, expected_origin(), "{name} origin");
        }
    }
    eprintln!(
        "{label}: shape nsg {nsg}, device nsg limit {limit}, default nsg {device_nsg}; row-exact m=1..8 under nr0 1 and 2; deterministic; max |err|/bound {}",
        checked.join(", ")
    );
}

const F16_SHAPES: [(usize, usize); 7] = [
    (24, 16384),
    (4, 16384),
    (256, 4096),
    (1024, 4096),
    (512, 4096),
    (8192, 1024),
    (64, 4096),
];

const Q8_0_SHAPES: [(usize, usize); 7] = [
    (32768, 1024),
    (4096, 8192),
    (2048, 4096),
    (4096, 2048),
    (1024, 4096),
    (512, 4096),
    (129_280, 4096),
];

#[test]
fn row_exact_matvec_f16_decode_shapes() {
    for (i, (n, k)) in F16_SHAPES.into_iter().enumerate() {
        check_shape(RowExactWeight::F16, n, k, 0x1600 + i as u64);
    }
    // Tails: K not a multiple of a full threadgroup pass, odd N.
    check_shape(RowExactWeight::F16, 37, 4100, 0x16ff);
    check_shape(RowExactWeight::F16, 3, 20, 0x16fe);
}

#[test]
fn row_exact_matvec_q8_0_decode_shapes() {
    for (i, (n, k)) in Q8_0_SHAPES.into_iter().enumerate() {
        check_shape(RowExactWeight::Q8_0, n, k, 0x0800 + i as u64);
    }
    check_shape(RowExactWeight::Q8_0, 37, 4128, 0x08ff);
    check_shape(RowExactWeight::Q8_0, 3, 32, 0x08fe);
}

/// DeepSeek-V4 attention output-A: 8 groups of Q8_0 [1024 x 4096] over a
/// token-major [m, 8 * 4096] input, written token-major [m, 8 * 1024].
#[test]
fn row_exact_matvec_q8_0_grouped_output_a() {
    let (groups, rank, gw) = (8usize, 1024usize, 4096usize);
    let ty = RowExactWeight::Q8_0;
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let w = weights(&device, ty, groups * rank, gw, 0x0a0a);
    let row_bytes = gw / 32 * 34;
    let mut rng = Rng(0x0a0b);
    let xv: Vec<f32> = (0..MAX_M * groups * gw).map(|_| rng.unit()).collect();
    let x = f32_buffer(&device, &xv);

    let grouped = |registry: &mut KernelRegistry, m: usize| {
        let shape = RowExactMatvecShape::grouped_token_major(
            m as u32,
            groups as u32,
            rank as u32,
            gw as u32,
        );
        let y = output(&device, m * groups * rank);
        run(&device, registry, ty, &w.buffer, &x, &y, &shape, None);
        bits(&y)
    };
    let full = grouped(&mut registry, MAX_M);
    // The grouped weight is one [groups * rank, gw] matrix: its reduction
    // nsg follows the total row count, so a per-group dispatch pins it.
    let grouped_schedule = RowExactMatvecSchedule::for_device(
        &mut registry,
        device.metal_device(),
        ty,
        &RowExactMatvecShape::grouped_token_major(
            MAX_M as u32,
            groups as u32,
            rank as u32,
            gw as u32,
        ),
    )
    .expect("grouped device schedule");
    let limit =
        row_exact_matvec_nsg_limit(&mut registry, device.metal_device(), ty).expect("nsg limit");
    assert_eq!(
        grouped_schedule.nsg,
        row_exact_reduction_nsg(ty, (groups * rank) as u32, gw as u32).min(limit)
    );

    // Per-group contiguous dispatch on gathered inputs must match bitwise.
    for g in 0..groups {
        let xg: Vec<f32> = (0..MAX_M)
            .flat_map(|j| xv[j * groups * gw + g * gw..j * groups * gw + (g + 1) * gw].to_vec())
            .collect();
        let xg_buf = f32_buffer(&device, &xg);
        let wg = w
            .buffer
            .slice_view((g * rank * row_bytes) as u64, rank * row_bytes);
        let yg = output(&device, MAX_M * rank);
        let shape = RowExactMatvecShape::contiguous(MAX_M as u32, rank as u32, gw as u32);
        run(
            &device,
            &mut registry,
            ty,
            &wg,
            &xg_buf,
            &yg,
            &shape,
            Some(grouped_schedule),
        );
        let yg_bits = bits(&yg);
        for j in 0..MAX_M {
            assert_eq!(
                full[j * groups * rank + g * rank..j * groups * rank + (g + 1) * rank],
                yg_bits[j * rank..(j + 1) * rank],
                "group {g} row {j}: grouped vs per-group dispatch"
            );
        }
        let yf: Vec<f32> = yg_bits.iter().map(|b| f32::from_bits(*b)).collect();
        let label = format!("grouped output-A group {g}");
        check_reference(&label, &w, g * rank, &xg, &yf, MAX_M, rank, gw);
    }

    // Every m < 8 matches the leading rows of m = 8; m = 8 is deterministic.
    for m in 1..=MAX_M {
        let got = grouped(&mut registry, m);
        assert_eq!(got[..], full[..m * groups * rank], "grouped m={m} vs m=8");
    }
}

/// Logs every pipeline's threadgroup limit and checks the derived nsg limit:
/// the largest power of two whose threads every pipeline of the weight type
/// can launch. A schedule above a pipeline's limit is refused, never
/// launched (Metal drops such a launch silently).
#[test]
fn row_exact_matvec_pipeline_thread_limits() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    for ty in [RowExactWeight::F16, RowExactWeight::Q8_0] {
        let prefix = match ty {
            RowExactWeight::F16 => "row_exact_mv_f16",
            RowExactWeight::Q8_0 => "row_exact_mv_q8_0",
        };
        let mut min_threads = u64::MAX;
        for name in row_exact_matvec_kernel_names().filter(|name| name.starts_with(prefix)) {
            let threads = registry
                .get_pipeline(name, device.metal_device())
                .expect("pipeline")
                .max_total_threads_per_threadgroup();
            eprintln!("{name}: max_total_threads_per_threadgroup {threads}");
            min_threads = min_threads.min(threads);
        }
        assert_ne!(
            min_threads,
            u64::MAX,
            "{ty:?}: no pipelines matched {prefix}"
        );
        let limit = row_exact_matvec_nsg_limit(&mut registry, device.metal_device(), ty)
            .expect("nsg limit");
        assert!(limit.is_power_of_two() && limit <= 32);
        assert!(u64::from(limit) * 32 <= min_threads);
        assert!(limit == 32 || u64::from(limit) * 64 > min_threads);
        eprintln!("{ty:?}: min pipeline threads {min_threads}, nsg limit {limit}");
    }
}

#[test]
fn row_exact_matvec_rejects_bad_arguments() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let w = f16_weights(&device, 8, 64, 1).buffer;
    let x = f32_buffer(&device, &[0.0; 9 * 64]);
    let y = output(&device, 9 * 8);
    let mut encoder = device.command_encoder().expect("encoder");
    let mut attempt = |ty, w: &MlxBuffer, x: &MlxBuffer, shape: RowExactMatvecShape| {
        dispatch_row_exact_matvec(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            ty,
            w,
            x,
            &y,
            &shape,
        )
    };
    assert!(attempt(
        RowExactWeight::F16,
        &w,
        &x,
        RowExactMatvecShape::contiguous(9, 8, 64)
    )
    .is_err());
    assert!(attempt(
        RowExactWeight::F16,
        &w,
        &x,
        RowExactMatvecShape::contiguous(0, 8, 64)
    )
    .is_err());
    assert!(attempt(
        RowExactWeight::F16,
        &w,
        &x,
        RowExactMatvecShape::contiguous(1, 9, 64)
    )
    .is_err());
    assert!(attempt(
        RowExactWeight::Q8_0,
        &w,
        &x,
        RowExactMatvecShape::contiguous(1, 8, 64)
    )
    .is_err());
    let unaligned = x.slice_view(4, 64);
    assert!(attempt(
        RowExactWeight::F16,
        &w,
        &unaligned,
        RowExactMatvecShape::contiguous(1, 8, 64)
    )
    .is_err());
    // Overlapping output rows (y_row_stride < n with m > 1) and colliding
    // batches are refused; overlapping input reads are allowed.
    let mut rows_overlap = RowExactMatvecShape::contiguous(2, 8, 64);
    rows_overlap.y_row_stride = 4;
    assert!(attempt(RowExactWeight::F16, &w, &x, rows_overlap).is_err());
    let mut batches_overlap = RowExactMatvecShape::grouped_token_major(2, 2, 4, 32);
    batches_overlap.y_batch_stride = 2;
    assert!(attempt(RowExactWeight::F16, &w, &x, batches_overlap).is_err());
    let mut shared_input = RowExactMatvecShape::contiguous(2, 8, 64);
    shared_input.x_row_stride = 0;
    assert!(attempt(RowExactWeight::F16, &w, &x, shared_input).is_ok());
}
