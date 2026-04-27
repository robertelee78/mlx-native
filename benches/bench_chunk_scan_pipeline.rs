//! Wave 5b.2 iter 1 — chunk-scan pipeline kernel-level benchmark harness.
//!
//! Six criterion benches, one per kernel in the chunk-parallel Gated DeltaNet
//! pipeline, at long-prefill shape:
//!
//!   B=1, T=4096, Hg=2, H=4, K=128, V=128, BT=64, NT=64
//!
//! Long-prefill is the regime where chunk-scan beats autoregressive forward
//! (autoregressive scales O(T) while chunk-scan scales O(T/BT) per
//! threadgroup), so this is the right operating point to baseline before
//! kernel-level optimization (commits 3+4 will introduce simdgroup_matrix
//! MMA on the dominant kernel).
//!
//! Each bench measures wall-time of the kernel's command-buffer dispatch
//! (encode + commit + wait); buffer allocation is hoisted out of the
//! `b.iter()` closure. The first iteration warms the pipeline cache.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use criterion::{criterion_group, criterion_main, Criterion};

use mlx_native::ops::chunk_gated_delta_rule_tri_solve_invert::{
    self as tri_solve, build_chunk_tri_solve_invert_params, dispatch_chunk_tri_solve_invert,
    ChunkTriSolveInvertParams,
};
use mlx_native::ops::gated_delta_net_chunk::{
    self as inter_state, build_gated_delta_net_chunk_params,
    dispatch_gated_delta_net_chunk_inter_state, GatedDeltaNetChunkParams,
};
use mlx_native::ops::gated_delta_net_chunk_o::{
    self as chunk_o, build_gated_delta_net_chunk_o_params, dispatch_gated_delta_net_chunk_o,
    GatedDeltaNetChunkOParams,
};
use mlx_native::ops::gated_delta_net_kkt::{
    self as kkt, build_gated_delta_net_kkt_params, dispatch_gated_delta_net_kkt,
    GatedDeltaNetKktParams,
};
use mlx_native::ops::gated_delta_net_recompute_wu::{
    self as recompute_wu, build_gated_delta_net_recompute_wu_params,
    dispatch_gated_delta_net_recompute_wu, GatedDeltaNetRecomputeWuParams,
};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

// Long-prefill shape constants.
const B: u32 = 1;
const T: u32 = 4096;
const HG: u32 = 2;
const H: u32 = 4;
const K: u32 = 128;
const V: u32 = 128;
const BT: u32 = 64;

fn alloc_bf16(device: &MlxDevice, n_elems: usize, fill: f32) -> MlxBuffer {
    let mut buf = device
        .alloc_buffer(n_elems * 2, DType::BF16, vec![n_elems])
        .expect("alloc bf16");
    {
        let dst = buf.as_mut_slice::<u16>().expect("mut bf16");
        // Convert fill -> bf16 (truncating round, identical to PyTorch's bf16 cast).
        let bf16_bits = (fill.to_bits() >> 16) as u16;
        for v in dst.iter_mut() {
            *v = bf16_bits;
        }
    }
    buf
}

fn alloc_f32(device: &MlxDevice, n_elems: usize, fill: f32) -> MlxBuffer {
    let mut buf = device
        .alloc_buffer(n_elems * 4, DType::F32, vec![n_elems])
        .expect("alloc f32");
    {
        let dst = buf.as_mut_slice::<f32>().expect("mut f32");
        for v in dst.iter_mut() {
            *v = fill;
        }
    }
    buf
}

/// `gated_delta_net_chunk_inter_state_bf16` — the dominant chunk-scan kernel.
fn bench_inter_state(c: &mut Criterion, device: &MlxDevice, registry: &mut KernelRegistry) {
    inter_state::register(registry);

    let p = GatedDeltaNetChunkParams {
        b: B,
        t: T,
        hg: HG,
        h: H,
        k: K,
        v: V,
        bt: BT,
    };
    let nt = p.num_chunks();

    let k_buf = alloc_bf16(device, (B * T * HG * K) as usize, 0.01);
    let w_buf = alloc_bf16(device, (B * T * H * K) as usize, 0.01);
    let u_buf = alloc_bf16(device, (B * T * H * V) as usize, 0.01);
    let g_buf = alloc_f32(device, (B * T * H) as usize, 0.0); // log-decay, neutral
    let h0_buf = alloc_f32(device, (B * H * V * K) as usize, 0.0);
    let h_out_buf = alloc_bf16(device, (B * nt * H * V * K) as usize, 0.0);
    let v_new_buf = alloc_bf16(device, (B * T * H * V) as usize, 0.0);
    let final_state_buf = alloc_f32(device, (B * H * V * K) as usize, 0.0);

    let params_buf =
        build_gated_delta_net_chunk_params(device, p).expect("build params");

    // Warm up.
    {
        let mut enc = device.command_encoder().expect("enc");
        dispatch_gated_delta_net_chunk_inter_state(
            &mut enc,
            registry,
            device.metal_device(),
            &k_buf,
            &w_buf,
            &u_buf,
            &g_buf,
            &h0_buf,
            &h_out_buf,
            &v_new_buf,
            &final_state_buf,
            &params_buf,
            p,
        )
        .expect("warmup dispatch");
        enc.commit_and_wait().expect("warmup commit");
    }

    c.bench_function("chunk_inter_state_b1_t4096_h4_k128_v128_bt64", |b| {
        b.iter(|| {
            let mut enc = device.command_encoder().expect("enc");
            dispatch_gated_delta_net_chunk_inter_state(
                &mut enc,
                registry,
                device.metal_device(),
                &k_buf,
                &w_buf,
                &u_buf,
                &g_buf,
                &h0_buf,
                &h_out_buf,
                &v_new_buf,
                &final_state_buf,
                &params_buf,
                p,
            )
            .expect("dispatch");
            enc.commit_and_wait().expect("commit");
        });
    });
}

/// `gated_delta_net_kkt_bf16` — A_strict = β·k·k^T·exp(g_i - g_j) per chunk.
fn bench_kkt(c: &mut Criterion, device: &MlxDevice, registry: &mut KernelRegistry) {
    kkt::register(registry);

    let p = GatedDeltaNetKktParams {
        b: B,
        t: T,
        hg: HG,
        h: H,
        k: K,
        bt: BT,
    };

    let k_buf = alloc_bf16(device, (B * T * HG * K) as usize, 0.01);
    let beta_buf = alloc_f32(device, (B * T * H) as usize, 0.5);
    let g_buf = alloc_f32(device, (B * T * H) as usize, 0.0);
    let a_buf = alloc_f32(device, (B * T * H * BT) as usize, 0.0);

    let params_buf = build_gated_delta_net_kkt_params(device, p).expect("build params");

    {
        let mut enc = device.command_encoder().expect("enc");
        dispatch_gated_delta_net_kkt(
            &mut enc,
            registry,
            device.metal_device(),
            &k_buf,
            &beta_buf,
            &g_buf,
            &a_buf,
            &params_buf,
            p,
        )
        .expect("warmup dispatch");
        enc.commit_and_wait().expect("warmup commit");
    }

    c.bench_function("chunk_kkt_b1_t4096_h4_k128_bt64", |b| {
        b.iter(|| {
            let mut enc = device.command_encoder().expect("enc");
            dispatch_gated_delta_net_kkt(
                &mut enc,
                registry,
                device.metal_device(),
                &k_buf,
                &beta_buf,
                &g_buf,
                &a_buf,
                &params_buf,
                p,
            )
            .expect("dispatch");
            enc.commit_and_wait().expect("commit");
        });
    });
}

/// `gated_delta_net_recompute_wu_bf16` — w = A·(β·g·k), u = A·(β·v).
fn bench_recompute_wu(c: &mut Criterion, device: &MlxDevice, registry: &mut KernelRegistry) {
    recompute_wu::register(registry);

    let p = GatedDeltaNetRecomputeWuParams {
        b: B,
        t: T,
        hg: HG,
        h: H,
        k: K,
        v: V,
        bt: BT,
    };

    let k_buf = alloc_bf16(device, (B * T * HG * K) as usize, 0.01);
    let v_buf = alloc_bf16(device, (B * T * H * V) as usize, 0.01);
    let beta_buf = alloc_f32(device, (B * T * H) as usize, 0.5);
    let g_buf = alloc_f32(device, (B * T * H) as usize, 0.0);
    let a_buf = alloc_f32(device, (B * T * H * BT) as usize, 0.0);
    let w_buf = alloc_bf16(device, (B * T * H * K) as usize, 0.0);
    let u_buf = alloc_bf16(device, (B * T * H * V) as usize, 0.0);

    let params_buf =
        build_gated_delta_net_recompute_wu_params(device, p).expect("build params");

    {
        let mut enc = device.command_encoder().expect("enc");
        dispatch_gated_delta_net_recompute_wu(
            &mut enc,
            registry,
            device.metal_device(),
            &k_buf,
            &v_buf,
            &beta_buf,
            &g_buf,
            &a_buf,
            &w_buf,
            &u_buf,
            &params_buf,
            p,
        )
        .expect("warmup dispatch");
        enc.commit_and_wait().expect("warmup commit");
    }

    c.bench_function("chunk_recompute_wu_b1_t4096_h4_k128_v128_bt64", |b| {
        b.iter(|| {
            let mut enc = device.command_encoder().expect("enc");
            dispatch_gated_delta_net_recompute_wu(
                &mut enc,
                registry,
                device.metal_device(),
                &k_buf,
                &v_buf,
                &beta_buf,
                &g_buf,
                &a_buf,
                &w_buf,
                &u_buf,
                &params_buf,
                p,
            )
            .expect("dispatch");
            enc.commit_and_wait().expect("commit");
        });
    });
}

/// `gated_delta_net_chunk_o_bf16` — final per-token output.
fn bench_chunk_o(c: &mut Criterion, device: &MlxDevice, registry: &mut KernelRegistry) {
    chunk_o::register(registry);

    let scale: f32 = (K as f32).powf(-0.5);
    let p = GatedDeltaNetChunkOParams {
        b: B,
        t: T,
        hg: HG,
        h: H,
        k: K,
        v: V,
        bt: BT,
        scale,
    };
    let nt = p.num_chunks();

    let q_buf = alloc_bf16(device, (B * T * HG * K) as usize, 0.01);
    let k_buf = alloc_bf16(device, (B * T * HG * K) as usize, 0.01);
    let v_new_buf = alloc_bf16(device, (B * T * H * V) as usize, 0.01);
    let h_buf = alloc_bf16(device, (B * nt * H * V * K) as usize, 0.01);
    let g_buf = alloc_f32(device, (B * T * H) as usize, 0.0);
    let o_buf = alloc_bf16(device, (B * T * H * V) as usize, 0.0);

    let params_buf =
        build_gated_delta_net_chunk_o_params(device, p).expect("build params");

    {
        let mut enc = device.command_encoder().expect("enc");
        dispatch_gated_delta_net_chunk_o(
            &mut enc,
            registry,
            device.metal_device(),
            &q_buf,
            &k_buf,
            &v_new_buf,
            &h_buf,
            &g_buf,
            &o_buf,
            &params_buf,
            p,
        )
        .expect("warmup dispatch");
        enc.commit_and_wait().expect("warmup commit");
    }

    c.bench_function("chunk_o_b1_t4096_h4_k128_v128_bt64", |b| {
        b.iter(|| {
            let mut enc = device.command_encoder().expect("enc");
            dispatch_gated_delta_net_chunk_o(
                &mut enc,
                registry,
                device.metal_device(),
                &q_buf,
                &k_buf,
                &v_new_buf,
                &h_buf,
                &g_buf,
                &o_buf,
                &params_buf,
                p,
            )
            .expect("dispatch");
            enc.commit_and_wait().expect("commit");
        });
    });
}

/// `chunk_tri_solve_invert_f32` — `(I + A_strict)^-1` per BT-block.
fn bench_tri_solve_invert(c: &mut Criterion, device: &MlxDevice, registry: &mut KernelRegistry) {
    tri_solve::register(registry);

    let p = ChunkTriSolveInvertParams {
        b: B,
        t: T,
        h: H,
        bt: BT,
    };

    let a_strict_buf = alloc_f32(device, (B * T * H * BT) as usize, 0.0);
    let a_inv_buf = alloc_f32(device, (B * T * H * BT) as usize, 0.0);

    let params_buf =
        build_chunk_tri_solve_invert_params(device, p).expect("build params");

    {
        let mut enc = device.command_encoder().expect("enc");
        dispatch_chunk_tri_solve_invert(
            &mut enc,
            registry,
            device.metal_device(),
            &a_strict_buf,
            &a_inv_buf,
            &params_buf,
            p,
        )
        .expect("warmup dispatch");
        enc.commit_and_wait().expect("warmup commit");
    }

    c.bench_function("chunk_tri_solve_invert_b1_t4096_h4_bt64", |b| {
        b.iter(|| {
            let mut enc = device.command_encoder().expect("enc");
            dispatch_chunk_tri_solve_invert(
                &mut enc,
                registry,
                device.metal_device(),
                &a_strict_buf,
                &a_inv_buf,
                &params_buf,
                p,
            )
            .expect("dispatch");
            enc.commit_and_wait().expect("commit");
        });
    });
}

/// `chunk_local_cumsum_g_f32` — per-chunk inclusive prefix sum on log-decay.
fn bench_cumsum_g(c: &mut Criterion, device: &MlxDevice, registry: &mut KernelRegistry) {
    use mlx_native::ops::chunk_gated_delta_rule;
    chunk_gated_delta_rule::register(registry);

    let g_in_buf = alloc_f32(device, (B * T * H) as usize, 0.0);
    let g_out_buf = alloc_f32(device, (B * T * H) as usize, 0.0);

    // Build the 5-u32 params buffer for the cumsum_g kernel: [B, T, H, BT, NT].
    let nt = T.div_ceil(BT);
    let mut params_buf = device
        .alloc_buffer(5 * 4, DType::U32, vec![5])
        .expect("alloc params");
    {
        let s = params_buf.as_mut_slice::<u32>().expect("mut params");
        s[0] = B;
        s[1] = T;
        s[2] = H;
        s[3] = BT;
        s[4] = nt;
    }

    // Mirror the orchestrator's dispatch shape exactly.
    let pipeline = registry
        .get_pipeline("chunk_local_cumsum_g_f32", device.metal_device())
        .expect("get_pipeline");
    let grid = metal::MTLSize::new(1, H as u64, (B as u64) * (nt as u64));
    let tg = metal::MTLSize::new(BT as u64, 1, 1);

    {
        let mut enc = device.command_encoder().expect("enc");
        enc.encode_threadgroups(
            &pipeline,
            &[(0, &g_in_buf), (1, &g_out_buf), (2, &params_buf)],
            grid,
            tg,
        );
        enc.commit_and_wait().expect("warmup commit");
    }

    c.bench_function("chunk_local_cumsum_g_b1_t4096_h4_bt64", |b| {
        b.iter(|| {
            let mut enc = device.command_encoder().expect("enc");
            enc.encode_threadgroups(
                &pipeline,
                &[(0, &g_in_buf), (1, &g_out_buf), (2, &params_buf)],
                grid,
                tg,
            );
            enc.commit_and_wait().expect("commit");
        });
    });
}

fn bench_chunk_scan_pipeline(c: &mut Criterion) {
    let device = match MlxDevice::new() {
        Ok(d) => d,
        Err(_) => {
            eprintln!("No Metal device available, skipping chunk-scan pipeline benchmarks");
            return;
        }
    };
    let mut registry = KernelRegistry::new();

    bench_inter_state(c, &device, &mut registry);
    bench_kkt(c, &device, &mut registry);
    bench_recompute_wu(c, &device, &mut registry);
    bench_chunk_o(c, &device, &mut registry);
    bench_tri_solve_invert(c, &device, &mut registry);
    bench_cumsum_g(c, &device, &mut registry);
}

criterion_group!(benches, bench_chunk_scan_pipeline);
criterion_main!(benches);
