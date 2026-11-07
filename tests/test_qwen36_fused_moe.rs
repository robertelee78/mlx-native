//! Correctness gates for the Qwen3.6 fused MoE decode kernels (ADR-063
//! Q4-QWEN36 F5):
//! - `quantized_matmul_id_ggml_mv_q4_k_nr2_swiglu` equals, bitwise, the
//!   unfused sequence it replaces: `quantized_matmul_id_ggml_mv_q4_k_nr2` on
//!   the gate weight, the same on the up weight, then `dispatch_silu_mul`;
//! - `quantized_matmul_id_ggml_mv_nr2_moe_reduce` (Q4_K and Q6_K) equals,
//!   bitwise, the down mv_id it replaces (Q4_K: nr2; Q6_K: the default
//!   `kernel_mul_mv_id_q6_K_f32_nr2` route) over `rows * top_k` one-row
//!   pairs followed by `dispatch_moe_weighted_reduce`, with and without the
//!   residual;
//!
//! both at the Qwen3.6 shapes (gate/up 512 x 2048, down 2048 x 512, top-8,
//! w = 1..4) and at row tails covering every Q4_K row-group fold, for edge
//! activations, shared and invalid expert ids;
//! - row independence, bitwise: every token row of a w-row dispatch equals
//!   the one-row dispatch of that row, for permuted row and slot orders,
//!   other rows' experts replaced, and a NaN/inf row elsewhere;
//! - argument validation.
//!
//! Since the unfused kernels carry their own F64 gates
//! (tests/test_mul_mv_id_q4_k_nr2.rs and the Q6_K mv_id suites), bitwise
//! equality makes the fused route a Class A change.

#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic, clippy::too_many_arguments)]

#[path = "support/gemma_expert_data.rs"]
mod gemma_expert_data;

use gemma_expert_data::*;
use mlx_native::ops::moe_weighted_reduce::dispatch_moe_weighted_reduce;
use mlx_native::ops::silu_mul::dispatch_silu_mul;
use mlx_native::{
    quantized_matmul_id_ggml_mv_nr2_moe_reduce, quantized_matmul_id_ggml_mv_q4_k_nr2,
    quantized_matmul_id_ggml_mv_q4_k_nr2_swiglu, quantized_matmul_id_ggml_mv_with_policy, DType,
    GgmlRoutingPolicy, GgmlType, KernelRegistry, MlxBuffer, MlxDevice, MoeDownReduceOperands,
};

const HIDDEN: usize = 2048;
const MOE_I: usize = 512;
const TOP_K: usize = 8;
const N_EXPERTS: usize = 48;

/// Q6_K experts (`[n_experts + 1, n, k]`, one padding expert) at a non-zero
/// offset, with random quants and scales and the support module's f16 `d`.
fn make_q6_k_weights(device: &MlxDevice, n: usize, k: usize, n_experts: usize, seed: u64) -> Weights {
    const BLOCK: usize = 210;
    let stride = n * (k / 256) * BLOCK;
    let lead = 4 * GUARD;
    let total = lead + (n_experts + 1) * stride;
    let mut parent = device.alloc_buffer(total, DType::U8, vec![total]).expect("q6 weights");
    {
        let bytes = parent.as_mut_slice::<u8>().expect("q6 slice");
        let mut rng = Rng(seed);
        bytes[..lead].fill(0xA5);
        for (index, block) in bytes[lead..].chunks_mut(BLOCK).enumerate() {
            for chunk in block[..208].chunks_mut(8) {
                let r = rng.next().to_le_bytes();
                chunk.copy_from_slice(&r[..chunk.len()]);
            }
            block[208..210].copy_from_slice(&half_scale_bits(&mut rng, index).to_le_bytes());
        }
    }
    let view = parent.slice_view(lead as u64, n_experts * stride);
    Weights { _parent: parent, view, ty: GgmlType::Q6_K, n, k, n_experts, stride }
}

fn weights(device: &MlxDevice, ty: GgmlType, n: usize, k: usize, seed: u64) -> Weights {
    match ty {
        GgmlType::Q4_K => make_weights(device, ty, n, k, N_EXPERTS, seed),
        GgmlType::Q6_K => make_q6_k_weights(device, n, k, N_EXPERTS, seed),
        other => panic!("unsupported {other:?}"),
    }
}

fn f32_buf(device: &MlxDevice, data: &[f32]) -> MlxBuffer {
    let mut b = device.alloc_buffer(data.len().max(1) * 4, DType::F32, vec![data.len().max(1)]).expect("f32");
    b.as_mut_slice::<f32>().expect("f32 slice")[..data.len()].copy_from_slice(data);
    b
}

fn bits(buf: &MlxBuffer, n: usize) -> Vec<u32> {
    buf.as_slice::<u32>().expect("bits")[..n].to_vec()
}

fn random_ids(rng: &mut Rng, rows: usize, top_k: usize) -> Vec<u32> {
    let pool: Vec<u32> = (0..N_EXPERTS as u32).collect();
    (0..rows).flat_map(|_| pick_distinct(rng, &pool, top_k)).collect()
}

fn first_difference(a: &[u32], b: &[u32]) -> Option<usize> {
    (0..a.len()).find(|&i| a[i] != b[i] && !(f32::from_bits(a[i]).is_nan() && f32::from_bits(b[i]).is_nan()))
}

struct Harness {
    device: MlxDevice,
    registry: KernelRegistry,
}

impl Harness {
    fn new() -> Self {
        Self { device: MlxDevice::new().expect("Metal device"), registry: KernelRegistry::new() }
    }

    /// Fused gate + up + SwiGLU; returns the guarded `[rows * top_k, n]`
    /// output (raw u32).
    fn swiglu_fused(&mut self, gate: &Weights, up: &Weights, rows: usize, top_k: usize, ids: &[u32], x: &[f32]) -> Vec<u32> {
        let pairs = rows * top_k;
        let (_xp, xv) = guarded_f32(&self.device, x);
        let (_ip, iv) = guarded_u32(&self.device, ids);
        let (op, ov) = guarded_f32(&self.device, &vec![f32::from_bits(SENTINEL); pairs * gate.n]);
        let mut enc = self.device.command_encoder().expect("encoder");
        quantized_matmul_id_ggml_mv_q4_k_nr2_swiglu(
            &mut enc, &mut self.registry, &self.device, &xv, &gate.view, &up.view, &iv, &ov,
            &params(gate, rows, top_k),
        )
        .expect("fused swiglu");
        enc.commit_and_wait().expect("GPU");
        op.as_slice::<u32>().expect("out").to_vec()
    }

    /// nr2 gate, nr2 up, silu_mul (the F2b sequence); `[pairs, n]` raw.
    fn swiglu_unfused(&mut self, gate: &Weights, up: &Weights, rows: usize, top_k: usize, ids: &[u32], x: &[f32]) -> Vec<u32> {
        let pairs = rows * top_k;
        let n = gate.n;
        let xb = f32_buf(&self.device, x);
        let (_ip, iv) = guarded_u32(&self.device, ids);
        let g = f32_buf(&self.device, &vec![0.0; pairs * n]);
        let u = f32_buf(&self.device, &vec![0.0; pairs * n]);
        let h = f32_buf(&self.device, &vec![0.0; pairs * n]);
        let mut pbuf = self.device.alloc_buffer(4, DType::U32, vec![1]).expect("params");
        pbuf.as_mut_slice::<u32>().expect("p")[0] = (pairs * n) as u32;
        let mut enc = self.device.command_encoder().expect("encoder");
        let p = params(gate, rows, top_k);
        quantized_matmul_id_ggml_mv_q4_k_nr2(&mut enc, &mut self.registry, &self.device, &xb, &gate.view, &iv, &g, &p)
            .expect("gate");
        quantized_matmul_id_ggml_mv_q4_k_nr2(&mut enc, &mut self.registry, &self.device, &xb, &up.view, &iv, &u, &p)
            .expect("up");
        enc.memory_barrier();
        dispatch_silu_mul(&mut enc, &mut self.registry, self.device.metal_device(), &g, &u, &h, &pbuf, (pairs * n) as u32)
            .expect("silu");
        enc.commit_and_wait().expect("GPU");
        bits(&h, pairs * n)
    }

    /// Fused down + reduce; returns the guarded `[rows, n]` output (raw).
    fn reduce_fused(&mut self, down: &Weights, rows: usize, top_k: usize, ids: &[u32], ops: &ReduceInputs) -> Vec<u32> {
        let n = down.n;
        let (_hp, hv) = guarded_f32(&self.device, &ops.h);
        let (_ip, iv) = guarded_u32(&self.device, ids);
        let (_rp, rv) = guarded_f32(&self.device, &ops.route_w);
        let (_sp, sv) = guarded_f32(&self.device, &ops.sh_logit);
        let (_yp, yv) = guarded_f32(&self.device, &ops.y_shared);
        let (_resp, resv) = guarded_f32(&self.device, &ops.residual);
        let (op, ov) = guarded_f32(&self.device, &vec![f32::from_bits(SENTINEL); rows * n]);
        let mut enc = self.device.command_encoder().expect("encoder");
        let operands = MoeDownReduceOperands {
            route_weights: &rv,
            sh_logit: &sv,
            y_shared: &yv,
            residual: ops.add_residual.then_some(&resv),
        };
        quantized_matmul_id_ggml_mv_nr2_moe_reduce(
            &mut enc, &mut self.registry, &self.device, &hv, &down.view, &iv, &operands, &ov,
            &params(down, rows, top_k),
        )
        .expect("fused down reduce");
        enc.commit_and_wait().expect("GPU");
        op.as_slice::<u32>().expect("out").to_vec()
    }

    /// Down mv_id over `rows * top_k` one-row pairs, then the weighted reduce.
    fn reduce_unfused(&mut self, down: &Weights, rows: usize, top_k: usize, ids: &[u32], ops: &ReduceInputs) -> Vec<u32> {
        let n = down.n;
        let pairs = rows * top_k;
        let hb = f32_buf(&self.device, &ops.h);
        let (_ip, iv) = guarded_u32(&self.device, ids);
        let y = f32_buf(&self.device, &vec![0.0; pairs * n]);
        let rw = f32_buf(&self.device, &ops.route_w);
        let sl = f32_buf(&self.device, &ops.sh_logit);
        let ys = f32_buf(&self.device, &ops.y_shared);
        let res = f32_buf(&self.device, &ops.residual);
        let mut out = f32_buf(&self.device, &vec![0.0; rows * n]);
        let mut enc = self.device.command_encoder().expect("encoder");
        let p = params(down, pairs, 1);
        if down.ty == GgmlType::Q4_K {
            quantized_matmul_id_ggml_mv_q4_k_nr2(&mut enc, &mut self.registry, &self.device, &hb, &down.view, &iv, &y, &p)
                .expect("down");
        } else {
            quantized_matmul_id_ggml_mv_with_policy(
                &mut enc, &mut self.registry, &self.device, &hb, &down.view, &iv, &y, &p,
                &GgmlRoutingPolicy::default(),
            )
            .expect("down");
        }
        enc.memory_barrier();
        dispatch_moe_weighted_reduce(
            &mut enc, &mut self.registry, &self.device, &rw, &y, &sl, &ys, &res, &mut out, rows as u32,
            top_k as u32, n as u32, ops.add_residual,
        )
        .expect("reduce");
        enc.commit_and_wait().expect("GPU");
        bits(&out, rows * n)
    }
}

#[derive(Clone)]
struct ReduceInputs {
    h: Vec<f32>,
    route_w: Vec<f32>,
    sh_logit: Vec<f32>,
    y_shared: Vec<f32>,
    residual: Vec<f32>,
    add_residual: bool,
}

impl ReduceInputs {
    fn new(rows: usize, top_k: usize, n: usize, k: usize, add_residual: bool, seed: u64) -> Self {
        let mut rng = Rng(seed);
        let mut route_w = Vec::with_capacity(rows * top_k);
        for _ in 0..rows {
            let raw: Vec<f32> = (0..top_k).map(|_| 0.05 + rng.unit().abs()).collect();
            let s: f32 = raw.iter().sum();
            route_w.extend(raw.iter().map(|v| v / s));
        }
        Self {
            h: make_input(rows * top_k, k, seed ^ 0x11),
            route_w,
            sh_logit: (0..rows).map(|_| rng.unit() * 6.0 - 3.0).collect(),
            y_shared: (0..rows * n).map(|_| rng.unit() * 2.0 - 1.0).collect(),
            residual: (0..rows * n).map(|_| rng.unit() * 4.0 - 2.0).collect(),
            add_residual,
        }
    }

    /// The operands of row `t` alone.
    fn row(&self, t: usize, top_k: usize, n: usize, k: usize) -> Self {
        Self {
            h: self.h[t * top_k * k..(t + 1) * top_k * k].to_vec(),
            route_w: self.route_w[t * top_k..(t + 1) * top_k].to_vec(),
            sh_logit: vec![self.sh_logit[t]],
            y_shared: self.y_shared[t * n..(t + 1) * n].to_vec(),
            residual: self.residual[t * n..(t + 1) * n].to_vec(),
            add_residual: self.add_residual,
        }
    }

    /// Rows reordered by `order` (slots of each row permuted by `slots[t]`).
    fn permuted(&self, order: &[usize], slots: &[Vec<usize>], top_k: usize, n: usize, k: usize) -> Self {
        let mut out = Self {
            h: Vec::new(),
            route_w: Vec::new(),
            sh_logit: Vec::new(),
            y_shared: Vec::new(),
            residual: Vec::new(),
            add_residual: self.add_residual,
        };
        for (i, &t) in order.iter().enumerate() {
            for &s in &slots[i] {
                let p = t * top_k + s;
                out.h.extend_from_slice(&self.h[p * k..(p + 1) * k]);
                out.route_w.push(self.route_w[p]);
            }
            out.sh_logit.push(self.sh_logit[t]);
            out.y_shared.extend_from_slice(&self.y_shared[t * n..(t + 1) * n]);
            out.residual.extend_from_slice(&self.residual[t * n..(t + 1) * n]);
        }
        out
    }
}

fn shuffle<T>(rng: &mut Rng, v: &mut [T]) {
    for i in (1..v.len()).rev() {
        v.swap(i, rng.below(i as u64 + 1) as usize);
    }
}

// ---- gate + up + SwiGLU ----

#[test]
fn swiglu_equals_unfused_sequence_bitwise() {
    let mut h = Harness::new();
    let mut rng = Rng(0xF5_0001);
    // (N, K, rows, top_k)
    let cases = [
        (MOE_I, HIDDEN, 1, TOP_K),
        (MOE_I, HIDDEN, 2, TOP_K),
        (MOE_I, HIDDEN, 3, TOP_K),
        (MOE_I, HIDDEN, 4, TOP_K),
        (7, 768, 3, 2),
        (5, 256, 2, 3),
        (6, 512, 2, 2),
        (13, 512, 3, 2),
        (11, 1024, 2, 2),
    ];
    for (c, &(n, k, rows, top_k)) in cases.iter().enumerate() {
        let gate = weights(&h.device, GgmlType::Q4_K, n, k, 0x6a7e + c as u64);
        let up = weights(&h.device, GgmlType::Q4_K, n, k, 0x0b0b + c as u64);
        let x = make_input(rows, k, 0x1234 + c as u64);
        let mut id_sets = vec![random_ids(&mut rng, rows, top_k)];
        let mut shared = id_sets[0].clone();
        for t in 1..rows {
            shared.copy_within(0..top_k, t * top_k);
        }
        id_sets.push(shared);
        let mut invalid = id_sets[0].clone();
        invalid[0] = INVALID_MAX;
        invalid[rows * top_k - 1] = N_EXPERTS as u32;
        id_sets.push(invalid);
        for ids in &id_sets {
            let label = format!("swiglu N={n} K={k} rows={rows} ids={ids:?}");
            let fused = h.swiglu_fused(&gate, &up, rows, top_k, ids, &x);
            assert_stock_frame(&fused, rows * top_k, n, &label);
            let want = h.swiglu_unfused(&gate, &up, rows, top_k, ids, &x);
            let got = &fused[GUARD..GUARD + rows * top_k * n];
            if let Some(i) = first_difference(got, &want) {
                panic!("{label}: element {i} (pair {}, col {}): fused {:#010x}, unfused {:#010x}", i / n, i % n, got[i], want[i]);
            }
            for (p, &id) in ids.iter().enumerate() {
                if id as usize >= N_EXPERTS {
                    assert!(got[p * n..(p + 1) * n].iter().all(|&b| f32::from_bits(b).is_nan()), "{label}: invalid pair {p} not NaN");
                }
            }
        }
    }
}

#[test]
fn swiglu_rows_independent_bitwise() {
    let mut h = Harness::new();
    let mut rng = Rng(0xF5_0002);
    let (n, k) = (MOE_I, HIDDEN);
    let gate = weights(&h.device, GgmlType::Q4_K, n, k, 0x91);
    let up = weights(&h.device, GgmlType::Q4_K, n, k, 0x92);
    let x = make_input(4, k, 0x93);
    let ids = random_ids(&mut rng, 4, TOP_K);
    let single = |h: &mut Harness, t: usize, slot_ids: &[u32]| -> Vec<u32> {
        let out = h.swiglu_fused(&gate, &up, 1, TOP_K, slot_ids, &x[t * k..(t + 1) * k]);
        out[GUARD..GUARD + TOP_K * n].to_vec()
    };
    let singles: Vec<Vec<u32>> = (0..4).map(|t| single(&mut h, t, &ids[t * TOP_K..(t + 1) * TOP_K])).collect();
    let mut checked = 0;
    for rows in 1..=4 {
        let out = h.swiglu_fused(&gate, &up, rows, TOP_K, &ids[..rows * TOP_K], &x[..rows * k]);
        for t in 0..rows {
            assert_eq!(&out[GUARD + t * TOP_K * n..GUARD + (t + 1) * TOP_K * n], &singles[t][..], "rows={rows} row {t}");
            checked += 1;
        }
        // Permuted rows and slots.
        let mut order: Vec<usize> = (0..rows).collect();
        shuffle(&mut rng, &mut order);
        let mut px = Vec::new();
        let mut pids = Vec::new();
        let mut perms = Vec::new();
        for &t in &order {
            px.extend_from_slice(&x[t * k..(t + 1) * k]);
            let mut s: Vec<usize> = (0..TOP_K).collect();
            shuffle(&mut rng, &mut s);
            pids.extend(s.iter().map(|&j| ids[t * TOP_K + j]));
            perms.push(s);
        }
        let out = h.swiglu_fused(&gate, &up, rows, TOP_K, &pids, &px);
        for (i, &t) in order.iter().enumerate() {
            for (j, &s) in perms[i].iter().enumerate() {
                let p = i * TOP_K + j;
                assert_eq!(
                    &out[GUARD + p * n..GUARD + (p + 1) * n],
                    &singles[t][s * n..(s + 1) * n],
                    "rows={rows} permuted row {t} slot {s}"
                );
            }
            checked += 1;
        }
        // Other rows: fresh experts, then row 0's experts, then a NaN/inf row.
        if rows > 1 {
            let mut other = ids[..rows * TOP_K].to_vec();
            other[TOP_K..].copy_from_slice(&random_ids(&mut rng, rows - 1, TOP_K));
            let out = h.swiglu_fused(&gate, &up, rows, TOP_K, &other, &x[..rows * k]);
            assert_eq!(&out[GUARD..GUARD + TOP_K * n], &singles[0][..], "rows={rows} others redrawn");
            let mut same = ids[..rows * TOP_K].to_vec();
            for t in 1..rows {
                same.copy_within(0..TOP_K, t * TOP_K);
            }
            let out = h.swiglu_fused(&gate, &up, rows, TOP_K, &same, &x[..rows * k]);
            assert_eq!(&out[GUARD..GUARD + TOP_K * n], &singles[0][..], "rows={rows} others share experts");
            let mut poisoned = x[..rows * k].to_vec();
            poison_row(&mut poisoned, rows - 1, k);
            let out = h.swiglu_fused(&gate, &up, rows, TOP_K, &ids[..rows * TOP_K], &poisoned);
            for t in 0..rows - 1 {
                assert_eq!(&out[GUARD + t * TOP_K * n..GUARD + (t + 1) * TOP_K * n], &singles[t][..], "rows={rows} NaN row elsewhere");
            }
            checked += 3;
        }
    }
    eprintln!("[f5 swiglu independence] {checked} row checks bitwise equal to one-row dispatches");
}

// ---- expert down + weighted reduce ----

fn reduce_equals_unfused(ty: GgmlType) {
    let mut h = Harness::new();
    let mut rng = Rng(0xF5_0003 ^ u64::from(ty == GgmlType::Q6_K));
    let mut cases = vec![
        (HIDDEN, MOE_I, 1, TOP_K),
        (HIDDEN, MOE_I, 2, TOP_K),
        (HIDDEN, MOE_I, 3, TOP_K),
        (HIDDEN, MOE_I, 4, TOP_K),
        (6, 512, 2, 3),
        (10, 256, 2, 2),
        (8, 768, 3, 5),
    ];
    if ty == GgmlType::Q4_K {
        cases.extend([(13, 512, 3, 2), (5, 256, 2, 3), (11, 1024, 2, 2)]);
    }
    for (c, &(n, k, rows, top_k)) in cases.iter().enumerate() {
        let down = weights(&h.device, ty, n, k, 0xd0 + c as u64);
        let mut id_sets = vec![random_ids(&mut rng, rows, top_k)];
        let mut shared = id_sets[0].clone();
        for t in 1..rows {
            shared.copy_within(0..top_k, t * top_k);
        }
        id_sets.push(shared);
        let mut invalid = id_sets[0].clone();
        invalid[rows * top_k - 1] = INVALID_MAX;
        id_sets.push(invalid);
        for add_residual in [true, false] {
            let ops = ReduceInputs::new(rows, top_k, n, k, add_residual, 0x77 + c as u64);
            for ids in &id_sets {
                let label = format!("{ty:?} down+reduce N={n} K={k} rows={rows} top_k={top_k} residual={add_residual} ids={ids:?}");
                let fused = h.reduce_fused(&down, rows, top_k, ids, &ops);
                assert_stock_frame(&fused, rows, n, &label);
                let want = h.reduce_unfused(&down, rows, top_k, ids, &ops);
                let got = &fused[GUARD..GUARD + rows * n];
                if let Some(i) = first_difference(got, &want) {
                    panic!("{label}: element {i} (row {}, col {}): fused {:#010x}, unfused {:#010x}", i / n, i % n, got[i], want[i]);
                }
                if ids[rows * top_k - 1] as usize >= N_EXPERTS {
                    assert!(got[(rows - 1) * n..].iter().all(|&b| f32::from_bits(b).is_nan()), "{label}: invalid id row not NaN");
                }
            }
        }
    }
}

#[test]
fn down_reduce_q4_k_equals_unfused_sequence_bitwise() {
    reduce_equals_unfused(GgmlType::Q4_K);
}

#[test]
fn down_reduce_q6_k_equals_unfused_sequence_bitwise() {
    reduce_equals_unfused(GgmlType::Q6_K);
}

fn reduce_rows_independent(ty: GgmlType) {
    let mut h = Harness::new();
    let mut rng = Rng(0xF5_0004 ^ u64::from(ty == GgmlType::Q6_K));
    let (n, k) = (HIDDEN, MOE_I);
    let down = weights(&h.device, ty, n, k, 0xe1);
    let ids = random_ids(&mut rng, 4, TOP_K);
    let ops = ReduceInputs::new(4, TOP_K, n, k, true, 0xe2);
    let singles: Vec<Vec<u32>> = (0..4)
        .map(|t| {
            let out = h.reduce_fused(&down, 1, TOP_K, &ids[t * TOP_K..(t + 1) * TOP_K], &ops.row(t, TOP_K, n, k));
            out[GUARD..GUARD + n].to_vec()
        })
        .collect();
    let mut checked = 0;
    for rows in 1..=4 {
        let sub = ReduceInputs {
            h: ops.h[..rows * TOP_K * k].to_vec(),
            route_w: ops.route_w[..rows * TOP_K].to_vec(),
            sh_logit: ops.sh_logit[..rows].to_vec(),
            y_shared: ops.y_shared[..rows * n].to_vec(),
            residual: ops.residual[..rows * n].to_vec(),
            add_residual: true,
        };
        let out = h.reduce_fused(&down, rows, TOP_K, &ids[..rows * TOP_K], &sub);
        for t in 0..rows {
            assert_eq!(&out[GUARD + t * n..GUARD + (t + 1) * n], &singles[t][..], "{ty:?} rows={rows} row {t}");
            checked += 1;
        }
        // Permuted rows (slot order within a row is the reduce order, kept).
        let mut order: Vec<usize> = (0..rows).collect();
        shuffle(&mut rng, &mut order);
        let keep: Vec<Vec<usize>> = order.iter().map(|_| (0..TOP_K).collect()).collect();
        let pops = ops.permuted(&order, &keep, TOP_K, n, k);
        let pids: Vec<u32> = order.iter().flat_map(|&t| ids[t * TOP_K..(t + 1) * TOP_K].iter().copied()).collect();
        let out = h.reduce_fused(&down, rows, TOP_K, &pids, &pops);
        for (i, &t) in order.iter().enumerate() {
            assert_eq!(&out[GUARD + i * n..GUARD + (i + 1) * n], &singles[t][..], "{ty:?} rows={rows} permuted row {t}");
            checked += 1;
        }
        if rows > 1 {
            // Other rows: redrawn experts, row 0's experts, NaN/inf input.
            let mut other = ids[..rows * TOP_K].to_vec();
            other[TOP_K..].copy_from_slice(&random_ids(&mut rng, rows - 1, TOP_K));
            let out = h.reduce_fused(&down, rows, TOP_K, &other, &sub);
            assert_eq!(&out[GUARD..GUARD + n], &singles[0][..], "{ty:?} rows={rows} others redrawn");
            let mut same = ids[..rows * TOP_K].to_vec();
            for t in 1..rows {
                same.copy_within(0..TOP_K, t * TOP_K);
            }
            let out = h.reduce_fused(&down, rows, TOP_K, &same, &sub);
            assert_eq!(&out[GUARD..GUARD + n], &singles[0][..], "{ty:?} rows={rows} others share experts");
            let mut poisoned = sub.clone();
            poison_row(&mut poisoned.h, rows * TOP_K - 1, k);
            let out = h.reduce_fused(&down, rows, TOP_K, &ids[..rows * TOP_K], &poisoned);
            for t in 0..rows - 1 {
                assert_eq!(&out[GUARD + t * n..GUARD + (t + 1) * n], &singles[t][..], "{ty:?} rows={rows} NaN row elsewhere");
            }
            checked += 3;
        }
    }
    eprintln!("[f5 down+reduce independence] {ty:?}: {checked} row checks bitwise equal to one-row dispatches");
}

#[test]
fn down_reduce_q4_k_rows_independent_bitwise() {
    reduce_rows_independent(GgmlType::Q4_K);
}

#[test]
fn down_reduce_q6_k_rows_independent_bitwise() {
    reduce_rows_independent(GgmlType::Q6_K);
}

// ---- validation ----

#[test]
fn rejects_unsupported_arguments() {
    let mut h = Harness::new();
    let q4 = weights(&h.device, GgmlType::Q4_K, 8, 256, 1);
    let q6 = weights(&h.device, GgmlType::Q6_K, 8, 256, 2);
    let x = f32_buf(&h.device, &vec![0.0; 256 * 64]);
    let ids = {
        let mut b = h.device.alloc_buffer(64 * 4, DType::U32, vec![64]).expect("ids");
        b.as_mut_slice::<u32>().expect("ids")[..].fill(0);
        b
    };
    let out = f32_buf(&h.device, &vec![0.0; 8 * 64]);
    let small = f32_buf(&h.device, &[0.0]);
    let mut enc = h.device.command_encoder().expect("encoder");
    // SwiGLU: Q6_K weights refused.
    assert!(quantized_matmul_id_ggml_mv_q4_k_nr2_swiglu(
        &mut enc, &mut h.registry, &h.device, &x, &q6.view, &q6.view, &ids, &out, &params(&q6, 1, 2)
    )
    .is_err());
    let ok = MoeDownReduceOperands { route_weights: &out, sh_logit: &out, y_shared: &out, residual: None };
    // top_k above 32.
    let mut p = params(&q4, 1, 2);
    p.top_k = 33;
    p.n_experts = 64;
    assert!(quantized_matmul_id_ggml_mv_nr2_moe_reduce(&mut enc, &mut h.registry, &h.device, &x, &q4.view, &ids, &ok, &out, &p).is_err());
    // Short operands.
    for (label, operands) in [
        ("route", MoeDownReduceOperands { route_weights: &small, ..ok }),
        ("sh_logit", MoeDownReduceOperands { sh_logit: &small, ..ok }),
        ("y_shared", MoeDownReduceOperands { y_shared: &small, ..ok }),
        ("residual", MoeDownReduceOperands { residual: Some(&small), ..ok }),
    ] {
        assert!(
            quantized_matmul_id_ggml_mv_nr2_moe_reduce(
                &mut enc, &mut h.registry, &h.device, &x, &q4.view, &ids, &operands, &out, &params(&q4, 2, 2)
            )
            .is_err(),
            "{label} accepted"
        );
    }
    // Short output ([rows, n] needed).
    assert!(quantized_matmul_id_ggml_mv_nr2_moe_reduce(
        &mut enc, &mut h.registry, &h.device, &x, &q4.view, &ids, &ok, &small, &params(&q4, 2, 2)
    )
    .is_err());
    // Q8_0 refused.
    let mut p8 = params(&q4, 1, 2);
    p8.ggml_type = GgmlType::Q8_0;
    assert!(quantized_matmul_id_ggml_mv_nr2_moe_reduce(&mut enc, &mut h.registry, &h.device, &x, &q4.view, &ids, &ok, &out, &p8).is_err());
}
