//! Cold-prefill attention microbenchmark at Gemma 4 served shapes: hf2q's
//! per-layer cold attention dispatch sequence for one sequence of `rows`
//! tokens, timed op-for-op so it can be compared with the peer's flash
//! attention on identical shapes.
//!
//! Loads no model, but times the GPU: run it only with the GPU lane and a
//! quiet host.  Each arm is timed as the full per-layer chain hf2q encodes
//! for that route (`chain`, scratch allocated once per prefill exactly as
//! the route does) and as its attention dispatches alone (`attn`, inputs
//! prepared once):
//!
//! | Kind | Arm | `chain` | `attn` |
//! |---|---|---|---|
//! | sliding (nh 16, nkv 8, d256, window 1024) | `fa_f16` | Q/K/V `permute_021_f32_to_f16`, mask `cast` BF16→F16, F16 d256 FA with blk, O `cast` F16→BF16 | the FA dispatch |
//! | global (nh 16, nkv 2, d512, causal) | `tensor_mm` (served for `64 < rows <= 8192`) | Q·Kᵀ `dense_matmul_bf16_f32_tensor`, in-place `scale_mask_softmax_f32`, V `transpose_last2_bf16`, scores·V `dense_matmul_bf16_f32_tensor`, O `permute_021_f32` | Q·Kᵀ, softmax, scores·V |
//! | global | `fa_f16_d512` (served only for `rows <= 64`; candidate) | as sliding `fa_f16` with the F16 d512 FA and the global mask/blk | the FA dispatch |
//!
//! Per kind the once-per-prefill `setup` is timed separately (`chain`: BF16
//! mask fill, its F16 copy, blk classifier; `blk`: the blk classifier
//! alone): sliding blk tiles (32, 16), global blk tiles (8, 64), both
//! classified from the BF16 mask as hf2q does.  Masks are the square
//! `[rows, rows]` single-sequence masks hf2q builds (`causal`, sliding also
//! `q - k < 1024`), scale 1.0, `do_causal = false` (the mask carries
//! causality).  Inputs are seeded standard-normal values in each kernel's
//! dtype: seq-major F32 Q `[rows, nh, hd]` and K/V `[rows, nkv, hd]` for
//! the FA arms, head-major F32 Q `[nh, rows, hd]` and BF16 K/V
//! `[nkv, rows, hd]` for `tensor_mm`.
//!
//! Cells: `rows` in {256, 458, 1024, 2048} for each kind;
//! `BENCH_ROWS=<comma list>` overrides.
//!
//! Method: per cell 10 warmup and 50 measured command buffers per arm (one
//! layer per command buffer), arms interleaved forward then reverse, GPU
//! timestamps; median and p90 in microseconds.  `BENCH_OUT=<path>` writes
//! the JSON receipt.  The receipt's commit is `GIT_COMMIT_SHA` from the run
//! environment, else `git rev-parse HEAD` of the crate checkout.  Run once
//! per metallib path (`MLX_PRECOMPILED_METALLIB=0` for the runtime-source
//! path).

use mlx_native::ops::dense_mm_bf16::{dense_matmul_bf16_f32_tensor, DenseMmBf16F32Params};
use mlx_native::ops::elementwise::{cast, CastDirection};
use mlx_native::ops::flash_attn_prefill::{self, FlashAttnPrefillParams};
use mlx_native::ops::flash_attn_prefill_blk::{self, alloc_blk_buffer, BlkParams};
use mlx_native::ops::flash_attn_prefill_d512;
use mlx_native::ops::flash_attn_prefill_mask::{self, SdpaMaskParams};
use mlx_native::ops::scale_mask_softmax::{dispatch_scale_mask_softmax_f32, ScaleMaskSoftmaxParams};
use mlx_native::ops::transpose::{permute_021_f32, permute_021_f32_to_f16, transpose_last2_bf16};
use mlx_native::{CommandEncoder, DType, KernelRegistry, MlxBuffer, MlxDevice};
use sha2::{Digest, Sha256};

const WARMUP: usize = 10;
const MEASURED: usize = 50;
const N_HEADS: usize = 16;
const SLIDING_KV_HEADS: usize = 8;
const GLOBAL_KV_HEADS: usize = 2;
const SLIDING_WINDOW: u32 = 1024;
const DEFAULT_ROWS: [usize; 4] = [256, 458, 1024, 2048];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Sliding,
    Global,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Sliding => "sliding",
            Kind::Global => "global",
        }
    }
    fn n_kv(self) -> usize {
        match self {
            Kind::Sliding => SLIDING_KV_HEADS,
            Kind::Global => GLOBAL_KV_HEADS,
        }
    }
    fn head_dim(self) -> usize {
        match self {
            Kind::Sliding => 256,
            Kind::Global => 512,
        }
    }
    fn window(self) -> Option<u32> {
        match self {
            Kind::Sliding => Some(SLIDING_WINDOW),
            Kind::Global => None,
        }
    }
    /// hf2q's blk tile shape: the main FA kernel's (BQ, BK).
    fn blk_params(self, rows: usize) -> BlkParams {
        let (bq, bk) = match self {
            Kind::Sliding => (32, 16),
            Kind::Global => (8, 64),
        };
        BlkParams {
            seq_len_q: rows as u32,
            seq_len_k: rows as u32,
            bq,
            bk,
        }
    }
    fn arms(self) -> &'static [Arm] {
        match self {
            Kind::Sliding => &[Arm::FaF16],
            Kind::Global => &[Arm::TensorMm, Arm::FaF16D512],
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    FaF16,
    TensorMm,
    FaF16D512,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Arm::FaF16 => "fa_f16",
            Arm::TensorMm => "tensor_mm",
            Arm::FaF16D512 => "fa_f16_d512",
        }
    }
}

/// splitmix64 + Box-Muller standard normals.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f32 {
        let (u1, u2) = (self.uniform(), self.uniform());
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }
}

fn alloc(dev: &MlxDevice, dtype: DType, n: usize) -> MlxBuffer {
    // SAFETY: every scratch buffer is fully written by the dispatch that
    // produces it before any dispatch reads it.
    unsafe { dev.alloc_buffer_for_overwrite(n.max(1) * dtype.size_of(), dtype, vec![n.max(1)]) }
        .unwrap()
}

fn upload_f32(dev: &MlxDevice, rng: &mut Rng, n: usize) -> MlxBuffer {
    let mut b = dev
        .alloc_buffer(n * DType::F32.size_of(), DType::F32, vec![n])
        .unwrap();
    for x in b.as_mut_slice::<f32>().unwrap() {
        *x = rng.normal();
    }
    b
}

fn upload_bf16(dev: &MlxDevice, rng: &mut Rng, n: usize) -> MlxBuffer {
    let mut b = dev
        .alloc_buffer(n * DType::BF16.size_of(), DType::BF16, vec![n])
        .unwrap();
    for x in b.as_mut_slice::<half::bf16>().unwrap() {
        *x = half::bf16::from_f32(rng.normal());
    }
    b
}

/// One cell's persistent inputs, per-prefill masks and per-prefill scratch.
struct Cell {
    kind: Kind,
    rows: usize,
    /// Seq-major F32 normed rows the FA route permutes: Q `[rows, nh, hd]`,
    /// K/V `[rows, nkv, hd]`.
    q_seq: MlxBuffer,
    k_seq: MlxBuffer,
    v_seq: MlxBuffer,
    /// Tensor-mm inputs (global only): head-major F32 Q `[nh, rows, hd]`,
    /// BF16 K/V `[nkv, rows, hd]`.
    q_hm_f32: Option<MlxBuffer>,
    k_hm_bf16: Option<MlxBuffer>,
    v_hm_bf16: Option<MlxBuffer>,
    /// BF16 `[rows, rows]` mask, its rank-2 F16 copy, and the blk bytes.
    mask_bf16: MlxBuffer,
    mask_f16: MlxBuffer,
    blk: MlxBuffer,
    blk_params: BlkParams,
    /// FA scratch (head-major F16 Q/K/V/O, BF16 O).
    q_f16: MlxBuffer,
    k_f16: MlxBuffer,
    v_f16: MlxBuffer,
    out_f16: MlxBuffer,
    out_bf16: MlxBuffer,
    /// Tensor-mm scratch (global only): kq `[nh, rows, rows]`, V_t
    /// `[nkv, hd, rows]`, attn `[nh, rows, hd]`, sdpa_out `[rows, nh, hd]`.
    kq: Option<MlxBuffer>,
    v_t: Option<MlxBuffer>,
    attn_f32: Option<MlxBuffer>,
    sdpa_out: Option<MlxBuffer>,
}

/// Encode hf2q's per-prefill mask setup for one sequence: BF16 mask fill,
/// F16 copy (rank 2), blk classifier over the BF16 mask.
fn encode_setup(
    dev: &MlxDevice,
    reg: &mut KernelRegistry,
    enc: &mut CommandEncoder,
    kind: Kind,
    rows: usize,
    mask_f16: &MlxBuffer,
    blk: &MlxBuffer,
    blk_params: &BlkParams,
) -> MlxBuffer {
    let mask = flash_attn_prefill_mask::build_sdpa_mask_bf16(
        dev,
        reg,
        enc,
        &SdpaMaskParams {
            seq_len_q: rows as u32,
            seq_len_k: rows as u32,
            window_size: kind.window(),
            causal: true,
            q_abs_offset: 0,
        },
    )
    .unwrap();
    enc.memory_barrier();
    cast(
        enc,
        reg,
        dev.metal_device(),
        &mask,
        mask_f16,
        rows * rows,
        CastDirection::BF16ToF16,
    )
    .unwrap();
    flash_attn_prefill_blk::dispatch_flash_attn_prefill_blk(enc, dev, reg, &mask, blk, blk_params)
        .unwrap();
    enc.memory_barrier();
    mask
}

fn cell(dev: &MlxDevice, reg: &mut KernelRegistry, kind: Kind, rows: usize, seed: u64) -> Cell {
    let (nh, nkv, hd) = (N_HEADS, kind.n_kv(), kind.head_dim());
    let mut rng = Rng(seed);
    let q_seq = upload_f32(dev, &mut rng, rows * nh * hd);
    let k_seq = upload_f32(dev, &mut rng, rows * nkv * hd);
    let v_seq = upload_f32(dev, &mut rng, rows * nkv * hd);
    let global = kind == Kind::Global;
    let q_hm_f32 = global.then(|| upload_f32(dev, &mut rng, nh * rows * hd));
    let k_hm_bf16 = global.then(|| upload_bf16(dev, &mut rng, nkv * rows * hd));
    let v_hm_bf16 = global.then(|| upload_bf16(dev, &mut rng, nkv * rows * hd));

    let mask_f16 = dev
        .alloc_buffer(rows * rows * DType::F16.size_of(), DType::F16, vec![rows, rows])
        .unwrap();
    let blk_params = kind.blk_params(rows);
    let blk = alloc_blk_buffer(dev, &blk_params).unwrap();
    let mut enc = dev.command_encoder().unwrap();
    let mask_bf16 = encode_setup(dev, reg, &mut enc, kind, rows, &mask_f16, &blk, &blk_params);
    enc.commit_and_wait().unwrap();

    Cell {
        kind,
        rows,
        q_seq,
        k_seq,
        v_seq,
        q_hm_f32,
        k_hm_bf16,
        v_hm_bf16,
        mask_bf16,
        mask_f16,
        blk,
        blk_params,
        q_f16: alloc(dev, DType::F16, nh * rows * hd),
        k_f16: alloc(dev, DType::F16, nkv * rows * hd),
        v_f16: alloc(dev, DType::F16, nkv * rows * hd),
        out_f16: alloc(dev, DType::F16, nh * rows * hd),
        out_bf16: alloc(dev, DType::BF16, nh * rows * hd),
        kq: global.then(|| alloc(dev, DType::F32, nh * rows * rows)),
        v_t: global.then(|| alloc(dev, DType::BF16, nkv * hd * rows)),
        attn_f32: global.then(|| alloc(dev, DType::F32, nh * rows * hd)),
        sdpa_out: global.then(|| alloc(dev, DType::F32, rows * nh * hd)),
    }
}

fn fa_params(c: &Cell) -> FlashAttnPrefillParams {
    FlashAttnPrefillParams {
        n_heads: N_HEADS as u32,
        n_kv_heads: c.kind.n_kv() as u32,
        head_dim: c.kind.head_dim() as u32,
        seq_len_q: c.rows as u32,
        seq_len_k: c.rows as u32,
        batch: 1,
        scale: 1.0,
        do_causal: false,
    }
}

fn encode_fa(dev: &MlxDevice, reg: &mut KernelRegistry, enc: &mut CommandEncoder, c: &Cell, arm: Arm) {
    let p = fa_params(c);
    match arm {
        Arm::FaF16 => flash_attn_prefill::dispatch_flash_attn_prefill_f16_d256_with_blk(
            enc,
            dev,
            reg,
            &c.q_f16,
            &c.k_f16,
            &c.v_f16,
            Some(&c.mask_f16),
            Some(&c.blk),
            &c.out_f16,
            &p,
        ),
        Arm::FaF16D512 => flash_attn_prefill_d512::dispatch_flash_attn_prefill_f16_d512_with_blk(
            enc,
            dev,
            reg,
            &c.q_f16,
            &c.k_f16,
            &c.v_f16,
            Some(&c.mask_f16),
            Some(&c.blk),
            &c.out_f16,
            &p,
        ),
        Arm::TensorMm => unreachable!("tensor_mm has no FA dispatch"),
    }
    .unwrap();
}

fn encode_qk_softmax_pv(dev: &MlxDevice, reg: &mut KernelRegistry, enc: &mut CommandEncoder, c: &Cell, with_v_transpose: bool) {
    let (nh, nkv, hd, rows) = (N_HEADS, c.kind.n_kv(), c.kind.head_dim(), c.rows);
    let kq = c.kq.as_ref().unwrap();
    let v_t = c.v_t.as_ref().unwrap();
    dense_matmul_bf16_f32_tensor(
        enc,
        reg,
        dev,
        c.k_hm_bf16.as_ref().unwrap(),
        c.q_hm_f32.as_ref().unwrap(),
        kq,
        &DenseMmBf16F32Params {
            m: rows as u32,
            n: rows as u32,
            k: hd as u32,
            src0_batch: nkv as u32,
            src1_batch: nh as u32,
        },
    )
    .unwrap();
    enc.memory_barrier();
    dispatch_scale_mask_softmax_f32(
        enc,
        reg,
        dev,
        kq,
        kq,
        &c.mask_bf16,
        &ScaleMaskSoftmaxParams {
            rows: (nh * rows) as u32,
            cols: rows as u32,
            seq_q: rows as u32,
            scale: 1.0,
        },
    )
    .unwrap();
    if with_v_transpose {
        // Independent of the softmax (hf2q orders it only against V).
        transpose_last2_bf16(
            enc,
            reg,
            dev.metal_device(),
            c.v_hm_bf16.as_ref().unwrap(),
            v_t,
            nkv,
            rows,
            hd,
        )
        .unwrap();
    }
    enc.memory_barrier();
    dense_matmul_bf16_f32_tensor(
        enc,
        reg,
        dev,
        v_t,
        kq,
        c.attn_f32.as_ref().unwrap(),
        &DenseMmBf16F32Params {
            m: rows as u32,
            n: hd as u32,
            k: rows as u32,
            src0_batch: nkv as u32,
            src1_batch: nh as u32,
        },
    )
    .unwrap();
}

/// Encode one layer's attention: the full chain, or the attention
/// dispatch(es) alone from prepared scratch.
fn encode_arm(dev: &MlxDevice, reg: &mut KernelRegistry, enc: &mut CommandEncoder, c: &Cell, arm: Arm, attn_only: bool) {
    let (nh, nkv, hd, rows) = (N_HEADS, c.kind.n_kv(), c.kind.head_dim(), c.rows);
    match arm {
        Arm::FaF16 | Arm::FaF16D512 => {
            if attn_only {
                encode_fa(dev, reg, enc, c, arm);
                return;
            }
            let md = dev.metal_device();
            permute_021_f32_to_f16(enc, reg, md, &c.q_seq, &c.q_f16, rows, nh, hd).unwrap();
            permute_021_f32_to_f16(enc, reg, md, &c.k_seq, &c.k_f16, rows, nkv, hd).unwrap();
            permute_021_f32_to_f16(enc, reg, md, &c.v_seq, &c.v_f16, rows, nkv, hd).unwrap();
            // Per-layer re-cast of the F16 mask from the BF16 source.
            cast(enc, reg, md, &c.mask_bf16, &c.mask_f16, rows * rows, CastDirection::BF16ToF16)
                .unwrap();
            enc.memory_barrier();
            encode_fa(dev, reg, enc, c, arm);
            enc.memory_barrier();
            cast(enc, reg, md, &c.out_f16, &c.out_bf16, nh * rows * hd, CastDirection::F16ToBF16)
                .unwrap();
        }
        Arm::TensorMm => {
            if attn_only {
                encode_qk_softmax_pv(dev, reg, enc, c, false);
                return;
            }
            encode_qk_softmax_pv(dev, reg, enc, c, true);
            enc.memory_barrier();
            permute_021_f32(
                enc,
                reg,
                dev.metal_device(),
                c.attn_f32.as_ref().unwrap(),
                c.sdpa_out.as_ref().unwrap(),
                nh,
                rows,
                hd,
            )
            .unwrap();
        }
    }
}

/// What one timed command buffer encodes.
#[derive(Clone, Copy)]
enum Job {
    Arm { arm: Arm, attn_only: bool },
    SetupChain,
    SetupBlk,
}

fn time_cb(dev: &MlxDevice, reg: &mut KernelRegistry, c: &Cell, job: Job) -> f64 {
    let mut enc = dev.command_encoder().unwrap();
    let mut keep = None;
    match job {
        Job::Arm { arm, attn_only } => encode_arm(dev, reg, &mut enc, c, arm, attn_only),
        Job::SetupChain => {
            keep = Some(encode_setup(
                dev,
                reg,
                &mut enc,
                c.kind,
                c.rows,
                &c.mask_f16,
                &c.blk,
                &c.blk_params,
            ))
        }
        Job::SetupBlk => flash_attn_prefill_blk::dispatch_flash_attn_prefill_blk(
            &mut enc,
            dev,
            reg,
            &c.mask_bf16,
            &c.blk,
            &c.blk_params,
        )
        .unwrap(),
    }
    let (start, end) = enc.commit_wait_with_gpu_time().unwrap();
    drop(keep);
    (end - start) * 1e6
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    let pos = q * (sorted.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

/// Time `jobs` interleaved forward then reverse; one stats object per job.
fn measure(dev: &MlxDevice, reg: &mut KernelRegistry, c: &Cell, jobs: &[Job]) -> Vec<serde_json::Value> {
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); jobs.len()];
    let order: Vec<usize> = (0..jobs.len()).chain((0..jobs.len()).rev()).collect();
    for round in 0..(WARMUP + MEASURED).div_ceil(2) {
        for &j in &order {
            let us = time_cb(dev, reg, c, jobs[j]);
            if round * 2 >= WARMUP {
                samples[j].push(us);
            }
        }
    }
    samples
        .into_iter()
        .map(|mut s| {
            s.sort_by(f64::total_cmp);
            s.truncate(MEASURED);
            serde_json::json!({"median_us": quantile(&s, 0.5), "p90_us": quantile(&s, 0.9), "n": s.len()})
        })
        .collect()
}

fn run_cell(dev: &MlxDevice, reg: &mut KernelRegistry, c: &Cell) -> serde_json::Value {
    // Prime every arm's scratch with one full chain so `attn` reads
    // prepared inputs (and the tensor-mm V_t).
    let mut enc = dev.command_encoder().unwrap();
    for &arm in c.kind.arms() {
        encode_arm(dev, reg, &mut enc, c, arm, false);
        enc.memory_barrier();
    }
    enc.commit_and_wait().unwrap();

    let mut jobs = Vec::new();
    for &arm in c.kind.arms() {
        jobs.push(Job::Arm { arm, attn_only: false });
        jobs.push(Job::Arm { arm, attn_only: true });
    }
    let stats = measure(dev, reg, c, &jobs);
    let mut arms = serde_json::Map::new();
    for (a, &arm) in c.kind.arms().iter().enumerate() {
        arms.insert(
            arm.name().into(),
            serde_json::json!({"chain": stats[2 * a], "attn": stats[2 * a + 1]}),
        );
    }
    let setup = measure(dev, reg, c, &[Job::SetupChain, Job::SetupBlk]);
    let bp = &c.blk_params;
    serde_json::json!({
        "kind": c.kind.name(),
        "rows": c.rows,
        "n_heads": N_HEADS, "n_kv_heads": c.kind.n_kv(), "head_dim": c.kind.head_dim(),
        "window": c.kind.window(),
        "arms": arms,
        "setup": {"chain": setup[0], "blk": setup[1], "blk_tile": [bp.bq, bp.bk]},
    })
}

fn sha256_file(path: &std::path::Path) -> String {
    std::fs::read(path)
        .map(|b| hex::encode(Sha256::digest(b)))
        .unwrap_or_default()
}

fn source_commit() -> Option<String> {
    if let Ok(sha) = std::env::var("GIT_COMMIT_SHA") {
        return Some(sha);
    }
    let out = std::process::Command::new("git")
        .args(["-C", env!("CARGO_MANIFEST_DIR"), "rev-parse", "HEAD"])
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Every pipeline label the arms can compile: plain kernels by name, and
/// the function-constant specializations by their registry cache key
/// (`name|idx:b0/b1|idx:iN`, bool constants then int constants).
fn candidate_labels() -> Vec<String> {
    let mut labels: Vec<String> = [
        "permute_021_f32_to_f16",
        "permute_021_f32",
        "transpose_last2_bf16",
        "cast_bf16_to_f16",
        "cast_f16_to_bf16",
        "scale_mask_softmax_f32",
        "scale_mask_softmax_f32_v4",
        "hf2q_dense_mm_bf16_f32_tensor",
        "hf2q_dense_mm_bf16_f32_tensor_v2",
        "hf2q_dense_mm_bf16_f32_fallback",
        flash_attn_prefill_mask::K_FILL_BF16,
    ]
    .into_iter()
    .map(String::from)
    .collect();
    let b = |v: bool| if v { "b1" } else { "b0" };
    for aq in [false, true] {
        for ak in [false, true] {
            labels.push(format!(
                "flash_attn_prefill_f16_d256|200:{}|201:{}|300:b1|301:b0|303:b1",
                b(aq),
                b(ak)
            ));
            labels.push(format!(
                "{}|200:{}|201:{}|300:b1|301:b0|303:b1|{}:b0|{}:i{}",
                flash_attn_prefill_d512::K_D512_F16,
                b(aq),
                b(ak),
                flash_attn_prefill_d512::FC_IDX_SINKS,
                flash_attn_prefill_d512::FC_IDX_NSG,
                flash_attn_prefill_d512::NSG_D512
            ));
        }
    }
    for (bq, bk) in [(32, 16), (8, 64)] {
        labels.push(format!("{}|400:i{bq}|401:i{bk}", flash_attn_prefill_blk::K_BLK_BF16));
    }
    labels
}

fn main() {
    let rows: Vec<usize> = match std::env::var("BENCH_ROWS") {
        Ok(s) => s
            .split(',')
            .map(|r| r.trim().parse().expect("BENCH_ROWS=<comma list of rows>"))
            .collect(),
        Err(_) => DEFAULT_ROWS.to_vec(),
    };
    let commit = source_commit();
    if commit.is_none() {
        eprintln!("WARNING: no GIT_COMMIT_SHA and no git checkout; the receipt has no commit");
    }
    let dev = MlxDevice::new().unwrap();
    let mut reg = KernelRegistry::new();
    flash_attn_prefill::register(&mut reg);
    flash_attn_prefill_d512::register(&mut reg);
    flash_attn_prefill_mask::register(&mut reg);
    flash_attn_prefill_blk::register(&mut reg);

    // Untimed warm-up pass over the smallest cell of each kind: the first
    // measured cell otherwise carries first-use cost (measured ~450 us extra
    // on the first sliding cell, whichever row count came first).
    let mut seed = 0xC01D_u64;
    let warm_rows = rows.iter().copied().min().unwrap_or(256);
    for kind in [Kind::Sliding, Kind::Global] {
        let c = cell(&dev, &mut reg, kind, warm_rows, seed ^ 0xFFFF);
        let _ = run_cell(&dev, &mut reg, &c);
    }

    let mut cells = Vec::new();
    for &r in &rows {
        for kind in [Kind::Sliding, Kind::Global] {
            seed += 1;
            let c = cell(&dev, &mut reg, kind, r, seed);
            let result = run_cell(&dev, &mut reg, &c);
            let mut line = format!("{} rows={r}:", kind.name());
            for &arm in kind.arms() {
                let a = &result["arms"][arm.name()];
                line += &format!(
                    " {} chain {:.1}us attn {:.1}us;",
                    arm.name(),
                    a["chain"]["median_us"].as_f64().unwrap(),
                    a["attn"]["median_us"].as_f64().unwrap()
                );
            }
            line += &format!(
                " setup {:.1}us (blk {:.1}us)",
                result["setup"]["chain"]["median_us"].as_f64().unwrap(),
                result["setup"]["blk"]["median_us"].as_f64().unwrap()
            );
            eprintln!("{line}");
            cells.push(result);
        }
    }

    let pipelines: Vec<serde_json::Value> = candidate_labels()
        .into_iter()
        .filter_map(|label| {
            let id = reg.pipeline_identity(&label).ok()?;
            Some(serde_json::json!({"label": label, "kernel_name": id.kernel_name,
                "origin": format!("{:?}", id.origin),
                "runtime_source_sha256": id.runtime_source_sha256,
                "embedded_metallib_sha256": id.embedded_metallib_sha256}))
        })
        .collect();
    let exe = std::env::current_exe().unwrap();
    let os = std::process::Command::new("sw_vers")
        .arg("-buildVersion")
        .output()
        .ok();
    let receipt = serde_json::json!({
        "schema": "mlx-native.bench_gemma_cold_attention.v1",
        "device": dev.name(),
        "macos_build": os.map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()),
        "mlx_native_version": env!("CARGO_PKG_VERSION"),
        "mlx_native_commit": commit,
        "bench_binary_sha256": sha256_file(&exe),
        "precompiled_metallib": std::env::var("MLX_PRECOMPILED_METALLIB").ok(),
        "pipelines": pipelines,
        "warmup": WARMUP, "measured": MEASURED,
        "cells": cells,
    });
    let text = serde_json::to_string_pretty(&receipt).unwrap();
    match std::env::var("BENCH_OUT") {
        Ok(path) => std::fs::write(path, text + "\n").unwrap(),
        Err(_) => println!("{text}"),
    }
}
