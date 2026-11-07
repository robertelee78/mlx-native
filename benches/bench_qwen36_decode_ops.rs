//! ADR-063 gap program: per-op GPU timing of one Qwen3.6-35B-A3B (qwen35moe)
//! decode step at joint widths w = 1..4, model-free.
//!
//! Every case calls the same mlx-native entry point with the same parameters,
//! dtypes and buffer shapes that hf2q's decode step uses (hf2q
//! `src/inference/models/qwen35/{joint_decode.rs, joint_decode_core.rs,
//! gpu_ffn.rs, gpu_full_attn.rs, gpu_delta_net.rs, kv_cache.rs}` at dc72f452;
//! inventory: `v0130/q4-qwen36/opparity/hf2q-inventory.{md,json}`). Weights are
//! random bytes in the exact GGML block layouts (Q4_K 144 B / Q6_K 210 B per
//! 256 values, f16 scale fields patched to a finite 0.01) or random BF16; no
//! GGUF is read.
//!
//! Modes (one JSON line per case on stdout, a `_meta` line first):
//! * `isolated_warm`: one unit (the dispatches hf2q issues for that op at
//!   width w, e.g. w per-row `mul_mv_id` calls separated by the row barrier)
//!   repeated U times in ONE command buffer with a memory barrier between
//!   units, always on the same buffers. Weights that fit the on-chip cache stay
//!   hot; this is the analogue of the peer's backend-op perf mode.
//! * `isolated_cold`: the same, but consecutive units rotate over enough
//!   distinct weight / state / KV copies (or, for expert matvecs and the
//!   embedding, distinct routed expert ids / token ids over the full stack) that
//!   a copy's bytes are not reused before `--cold-mib` (default 256 MiB) of
//!   other data has been touched. Copies are capped at `MAX_SETS`; the
//!   resulting footprint is reported per case.
//! * `sequence`: one GDN layer, one full-attention layer, one FFN layer (fused
//!   residual norm, the serial MoE builder once per row behind the row barrier,
//!   the per-row gather copies) and the output head, encoded exactly as hf2q's
//!   joint step does (serial step at w = 1: no joint-only row copies), repeated
//!   with the cross-layer barrier, rotating over layer copies (cold).
//!   The full-attention layer also runs row-batched: `layer_attn_rows_*` (one
//!   batched K and V encode, one Q premultiply, `flash_attn_vec_tq_hb_rows`
//!   straight into the packed rows, one fused undo + sigmoid gate; no row
//!   copies or row barriers) and `layer_attn_batched_*` (the same with the
//!   one-dispatch `flash_attn_vec_tq_hb_batched` kernel, which has no GQA tile).
//! * `cb_overhead`: empty command buffer commit+wait, and the 22-CB step pattern
//!   (21 async `commit_labeled` + 1 `commit_and_wait`), empty and with one tiny
//!   dispatch per CB.
//!
//! GPU time: `MTLCommandBuffer` `GPUEndTime - GPUStartTime` of each measured CB
//! divided by its unit count. Wall: CPU encode + commit + wait per unit.
//! Kernel names: the per-pipeline dispatch buckets (`MLX_DISP_BUCKET=1`, set by
//! this bench before the first dispatch) of one probe unit. The bucket adds a
//! mutex-guarded map update per encode, so `wall_us`/`encode_us` carry that
//! CPU cost; GPU times do not.
//!
//! Run (needs the GPU lane; HOST-LOCK held by the caller):
//!   cargo bench -p mlx-native --bench bench_qwen36_decode_ops -- [--filter S[,S..]]
//!     [--samples N] [--target-cb-us U] [--cold-mib M] [--list]

use std::time::Instant;

use mlx_native::ops::argmax::dispatch_argmax_f32;
use mlx_native::ops::compute_g_beta::dispatch_compute_g_beta;
use mlx_native::ops::copy::dispatch_copy_f32;
use mlx_native::ops::dense_gemv_bf16::dense_gemv_bf16_f32;
use mlx_native::ops::dense_mm_bf16::DenseMmBf16F32Params;
use mlx_native::ops::elementwise::scalar_mul_f32;
use mlx_native::ops::flash_attn_vec::tmp_buffer_bytes as flash_attn_vec_tmp_bytes;
use mlx_native::ops::flash_attn_vec_tq_hb::{
    compute_nsg, flash_attn_vec_tq_hb, flash_attn_vec_tq_hb_batched, flash_attn_vec_tq_hb_gqa,
    flash_attn_vec_tq_hb_rows, tmp_buffer_bytes as tq_hb_tmp_bytes, FlashAttnVecTqHbParams,
    GqaTile, TqHbDecodeRow, TqHbRowKernel,
};
use mlx_native::ops::fused_norm_add::dispatch_fused_residual_norm_f32;
use mlx_native::ops::fwht_standalone::{
    dispatch_fwht_sign_premult_f32, dispatch_fwht_sign_undo_f32,
    dispatch_fwht_sign_undo_sigmoid_mul_f32,
};
use mlx_native::ops::gated_delta_net::GatedDeltaNetParams;
use mlx_native::ops::gated_delta_net_decode::dispatch_gated_delta_net_decode;
use mlx_native::ops::hadamard_quantize_kv::{
    dispatch_hadamard_quantize_kv_hb_batched, dispatch_hadamard_quantize_kv_hb_seq,
};
use mlx_native::ops::l2_norm::dispatch_l2_norm;
use mlx_native::ops::moe_softmax_topk::dispatch_moe_softmax_topk;
use mlx_native::ops::moe_weighted_reduce::dispatch_moe_weighted_reduce;
use mlx_native::ops::quantized_matmul_ggml::{
    quantized_matmul_ggml, GgmlQuantizedMatmulParams, GgmlType,
};
use mlx_native::ops::quantized_matmul_id_ggml::{
    quantized_matmul_id_ggml_pooled, GgmlQuantizedMatmulIdParams,
};
use mlx_native::ops::rms_norm::dispatch_rms_norm;
use mlx_native::ops::rope_multi::{dispatch_rope_multi_cached, RopeMultiMode, RopeMultiParams};
use mlx_native::ops::sigmoid_mul::dispatch_sigmoid_mul;
use mlx_native::ops::silu_mul::dispatch_silu_mul;
use mlx_native::ops::ssm_conv::{dispatch_ssm_conv, SsmConvParams};
use mlx_native::ops::ssm_norm_gate::dispatch_ssm_norm_gate;
use mlx_native::{
    dispatch_q_gate_deinterleave_f32, embedding_gather_q4_k, CommandEncoder,
    CompletedGpuTimestamps, DType, EmbeddingQ4KParams, IdMmScratch, KernelRegistry, MlxBuffer,
    MlxDevice, QGateDeinterleaveParams,
};
use serde_json::{json, Value};

type R<T> = Result<T, Box<dyn std::error::Error>>;

// ---- Qwen3.6-35B-A3B (qwen35moe) decode shapes ----
const HIDDEN: u32 = 2048;
const EPS: f32 = 1e-6;
const VOCAB: u32 = 248_320;
// GatedDeltaNet
const N_K_HEADS: u32 = 16;
const N_V_HEADS: u32 = 32;
const D_K: u32 = 128;
const D_V: u32 = 128;
const CONV_K: u32 = 4;
const QKV_CH: u32 = 2 * N_K_HEADS * D_K + N_V_HEADS * D_V; // 8192
const Z_CH: u32 = N_V_HEADS * D_V; // 4096
// Gated full attention
const N_HEADS: u32 = 16;
const N_KV_HEADS: u32 = 2;
const HEAD_DIM: u32 = 256;
const ROTARY_DIM: u32 = 64;
const ROPE_THETA: f32 = 10_000_000.0;
const MROPE_SECTION: [u32; 4] = [11, 11, 10, 0];
const Q_TOTAL: u32 = N_HEADS * HEAD_DIM; // 4096
const KV_TOTAL: u32 = N_KV_HEADS * HEAD_DIM; // 512
const TQ_CODEBOOK_BITS: u32 = 8;
const GQA_Q2_MIN_KV_SEQ_LEN: u32 = 8_192;
const KV_LENS: [u32; 3] = [2048, 8192, 16384];
// MoE
const N_EXPERTS: u32 = 256;
const TOP_K: u32 = 8;
const MOE_I: u32 = 512;
const SHARED_I: u32 = 512;
const WIDTHS: [u32; 4] = [1, 2, 3, 4];

const MAX_SETS: usize = 512;
const ID_SETS: usize = 1024;
const MIB: u64 = 1 << 20;

// ======================================================================
// Random data in exact layouts
// ======================================================================

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn below(&mut self, n: u32) -> u32 {
        ((self.next() >> 32) % n as u64) as u32
    }
}

fn qname(qt: GgmlType) -> &'static str {
    match qt {
        GgmlType::Q4_K => "q4_K",
        GgmlType::Q6_K => "q6_K",
        _ => "other",
    }
}

fn qbytes(qt: GgmlType, rows: u32, k: u32) -> u64 {
    rows as u64 * (k as u64 / qt.block_values() as u64) * qt.block_bytes() as u64
}

fn alloc(dev: &MlxDevice, dtype: DType, shape: Vec<usize>) -> R<MlxBuffer> {
    let n: usize = shape.iter().product();
    Ok(dev.alloc_buffer(n * dtype.size_of(), dtype, shape)?)
}

fn fill_bytes(buf: &mut MlxBuffer, rng: &mut Rng) -> R<()> {
    let s = buf.as_mut_slice::<u8>()?;
    let mut chunks = s.chunks_exact_mut(8);
    for c in &mut chunks {
        c.copy_from_slice(&rng.next().to_le_bytes());
    }
    for b in chunks.into_remainder() {
        *b = rng.next() as u8;
    }
    Ok(())
}

fn f32_buf(dev: &MlxDevice, shape: Vec<usize>, rng: &mut Rng, lo: f32, hi: f32) -> R<MlxBuffer> {
    let mut b = alloc(dev, DType::F32, shape)?;
    for v in b.as_mut_slice::<f32>()? {
        *v = lo + (hi - lo) * 0.5 * (rng.unit() + 1.0);
    }
    Ok(b)
}

fn zeros(dev: &MlxDevice, dtype: DType, shape: Vec<usize>) -> R<MlxBuffer> {
    let mut b = alloc(dev, dtype, shape)?;
    b.as_mut_slice::<u8>()?.fill(0);
    Ok(b)
}

fn u32_buf(dev: &MlxDevice, values: &[u32]) -> R<MlxBuffer> {
    let mut b = alloc(dev, DType::U32, vec![values.len()])?;
    b.as_mut_slice::<u32>()?.copy_from_slice(values);
    Ok(b)
}

fn f32_params(dev: &MlxDevice, values: &[f32]) -> R<MlxBuffer> {
    let mut b = alloc(dev, DType::F32, vec![values.len()])?;
    b.as_mut_slice::<f32>()?.copy_from_slice(values);
    Ok(b)
}

/// Random GGML K-quant blocks (`rows` x `k`), finite f16 scales.
fn quant_buf(dev: &MlxDevice, qt: GgmlType, rows: u32, k: u32, rng: &mut Rng) -> R<MlxBuffer> {
    let bytes = qbytes(qt, rows, k) as usize;
    let mut b = alloc(dev, DType::U8, vec![bytes])?;
    fill_bytes(&mut b, rng)?;
    let scale = half::f16::from_f32(0.01).to_bits().to_le_bytes();
    let block = qt.block_bytes() as usize;
    let s = b.as_mut_slice::<u8>()?;
    for blk in s.chunks_exact_mut(block) {
        match qt {
            // block_q4_K: d (f16), dmin (f16), scales[12], qs[128]
            GgmlType::Q4_K => {
                blk[0..2].copy_from_slice(&scale);
                blk[2..4].copy_from_slice(&scale);
            }
            // block_q6_K: ql[128], qh[64], scales[16] (i8), d (f16)
            GgmlType::Q6_K => blk[208..210].copy_from_slice(&scale),
            _ => {}
        }
    }
    Ok(b)
}

fn bf16_buf(dev: &MlxDevice, rows: u32, k: u32, rng: &mut Rng) -> R<MlxBuffer> {
    let mut b = alloc(dev, DType::BF16, vec![rows as usize, k as usize])?;
    for v in b.as_mut_slice::<u16>()? {
        *v = half::bf16::from_f32(0.05 * rng.unit()).to_bits();
    }
    Ok(b)
}

fn row_view(buf: &MlxBuffer, row: usize, shape: Vec<usize>) -> R<MlxBuffer> {
    let n: usize = shape.iter().product();
    let off = (row * n * buf.dtype().size_of()) as u64;
    Ok(buf.slice_view(off, n).with_shape(shape)?)
}

fn random_ids(rng: &mut Rng, count: usize) -> Vec<u32> {
    // `count` distinct experts (top-k never repeats an expert).
    let mut ids = Vec::with_capacity(count);
    while ids.len() < count {
        let e = rng.below(N_EXPERTS);
        if !ids.contains(&e) {
            ids.push(e);
        }
    }
    ids
}

fn sets_for(per_set_bytes: u64, cold: bool, cold_bytes: u64, min: usize, max: usize) -> usize {
    if !cold {
        return 1;
    }
    let need = cold_bytes.div_ceil(per_set_bytes.max(1)) as usize;
    need.clamp(min.max(1), max)
}

// ======================================================================
// Case catalogue
// ======================================================================

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Warm,
    Cold,
    Sequence,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Mode::Warm => "isolated_warm",
            Mode::Cold => "isolated_cold",
            Mode::Sequence => "sequence",
        }
    }
}

#[derive(Clone, Copy)]
enum Bf16Role {
    Router,
    SharedGateLogit,
    SharedGateUp,
    SharedDown,
}

#[derive(Clone, Copy)]
enum Op {
    Embedding,
    RmsNorm { rows_per_w: u32, dim: u32 },
    FusedResidualNorm,
    QMatmul { n: u32, k: u32, qt: GgmlType },
    QGateDeinterleave,
    SsmConv,
    L2Norm,
    QScale,
    ComputeGBeta,
    GatedDeltaNet,
    SsmNormGate,
    Rope { heads: u32 },
    TqEncode,
    FwhtPremult,
    FwhtUndo,
    TqSdpa { kv: u32 },
    /// The w rows' attention of the row-batched layer (`route`).
    TqSdpaRows { kv: u32, route: AttnRoute },
    /// Batched K and V encode of the w rows into a multi-slot cache.
    TqEncodeRows,
    /// Fused FWHT undo + sigmoid gate over the w rows' heads.
    FwhtUndoSigmoid,
    RowCopy { width: u32 },
    SigmoidMul,
    Bf16Gemv(Bf16Role),
    SoftmaxTopk,
    SiluMul { rows: u32 },
    MoeId { down: bool, qt: GgmlType },
    WeightedReduce,
    Argmax,
    LayerGdn { qkv: GgmlType },
    LayerAttn { v: GgmlType, kv: u32 },
    /// The full-attention layer with the per-row small ops batched over w.
    LayerAttnRows { v: GgmlType, kv: u32, route: AttnRoute },
    LayerFfn { down: GgmlType },
    Head,
}

/// Attention core of the row-batched full-attention layer.
#[derive(Clone, Copy)]
enum AttnRoute {
    /// `flash_attn_vec_tq_hb_rows`: each row's own scalar / GQA-Q2 kernel,
    /// all mains concurrent, one barrier, all reduces.
    Rows,
    /// `flash_attn_vec_tq_hb_batched`: one dispatch over the w rows (scalar
    /// kernel, no GQA tile) plus its fused reduce + undo.
    Batched,
}

impl AttnRoute {
    fn tag(self) -> &'static str {
        match self {
            Self::Rows => "rows",
            Self::Batched => "batched",
        }
    }
}

#[derive(Clone, Copy)]
struct Case {
    op: Op,
    w: u32,
    mode: Mode,
}

fn bf16_shape(role: Bf16Role) -> (u32, u32) {
    match role {
        Bf16Role::Router => (N_EXPERTS, HIDDEN),
        Bf16Role::SharedGateLogit => (1, HIDDEN),
        Bf16Role::SharedGateUp => (SHARED_I, HIDDEN),
        Bf16Role::SharedDown => (HIDDEN, SHARED_I),
    }
}

impl Case {
    fn name(&self) -> String {
        let w = self.w;
        match self.op {
            Op::Embedding => format!("get_rows_q4_K_{HIDDEN}x{VOCAB}_w{w}"),
            Op::RmsNorm { rows_per_w: 1, dim } => format!("rms_norm_f32_{dim}_w{w}"),
            Op::RmsNorm { rows_per_w, dim } => format!("rms_norm_f32_{dim}x{rows_per_w}_w{w}"),
            Op::FusedResidualNorm => format!("fused_residual_norm_f32_{HIDDEN}_w{w}"),
            Op::QMatmul { n, k, qt } => format!("mul_mat_{}_{n}x{k}_w{w}", qname(qt)),
            Op::QGateDeinterleave => format!("q_gate_deinterleave_f32_{N_HEADS}x{HEAD_DIM}_w{w}"),
            Op::SsmConv => format!("ssm_conv_f32_{QKV_CH}x{CONV_K}_w{w}"),
            Op::L2Norm => format!("l2_norm_f32_{D_K}x{N_K_HEADS}_w{w}"),
            Op::QScale => format!("scale_f32_{}_w{w}", N_K_HEADS * D_K),
            Op::ComputeGBeta => format!("compute_g_beta_f32_{N_V_HEADS}_w{w}"),
            Op::GatedDeltaNet => format!("gated_delta_net_f32_{N_V_HEADS}x{D_V}_w{w}"),
            Op::SsmNormGate => format!("ssm_norm_gate_f32_{D_V}x{N_V_HEADS}_w{w}"),
            Op::Rope { heads } => format!("rope_imrope_f32_{HEAD_DIM}x{heads}_w{w}"),
            Op::TqEncode => format!("tq_encode_kv_d{HEAD_DIM}x{N_KV_HEADS}_w{w}"),
            Op::FwhtPremult => format!("fwht_premult_f32_{N_HEADS}x{HEAD_DIM}_w{w}"),
            Op::FwhtUndo => format!("fwht_undo_f32_{N_HEADS}x{HEAD_DIM}_w{w}"),
            Op::TqSdpa { kv } => format!("flash_attn_tq_d{HEAD_DIM}_kv{kv}_w{w}"),
            Op::TqSdpaRows { kv, route } => {
                format!("flash_attn_tq_{}_d{HEAD_DIM}_kv{kv}_w{w}", route.tag())
            }
            Op::TqEncodeRows => format!("tq_encode_kv_rows_d{HEAD_DIM}x{N_KV_HEADS}_w{w}"),
            Op::FwhtUndoSigmoid => format!("fwht_undo_sigmoid_mul_f32_{N_HEADS}x{HEAD_DIM}_w{w}"),
            Op::RowCopy { width } => format!("copy_f32_{width}_w{w}"),
            Op::SigmoidMul => format!("sigmoid_mul_f32_{Q_TOTAL}_w{w}"),
            Op::Bf16Gemv(role) => {
                let (n, k) = bf16_shape(role);
                format!("mul_mat_bf16_{n}x{k}_w{w}")
            }
            Op::SoftmaxTopk => format!("topk_moe_f32_{N_EXPERTS}x{TOP_K}_w{w}"),
            Op::SiluMul { rows: 1 } => format!("swiglu_f32_{SHARED_I}_w{w}"),
            Op::SiluMul { rows } => format!("swiglu_f32_{MOE_I}x{rows}_w{w}"),
            Op::MoeId { down: false, qt } => {
                format!("mul_mat_id_{}_{MOE_I}x{HIDDEN}_w{w}", qname(qt))
            }
            Op::MoeId { down: true, qt } => {
                format!("mul_mat_id_{}_{HIDDEN}x{MOE_I}_w{w}", qname(qt))
            }
            Op::WeightedReduce => format!("moe_reduce_f32_{HIDDEN}x{TOP_K}_w{w}"),
            Op::Argmax => format!("argmax_f32_{VOCAB}_w{w}"),
            Op::LayerGdn { qkv } => format!("layer_gdn_qkv{}_w{w}", qname(qkv)),
            Op::LayerAttn { v, kv } => format!("layer_attn_v{}_kv{kv}_w{w}", qname(v)),
            Op::LayerAttnRows { v, kv, route } => {
                format!("layer_attn_{}_v{}_kv{kv}_w{w}", route.tag(), qname(v))
            }
            Op::LayerFfn { down } => format!("layer_ffn_down{}_w{w}", qname(down)),
            Op::Head => format!("head_lm_q6_K_{VOCAB}x{HIDDEN}_w{w}"),
        }
    }

    /// Closest case of the peer's backend-op perf list (peer_reference.sh).
    fn peer_case(&self) -> String {
        let w = self.w;
        match self.op {
            Op::Embedding => format!("GET_ROWS(type=q4_K,n={HIDDEN},m={VOCAB},r={w})"),
            Op::RmsNorm { rows_per_w: 1, dim } => format!("RMS_NORM(type=f32,ne=[{dim},{w},1,1])"),
            Op::RmsNorm { rows_per_w, dim } => {
                format!("RMS_NORM(type=f32,ne=[{dim},{rows_per_w},{w},1])")
            }
            Op::FusedResidualNorm => format!(
                "ADD(ne=[{HIDDEN},{w},1,1]) + RMS_NORM(ne=[{HIDDEN},{w},1,1]) (peer fuses norm*weight)"
            ),
            Op::QMatmul { n, k, qt } => {
                format!("MUL_MAT(type_a={},type_b=f32,m={n},n={w},k={k})", qname(qt))
            }
            Op::QGateDeinterleave => "none (peer splits q/gate by views; CONT of gate)".into(),
            Op::SsmConv => format!(
                "CONCAT(ne=[3,{QKV_CH},{w},1],dim=0) + SSM_CONV(ne_a=[4,{QKV_CH},{w},1],ne_b=[4,{QKV_CH},1,1])"
            ),
            Op::L2Norm => format!("RMS_NORM(ne=[{D_K},{N_K_HEADS},{w},1]) + SCALE (q and k)"),
            Op::QScale => format!("SCALE(ne=[{D_K},{N_K_HEADS},{w},1])"),
            Op::ComputeGBeta => format!(
                "ADD(ne=[32,1]+[1,{w}]) + SOFTPLUS(ne=[32,{w}]) + MUL(ne=[32,1]x[1,{w}]) + SIGMOID(ne=[32,{w}])"
            ),
            Op::GatedDeltaNet => format!(
                "GATED_DELTA_NET(head_count={N_K_HEADS},head_size={D_K},n_seq_tokens=1,n_seqs={w},v_repeat=2)"
            ),
            Op::SsmNormGate => format!(
                "RMS_NORM(ne=[{D_V},{N_V_HEADS},{w},1]) + SILU + MUL(ne=[{D_V},{N_V_HEADS},{w},1])"
            ),
            Op::Rope { heads } => format!(
                "ROPE(type=f32,ne=[{HEAD_DIM},{heads},{w},1],n_dims={ROTARY_DIM},mode=IMROPE)"
            ),
            Op::TqEncode => format!(
                "SET_ROWS(type=f32->f16,ne=[{KV_TOTAL},16384,1,1],r={w}) x2 (peer F16 KV)"
            ),
            Op::FwhtPremult | Op::FwhtUndo => "none (TQ-KV specific)".into(),
            Op::TqEncodeRows => format!(
                "SET_ROWS(type=f32->f16,ne=[{KV_TOTAL},16384,1,1],r={w}) x2 (peer F16 KV)"
            ),
            Op::FwhtUndoSigmoid => {
                format!("SIGMOID(ne=[{Q_TOTAL},{w},1,1]) + MUL (undo is TQ-KV specific)")
            }
            Op::TqSdpaRows { kv, .. } => format!(
                "FLASH_ATTN_EXT(hsk=256,hsv=256,nh=2,nr23=[8,{w}],kv={kv},nb=1,type_KV=f16) (per-stream)"
            ),
            Op::TqSdpa { kv } if w == 1 => format!(
                "FLASH_ATTN_EXT(hsk=256,hsv=256,nh=2,nr23=[8,1],kv={kv},nb=1,type_KV=f16)"
            ),
            Op::TqSdpa { kv } => format!(
                "FLASH_ATTN_EXT(hsk=256,hsv=256,nh=2,nr23=[8,{w}],kv={kv},nb=1,type_KV=f16) (per-stream)"
            ),
            Op::RowCopy { .. } => "none (peer graph is batched over w)".into(),
            Op::SigmoidMul => format!("SIGMOID(ne=[{Q_TOTAL},{w},1,1]) + MUL(ne=[{Q_TOTAL},{w},1,1])"),
            Op::Bf16Gemv(Bf16Role::Router) => {
                format!("MUL_MAT(type_a=f32,type_b=f32,m={N_EXPERTS},n={w},k={HIDDEN})")
            }
            Op::Bf16Gemv(Bf16Role::SharedGateLogit) => {
                format!("MUL_MAT(type_a=f32,type_b=f32,m=1,n={w},k={HIDDEN})")
            }
            Op::Bf16Gemv(Bf16Role::SharedGateUp) => {
                format!("MUL_MAT(type_a=q4_K,type_b=f32,m={SHARED_I},n={w},k={HIDDEN})")
            }
            Op::Bf16Gemv(Bf16Role::SharedDown) => format!(
                "MUL_MAT(type_a=q6_K|q4_K,type_b=f32,m={HIDDEN},n={w},k={SHARED_I})"
            ),
            Op::SoftmaxTopk => format!(
                "TOPK_MOE(ne=[{N_EXPERTS},{w},1,1],n_expert_used={TOP_K},with_norm=1,gating=softmax)"
            ),
            Op::SiluMul { rows: 1 } => format!("SWIGLU split(ne=[{SHARED_I},{w},1,1])"),
            Op::SiluMul { .. } => format!("SWIGLU split(ne=[{MOE_I},{TOP_K},{w},1])"),
            Op::MoeId { down: false, qt } => format!(
                "MUL_MAT_ID(type_a={},n_mats={N_EXPERTS},n_used={TOP_K},b=1,m={MOE_I},n={w},k={HIDDEN})",
                qname(qt)
            ),
            Op::MoeId { down: true, qt } => format!(
                "MUL_MAT_ID(type_a={},n_mats={N_EXPERTS},n_used={TOP_K},b=0,m={HIDDEN},n={w},k={MOE_I})",
                qname(qt)
            ),
            Op::WeightedReduce => {
                format!("MOE_REDUCE(n_embd={HIDDEN},n_expert_used={TOP_K},n_tokens={w})")
            }
            Op::Argmax => "none (not in the peer case list)".into(),
            Op::LayerGdn { .. } => "peer GDN layer (sum of its per-op cases)".into(),
            Op::LayerAttn { .. } | Op::LayerAttnRows { .. } => {
                "peer full-attention layer (sum of its per-op cases)".into()
            }
            Op::LayerFfn { .. } => "peer MoE FFN layer (sum of its per-op cases)".into(),
            Op::Head => "peer output head (RMS_NORM + MUL_MAT q6_K 248320x2048)".into(),
        }
    }
}

fn catalogue() -> Vec<Case> {
    let mut ops: Vec<(Op, bool)> = vec![
        // (op, has a cold variant)
        (Op::Embedding, true),
        (Op::RmsNorm { rows_per_w: 1, dim: HIDDEN }, false),
        (Op::FusedResidualNorm, false),
        (Op::QMatmul { n: QKV_CH, k: HIDDEN, qt: GgmlType::Q6_K }, true),
        (Op::QMatmul { n: QKV_CH, k: HIDDEN, qt: GgmlType::Q4_K }, true),
        (Op::QMatmul { n: Z_CH, k: HIDDEN, qt: GgmlType::Q4_K }, true),
        (Op::QMatmul { n: N_V_HEADS, k: HIDDEN, qt: GgmlType::Q4_K }, true),
        (Op::QMatmul { n: HIDDEN, k: Z_CH, qt: GgmlType::Q4_K }, true),
        (Op::QMatmul { n: KV_TOTAL, k: HIDDEN, qt: GgmlType::Q4_K }, true),
        (Op::QMatmul { n: KV_TOTAL, k: HIDDEN, qt: GgmlType::Q6_K }, true),
        (Op::QMatmul { n: VOCAB, k: HIDDEN, qt: GgmlType::Q6_K }, true),
        (Op::SsmConv, false),
        (Op::L2Norm, false),
        (Op::QScale, false),
        (Op::ComputeGBeta, false),
        (Op::GatedDeltaNet, true),
        (Op::SsmNormGate, false),
        (Op::RmsNorm { rows_per_w: N_HEADS, dim: HEAD_DIM }, false),
        (Op::RmsNorm { rows_per_w: N_KV_HEADS, dim: HEAD_DIM }, false),
        (Op::QGateDeinterleave, false),
        (Op::Rope { heads: N_HEADS }, false),
        (Op::Rope { heads: N_KV_HEADS }, false),
        (Op::TqEncode, false),
        (Op::FwhtPremult, false),
    ];
    for kv in KV_LENS {
        ops.push((Op::TqSdpa { kv }, true));
        ops.push((Op::TqSdpaRows { kv, route: AttnRoute::Rows }, true));
        ops.push((Op::TqSdpaRows { kv, route: AttnRoute::Batched }, true));
    }
    ops.extend([
        (Op::TqEncodeRows, false),
        (Op::FwhtUndo, false),
        (Op::FwhtUndoSigmoid, false),
        (Op::RowCopy { width: Q_TOTAL }, false),
        (Op::SigmoidMul, false),
        (Op::Bf16Gemv(Bf16Role::Router), true),
        (Op::Bf16Gemv(Bf16Role::SharedGateLogit), true),
        (Op::Bf16Gemv(Bf16Role::SharedGateUp), true),
        (Op::Bf16Gemv(Bf16Role::SharedDown), true),
        (Op::SoftmaxTopk, false),
        (Op::SiluMul { rows: 1 }, false),
        (Op::MoeId { down: false, qt: GgmlType::Q4_K }, true),
        (Op::SiluMul { rows: TOP_K }, false),
        (Op::MoeId { down: true, qt: GgmlType::Q6_K }, true),
        (Op::MoeId { down: true, qt: GgmlType::Q4_K }, true),
        (Op::WeightedReduce, false),
        (Op::RowCopy { width: HIDDEN }, false),
        (Op::Argmax, false),
    ]);
    let mut seq = vec![
        Op::LayerGdn { qkv: GgmlType::Q6_K },
        Op::LayerGdn { qkv: GgmlType::Q4_K },
    ];
    for kv in KV_LENS {
        seq.push(Op::LayerAttn { v: GgmlType::Q6_K, kv });
        seq.push(Op::LayerAttn { v: GgmlType::Q4_K, kv });
        for route in [AttnRoute::Rows, AttnRoute::Batched] {
            seq.push(Op::LayerAttnRows { v: GgmlType::Q6_K, kv, route });
            seq.push(Op::LayerAttnRows { v: GgmlType::Q4_K, kv, route });
        }
    }
    seq.push(Op::LayerFfn { down: GgmlType::Q6_K });
    seq.push(Op::LayerFfn { down: GgmlType::Q4_K });
    seq.push(Op::Head);

    let mut cases = Vec::new();
    for w in WIDTHS {
        for &(op, cold) in &ops {
            // Joint-only row copies do not exist on the serial w = 1 path.
            if matches!(op, Op::RowCopy { .. }) && w == 1 {
                continue;
            }
            cases.push(Case { op, w, mode: Mode::Warm });
            if cold {
                cases.push(Case { op, w, mode: Mode::Cold });
            }
        }
        for &op in &seq {
            cases.push(Case { op, w, mode: Mode::Sequence });
        }
    }
    cases
}

// ======================================================================
// Prepared case: buffers + per-unit encoder
// ======================================================================

type EncodeFn = Box<dyn FnMut(&mut CommandEncoder, &mut KernelRegistry, &MlxDevice, usize) -> R<()>>;

/// Box a unit encoder; the `FnMut` bound gives the closure its higher-ranked
/// signature.
fn unit<F>(f: F) -> EncodeFn
where
    F: FnMut(&mut CommandEncoder, &mut KernelRegistry, &MlxDevice, usize) -> R<()> + 'static,
{
    Box::new(f)
}

struct Prepared {
    kernel_hint: &'static str,
    m: Value,
    n: Value,
    k: Value,
    qtype: &'static str,
    /// Compulsory bytes moved by one unit (weights + state/KV + activations).
    bytes: u64,
    sets: usize,
    footprint: u64,
    extra: Value,
    encode: EncodeFn,
}

/// Model-lifetime buffers shared by many cases.
struct Shared {
    embd: MlxBuffer,
    lm_head: MlxBuffer,
    gate_exps: MlxBuffer,
    up_exps: MlxBuffer,
    down_exps_q6: MlxBuffer,
    down_exps_q4: MlxBuffer,
}

impl Shared {
    fn new(dev: &MlxDevice, rng: &mut Rng) -> R<Self> {
        Ok(Self {
            embd: quant_buf(dev, GgmlType::Q4_K, VOCAB, HIDDEN, rng)?,
            lm_head: quant_buf(dev, GgmlType::Q6_K, VOCAB, HIDDEN, rng)?,
            gate_exps: quant_buf(dev, GgmlType::Q4_K, N_EXPERTS * MOE_I, HIDDEN, rng)?,
            up_exps: quant_buf(dev, GgmlType::Q4_K, N_EXPERTS * MOE_I, HIDDEN, rng)?,
            down_exps_q6: quant_buf(dev, GgmlType::Q6_K, N_EXPERTS * HIDDEN, MOE_I, rng)?,
            down_exps_q4: quant_buf(dev, GgmlType::Q4_K, N_EXPERTS * HIDDEN, MOE_I, rng)?,
        })
    }
    fn down(&self, qt: GgmlType) -> &MlxBuffer {
        if qt == GgmlType::Q6_K {
            &self.down_exps_q6
        } else {
            &self.down_exps_q4
        }
    }
}

fn qmm(
    enc: &mut CommandEncoder,
    reg: &mut KernelRegistry,
    dev: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    m: u32,
    n: u32,
    k: u32,
    qt: GgmlType,
) -> R<()> {
    let params = GgmlQuantizedMatmulParams { m, n, k, ggml_type: qt };
    quantized_matmul_ggml(enc, reg, dev, input, weight, output, &params)?;
    Ok(())
}

fn gemv_bf16(
    enc: &mut CommandEncoder,
    reg: &mut KernelRegistry,
    dev: &MlxDevice,
    weight: &MlxBuffer,
    input: &MlxBuffer,
    output: &MlxBuffer,
    n: u32,
    k: u32,
) -> R<()> {
    let params = DenseMmBf16F32Params { m: 1, n, k, src0_batch: 1, src1_batch: 1 };
    dense_gemv_bf16_f32(enc, reg, dev, weight, input, output, &params)?;
    Ok(())
}

fn rms(
    enc: &mut CommandEncoder,
    reg: &mut KernelRegistry,
    dev: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    params: &MlxBuffer,
    rows: u32,
    dim: u32,
) -> R<()> {
    dispatch_rms_norm(enc, reg, dev.metal_device(), input, weight, output, params, rows, dim)?;
    Ok(())
}

fn rope_params(heads: u32, w: u32) -> RopeMultiParams {
    RopeMultiParams {
        head_dim: HEAD_DIM,
        rope_dim: ROTARY_DIM,
        n_heads: heads,
        seq_len: w,
        freq_base: ROPE_THETA,
        mode: RopeMultiMode::Imrope,
        sections: MROPE_SECTION,
    }
}

fn positions_buf(dev: &MlxDevice, w: u32, pos: i32) -> R<MlxBuffer> {
    let mut b = alloc(dev, DType::I32, vec![4 * w as usize])?;
    for (i, v) in b.as_mut_slice::<i32>()?.iter_mut().enumerate() {
        *v = pos + (i % w as usize) as i32;
    }
    Ok(b)
}

fn tq_params(kv: u32) -> FlashAttnVecTqHbParams {
    FlashAttnVecTqHbParams {
        num_heads: N_HEADS,
        num_kv_heads: N_KV_HEADS,
        head_dim: HEAD_DIM,
        kv_seq_len: kv,
        kv_capacity: kv,
        scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        mask_type: 0,
        sliding_window: 0,
        softcap: 0.0,
        ring_start: 0,
        scale_factor_d512: 1.0,
        codebook_bits: TQ_CODEBOOK_BITS,
        fuse_fwht_pre: 0,
        nsg: compute_nsg(kv),
    }
}

/// hf2q `use_gqa_q2_tq_sdpa` under the default `HF2Q_QWEN_GQA_Q2=auto`.
fn tq_uses_gqa_q2(kv: u32) -> bool {
    match std::env::var("HF2Q_QWEN_GQA_Q2").ok().as_deref().map(str::trim) {
        Some("0" | "off" | "false") => false,
        Some("1" | "on" | "true") => true,
        _ => kv >= GQA_Q2_MIN_KV_SEQ_LEN,
    }
}

/// One slot's TQ K/V cache (`[n_kv_heads, capacity, head_dim]` u8 codes plus
/// `[n_kv_heads, capacity]` F32 norms for each of K and V).
#[derive(Clone)]
struct TqSlot {
    k_packed: MlxBuffer,
    k_norms: MlxBuffer,
    v_packed: MlxBuffer,
    v_norms: MlxBuffer,
}

impl TqSlot {
    fn new(dev: &MlxDevice, kv: u32, rng: &mut Rng) -> R<Self> {
        let packed = (N_KV_HEADS * kv * HEAD_DIM) as usize;
        let norms = (N_KV_HEADS * kv) as usize;
        let mut k_packed = alloc(dev, DType::U8, vec![packed])?;
        let mut v_packed = alloc(dev, DType::U8, vec![packed])?;
        fill_bytes(&mut k_packed, rng)?;
        fill_bytes(&mut v_packed, rng)?;
        Ok(Self {
            k_packed,
            k_norms: f32_buf(dev, vec![norms], rng, 0.5, 1.5)?,
            v_packed,
            v_norms: f32_buf(dev, vec![norms], rng, 0.5, 1.5)?,
        })
    }
    fn bytes(kv: u32) -> u64 {
        2 * (N_KV_HEADS * kv * HEAD_DIM) as u64 + 2 * 4 * (N_KV_HEADS * kv) as u64
    }
}

fn tq_encode_kv(
    enc: &mut CommandEncoder,
    reg: &mut KernelRegistry,
    dev: &MlxDevice,
    k_row: &MlxBuffer,
    v_row: &MlxBuffer,
    slot: &TqSlot,
    kv: u32,
) -> R<()> {
    // hf2q writes the new token at cursor kv - 1 (cache already holds kv - 1).
    for (src, packed, norms) in [
        (k_row, &slot.k_packed, &slot.k_norms),
        (v_row, &slot.v_packed, &slot.v_norms),
    ] {
        dispatch_hadamard_quantize_kv_hb_seq(
            enc,
            reg,
            dev.metal_device(),
            src,
            packed,
            norms,
            N_KV_HEADS,
            HEAD_DIM,
            kv,
            kv - 1,
            1,
            0,
            false,
            1.0,
            TQ_CODEBOOK_BITS,
        )?;
    }
    Ok(())
}

fn tq_sdpa(
    enc: &mut CommandEncoder,
    reg: &mut KernelRegistry,
    dev: &MlxDevice,
    q: &MlxBuffer,
    slot: &TqSlot,
    out: &MlxBuffer,
    tmp: &MlxBuffer,
    kv: u32,
) -> R<()> {
    let p = tq_params(kv);
    if tq_uses_gqa_q2(kv) {
        flash_attn_vec_tq_hb_gqa(
            enc, reg, dev, q, &slot.k_packed, &slot.k_norms, &slot.v_packed, &slot.v_norms, out,
            tmp, &p, GqaTile::Q2,
        )?;
    } else {
        flash_attn_vec_tq_hb(
            enc, reg, dev, q, &slot.k_packed, &slot.k_norms, &slot.v_packed, &slot.v_norms, out,
            tmp, &p,
        )?;
    }
    Ok(())
}

fn sdpa_tmp(dev: &MlxDevice) -> R<MlxBuffer> {
    let bytes = flash_attn_vec_tmp_bytes(N_HEADS, HEAD_DIM);
    alloc(dev, DType::F32, vec![bytes / 4])
}

/// The w slots of one layer in one multi-slot TQ cache
/// (`[w, n_kv_heads, capacity, head_dim]` codes, `[w, n_kv_heads, capacity]`
/// norms): hf2q's `HybridKvCache` layout, sliced per slot by `slot_views`.
struct TqSlots {
    all: TqSlot,
    kv: u32,
}

impl TqSlots {
    fn new(dev: &MlxDevice, w: u32, kv: u32, rng: &mut Rng) -> R<Self> {
        let packed = (w * N_KV_HEADS * kv * HEAD_DIM) as usize;
        let norms = (w * N_KV_HEADS * kv) as usize;
        let mut k_packed = alloc(dev, DType::U8, vec![packed])?;
        let mut v_packed = alloc(dev, DType::U8, vec![packed])?;
        fill_bytes(&mut k_packed, rng)?;
        fill_bytes(&mut v_packed, rng)?;
        Ok(Self {
            all: TqSlot {
                k_packed,
                k_norms: f32_buf(dev, vec![norms], rng, 0.5, 1.5)?,
                v_packed,
                v_norms: f32_buf(dev, vec![norms], rng, 0.5, 1.5)?,
            },
            kv,
        })
    }

    /// Zero-copy views of slot `s` (hf2q `TqFullAttnKvBuffers::slot_views`).
    fn slot(&self, s: usize) -> TqSlot {
        let packed = (N_KV_HEADS * self.kv * HEAD_DIM) as usize;
        let norms = (N_KV_HEADS * self.kv) as usize;
        TqSlot {
            k_packed: self.all.k_packed.slice_view((s * packed) as u64, packed),
            k_norms: self.all.k_norms.slice_view((s * norms * 4) as u64, norms),
            v_packed: self.all.v_packed.slice_view((s * packed) as u64, packed),
            v_norms: self.all.v_norms.slice_view((s * norms * 4) as u64, norms),
        }
    }
}

/// Batched K and V encode of the w rows at their cursors (one dispatch each).
#[allow(clippy::too_many_arguments)]
fn tq_encode_kv_rows(
    enc: &mut CommandEncoder,
    reg: &mut KernelRegistry,
    dev: &MlxDevice,
    k_rows: &MlxBuffer,
    v_rows: &MlxBuffer,
    slots: &TqSlots,
    slot_ids: &MlxBuffer,
    cursors: &MlxBuffer,
    w: u32,
) -> R<()> {
    for (src, packed, norms) in [
        (k_rows, &slots.all.k_packed, &slots.all.k_norms),
        (v_rows, &slots.all.v_packed, &slots.all.v_norms),
    ] {
        dispatch_hadamard_quantize_kv_hb_batched(
            enc,
            reg,
            dev.metal_device(),
            src,
            packed,
            norms,
            slot_ids,
            cursors,
            w,
            N_KV_HEADS,
            HEAD_DIM,
            slots.kv,
            false,
            1.0,
            TQ_CODEBOOK_BITS,
        )?;
    }
    Ok(())
}

/// The w rows' attention of the row-batched layer. Rows route: per-row
/// `TqHbDecodeRow`s over the slot views, outputs straight into the packed
/// `out` rows (rotated domain). Batched route: one batched dispatch plus its
/// fused reduce + undo (output already un-rotated).
#[allow(clippy::too_many_arguments)]
fn tq_sdpa_rows(
    enc: &mut CommandEncoder,
    reg: &mut KernelRegistry,
    dev: &MlxDevice,
    route: AttnRoute,
    q: &MlxBuffer,
    slots: &TqSlots,
    out: &MlxBuffer,
    tmps: &[MlxBuffer],
    batched_tmp: &MlxBuffer,
    slot_ids: &MlxBuffer,
    cursors: &MlxBuffer,
    w: u32,
) -> R<()> {
    let p = tq_params(slots.kv);
    match route {
        AttnRoute::Rows => {
            let kernel = if tq_uses_gqa_q2(slots.kv) {
                TqHbRowKernel::Gqa(GqaTile::Q2)
            } else {
                TqHbRowKernel::Scalar
            };
            let nh = N_HEADS as usize;
            let hd = HEAD_DIM as usize;
            let views: Vec<(MlxBuffer, MlxBuffer, TqSlot)> = (0..w as usize)
                .map(|r| {
                    Ok((
                        row_view(q, r, vec![1, nh, hd])?,
                        row_view(out, r, vec![1, nh, hd])?,
                        slots.slot(r),
                    ))
                })
                .collect::<R<_>>()?;
            let rows: Vec<TqHbDecodeRow<'_>> = views
                .iter()
                .zip(tmps)
                .map(|((q, o, s), tmp)| TqHbDecodeRow {
                    q,
                    k_packed: &s.k_packed,
                    k_norms: &s.k_norms,
                    v_packed: &s.v_packed,
                    v_norms: &s.v_norms,
                    output: o,
                    tmp,
                    params: p,
                    kernel,
                })
                .collect();
            flash_attn_vec_tq_hb_rows(enc, reg, dev, &rows)?;
        }
        AttnRoute::Batched => {
            flash_attn_vec_tq_hb_batched(
                enc,
                reg,
                dev,
                w,
                q,
                &slots.all.k_packed,
                &slots.all.k_norms,
                &slots.all.v_packed,
                &slots.all.v_norms,
                out,
                batched_tmp,
                slot_ids,
                cursors,
                &p,
            )?;
        }
    }
    Ok(())
}

fn batched_tmp(dev: &MlxDevice, w: u32) -> R<MlxBuffer> {
    let bytes = tq_hb_tmp_bytes(w * N_HEADS, HEAD_DIM);
    alloc(dev, DType::F32, vec![bytes / 4])
}

fn hint(s: &'static str) -> &'static str {
    s
}

fn prepare(
    case: &Case,
    dev: &MlxDevice,
    shared: &Shared,
    rng: &mut Rng,
    cold_bytes: u64,
) -> R<Prepared> {
    let w = case.w;
    let wu = w as usize;
    let cold = case.mode != Mode::Warm;
    let h = HIDDEN as usize;
    let mut extra = json!({});
    let p = match case.op {
        Op::Embedding => {
            let n_sets = if cold { ID_SETS } else { 1 };
            let mut ids = Vec::with_capacity(n_sets);
            for _ in 0..n_sets {
                let toks: Vec<u32> = (0..w).map(|_| rng.below(VOCAB)).collect();
                ids.push(u32_buf(dev, &toks)?);
            }
            let out = zeros(dev, DType::F32, vec![wu, h])?;
            let table = shared.embd.clone();
            let row = qbytes(GgmlType::Q4_K, 1, HIDDEN);
            Prepared {
                kernel_hint: hint("embedding_gather_q4_k_f32"),
                m: json!(w),
                n: json!(HIDDEN),
                k: json!(VOCAB),
                qtype: "q4_K",
                bytes: w as u64 * (row + 4 * HIDDEN as u64 + 4),
                sets: n_sets,
                footprint: if cold { table.byte_len() as u64 } else { w as u64 * row },
                extra,
                encode: unit(move |enc, reg, dev, i| {
                    let params = EmbeddingQ4KParams {
                        vocab_size: VOCAB as usize,
                        embed_dim: HIDDEN as usize,
                        n_tokens: wu,
                    };
                    embedding_gather_q4_k(enc, reg, dev, &table, &ids[i % ids.len()], &out, &params)?;
                    Ok(())
                }),
            }
        }
        Op::RmsNorm { rows_per_w, dim } => {
            let rows = rows_per_w * w;
            let input = f32_buf(dev, vec![rows as usize, dim as usize], rng, -1.0, 1.0)?;
            let weight = f32_buf(dev, vec![dim as usize], rng, 0.5, 1.5)?;
            let out = zeros(dev, DType::F32, vec![rows as usize, dim as usize])?;
            let params = f32_params(dev, &[EPS, dim as f32])?;
            Prepared {
                kernel_hint: hint("rms_norm_f32_v2"),
                m: json!(rows),
                n: json!(dim),
                k: json!(null),
                qtype: "f32",
                bytes: (2 * rows * dim + dim) as u64 * 4,
                sets: 1,
                footprint: (2 * rows * dim + dim) as u64 * 4,
                extra,
                encode: unit(move |enc, reg, dev, _| {
                    rms(enc, reg, dev, &input, &weight, &out, &params, rows, dim)
                }),
            }
        }
        Op::FusedResidualNorm => {
            let resid = f32_buf(dev, vec![wu, h], rng, -1.0, 1.0)?;
            let input = f32_buf(dev, vec![wu, h], rng, -1.0, 1.0)?;
            let weight = f32_buf(dev, vec![h], rng, 0.5, 1.5)?;
            let normed = zeros(dev, DType::F32, vec![wu, h])?;
            let sum = zeros(dev, DType::F32, vec![wu, h])?;
            Prepared {
                kernel_hint: hint("fused_residual_norm_f32"),
                m: json!(w),
                n: json!(HIDDEN),
                k: json!(null),
                qtype: "f32",
                bytes: (4 * w * HIDDEN + HIDDEN) as u64 * 4,
                sets: 1,
                footprint: (4 * w * HIDDEN + HIDDEN) as u64 * 4,
                extra,
                encode: unit(move |enc, reg, dev, _| {
                    dispatch_fused_residual_norm_f32(
                        enc,
                        reg,
                        dev.metal_device(),
                        &resid,
                        &input,
                        &weight,
                        &normed,
                        Some(&sum),
                        w,
                        HIDDEN,
                        EPS,
                    )?;
                    Ok(())
                }),
            }
        }
        Op::QMatmul { n, k, qt } => {
            let wbytes = qbytes(qt, n, k);
            let act = 4 * w as u64 * (n + k) as u64;
            let n_sets = if n == VOCAB {
                1
            } else {
                sets_for(wbytes + act, cold, cold_bytes, 1, MAX_SETS)
            };
            let mut sets = Vec::with_capacity(n_sets);
            for _ in 0..n_sets {
                let weight = if n == VOCAB { shared.lm_head.clone() } else { quant_buf(dev, qt, n, k, rng)? };
                let input = f32_buf(dev, vec![wu, k as usize], rng, -1.0, 1.0)?;
                let out = zeros(dev, DType::F32, vec![wu, n as usize])?;
                sets.push((weight, input, out));
            }
            if n == VOCAB {
                extra = json!({"cold_note": "417 MB weight exceeds the on-chip cache; warm == cold"});
            }
            Prepared {
                kernel_hint: if w == 1 {
                    if qt == GgmlType::Q6_K { "kernel_mul_mv_q6_K_f32_nr2" } else { "kernel_mul_mv_q4_K_f32" }
                } else if qt == GgmlType::Q6_K {
                    "kernel_mul_mv_q6_K_f32_mN_r1_<w>"
                } else {
                    "kernel_mul_mv_q4_K_f32_mN_r1_<w>"
                },
                m: json!(w),
                n: json!(n),
                k: json!(k),
                qtype: qname(qt),
                bytes: wbytes + act,
                sets: n_sets,
                footprint: n_sets as u64 * (wbytes + act),
                extra,
                encode: unit(move |enc, reg, dev, i| {
                    let (weight, input, out) = &sets[i % sets.len()];
                    qmm(enc, reg, dev, input, weight, out, w, n, k, qt)
                }),
            }
        }
        Op::QGateDeinterleave => {
            let fused = f32_buf(dev, vec![wu, N_HEADS as usize, 2 * HEAD_DIM as usize], rng, -1.0, 1.0)?;
            let q = zeros(dev, DType::F32, vec![wu, N_HEADS as usize, HEAD_DIM as usize])?;
            let gate = zeros(dev, DType::F32, vec![wu, N_HEADS as usize, HEAD_DIM as usize])?;
            Prepared {
                kernel_hint: hint("q_gate_deinterleave_f32"),
                m: json!(w),
                n: json!(2 * Q_TOTAL),
                k: json!(null),
                qtype: "f32",
                bytes: 4 * 4 * (w * Q_TOTAL) as u64,
                sets: 1,
                footprint: 4 * 4 * (w * Q_TOTAL) as u64,
                extra,
                encode: unit(move |enc, reg, dev, _| {
                    dispatch_q_gate_deinterleave_f32(
                        enc,
                        reg,
                        dev,
                        &fused,
                        &q,
                        &gate,
                        QGateDeinterleaveParams { m: w, n_heads: N_HEADS, head_dim: HEAD_DIM },
                    )?;
                    Ok(())
                }),
            }
        }
        Op::SsmConv => {
            let ch = QKV_CH as usize;
            let st = ((CONV_K - 1) * QKV_CH) as usize;
            let x = f32_buf(dev, vec![wu, ch], rng, -1.0, 1.0)?;
            let y = zeros(dev, DType::F32, vec![wu, ch])?;
            let kernel = f32_buf(dev, vec![(CONV_K * QKV_CH) as usize], rng, -0.5, 0.5)?;
            let mut states = Vec::new();
            for _ in 0..w {
                states.push((f32_buf(dev, vec![st], rng, -1.0, 1.0)?, zeros(dev, DType::F32, vec![st])?));
            }
            let params = u32_buf(dev, &[QKV_CH, 1, 1, CONV_K])?;
            let per_row = 4 * (2 * QKV_CH + 2 * (CONV_K - 1) * QKV_CH) as u64;
            Prepared {
                kernel_hint: hint("ssm_conv_forward_f32 + ssm_conv_state_update_f32 (per row)"),
                m: json!(1),
                n: json!(QKV_CH),
                k: json!(CONV_K),
                qtype: "f32",
                bytes: w as u64 * per_row + 4 * (CONV_K * QKV_CH) as u64,
                sets: 1,
                footprint: w as u64 * per_row + 4 * (CONV_K * QKV_CH) as u64,
                extra: json!({"per_row_dispatches": 2, "row_barrier": false}),
                encode: unit(move |enc, reg, dev, _| {
                    let p = SsmConvParams { channels: QKV_CH, n_tokens: 1, n_seqs: 1, k_width: CONV_K };
                    for (row, (old, new)) in states.iter().enumerate() {
                        dispatch_ssm_conv(
                            enc,
                            reg,
                            dev.metal_device(),
                            &row_view(&x, row, vec![1, ch])?,
                            &kernel,
                            old,
                            new,
                            &row_view(&y, row, vec![1, ch])?,
                            &params,
                            p,
                        )?;
                    }
                    Ok(())
                }),
            }
        }
        Op::L2Norm => {
            let span = (N_K_HEADS * D_K) as usize;
            let src = f32_buf(dev, vec![wu * QKV_CH as usize], rng, -1.0, 1.0)?;
            let mut outs = Vec::new();
            for _ in 0..2 * w {
                outs.push(zeros(dev, DType::F32, vec![N_K_HEADS as usize, D_K as usize])?);
            }
            let params = f32_params(dev, &[EPS, D_K as f32])?;
            Prepared {
                kernel_hint: hint("l2_norm_f32 (q and k per row)"),
                m: json!(N_K_HEADS),
                n: json!(D_K),
                k: json!(null),
                qtype: "f32",
                bytes: 2 * w as u64 * 2 * 4 * span as u64,
                sets: 1,
                footprint: 2 * w as u64 * 2 * 4 * span as u64,
                extra: json!({"per_row_dispatches": 2, "row_barrier": false}),
                encode: unit(move |enc, reg, dev, _| {
                    for row in 0..wu {
                        let base = row * QKV_CH as usize;
                        let q = src.slice_view((base * 4) as u64, span);
                        let k = src.slice_view(((base + span) * 4) as u64, span);
                        for (j, v) in [q, k].iter().enumerate() {
                            dispatch_l2_norm(
                                enc,
                                reg,
                                dev.metal_device(),
                                v,
                                &outs[2 * row + j],
                                &params,
                                N_K_HEADS,
                                D_K,
                            )?;
                        }
                    }
                    Ok(())
                }),
            }
        }
        Op::QScale => {
            let span = (N_K_HEADS * D_K) as usize;
            let mut bufs = Vec::new();
            for _ in 0..w {
                bufs.push((
                    f32_buf(dev, vec![N_K_HEADS as usize, D_K as usize], rng, -1.0, 1.0)?,
                    zeros(dev, DType::F32, vec![span])?,
                ));
            }
            Prepared {
                kernel_hint: hint("scalar_mul_f32 (per row)"),
                m: json!(1),
                n: json!(span),
                k: json!(null),
                qtype: "f32",
                bytes: w as u64 * 8 * span as u64,
                sets: 1,
                footprint: w as u64 * 8 * span as u64,
                extra: json!({"per_row_dispatches": 1, "row_barrier": false}),
                encode: unit(move |enc, reg, dev, _| {
                    let scale = 1.0 / (D_K as f32).sqrt();
                    for (src, dst) in &bufs {
                        scalar_mul_f32(enc, reg, dev.metal_device(), src, dst, span, scale)?;
                    }
                    Ok(())
                }),
            }
        }
        Op::ComputeGBeta => {
            let nv = N_V_HEADS as usize;
            let alpha = f32_buf(dev, vec![wu, nv], rng, -1.0, 1.0)?;
            let beta_l = f32_buf(dev, vec![wu, nv], rng, -1.0, 1.0)?;
            let dt_bias = f32_buf(dev, vec![nv], rng, -1.0, 1.0)?;
            let ssm_a = f32_buf(dev, vec![nv], rng, -1.0, -0.1)?;
            let g = zeros(dev, DType::F32, vec![wu * nv])?;
            let beta = zeros(dev, DType::F32, vec![wu * nv])?;
            let params = u32_buf(dev, &[N_V_HEADS, w])?;
            Prepared {
                kernel_hint: hint("compute_g_beta_f32"),
                m: json!(w),
                n: json!(N_V_HEADS),
                k: json!(null),
                qtype: "f32",
                bytes: 4 * (4 * w * N_V_HEADS + 2 * N_V_HEADS) as u64,
                sets: 1,
                footprint: 4 * (4 * w * N_V_HEADS + 2 * N_V_HEADS) as u64,
                extra,
                encode: unit(move |enc, reg, dev, _| {
                    dispatch_compute_g_beta(
                        enc,
                        reg,
                        dev.metal_device(),
                        &alpha,
                        &beta_l,
                        &dt_bias,
                        &ssm_a,
                        &g,
                        &beta,
                        &params,
                        w,
                        N_V_HEADS,
                    )?;
                    Ok(())
                }),
            }
        }
        Op::GatedDeltaNet => {
            let qk = (N_K_HEADS * D_K) as usize;
            let v_elems = (N_V_HEADS * D_V) as usize;
            let nv = N_V_HEADS as usize;
            let state = (D_K * D_V * N_V_HEADS) as usize;
            let per_row_state = 2 * 4 * state as u64;
            let n_sets = sets_for(w as u64 * per_row_state, cold, cold_bytes, 1, 64);
            let q = f32_buf(dev, vec![wu * qk], rng, -0.1, 0.1)?;
            let k = f32_buf(dev, vec![wu * qk], rng, -0.1, 0.1)?;
            let v = f32_buf(dev, vec![wu * v_elems], rng, -1.0, 1.0)?;
            let g = f32_buf(dev, vec![wu * nv], rng, -0.2, -0.01)?;
            let beta = f32_buf(dev, vec![wu * nv], rng, 0.1, 0.9)?;
            let out = zeros(dev, DType::F32, vec![wu * v_elems])?;
            let mut sets = Vec::with_capacity(n_sets);
            for _ in 0..n_sets {
                let mut rows = Vec::with_capacity(wu);
                for _ in 0..w {
                    rows.push((f32_buf(dev, vec![state], rng, -0.1, 0.1)?, zeros(dev, DType::F32, vec![state])?));
                }
                sets.push(rows);
            }
            let params = u32_buf(dev, &[D_K, D_V, N_K_HEADS, N_V_HEADS, 1, 1, 0, 0, 0])?;
            Prepared {
                kernel_hint: hint("gated_delta_net_decode_f32_4 (per row)"),
                m: json!(1),
                n: json!(N_V_HEADS * D_V),
                k: json!(D_K),
                qtype: "f32",
                bytes: w as u64 * (per_row_state + 4 * (2 * qk + 2 * v_elems + 2 * nv) as u64),
                sets: n_sets,
                footprint: n_sets as u64 * w as u64 * per_row_state,
                extra: json!({"per_row_dispatches": 1, "row_barrier": false}),
                encode: unit(move |enc, reg, dev, i| {
                    let p = GatedDeltaNetParams {
                        d_k: D_K,
                        d_v: D_V,
                        n_k_heads: N_K_HEADS,
                        n_v_heads: N_V_HEADS,
                        n_tokens: 1,
                        n_seqs: 1,
                    };
                    for (row, (s_in, s_out)) in sets[i % sets.len()].iter().enumerate() {
                        dispatch_gated_delta_net_decode(
                            enc,
                            reg,
                            dev.metal_device(),
                            &q.slice_view((row * qk * 4) as u64, qk),
                            &k.slice_view((row * qk * 4) as u64, qk),
                            &v.slice_view((row * v_elems * 4) as u64, v_elems),
                            &g.slice_view((row * nv * 4) as u64, nv),
                            &beta.slice_view((row * nv * 4) as u64, nv),
                            s_in,
                            &out.slice_view((row * v_elems * 4) as u64, v_elems),
                            s_out,
                            &params,
                            p,
                        )?;
                    }
                    Ok(())
                }),
            }
        }
        Op::SsmNormGate => {
            let n = wu * Z_CH as usize;
            let attn = f32_buf(dev, vec![n], rng, -1.0, 1.0)?;
            let weight = f32_buf(dev, vec![D_V as usize], rng, 0.5, 1.5)?;
            let z = f32_buf(dev, vec![wu, Z_CH as usize], rng, -1.0, 1.0)?;
            let out = zeros(dev, DType::F32, vec![n])?;
            let params = f32_params(dev, &[EPS, D_V as f32])?;
            Prepared {
                kernel_hint: hint("ssm_norm_gate_f32"),
                m: json!(w * N_V_HEADS),
                n: json!(D_V),
                k: json!(null),
                qtype: "f32",
                bytes: 3 * 4 * n as u64,
                sets: 1,
                footprint: 3 * 4 * n as u64,
                extra,
                encode: unit(move |enc, reg, dev, _| {
                    dispatch_ssm_norm_gate(
                        enc,
                        reg,
                        dev.metal_device(),
                        &attn,
                        &weight,
                        &z,
                        &out,
                        &params,
                        w * N_V_HEADS,
                        D_V,
                    )?;
                    Ok(())
                }),
            }
        }
        Op::Rope { heads } => {
            let input = f32_buf(dev, vec![(w * heads) as usize, HEAD_DIM as usize], rng, -1.0, 1.0)?;
            let out = zeros(dev, DType::F32, vec![wu, heads as usize, HEAD_DIM as usize])?;
            let pos = positions_buf(dev, w, 4096)?;
            Prepared {
                kernel_hint: hint("rope_multi_f32 (IMROPE)"),
                m: json!(w * heads),
                n: json!(HEAD_DIM),
                k: json!(ROTARY_DIM),
                qtype: "f32",
                bytes: 2 * 4 * (w * heads * HEAD_DIM) as u64,
                sets: 1,
                footprint: 2 * 4 * (w * heads * HEAD_DIM) as u64,
                extra,
                encode: unit(move |enc, reg, dev, _| {
                    dispatch_rope_multi_cached(enc, reg, dev, &input, &out, &pos, rope_params(heads, w))?;
                    Ok(())
                }),
            }
        }
        Op::TqEncode => {
            let kv = 8192;
            let k_rows = f32_buf(dev, vec![wu, KV_TOTAL as usize], rng, -1.0, 1.0)?;
            let v_rows = f32_buf(dev, vec![wu, KV_TOTAL as usize], rng, -1.0, 1.0)?;
            let mut slots = Vec::new();
            for _ in 0..w {
                slots.push(TqSlot::new(dev, kv, rng)?);
            }
            Prepared {
                kernel_hint: hint("hadamard_quantize_kv_hb_d256 (K and V per row)"),
                m: json!(1),
                n: json!(KV_TOTAL),
                k: json!(null),
                qtype: "tq8",
                bytes: w as u64 * 2 * (4 * KV_TOTAL as u64 + KV_TOTAL as u64 + 4 * N_KV_HEADS as u64),
                sets: 1,
                footprint: w as u64 * TqSlot::bytes(kv),
                extra: json!({"per_row_dispatches": 2, "row_barrier": true, "kv_capacity": kv}),
                encode: unit(move |enc, reg, dev, _| {
                    for (row, slot) in slots.iter().enumerate() {
                        if row > 0 {
                            enc.memory_barrier();
                        }
                        tq_encode_kv(
                            enc,
                            reg,
                            dev,
                            &row_view(&k_rows, row, vec![1, KV_TOTAL as usize])?,
                            &row_view(&v_rows, row, vec![1, KV_TOTAL as usize])?,
                            slot,
                            kv,
                        )?;
                    }
                    Ok(())
                }),
            }
        }
        Op::FwhtPremult | Op::FwhtUndo => {
            let premult = matches!(case.op, Op::FwhtPremult);
            let mut bufs = Vec::new();
            for _ in 0..w {
                bufs.push(f32_buf(dev, vec![1, N_HEADS as usize, HEAD_DIM as usize], rng, -1.0, 1.0)?);
            }
            Prepared {
                kernel_hint: if premult { "fwht_sign_premult_f32_d256 (per row)" } else { "fwht_sign_undo_f32_d256 (per row)" },
                m: json!(N_HEADS),
                n: json!(HEAD_DIM),
                k: json!(null),
                qtype: "f32",
                bytes: w as u64 * 2 * 4 * Q_TOTAL as u64,
                sets: 1,
                footprint: w as u64 * 4 * Q_TOTAL as u64,
                extra: json!({"per_row_dispatches": 1, "row_barrier": true}),
                encode: unit(move |enc, reg, dev, _| {
                    for (row, b) in bufs.iter().enumerate() {
                        if row > 0 {
                            enc.memory_barrier();
                        }
                        if premult {
                            dispatch_fwht_sign_premult_f32(enc, reg, dev.metal_device(), b, N_HEADS, HEAD_DIM)?;
                        } else {
                            dispatch_fwht_sign_undo_f32(enc, reg, dev.metal_device(), b, N_HEADS, HEAD_DIM)?;
                        }
                    }
                    Ok(())
                }),
            }
        }
        Op::TqSdpa { kv } => {
            let n_sets = sets_for(w as u64 * TqSlot::bytes(kv), cold, cold_bytes, 1, 64);
            let mut sets = Vec::with_capacity(n_sets);
            for _ in 0..n_sets {
                let mut rows = Vec::with_capacity(wu);
                for _ in 0..w {
                    rows.push(TqSlot::new(dev, kv, rng)?);
                }
                sets.push(rows);
            }
            let mut rows = Vec::new();
            for _ in 0..w {
                rows.push((
                    f32_buf(dev, vec![1, N_HEADS as usize, HEAD_DIM as usize], rng, -1.0, 1.0)?,
                    zeros(dev, DType::F32, vec![1, N_HEADS as usize, 1, HEAD_DIM as usize])?,
                    sdpa_tmp(dev)?,
                ));
            }
            Prepared {
                kernel_hint: if tq_uses_gqa_q2(kv) {
                    "flash_attn_vec_tq_hb_gqa_q2_dk256 + flash_attn_vec_reduce_dk256 (per row)"
                } else {
                    "flash_attn_vec_tq_hb_dk256 + flash_attn_vec_reduce_dk256 (per row)"
                },
                m: json!(N_HEADS),
                n: json!(HEAD_DIM),
                k: json!(kv),
                qtype: "tq8",
                bytes: w as u64 * (TqSlot::bytes(kv) + 2 * 4 * Q_TOTAL as u64),
                sets: n_sets,
                footprint: n_sets as u64 * w as u64 * TqSlot::bytes(kv),
                extra: json!({"per_row_dispatches": 2, "row_barrier": true, "kv_len": kv,
                    "kv_capacity": kv, "gqa_q2": tq_uses_gqa_q2(kv), "nsg": compute_nsg(kv),
                    "peer_case_unified": format!(
                        "FLASH_ATTN_EXT(hsk=256,hsv=256,nh=2,nr23=[8,1],kv={},nb={w},type_K=f16,type_V=f16) (peer serving layout: unified KV, kv = w*L)",
                        w * kv
                    ),
                    "peer_match_note": "match peer perf lines on the named fields only; test-backend-ops prints more vars"}),
                encode: unit(move |enc, reg, dev, i| {
                    for (row, slot) in sets[i % sets.len()].iter().enumerate() {
                        if row > 0 {
                            enc.memory_barrier();
                        }
                        let (q, out, tmp) = &rows[row];
                        tq_sdpa(enc, reg, dev, q, slot, out, tmp, kv)?;
                    }
                    Ok(())
                }),
            }
        }
        Op::TqSdpaRows { kv, route } => {
            let n_sets = sets_for(w as u64 * TqSlot::bytes(kv), cold, cold_bytes, 1, 64);
            let mut sets = Vec::with_capacity(n_sets);
            for _ in 0..n_sets {
                sets.push(TqSlots::new(dev, w, kv, rng)?);
            }
            let q = f32_buf(dev, vec![wu, N_HEADS as usize, HEAD_DIM as usize], rng, -1.0, 1.0)?;
            let out = zeros(dev, DType::F32, vec![wu, N_HEADS as usize, HEAD_DIM as usize])?;
            let tmps = (0..w).map(|_| sdpa_tmp(dev)).collect::<R<Vec<_>>>()?;
            let btmp = batched_tmp(dev, w)?;
            let slot_ids = u32_buf(dev, &(0..w).collect::<Vec<_>>())?;
            let cursors = u32_buf(dev, &vec![kv - 1; wu])?;
            Prepared {
                kernel_hint: match route {
                    AttnRoute::Rows => "flash_attn_vec_tq_hb_rows: w mains (scalar / gqa_q2), barrier, w reduces",
                    AttnRoute::Batched => "flash_attn_vec_tq_hb_batched_dk256 + reduce_tq_hb_undo (one dispatch each)",
                },
                m: json!(N_HEADS),
                n: json!(HEAD_DIM),
                k: json!(kv),
                qtype: "tq8",
                bytes: w as u64 * (TqSlot::bytes(kv) + 2 * 4 * Q_TOTAL as u64),
                sets: n_sets,
                footprint: n_sets as u64 * w as u64 * TqSlot::bytes(kv),
                extra: json!({"route": route.tag(), "kv_len": kv, "kv_capacity": kv,
                    "gqa_q2": matches!(route, AttnRoute::Rows) && tq_uses_gqa_q2(kv),
                    "nsg": compute_nsg(kv)}),
                encode: unit(move |enc, reg, dev, i| {
                    let slots = &sets[i % sets.len()];
                    tq_sdpa_rows(enc, reg, dev, route, &q, slots, &out, &tmps, &btmp, &slot_ids, &cursors, w)
                }),
            }
        }
        Op::TqEncodeRows => {
            let kv = 8192;
            let k_rows = f32_buf(dev, vec![wu, KV_TOTAL as usize], rng, -1.0, 1.0)?;
            let v_rows = f32_buf(dev, vec![wu, KV_TOTAL as usize], rng, -1.0, 1.0)?;
            let slots = TqSlots::new(dev, w, kv, rng)?;
            let slot_ids = u32_buf(dev, &(0..w).collect::<Vec<_>>())?;
            let cursors = u32_buf(dev, &vec![kv - 1; wu])?;
            Prepared {
                kernel_hint: hint("hadamard_quantize_kv_hb_batched_d256 (K, V; all rows)"),
                m: json!(w),
                n: json!(KV_TOTAL),
                k: json!(null),
                qtype: "tq8",
                bytes: w as u64 * 2 * (4 * KV_TOTAL as u64 + KV_TOTAL as u64 + 4 * N_KV_HEADS as u64),
                sets: 1,
                footprint: w as u64 * TqSlot::bytes(kv),
                extra: json!({"dispatches": 2, "row_barrier": false, "kv_capacity": kv}),
                encode: unit(move |enc, reg, dev, _| {
                    tq_encode_kv_rows(enc, reg, dev, &k_rows, &v_rows, &slots, &slot_ids, &cursors, w)
                }),
            }
        }
        Op::FwhtUndoSigmoid => {
            let n = wu * Q_TOTAL as usize;
            let data = f32_buf(dev, vec![n], rng, -1.0, 1.0)?;
            let gate = f32_buf(dev, vec![n], rng, -1.0, 1.0)?;
            let out = zeros(dev, DType::F32, vec![n])?;
            Prepared {
                kernel_hint: hint("fwht_sign_undo_sigmoid_mul_f32_d256 (all rows)"),
                m: json!(w * N_HEADS),
                n: json!(HEAD_DIM),
                k: json!(null),
                qtype: "f32",
                bytes: 3 * 4 * n as u64,
                sets: 1,
                footprint: 3 * 4 * n as u64,
                extra: json!({"dispatches": 1}),
                encode: unit(move |enc, reg, dev, _| {
                    dispatch_fwht_sign_undo_sigmoid_mul_f32(
                        enc,
                        reg,
                        dev.metal_device(),
                        &data,
                        &gate,
                        &out,
                        w * N_HEADS,
                        HEAD_DIM,
                    )?;
                    Ok(())
                }),
            }
        }
        Op::RowCopy { width } => {
            let mut srcs = Vec::new();
            for _ in 0..w {
                srcs.push(f32_buf(dev, vec![width as usize], rng, -1.0, 1.0)?);
            }
            let dst = zeros(dev, DType::F32, vec![wu, width as usize])?;
            Prepared {
                kernel_hint: hint("offset_copy_f32 (per row)"),
                m: json!(1),
                n: json!(width),
                k: json!(null),
                qtype: "f32",
                bytes: w as u64 * 8 * width as u64,
                sets: 1,
                footprint: w as u64 * 8 * width as u64,
                extra: json!({"per_row_dispatches": 1, "row_barrier": false}),
                encode: unit(move |enc, reg, dev, _| {
                    for (row, src) in srcs.iter().enumerate() {
                        dispatch_copy_f32(
                            enc,
                            reg,
                            dev.metal_device(),
                            src,
                            &dst,
                            0,
                            row * width as usize,
                            width as usize,
                        )?;
                    }
                    Ok(())
                }),
            }
        }
        Op::SigmoidMul => {
            let n = w * Q_TOTAL;
            let x = f32_buf(dev, vec![wu, Q_TOTAL as usize], rng, -1.0, 1.0)?;
            let gate = f32_buf(dev, vec![wu, Q_TOTAL as usize], rng, -1.0, 1.0)?;
            let out = zeros(dev, DType::F32, vec![n as usize])?;
            let params = u32_buf(dev, &[n])?;
            // hf2q passes the packed attention `[w, 4096]` and gate `[w, 4096]`.
            let x = x.with_shape(vec![n as usize])?;
            let gate = gate.with_shape(vec![n as usize])?;
            Prepared {
                kernel_hint: hint("sigmoid_mul_f32"),
                m: json!(w),
                n: json!(Q_TOTAL),
                k: json!(null),
                qtype: "f32",
                bytes: 3 * 4 * n as u64,
                sets: 1,
                footprint: 3 * 4 * n as u64,
                extra,
                encode: unit(move |enc, reg, dev, _| {
                    dispatch_sigmoid_mul(enc, reg, dev.metal_device(), &x, &gate, &out, &params, n)?;
                    Ok(())
                }),
            }
        }
        Op::Bf16Gemv(role) => {
            let (n, k) = bf16_shape(role);
            let wbytes = 2 * (n * k) as u64;
            let n_sets = sets_for(wbytes, cold, cold_bytes, 1, MAX_SETS);
            let mut weights = Vec::with_capacity(n_sets);
            for _ in 0..n_sets {
                weights.push(bf16_buf(dev, n, k, rng)?);
            }
            let mut rows = Vec::new();
            for _ in 0..w {
                rows.push((
                    f32_buf(dev, vec![1, k as usize], rng, -1.0, 1.0)?,
                    zeros(dev, DType::F32, vec![1, n as usize])?,
                ));
            }
            Prepared {
                kernel_hint: hint("hf2q_dense_gemv_bf16_f32_4 (per row)"),
                m: json!(1),
                n: json!(n),
                k: json!(k),
                qtype: "bf16",
                bytes: w as u64 * (wbytes + 4 * (n + k) as u64),
                sets: n_sets,
                footprint: n_sets as u64 * wbytes,
                extra: json!({"per_row_dispatches": 1, "row_barrier": true,
                    "weight_note": "hf2q dequantizes this tensor at load and uploads BF16"}),
                encode: unit(move |enc, reg, dev, i| {
                    // One BF16 weight per layer; the w rows read the same weight.
                    let weight = &weights[i % weights.len()];
                    for (row, (x, y)) in rows.iter().enumerate() {
                        if row > 0 {
                            enc.memory_barrier();
                        }
                        gemv_bf16(enc, reg, dev, weight, x, y, n, k)?;
                    }
                    Ok(())
                }),
            }
        }
        Op::SoftmaxTopk => {
            let mut rows = Vec::new();
            for _ in 0..w {
                rows.push((
                    f32_buf(dev, vec![1, N_EXPERTS as usize], rng, -2.0, 2.0)?,
                    zeros(dev, DType::U32, vec![TOP_K as usize])?,
                    zeros(dev, DType::F32, vec![TOP_K as usize])?,
                ));
            }
            Prepared {
                kernel_hint: hint("moe_softmax_topk_f32 (per row)"),
                m: json!(1),
                n: json!(N_EXPERTS),
                k: json!(TOP_K),
                qtype: "f32",
                bytes: w as u64 * 4 * (N_EXPERTS + 2 * TOP_K) as u64,
                sets: 1,
                footprint: w as u64 * 4 * (N_EXPERTS + 2 * TOP_K) as u64,
                extra: json!({"per_row_dispatches": 1, "row_barrier": true}),
                encode: unit(move |enc, reg, dev, _| {
                    for (row, (logits, ids, wts)) in rows.iter().enumerate() {
                        if row > 0 {
                            enc.memory_barrier();
                        }
                        dispatch_moe_softmax_topk(enc, reg, dev, logits, ids, wts, 1, N_EXPERTS, TOP_K)?;
                    }
                    Ok(())
                }),
            }
        }
        Op::SiluMul { rows: r } => {
            let width = if r == 1 { SHARED_I } else { MOE_I };
            let n = r * width;
            let mut rows = Vec::new();
            for _ in 0..w {
                rows.push((
                    f32_buf(dev, vec![r as usize, width as usize], rng, -1.0, 1.0)?,
                    f32_buf(dev, vec![r as usize, width as usize], rng, -1.0, 1.0)?,
                    zeros(dev, DType::F32, vec![r as usize, width as usize])?,
                ));
            }
            let params = u32_buf(dev, &[n])?;
            Prepared {
                kernel_hint: hint("silu_mul_f32 (per row)"),
                m: json!(r),
                n: json!(width),
                k: json!(null),
                qtype: "f32",
                bytes: w as u64 * 3 * 4 * n as u64,
                sets: 1,
                footprint: w as u64 * 3 * 4 * n as u64,
                extra: json!({"per_row_dispatches": 1, "row_barrier": true}),
                encode: unit(move |enc, reg, dev, _| {
                    for (row, (a, b, out)) in rows.iter().enumerate() {
                        if row > 0 {
                            enc.memory_barrier();
                        }
                        dispatch_silu_mul(enc, reg, dev.metal_device(), a, b, out, &params, n)?;
                    }
                    Ok(())
                }),
            }
        }
        Op::MoeId { down, qt } => {
            let (n, k) = if down { (HIDDEN, MOE_I) } else { (MOE_I, HIDDEN) };
            let stack = if down { shared.down(qt).clone() } else { shared.gate_exps.clone() };
            let stride = qbytes(qt, n, 1 * k);
            let (n_tokens, top_k) = if down { (TOP_K, 1) } else { (1, TOP_K) };
            let n_id_sets = if cold { ID_SETS } else { 1 };
            let mut ids = Vec::with_capacity(n_id_sets);
            for _ in 0..n_id_sets {
                let mut per_row = Vec::with_capacity(wu);
                for _ in 0..w {
                    per_row.push(u32_buf(dev, &random_ids(rng, TOP_K as usize))?);
                }
                ids.push(per_row);
            }
            let mut rows = Vec::new();
            for _ in 0..w {
                rows.push((
                    f32_buf(dev, vec![n_tokens as usize, k as usize], rng, -1.0, 1.0)?,
                    zeros(dev, DType::F32, vec![TOP_K as usize, n as usize])?,
                ));
            }
            let mut scratch = IdMmScratch::alloc(dev, N_EXPERTS, TOP_K)?;
            let params = GgmlQuantizedMatmulIdParams {
                n_tokens,
                top_k,
                n,
                k,
                n_experts: N_EXPERTS,
                expert_stride: stride,
                ggml_type: qt,
            };
            Prepared {
                kernel_hint: if qt == GgmlType::Q6_K {
                    "kernel_mul_mv_id_q6_K_f32_nr2 (per row)"
                } else {
                    "kernel_mul_mv_id_q4_K_f32 (per row)"
                },
                m: json!(TOP_K),
                n: json!(n),
                k: json!(k),
                qtype: qname(qt),
                bytes: w as u64 * (TOP_K as u64 * stride + 4 * (n_tokens * k + TOP_K * n) as u64),
                sets: n_id_sets,
                footprint: if cold { stack.byte_len() as u64 } else { w as u64 * TOP_K as u64 * stride },
                extra: json!({"per_row_dispatches": 1, "row_barrier": true,
                    "pairs_per_row": TOP_K, "n_tokens": n_tokens, "top_k": top_k,
                    "cold_note": "cold rotates distinct random top-8 expert sets over the full 256-expert stack"}),
                encode: unit(move |enc, reg, dev, i| {
                    let set = &ids[i % ids.len()];
                    for (row, (x, y)) in rows.iter().enumerate() {
                        if row > 0 {
                            enc.memory_barrier();
                        }
                        quantized_matmul_id_ggml_pooled(enc, reg, dev, x, &stack, &set[row], y, &mut scratch, &params)?;
                    }
                    Ok(())
                }),
            }
        }
        Op::WeightedReduce => {
            let mut rows = Vec::new();
            for _ in 0..w {
                rows.push((
                    f32_buf(dev, vec![TOP_K as usize], rng, 0.0, 0.25)?,
                    f32_buf(dev, vec![TOP_K as usize, h], rng, -1.0, 1.0)?,
                    f32_buf(dev, vec![1, 1], rng, -1.0, 1.0)?,
                    f32_buf(dev, vec![1, h], rng, -1.0, 1.0)?,
                    f32_buf(dev, vec![1, h], rng, -1.0, 1.0)?,
                    zeros(dev, DType::F32, vec![1, h])?,
                ));
            }
            Prepared {
                kernel_hint: hint("moe_weighted_reduce_f32 (per row)"),
                m: json!(1),
                n: json!(HIDDEN),
                k: json!(TOP_K),
                qtype: "f32",
                bytes: w as u64 * 4 * ((TOP_K + 3) * HIDDEN) as u64,
                sets: 1,
                footprint: w as u64 * 4 * ((TOP_K + 3) * HIDDEN) as u64,
                extra: json!({"per_row_dispatches": 1, "row_barrier": true}),
                encode: unit(move |enc, reg, dev, _| {
                    for (row, (wts, y, sh, ys, resid, out)) in rows.iter_mut().enumerate() {
                        if row > 0 {
                            enc.memory_barrier();
                        }
                        dispatch_moe_weighted_reduce(enc, reg, dev, wts, y, sh, ys, resid, out, 1, TOP_K, HIDDEN, true)?;
                    }
                    Ok(())
                }),
            }
        }
        Op::Argmax => {
            let logits = f32_buf(dev, vec![wu, VOCAB as usize], rng, -10.0, 10.0)?;
            let mut outs = Vec::new();
            for _ in 0..w {
                outs.push((zeros(dev, DType::U32, vec![1])?, zeros(dev, DType::F32, vec![1])?));
            }
            let params = u32_buf(dev, &[VOCAB])?;
            Prepared {
                kernel_hint: hint("argmax_f32 (per greedy row)"),
                m: json!(1),
                n: json!(VOCAB),
                k: json!(null),
                qtype: "f32",
                bytes: w as u64 * 4 * VOCAB as u64,
                sets: 1,
                footprint: w as u64 * 4 * VOCAB as u64,
                extra: json!({"per_row_dispatches": 1, "row_barrier": false}),
                encode: unit(move |enc, reg, dev, _| {
                    for (row, (idx, val)) in outs.iter().enumerate() {
                        dispatch_argmax_f32(
                            enc,
                            reg,
                            dev.metal_device(),
                            &row_view(&logits, row, vec![1, VOCAB as usize])?,
                            idx,
                            val,
                            &params,
                            VOCAB,
                        )?;
                    }
                    Ok(())
                }),
            }
        }
        Op::LayerGdn { qkv } => prepare_layer_gdn(dev, rng, w, qkv, cold_bytes)?,
        Op::LayerAttn { v, kv } => prepare_layer_attn(dev, rng, w, v, kv, cold_bytes)?,
        Op::LayerAttnRows { v, kv, route } => {
            prepare_layer_attn_rows(dev, rng, w, v, kv, route, cold_bytes)?
        }
        Op::LayerFfn { down } => prepare_layer_ffn(dev, shared, rng, w, down, cold_bytes)?,
        Op::Head => prepare_head(dev, shared, rng, w)?,
    };
    Ok(p)
}

// ======================================================================
// As-dispatched layer sequences (hf2q joint step; serial step at w = 1)
// ======================================================================

struct GdnLayerWeights {
    attn_norm: MlxBuffer,
    qkv: MlxBuffer,
    gate: MlxBuffer,
    alpha: MlxBuffer,
    beta: MlxBuffer,
    conv1d: MlxBuffer,
    dt_bias: MlxBuffer,
    ssm_a: MlxBuffer,
    ssm_norm: MlxBuffer,
    ssm_out: MlxBuffer,
    /// Per row: conv state in/out, recurrent state in/out (ping-pong halves).
    slots: Vec<(MlxBuffer, MlxBuffer, MlxBuffer, MlxBuffer)>,
    x: MlxBuffer,
}

fn prepare_layer_gdn(dev: &MlxDevice, rng: &mut Rng, w: u32, qkv_t: GgmlType, cold_bytes: u64) -> R<Prepared> {
    let wu = w as usize;
    let h = HIDDEN as usize;
    let st_conv = ((CONV_K - 1) * QKV_CH) as usize;
    let st_rec = (D_K * D_V * N_V_HEADS) as usize;
    let wbytes = qbytes(qkv_t, QKV_CH, HIDDEN)
        + qbytes(GgmlType::Q4_K, Z_CH, HIDDEN)
        + 2 * qbytes(GgmlType::Q4_K, N_V_HEADS, HIDDEN)
        + qbytes(GgmlType::Q4_K, HIDDEN, Z_CH)
        + 4 * (CONV_K * QKV_CH) as u64;
    let state_bytes = w as u64 * 4 * 2 * (st_conv + st_rec) as u64;
    let n_sets = sets_for(wbytes + state_bytes, true, cold_bytes, 2, 32);
    let mut sets = Vec::with_capacity(n_sets);
    for _ in 0..n_sets {
        let mut slots = Vec::with_capacity(wu);
        for _ in 0..w {
            slots.push((
                f32_buf(dev, vec![st_conv], rng, -1.0, 1.0)?,
                zeros(dev, DType::F32, vec![st_conv])?,
                f32_buf(dev, vec![st_rec], rng, -0.1, 0.1)?,
                zeros(dev, DType::F32, vec![st_rec])?,
            ));
        }
        sets.push(GdnLayerWeights {
            attn_norm: f32_buf(dev, vec![h], rng, 0.5, 1.5)?,
            qkv: quant_buf(dev, qkv_t, QKV_CH, HIDDEN, rng)?,
            gate: quant_buf(dev, GgmlType::Q4_K, Z_CH, HIDDEN, rng)?,
            alpha: quant_buf(dev, GgmlType::Q4_K, N_V_HEADS, HIDDEN, rng)?,
            beta: quant_buf(dev, GgmlType::Q4_K, N_V_HEADS, HIDDEN, rng)?,
            conv1d: f32_buf(dev, vec![(CONV_K * QKV_CH) as usize], rng, -0.5, 0.5)?,
            dt_bias: f32_buf(dev, vec![N_V_HEADS as usize], rng, -1.0, 1.0)?,
            ssm_a: f32_buf(dev, vec![N_V_HEADS as usize], rng, -1.0, -0.1)?,
            ssm_norm: f32_buf(dev, vec![D_V as usize], rng, 0.5, 1.5)?,
            ssm_out: quant_buf(dev, GgmlType::Q4_K, HIDDEN, Z_CH, rng)?,
            slots,
            x: f32_buf(dev, vec![wu, h], rng, -1.0, 1.0)?,
        });
    }
    // Scratch activations (hf2q: pooled per step).
    let x_norm = zeros(dev, DType::F32, vec![wu, h])?;
    let qkv_raw = zeros(dev, DType::F32, vec![wu, QKV_CH as usize])?;
    let z = zeros(dev, DType::F32, vec![wu, Z_CH as usize])?;
    let alpha_logit = zeros(dev, DType::F32, vec![wu, N_V_HEADS as usize])?;
    let beta_logit = zeros(dev, DType::F32, vec![wu, N_V_HEADS as usize])?;
    let qkv_conv = zeros(dev, DType::F32, vec![wu, QKV_CH as usize])?;
    let span = (N_K_HEADS * D_K) as usize;
    let mut q_l2 = Vec::new();
    let mut k_l2 = Vec::new();
    let mut q_scaled = Vec::new();
    for _ in 0..w {
        q_l2.push(zeros(dev, DType::F32, vec![N_K_HEADS as usize, D_K as usize])?);
        k_l2.push(zeros(dev, DType::F32, vec![N_K_HEADS as usize, D_K as usize])?);
        q_scaled.push(zeros(dev, DType::F32, vec![span])?);
    }
    let nv = N_V_HEADS as usize;
    let v_elems = (N_V_HEADS * D_V) as usize;
    let g_buf = zeros(dev, DType::F32, vec![wu * nv])?;
    let beta_buf = zeros(dev, DType::F32, vec![wu * nv])?;
    let attn_out = zeros(dev, DType::F32, vec![wu * v_elems])?;
    let gated = zeros(dev, DType::F32, vec![wu * v_elems])?;
    let gated_2d = gated.with_shape(vec![wu, Z_CH as usize])?;
    let output = zeros(dev, DType::F32, vec![wu, h])?;
    let norm_params = f32_params(dev, &[EPS, HIDDEN as f32])?;
    let ssm_params = u32_buf(dev, &[QKV_CH, 1, 1, CONV_K])?;
    let l2_params = f32_params(dev, &[EPS, D_K as f32])?;
    let g_params = u32_buf(dev, &[N_V_HEADS, w])?;
    let gdn_params = u32_buf(dev, &[D_K, D_V, N_K_HEADS, N_V_HEADS, 1, 1, 0, 0, 0])?;
    let gate_params = f32_params(dev, &[EPS, D_V as f32])?;
    let act = 4 * (w as u64) * (2 * HIDDEN + 3 * QKV_CH + 4 * Z_CH + 4 * N_V_HEADS) as u64;
    Ok(Prepared {
        kernel_hint: hint("GDN layer: 8 + 6w dispatches"),
        m: json!(w),
        n: json!(null),
        k: json!(null),
        qtype: qname(qkv_t),
        bytes: wbytes + state_bytes + act,
        sets: n_sets,
        footprint: n_sets as u64 * (wbytes + state_bytes),
        extra: json!({"layers_per_step": if qkv_t == GgmlType::Q6_K { 14 } else { 16 },
            "barriers_per_layer": 7}),
        encode: unit(move |enc, reg, dev, i| {
            let lw = &sets[i % sets.len()];
            let md = dev.metal_device();
            // ---- packed pre-norm and the four input projections ----
            rms(enc, reg, dev, &lw.x, &lw.attn_norm, &x_norm, &norm_params, w, HIDDEN)?;
            enc.memory_barrier();
            qmm(enc, reg, dev, &x_norm, &lw.qkv, &qkv_raw, w, QKV_CH, HIDDEN, qkv_t)?;
            qmm(enc, reg, dev, &x_norm, &lw.gate, &z, w, Z_CH, HIDDEN, GgmlType::Q4_K)?;
            qmm(enc, reg, dev, &x_norm, &lw.alpha, &alpha_logit, w, N_V_HEADS, HIDDEN, GgmlType::Q4_K)?;
            qmm(enc, reg, dev, &x_norm, &lw.beta, &beta_logit, w, N_V_HEADS, HIDDEN, GgmlType::Q4_K)?;
            enc.memory_barrier();
            // ---- per slot causal conv ----
            let qkv_ch = QKV_CH as usize;
            let conv_p = SsmConvParams { channels: QKV_CH, n_tokens: 1, n_seqs: 1, k_width: CONV_K };
            for (row, (c_in, c_out, _, _)) in lw.slots.iter().enumerate() {
                dispatch_ssm_conv(
                    enc,
                    reg,
                    md,
                    &row_view(&qkv_raw, row, vec![1, qkv_ch])?,
                    &lw.conv1d,
                    c_in,
                    c_out,
                    &row_view(&qkv_conv, row, vec![1, qkv_ch])?,
                    &ssm_params,
                    conv_p,
                )?;
            }
            enc.memory_barrier();
            // ---- per slot L2 norms of q and k ----
            for row in 0..wu {
                let base = row * qkv_ch;
                let qv = qkv_conv.slice_view((base * 4) as u64, span);
                let kvw = qkv_conv.slice_view(((base + span) * 4) as u64, span);
                dispatch_l2_norm(enc, reg, md, &qv, &q_l2[row], &l2_params, N_K_HEADS, D_K)?;
                dispatch_l2_norm(enc, reg, md, &kvw, &k_l2[row], &l2_params, N_K_HEADS, D_K)?;
            }
            enc.memory_barrier();
            // ---- per slot q scale; packed g/beta ----
            let q_scale = 1.0 / (D_K as f32).sqrt();
            for row in 0..wu {
                scalar_mul_f32(enc, reg, md, &q_l2[row], &q_scaled[row], span, q_scale)?;
            }
            dispatch_compute_g_beta(
                enc, reg, md, &alpha_logit, &beta_logit, &lw.dt_bias, &lw.ssm_a, &g_buf, &beta_buf,
                &g_params, w, N_V_HEADS,
            )?;
            enc.memory_barrier();
            // ---- per slot gated delta net ----
            let p = GatedDeltaNetParams {
                d_k: D_K,
                d_v: D_V,
                n_k_heads: N_K_HEADS,
                n_v_heads: N_V_HEADS,
                n_tokens: 1,
                n_seqs: 1,
            };
            for (row, (_, _, r_in, r_out)) in lw.slots.iter().enumerate() {
                let base = row * qkv_ch;
                let v_view = qkv_conv.slice_view(((base + 2 * span) * 4) as u64, v_elems);
                dispatch_gated_delta_net_decode(
                    enc,
                    reg,
                    md,
                    &q_scaled[row],
                    &k_l2[row],
                    &v_view,
                    &g_buf.slice_view((row * nv * 4) as u64, nv),
                    &beta_buf.slice_view((row * nv * 4) as u64, nv),
                    r_in,
                    &attn_out.slice_view((row * v_elems * 4) as u64, v_elems),
                    r_out,
                    &gdn_params,
                    p,
                )?;
            }
            enc.memory_barrier();
            // ---- packed gated norm and output projection ----
            dispatch_ssm_norm_gate(
                enc, reg, md, &attn_out, &lw.ssm_norm, &z, &gated, &gate_params, w * N_V_HEADS, D_V,
            )?;
            enc.memory_barrier();
            qmm(enc, reg, dev, &gated_2d, &lw.ssm_out, &output, w, HIDDEN, Z_CH, GgmlType::Q4_K)
        }),
    })
}

struct AttnLayerWeights {
    attn_norm: MlxBuffer,
    q_gate: MlxBuffer,
    wk: MlxBuffer,
    wv: MlxBuffer,
    q_norm: MlxBuffer,
    k_norm: MlxBuffer,
    wo: MlxBuffer,
    slots: Vec<TqSlot>,
    x: MlxBuffer,
}

fn prepare_layer_attn(
    dev: &MlxDevice,
    rng: &mut Rng,
    w: u32,
    v_t: GgmlType,
    kv: u32,
    cold_bytes: u64,
) -> R<Prepared> {
    let wu = w as usize;
    let h = HIDDEN as usize;
    let hd = HEAD_DIM as usize;
    let nh = N_HEADS as usize;
    let nkv = N_KV_HEADS as usize;
    let q_total = Q_TOTAL as usize;
    let wbytes = qbytes(GgmlType::Q4_K, 2 * Q_TOTAL, HIDDEN)
        + qbytes(GgmlType::Q4_K, KV_TOTAL, HIDDEN)
        + qbytes(v_t, KV_TOTAL, HIDDEN)
        + qbytes(GgmlType::Q4_K, HIDDEN, Q_TOTAL);
    let kv_bytes = w as u64 * TqSlot::bytes(kv);
    let n_sets = sets_for(wbytes + kv_bytes, true, cold_bytes, 2, 32);
    let mut sets = Vec::with_capacity(n_sets);
    for _ in 0..n_sets {
        let mut slots = Vec::with_capacity(wu);
        for _ in 0..w {
            slots.push(TqSlot::new(dev, kv, rng)?);
        }
        sets.push(AttnLayerWeights {
            attn_norm: f32_buf(dev, vec![h], rng, 0.5, 1.5)?,
            q_gate: quant_buf(dev, GgmlType::Q4_K, 2 * Q_TOTAL, HIDDEN, rng)?,
            wk: quant_buf(dev, GgmlType::Q4_K, KV_TOTAL, HIDDEN, rng)?,
            wv: quant_buf(dev, v_t, KV_TOTAL, HIDDEN, rng)?,
            q_norm: f32_buf(dev, vec![hd], rng, 0.5, 1.5)?,
            k_norm: f32_buf(dev, vec![hd], rng, 0.5, 1.5)?,
            wo: quant_buf(dev, GgmlType::Q4_K, HIDDEN, Q_TOTAL, rng)?,
            slots,
            x: f32_buf(dev, vec![wu, h], rng, -1.0, 1.0)?,
        });
    }
    let positions = positions_buf(dev, w, (kv - 1) as i32)?;
    let x_norm = zeros(dev, DType::F32, vec![wu, h])?;
    let fused = zeros(dev, DType::F32, vec![wu, 2 * q_total])?;
    let fused_3d = fused.with_shape(vec![wu, nh, 2 * hd])?;
    let q = zeros(dev, DType::F32, vec![wu, nh, hd])?;
    let gate = zeros(dev, DType::F32, vec![wu, nh, hd])?;
    let q_flat = q.with_shape(vec![wu, q_total])?;
    let gate_flat = gate.with_shape(vec![wu, q_total])?;
    let k_flat = zeros(dev, DType::F32, vec![wu, KV_TOTAL as usize])?;
    let v_flat = zeros(dev, DType::F32, vec![wu, KV_TOTAL as usize])?;
    let q_normed = zeros(dev, DType::F32, vec![wu * nh, hd])?;
    let k_normed = zeros(dev, DType::F32, vec![wu * nkv, hd])?;
    let q_rope = zeros(dev, DType::F32, vec![wu, nh, hd])?;
    let k_rope = zeros(dev, DType::F32, vec![wu, nkv, hd])?;
    let mut row_out = Vec::new();
    for _ in 0..w {
        row_out.push((zeros(dev, DType::F32, vec![1, nh, 1, hd])?, sdpa_tmp(dev)?));
    }
    let attention = zeros(dev, DType::F32, vec![wu, q_total])?;
    let gated = zeros(dev, DType::F32, vec![wu * q_total])?;
    let gated_2d = gated.with_shape(vec![wu, q_total])?;
    let output = zeros(dev, DType::F32, vec![wu, h])?;
    let norm_params = f32_params(dev, &[EPS, HIDDEN as f32])?;
    let head_params = f32_params(dev, &[EPS, HEAD_DIM as f32])?;
    let sig_params = u32_buf(dev, &[w * Q_TOTAL])?;
    let joint = w > 1;
    Ok(Prepared {
        kernel_hint: if joint { "full-attn layer: 11 + 7w dispatches" } else { "full-attn layer (serial w=1): 17 dispatches" },
        m: json!(w),
        n: json!(null),
        k: json!(kv),
        qtype: qname(v_t),
        bytes: wbytes + kv_bytes + 4 * w as u64 * (2 * HIDDEN + 6 * Q_TOTAL + 4 * KV_TOTAL) as u64,
        sets: n_sets,
        footprint: n_sets as u64 * (wbytes + kv_bytes),
        extra: json!({"kv_len": kv, "kv_capacity": kv, "gqa_q2": tq_uses_gqa_q2(kv),
            "peer_attention_unified": format!(
                "FLASH_ATTN_EXT(hsk=256,hsv=256,nh=2,nr23=[8,1],kv={},nb={w},type_K=f16,type_V=f16)",
                w * kv
            ),
            "layers_per_step": if v_t == GgmlType::Q6_K { 6 } else { 4 },
            "row_copies": joint}),
        encode: unit(move |enc, reg, dev, i| {
            let lw = &sets[i % sets.len()];
            let md = dev.metal_device();
            // ---- packed pre-norm, projections, Q/K norms, IMROPE ----
            rms(enc, reg, dev, &lw.x, &lw.attn_norm, &x_norm, &norm_params, w, HIDDEN)?;
            enc.memory_barrier();
            qmm(enc, reg, dev, &x_norm, &lw.q_gate, &fused, w, 2 * Q_TOTAL, HIDDEN, GgmlType::Q4_K)?;
            enc.memory_barrier();
            dispatch_q_gate_deinterleave_f32(
                enc,
                reg,
                dev,
                &fused_3d,
                &q,
                &gate,
                QGateDeinterleaveParams { m: w, n_heads: N_HEADS, head_dim: HEAD_DIM },
            )?;
            qmm(enc, reg, dev, &x_norm, &lw.wk, &k_flat, w, KV_TOTAL, HIDDEN, GgmlType::Q4_K)?;
            qmm(enc, reg, dev, &x_norm, &lw.wv, &v_flat, w, KV_TOTAL, HIDDEN, v_t)?;
            enc.memory_barrier();
            rms(enc, reg, dev, &q_flat, &lw.q_norm, &q_normed, &head_params, w * N_HEADS, HEAD_DIM)?;
            rms(enc, reg, dev, &k_flat, &lw.k_norm, &k_normed, &head_params, w * N_KV_HEADS, HEAD_DIM)?;
            enc.memory_barrier();
            dispatch_rope_multi_cached(enc, reg, dev, &q_normed, &q_rope, &positions, rope_params(N_HEADS, w))?;
            dispatch_rope_multi_cached(enc, reg, dev, &k_normed, &k_rope, &positions, rope_params(N_KV_HEADS, w))?;
            enc.memory_barrier();
            // ---- per slot: TQ K/V write at the cursor + TQ SDPA ----
            for (row, slot) in lw.slots.iter().enumerate() {
                if row > 0 {
                    enc.memory_barrier();
                }
                let q_row = row_view(&q_rope, row, vec![1, nh, hd])?;
                let k_row = row_view(&k_rope, row, vec![1, nkv, hd])?;
                let v_row = row_view(&v_flat, row, vec![1, KV_TOTAL as usize])?;
                let (out, tmp) = &row_out[row];
                // write_kv_with_optional_tq_encode (TQ-only: no F32 copy)
                enc.memory_barrier();
                tq_encode_kv(enc, reg, dev, &k_row, &v_row, slot, kv)?;
                enc.memory_barrier();
                // dispatch_decode_sdpa_with_optional_tq
                dispatch_fwht_sign_premult_f32(enc, reg, md, &q_row, N_HEADS, HEAD_DIM)?;
                enc.memory_barrier();
                tq_sdpa(enc, reg, dev, &q_row, slot, out, tmp, kv)?;
                enc.memory_barrier();
                dispatch_fwht_sign_undo_f32(enc, reg, md, out, N_HEADS, HEAD_DIM)?;
            }
            enc.memory_barrier();
            if joint {
                for (row, (out, _)) in row_out.iter().enumerate() {
                    dispatch_copy_f32(enc, reg, md, out, &attention, 0, row * q_total, q_total)?;
                }
                enc.memory_barrier();
            }
            // ---- packed sigmoid gate and output projection ----
            // Serial w = 1 reads the SDPA output directly (no row copy).
            let attn_src = if joint { attention.clone() } else { row_out[0].0.with_shape(vec![q_total])? };
            let attn_src = attn_src.with_shape(vec![wu * q_total])?;
            let gate_src = gate_flat.with_shape(vec![wu * q_total])?;
            dispatch_sigmoid_mul(enc, reg, md, &attn_src, &gate_src, &gated, &sig_params, w * Q_TOTAL)?;
            enc.memory_barrier();
            qmm(enc, reg, dev, &gated_2d, &lw.wo, &output, w, HIDDEN, Q_TOTAL, GgmlType::Q4_K)
        }),
    })
}

struct AttnRowsLayerWeights {
    attn_norm: MlxBuffer,
    q_gate: MlxBuffer,
    wk: MlxBuffer,
    wv: MlxBuffer,
    q_norm: MlxBuffer,
    k_norm: MlxBuffer,
    wo: MlxBuffer,
    slots: TqSlots,
    x: MlxBuffer,
}

/// The full-attention layer with the per-row small ops batched over the w
/// rows (hf2q `encode_joint_full_attn_layer`, row-batched): one batched K
/// and one V encode, one Q premultiply over all heads, the rows' attention
/// (`route`) writing straight into the packed attention rows, then one fused
/// undo + sigmoid gate (rows route; the batched route's reduce already
/// un-rotates, so it keeps `sigmoid_mul`). No row copies, no row barriers.
fn prepare_layer_attn_rows(
    dev: &MlxDevice,
    rng: &mut Rng,
    w: u32,
    v_t: GgmlType,
    kv: u32,
    route: AttnRoute,
    cold_bytes: u64,
) -> R<Prepared> {
    let wu = w as usize;
    let h = HIDDEN as usize;
    let hd = HEAD_DIM as usize;
    let nh = N_HEADS as usize;
    let nkv = N_KV_HEADS as usize;
    let q_total = Q_TOTAL as usize;
    let wbytes = qbytes(GgmlType::Q4_K, 2 * Q_TOTAL, HIDDEN)
        + qbytes(GgmlType::Q4_K, KV_TOTAL, HIDDEN)
        + qbytes(v_t, KV_TOTAL, HIDDEN)
        + qbytes(GgmlType::Q4_K, HIDDEN, Q_TOTAL);
    let kv_bytes = w as u64 * TqSlot::bytes(kv);
    let n_sets = sets_for(wbytes + kv_bytes, true, cold_bytes, 2, 32);
    let mut sets = Vec::with_capacity(n_sets);
    for _ in 0..n_sets {
        sets.push(AttnRowsLayerWeights {
            attn_norm: f32_buf(dev, vec![h], rng, 0.5, 1.5)?,
            q_gate: quant_buf(dev, GgmlType::Q4_K, 2 * Q_TOTAL, HIDDEN, rng)?,
            wk: quant_buf(dev, GgmlType::Q4_K, KV_TOTAL, HIDDEN, rng)?,
            wv: quant_buf(dev, v_t, KV_TOTAL, HIDDEN, rng)?,
            q_norm: f32_buf(dev, vec![hd], rng, 0.5, 1.5)?,
            k_norm: f32_buf(dev, vec![hd], rng, 0.5, 1.5)?,
            wo: quant_buf(dev, GgmlType::Q4_K, HIDDEN, Q_TOTAL, rng)?,
            slots: TqSlots::new(dev, w, kv, rng)?,
            x: f32_buf(dev, vec![wu, h], rng, -1.0, 1.0)?,
        });
    }
    let positions = positions_buf(dev, w, (kv - 1) as i32)?;
    let slot_ids = u32_buf(dev, &(0..w).collect::<Vec<_>>())?;
    let cursors = u32_buf(dev, &vec![kv - 1; wu])?;
    let x_norm = zeros(dev, DType::F32, vec![wu, h])?;
    let fused = zeros(dev, DType::F32, vec![wu, 2 * q_total])?;
    let fused_3d = fused.with_shape(vec![wu, nh, 2 * hd])?;
    let q = zeros(dev, DType::F32, vec![wu, nh, hd])?;
    let gate = zeros(dev, DType::F32, vec![wu, nh, hd])?;
    let q_flat = q.with_shape(vec![wu, q_total])?;
    let gate_flat = gate.with_shape(vec![wu * q_total])?;
    let k_flat = zeros(dev, DType::F32, vec![wu, KV_TOTAL as usize])?;
    let v_flat = zeros(dev, DType::F32, vec![wu, KV_TOTAL as usize])?;
    let q_normed = zeros(dev, DType::F32, vec![wu * nh, hd])?;
    let k_normed = zeros(dev, DType::F32, vec![wu * nkv, hd])?;
    let q_rope = zeros(dev, DType::F32, vec![wu, nh, hd])?;
    let k_rope = zeros(dev, DType::F32, vec![wu, nkv, hd])?;
    let tmps = (0..w).map(|_| sdpa_tmp(dev)).collect::<R<Vec<_>>>()?;
    let btmp = batched_tmp(dev, w)?;
    let attention = zeros(dev, DType::F32, vec![wu * q_total])?;
    let gated = zeros(dev, DType::F32, vec![wu * q_total])?;
    let gated_2d = gated.with_shape(vec![wu, q_total])?;
    let output = zeros(dev, DType::F32, vec![wu, h])?;
    let norm_params = f32_params(dev, &[EPS, HIDDEN as f32])?;
    let head_params = f32_params(dev, &[EPS, HEAD_DIM as f32])?;
    let sig_params = u32_buf(dev, &[w * Q_TOTAL])?;
    Ok(Prepared {
        kernel_hint: match route {
            AttnRoute::Rows => "full-attn layer, rows batched: 13 + 2w dispatches",
            AttnRoute::Batched => "full-attn layer, batched attention kernel: 15 dispatches",
        },
        m: json!(w),
        n: json!(null),
        k: json!(kv),
        qtype: qname(v_t),
        bytes: wbytes + kv_bytes + 4 * w as u64 * (2 * HIDDEN + 6 * Q_TOTAL + 4 * KV_TOTAL) as u64,
        sets: n_sets,
        footprint: n_sets as u64 * (wbytes + kv_bytes),
        extra: json!({"route": route.tag(), "kv_len": kv, "kv_capacity": kv,
            "gqa_q2": matches!(route, AttnRoute::Rows) && tq_uses_gqa_q2(kv),
            "layers_per_step": if v_t == GgmlType::Q6_K { 6 } else { 4 },
            "row_copies": false}),
        encode: unit(move |enc, reg, dev, i| {
            let lw = &sets[i % sets.len()];
            let md = dev.metal_device();
            // ---- packed pre-norm, projections, Q/K norms, IMROPE (unchanged) ----
            rms(enc, reg, dev, &lw.x, &lw.attn_norm, &x_norm, &norm_params, w, HIDDEN)?;
            enc.memory_barrier();
            qmm(enc, reg, dev, &x_norm, &lw.q_gate, &fused, w, 2 * Q_TOTAL, HIDDEN, GgmlType::Q4_K)?;
            enc.memory_barrier();
            dispatch_q_gate_deinterleave_f32(
                enc,
                reg,
                dev,
                &fused_3d,
                &q,
                &gate,
                QGateDeinterleaveParams { m: w, n_heads: N_HEADS, head_dim: HEAD_DIM },
            )?;
            qmm(enc, reg, dev, &x_norm, &lw.wk, &k_flat, w, KV_TOTAL, HIDDEN, GgmlType::Q4_K)?;
            qmm(enc, reg, dev, &x_norm, &lw.wv, &v_flat, w, KV_TOTAL, HIDDEN, v_t)?;
            enc.memory_barrier();
            rms(enc, reg, dev, &q_flat, &lw.q_norm, &q_normed, &head_params, w * N_HEADS, HEAD_DIM)?;
            rms(enc, reg, dev, &k_flat, &lw.k_norm, &k_normed, &head_params, w * N_KV_HEADS, HEAD_DIM)?;
            enc.memory_barrier();
            dispatch_rope_multi_cached(enc, reg, dev, &q_normed, &q_rope, &positions, rope_params(N_HEADS, w))?;
            dispatch_rope_multi_cached(enc, reg, dev, &k_normed, &k_rope, &positions, rope_params(N_KV_HEADS, w))?;
            enc.memory_barrier();
            // ---- all rows: K/V encode at the cursors + Q premultiply ----
            tq_encode_kv_rows(enc, reg, dev, &k_rope, &v_flat, &lw.slots, &slot_ids, &cursors, w)?;
            dispatch_fwht_sign_premult_f32(enc, reg, md, &q_rope, w * N_HEADS, HEAD_DIM)?;
            enc.memory_barrier();
            tq_sdpa_rows(
                enc, reg, dev, route, &q_rope, &lw.slots, &attention, &tmps, &btmp, &slot_ids,
                &cursors, w,
            )?;
            enc.memory_barrier();
            // ---- packed un-rotation + sigmoid gate, output projection ----
            match route {
                AttnRoute::Rows => dispatch_fwht_sign_undo_sigmoid_mul_f32(
                    enc, reg, md, &attention, &gate_flat, &gated, w * N_HEADS, HEAD_DIM,
                )?,
                AttnRoute::Batched => dispatch_sigmoid_mul(
                    enc, reg, md, &attention, &gate_flat, &gated, &sig_params, w * Q_TOTAL,
                )?,
            }
            enc.memory_barrier();
            qmm(enc, reg, dev, &gated_2d, &lw.wo, &output, w, HIDDEN, Q_TOTAL, GgmlType::Q4_K)
        }),
    })
}

struct FfnLayerWeights {
    post_norm: MlxBuffer,
    router: MlxBuffer,
    shared_gate_inp: MlxBuffer,
    shared_gate: MlxBuffer,
    shared_up: MlxBuffer,
    shared_down: MlxBuffer,
    hidden: MlxBuffer,
    attention: MlxBuffer,
}

struct MoeRowScratch {
    logits: MlxBuffer,
    sh_logit: MlxBuffer,
    a_s: MlxBuffer,
    b_s: MlxBuffer,
    ids: MlxBuffer,
    weights: MlxBuffer,
    h_s: MlxBuffer,
    gate_all: MlxBuffer,
    up_all: MlxBuffer,
    y_s: MlxBuffer,
    h_all: MlxBuffer,
    y_all: MlxBuffer,
    out: MlxBuffer,
}

fn prepare_layer_ffn(
    dev: &MlxDevice,
    shared: &Shared,
    rng: &mut Rng,
    w: u32,
    down_t: GgmlType,
    cold_bytes: u64,
) -> R<Prepared> {
    let wu = w as usize;
    let h = HIDDEN as usize;
    let topk = TOP_K as usize;
    let dense_bytes = 2 * (N_EXPERTS * HIDDEN + HIDDEN + 3 * SHARED_I * HIDDEN) as u64;
    let n_sets = sets_for(dense_bytes, true, cold_bytes, 2, 64);
    let mut sets = Vec::with_capacity(n_sets);
    for _ in 0..n_sets {
        sets.push(FfnLayerWeights {
            post_norm: f32_buf(dev, vec![h], rng, 0.5, 1.5)?,
            router: bf16_buf(dev, N_EXPERTS, HIDDEN, rng)?,
            shared_gate_inp: bf16_buf(dev, 1, HIDDEN, rng)?,
            shared_gate: bf16_buf(dev, SHARED_I, HIDDEN, rng)?,
            shared_up: bf16_buf(dev, SHARED_I, HIDDEN, rng)?,
            shared_down: bf16_buf(dev, HIDDEN, SHARED_I, rng)?,
            hidden: f32_buf(dev, vec![wu, h], rng, -1.0, 1.0)?,
            attention: f32_buf(dev, vec![wu, h], rng, -1.0, 1.0)?,
        });
    }
    let gate_exps = shared.gate_exps.clone();
    let up_exps = shared.up_exps.clone();
    let down_exps = shared.down(down_t).clone();
    let gate_stride = qbytes(GgmlType::Q4_K, MOE_I, HIDDEN);
    let down_stride = qbytes(down_t, HIDDEN, MOE_I);
    let ffn_input = zeros(dev, DType::F32, vec![wu, h])?;
    let ffn_residual = zeros(dev, DType::F32, vec![wu, h])?;
    let mut rows = Vec::with_capacity(wu);
    for _ in 0..w {
        rows.push(MoeRowScratch {
            logits: zeros(dev, DType::F32, vec![1, N_EXPERTS as usize])?,
            sh_logit: zeros(dev, DType::F32, vec![1, 1])?,
            a_s: zeros(dev, DType::F32, vec![1, SHARED_I as usize])?,
            b_s: zeros(dev, DType::F32, vec![1, SHARED_I as usize])?,
            ids: zeros(dev, DType::U32, vec![topk])?,
            weights: zeros(dev, DType::F32, vec![topk])?,
            h_s: zeros(dev, DType::F32, vec![1, SHARED_I as usize])?,
            gate_all: zeros(dev, DType::F32, vec![topk, MOE_I as usize])?,
            up_all: zeros(dev, DType::F32, vec![topk, MOE_I as usize])?,
            y_s: zeros(dev, DType::F32, vec![1, h])?,
            h_all: zeros(dev, DType::F32, vec![topk, MOE_I as usize])?,
            y_all: zeros(dev, DType::F32, vec![topk, h])?,
            out: zeros(dev, DType::F32, vec![1, h])?,
        });
    }
    let packed_out = zeros(dev, DType::F32, vec![wu, h])?;
    // hf2q: thread-local Gate / Up / Down `IdMmScratch` slots (decode_pool).
    let mut scratch_gate = IdMmScratch::alloc(dev, N_EXPERTS, TOP_K)?;
    let mut scratch_up = IdMmScratch::alloc(dev, N_EXPERTS, TOP_K)?;
    let mut scratch_down = IdMmScratch::alloc(dev, N_EXPERTS, TOP_K)?;
    let silu_params = u32_buf(dev, &[TOP_K * MOE_I])?;
    let silu_sh_params = u32_buf(dev, &[SHARED_I])?;
    let gate_params = GgmlQuantizedMatmulIdParams {
        n_tokens: 1,
        top_k: TOP_K,
        n: MOE_I,
        k: HIDDEN,
        n_experts: N_EXPERTS,
        expert_stride: gate_stride,
        ggml_type: GgmlType::Q4_K,
    };
    let down_params = GgmlQuantizedMatmulIdParams {
        n_tokens: TOP_K,
        top_k: 1,
        n: HIDDEN,
        k: MOE_I,
        n_experts: N_EXPERTS,
        expert_stride: down_stride,
        ggml_type: down_t,
    };
    let expert_bytes = TOP_K as u64 * (2 * gate_stride + down_stride);
    let joint = w > 1;
    Ok(Prepared {
        kernel_hint: if joint { "FFN layer: 1 + 13w dispatches" } else { "FFN layer (serial w=1): 13 dispatches" },
        m: json!(w),
        n: json!(null),
        k: json!(null),
        qtype: qname(down_t),
        bytes: w as u64 * (dense_bytes + expert_bytes),
        sets: n_sets,
        footprint: n_sets as u64 * dense_bytes
            + (gate_exps.byte_len() + up_exps.byte_len() + down_exps.byte_len()) as u64,
        extra: json!({"layers_per_step": 20, "row_copies": joint,
            "routing": "GPU top-8 over random router logits; router copies rotate per repetition",
            "note": "per row: router + 4 shared-expert BF16 gemv, softmax_topk, 3 mul_mv_id, 2 silu_mul, weighted reduce; 6 barriers"}),
        encode: unit(move |enc, reg, dev, i| {
            let lw = &sets[i % sets.len()];
            let md = dev.metal_device();
            dispatch_fused_residual_norm_f32(
                enc,
                reg,
                md,
                &lw.hidden,
                &lw.attention,
                &lw.post_norm,
                &ffn_input,
                Some(&ffn_residual),
                w,
                HIDDEN,
                EPS,
            )?;
            enc.memory_barrier();
            for (row, s) in rows.iter_mut().enumerate() {
                if row > 0 {
                    enc.memory_barrier();
                }
                let x = row_view(&ffn_input, row, vec![1, h])?;
                let residual = row_view(&ffn_residual, row, vec![1, h])?;
                // Phase A: router + shared projections
                gemv_bf16(enc, reg, dev, &lw.router, &x, &s.logits, N_EXPERTS, HIDDEN)?;
                gemv_bf16(enc, reg, dev, &lw.shared_gate_inp, &x, &s.sh_logit, 1, HIDDEN)?;
                gemv_bf16(enc, reg, dev, &lw.shared_gate, &x, &s.a_s, SHARED_I, HIDDEN)?;
                gemv_bf16(enc, reg, dev, &lw.shared_up, &x, &s.b_s, SHARED_I, HIDDEN)?;
                enc.memory_barrier();
                // Phase B: GPU softmax + top-k, shared silu_mul
                dispatch_moe_softmax_topk(enc, reg, dev, &s.logits, &s.ids, &s.weights, 1, N_EXPERTS, TOP_K)?;
                dispatch_silu_mul(enc, reg, md, &s.a_s, &s.b_s, &s.h_s, &silu_sh_params, SHARED_I)?;
                enc.memory_barrier();
                // Phase C: expert gate + up, shared down
                quantized_matmul_id_ggml_pooled(
                    enc, reg, dev, &x, &gate_exps, &s.ids, &s.gate_all, &mut scratch_gate, &gate_params,
                )?;
                let up_params = GgmlQuantizedMatmulIdParams { expert_stride: gate_stride, ..gate_params };
                quantized_matmul_id_ggml_pooled(
                    enc, reg, dev, &x, &up_exps, &s.ids, &s.up_all, &mut scratch_up, &up_params,
                )?;
                gemv_bf16(enc, reg, dev, &lw.shared_down, &s.h_s, &s.y_s, HIDDEN, SHARED_I)?;
                enc.memory_barrier();
                // Phase D
                dispatch_silu_mul(enc, reg, md, &s.gate_all, &s.up_all, &s.h_all, &silu_params, TOP_K * MOE_I)?;
                enc.memory_barrier();
                // Phase E
                quantized_matmul_id_ggml_pooled(
                    enc, reg, dev, &s.h_all, &down_exps, &s.ids, &s.y_all, &mut scratch_down, &down_params,
                )?;
                enc.memory_barrier();
                // Phase F
                dispatch_moe_weighted_reduce(
                    enc, reg, dev, &s.weights, &s.y_all, &s.sh_logit, &s.y_s, &residual, &mut s.out, 1,
                    TOP_K, HIDDEN, true,
                )?;
            }
            if joint {
                enc.memory_barrier();
                for (row, s) in rows.iter().enumerate() {
                    dispatch_copy_f32(enc, reg, md, &s.out, &packed_out, 0, row * h, h)?;
                }
            }
            Ok(())
        }),
    })
}

fn prepare_head(dev: &MlxDevice, shared: &Shared, rng: &mut Rng, w: u32) -> R<Prepared> {
    let wu = w as usize;
    let h = HIDDEN as usize;
    let hidden = f32_buf(dev, vec![wu, h], rng, -1.0, 1.0)?;
    let norm_w = f32_buf(dev, vec![h], rng, 0.5, 1.5)?;
    let normed = zeros(dev, DType::F32, vec![wu, h])?;
    let norm_params = f32_params(dev, &[EPS, HIDDEN as f32])?;
    let logits = zeros(dev, DType::F32, vec![wu, VOCAB as usize])?;
    let lm_head = shared.lm_head.clone();
    let argmax_params = u32_buf(dev, &[VOCAB])?;
    let mut outs = Vec::new();
    for _ in 0..w {
        outs.push((zeros(dev, DType::U32, vec![1])?, zeros(dev, DType::F32, vec![1])?));
    }
    let wbytes = qbytes(GgmlType::Q6_K, VOCAB, HIDDEN);
    Ok(Prepared {
        kernel_hint: hint("output head: rms_norm + lm_head + argmax per greedy row"),
        m: json!(w),
        n: json!(VOCAB),
        k: json!(HIDDEN),
        qtype: "q6_K",
        bytes: wbytes + 4 * 2 * w as u64 * VOCAB as u64,
        sets: 1,
        footprint: wbytes,
        extra: json!({"note": "all rows greedy; sampled rows skip argmax and copy logits on the host"}),
        encode: unit(move |enc, reg, dev, _| {
            let md = dev.metal_device();
            rms(enc, reg, dev, &hidden, &norm_w, &normed, &norm_params, w, HIDDEN)?;
            enc.memory_barrier();
            qmm(enc, reg, dev, &normed, &lm_head, &logits, w, VOCAB, HIDDEN, GgmlType::Q6_K)?;
            enc.memory_barrier();
            for (row, (idx, val)) in outs.iter().enumerate() {
                dispatch_argmax_f32(
                    enc,
                    reg,
                    md,
                    &row_view(&logits, row, vec![1, VOCAB as usize])?,
                    idx,
                    val,
                    &argmax_params,
                    VOCAB,
                )?;
            }
            Ok(())
        }),
    })
}

// ======================================================================
// Timing driver
// ======================================================================

struct Config {
    samples: usize,
    target_cb_us: f64,
    cold_bytes: u64,
    filter: Option<String>,
    list: bool,
}

fn parse_args() -> Config {
    let mut cfg = Config {
        samples: 25,
        target_cb_us: 4000.0,
        cold_bytes: 256 * MIB,
        filter: None,
        list: false,
    };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--samples" => cfg.samples = args.next().and_then(|v| v.parse().ok()).unwrap_or(cfg.samples),
            "--target-cb-us" => {
                cfg.target_cb_us = args.next().and_then(|v| v.parse().ok()).unwrap_or(cfg.target_cb_us)
            }
            "--cold-mib" => {
                cfg.cold_bytes = args.next().and_then(|v| v.parse::<u64>().ok()).map_or(cfg.cold_bytes, |m| m * MIB)
            }
            "--filter" => cfg.filter = args.next(),
            "--list" => cfg.list = true,
            // `cargo bench` passes `--bench`; ignore unknown flags.
            _ => {}
        }
    }
    cfg
}

struct Stats {
    mean: f64,
    p50: f64,
    min: f64,
    max: f64,
}

fn stats(v: &mut [f64]) -> Stats {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len().max(1);
    Stats {
        mean: v.iter().sum::<f64>() / n as f64,
        p50: v[v.len() / 2],
        min: v[0],
        max: v[v.len() - 1],
    }
}

/// Encode `units` units (barrier-separated) into one CB; returns
/// (gpu_us, wall_us, encode_us) for the whole CB.
fn run_cb(
    dev: &MlxDevice,
    reg: &mut KernelRegistry,
    encode: &mut EncodeFn,
    next: &mut usize,
    units: usize,
) -> R<(f64, f64, f64)> {
    let mut enc = dev.command_encoder()?;
    let t0 = Instant::now();
    for u in 0..units {
        if u > 0 {
            enc.memory_barrier();
        }
        encode(&mut enc, reg, dev, *next)?;
        *next += 1;
    }
    let t_enc = t0.elapsed().as_secs_f64() * 1e6;
    let (gs, ge) = enc.commit_wait_with_gpu_time()?;
    let wall = t0.elapsed().as_secs_f64() * 1e6;
    Ok(((ge - gs) * 1e6, wall, t_enc))
}

fn measure(case: &Case, dev: &MlxDevice, reg: &mut KernelRegistry, mut p: Prepared, cfg: &Config) -> R<Value> {
    let mut next = 0usize;
    // Probe: one unit, collects the dispatched pipeline labels (and compiles).
    mlx_native::reset_pipeline_dispatch_buckets();
    let d0 = mlx_native::dispatch_count();
    run_cb(dev, reg, &mut p.encode, &mut next, 1)?;
    let dispatches = mlx_native::dispatch_count() - d0;
    let buckets = mlx_native::pipeline_dispatch_buckets();
    let kernel = if buckets.is_empty() {
        p.kernel_hint.to_string()
    } else {
        buckets.iter().map(|(k, c)| format!("{k}x{c}")).collect::<Vec<_>>().join("+")
    };
    // Warm-up and unit-count calibration.
    let mut t_unit = f64::MAX;
    for _ in 0..3 {
        let (g, _, _) = run_cb(dev, reg, &mut p.encode, &mut next, 4)?;
        t_unit = t_unit.min(g / 4.0);
    }
    let units = ((cfg.target_cb_us / t_unit.max(0.5)).ceil() as usize).clamp(4, 512);
    let (mut gpu, mut wall, mut encv) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..cfg.samples {
        let (g, wl, e) = run_cb(dev, reg, &mut p.encode, &mut next, units)?;
        gpu.push(g / units as f64);
        wall.push(wl / units as f64);
        encv.push(e / units as f64);
    }
    let g = stats(&mut gpu);
    let wl = stats(&mut wall);
    let e = stats(&mut encv);
    let mut rec = json!({
        "case": case.name(),
        "op": op_name(case.op),
        "kernel": kernel,
        "w": case.w,
        "M": p.m,
        "N": p.n,
        "K": p.k,
        "qtype": p.qtype,
        "mode": case.mode.name(),
        "gpu_us_mean": g.mean,
        "gpu_us_p50": g.p50,
        "gpu_us_min": g.min,
        "gpu_us_max": g.max,
        "wall_us": wl.mean,
        "wall_us_p50": wl.p50,
        "encode_us": e.mean,
        "iters": cfg.samples * units,
        "samples": cfg.samples,
        "units_per_cb": units,
        "dispatches_per_unit": dispatches,
        "bytes": p.bytes,
        "gbps": p.bytes as f64 / (g.p50 * 1e3),
        "sets": p.sets,
        "footprint_bytes": p.footprint,
        "peer_case": case.peer_case(),
    });
    if let (Some(obj), Value::Object(extra)) = (rec.as_object_mut(), std::mem::take(&mut p.extra)) {
        for (k, v) in extra {
            obj.insert(k, v);
        }
    }
    Ok(rec)
}

fn op_name(op: Op) -> &'static str {
    match op {
        Op::Embedding => "embedding_gather_q4_k",
        Op::RmsNorm { .. } => "rms_norm",
        Op::FusedResidualNorm => "fused_residual_norm",
        Op::QMatmul { .. } => "quantized_matmul_ggml",
        Op::QGateDeinterleave => "q_gate_deinterleave",
        Op::SsmConv => "ssm_conv",
        Op::L2Norm => "l2_norm",
        Op::QScale => "scalar_mul",
        Op::ComputeGBeta => "compute_g_beta",
        Op::GatedDeltaNet => "gated_delta_net_decode",
        Op::SsmNormGate => "ssm_norm_gate",
        Op::Rope { .. } => "rope_multi",
        Op::TqEncode => "hadamard_quantize_kv_hb_seq",
        Op::FwhtPremult => "fwht_sign_premult",
        Op::FwhtUndo => "fwht_sign_undo",
        Op::TqSdpa { .. } => "flash_attn_vec_tq_hb",
        Op::TqSdpaRows { route: AttnRoute::Rows, .. } => "flash_attn_vec_tq_hb_rows",
        Op::TqSdpaRows { route: AttnRoute::Batched, .. } => "flash_attn_vec_tq_hb_batched",
        Op::TqEncodeRows => "hadamard_quantize_kv_hb_batched",
        Op::FwhtUndoSigmoid => "fwht_sign_undo_sigmoid_mul",
        Op::RowCopy { .. } => "copy_f32",
        Op::SigmoidMul => "sigmoid_mul",
        Op::Bf16Gemv(_) => "dense_gemv_bf16_f32",
        Op::SoftmaxTopk => "moe_softmax_topk",
        Op::SiluMul { .. } => "silu_mul",
        Op::MoeId { .. } => "quantized_matmul_id_ggml",
        Op::WeightedReduce => "moe_weighted_reduce",
        Op::Argmax => "argmax",
        Op::LayerGdn { .. } => "layer_gdn",
        Op::LayerAttn { .. } => "layer_full_attn",
        Op::LayerAttnRows { .. } => "layer_full_attn_rows",
        Op::LayerFfn { .. } => "layer_moe_ffn",
        Op::Head => "output_head",
    }
}

// ======================================================================
// Command-buffer overhead (22 CBs per decode step)
// ======================================================================

fn cb_record(case: &str, samples: usize, wall: &mut [f64], gpu: Option<&mut [f64]>, extra: Value) -> Value {
    let wl = stats(wall);
    let (gm, gp) = match gpu {
        Some(g) => {
            let s = stats(g);
            (json!(s.mean), json!(s.p50))
        }
        None => (Value::Null, Value::Null),
    };
    let mut rec = json!({
        "case": case, "op": "command_buffer", "kernel": Value::Null, "w": 0,
        "M": Value::Null, "N": Value::Null, "K": Value::Null, "qtype": Value::Null,
        "mode": "cb_overhead", "gpu_us_mean": gm, "gpu_us_p50": gp,
        "wall_us": wl.mean, "wall_us_p50": wl.p50, "wall_us_min": wl.min,
        "iters": samples, "bytes": 0, "gbps": Value::Null,
    });
    if let (Some(obj), Value::Object(extra)) = (rec.as_object_mut(), extra) {
        for (k, v) in extra {
            obj.insert(k, v);
        }
    }
    rec
}

fn gpu_span(encoders: &[CommandEncoder]) -> Option<(f64, f64)> {
    let mut first = f64::MAX;
    let mut last = 0.0f64;
    let mut busy = 0.0f64;
    for e in encoders {
        match e.completed_gpu_timestamps() {
            CompletedGpuTimestamps::Available { start_seconds, end_seconds, .. } => {
                first = first.min(start_seconds);
                last = last.max(end_seconds);
                busy += end_seconds - start_seconds;
            }
            CompletedGpuTimestamps::Unknown { .. } => return None,
        }
    }
    Some(((last - first) * 1e6, busy * 1e6))
}

fn cb_overhead(dev: &MlxDevice, reg: &mut KernelRegistry, cfg: &Config) -> R<Vec<Value>> {
    let samples = (cfg.samples * 8).max(100);
    let mut out = Vec::new();
    // (1) one empty CB, commit + wait.
    for _ in 0..10 {
        dev.command_encoder()?.commit_and_wait()?;
    }
    let mut wall = Vec::with_capacity(samples);
    for _ in 0..samples {
        let t0 = Instant::now();
        let mut enc = dev.command_encoder()?;
        enc.commit_and_wait()?;
        wall.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    out.push(cb_record("cb_empty_commit_wait", samples, &mut wall, None, json!({"cbs_per_unit": 1})));

    // (2)/(3) the decode step's CB pattern: 21 async commits + 1 commit_and_wait.
    let src = zeros(dev, DType::F32, vec![64])?;
    let dst = zeros(dev, DType::F32, vec![64])?;
    for tiny in [false, true] {
        let mut wall = Vec::with_capacity(samples);
        let mut span = Vec::with_capacity(samples);
        let mut busy = Vec::with_capacity(samples);
        for s in 0..samples + 5 {
            let t0 = Instant::now();
            let mut encs = Vec::with_capacity(22);
            for cb in 0..22 {
                let mut enc = dev.command_encoder()?;
                if tiny {
                    dispatch_copy_f32(&mut enc, reg, dev.metal_device(), &src, &dst, 0, 0, 64)?;
                }
                if cb < 21 {
                    enc.commit_labeled("qwen35.joint_decode.layers");
                } else {
                    enc.commit_and_wait_labeled("qwen35.joint_decode.output_head")?;
                }
                encs.push(enc);
            }
            let elapsed = t0.elapsed().as_secs_f64() * 1e6;
            for e in &encs[..21] {
                e.wait_until_completed()?;
            }
            if s < 5 {
                continue;
            }
            wall.push(elapsed);
            if tiny {
                if let Some((sp, b)) = gpu_span(&encs) {
                    span.push(sp);
                    busy.push(b);
                }
            }
        }
        if tiny && span.len() == wall.len() {
            let b = stats(&mut busy);
            out.push(cb_record(
                "cb_step22_tiny",
                samples,
                &mut wall,
                Some(&mut span),
                json!({"cbs_per_unit": 22, "dispatches_per_unit": 22,
                    "gpu_busy_sum_us_mean": b.mean,
                    "gpu_note": "gpu_us = first CB GPUStartTime to last CB GPUEndTime per step"}),
            ));
        } else {
            out.push(cb_record(
                if tiny { "cb_step22_tiny" } else { "cb_step22_empty" },
                samples,
                &mut wall,
                None,
                json!({"cbs_per_unit": 22, "dispatches_per_unit": if tiny { 22 } else { 0 }}),
            ));
        }
    }
    Ok(out)
}

// ======================================================================

fn main() {
    // Per-pipeline dispatch buckets name the kernels of each case; the flag
    // is read once at the first encode.
    std::env::set_var("MLX_DISP_BUCKET", "1");
    let cfg = parse_args();
    let cases: Vec<Case> = catalogue()
        .into_iter()
        // `--filter a,b`: cases whose name contains any of the substrings.
        .filter(|c| {
            cfg.filter
                .as_deref()
                .map_or(true, |f| f.split(',').any(|part| c.name().contains(part)))
        })
        .collect();
    if cfg.list {
        for c in &cases {
            println!("{}\t{}", c.name(), c.mode.name());
        }
        for name in ["cb_empty_commit_wait", "cb_step22_empty", "cb_step22_tiny"] {
            println!("{name}\tcb_overhead");
        }
        return;
    }

    let dev = MlxDevice::new().expect("MlxDevice::new");
    let mut reg = KernelRegistry::new();
    let policy = mlx_native::ggml_routing_policy_from_environment();
    reg.freeze_ggml_routing_policy(policy).expect("freeze routing policy");
    mlx_native::ops::embedding_q4_k::register(&mut reg);
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    println!(
        "{}",
        json!({
            "case": "_meta",
            "schema": "hf2q.adr063.qwen36-opparity.hf2q-bench.v1",
            "device": dev.name(),
            "mlx_native_version": env!("CARGO_PKG_VERSION"),
            "source_commit": option_env!("GIT_COMMIT_SHA"),
            "routing_policy": serde_json::to_value(policy).unwrap_or(Value::Null),
            "samples": cfg.samples,
            "target_cb_us": cfg.target_cb_us,
            "cold_bytes": cfg.cold_bytes,
            "cases": cases.len(),
            "tq_codebook_bits": TQ_CODEBOOK_BITS,
            "gpu_clock": "MTLCommandBuffer GPUEndTime - GPUStartTime per CB / units_per_cb",
        })
    );

    let t_start = Instant::now();
    let shared = Shared::new(&dev, &mut rng).expect("allocate shared model-size buffers");
    for case in &cases {
        let rec = prepare(case, &dev, &shared, &mut rng, cfg.cold_bytes)
            .and_then(|p| measure(case, &dev, &mut reg, p, &cfg));
        match rec {
            Ok(v) => println!("{v}"),
            Err(e) => println!(
                "{}",
                json!({"case": case.name(), "mode": case.mode.name(), "w": case.w, "error": e.to_string()})
            ),
        }
    }
    if cfg.filter.as_deref().map_or(true, |f| "cb_".contains(f) || f.starts_with("cb_")) {
        match cb_overhead(&dev, &mut reg, &cfg) {
            Ok(recs) => recs.iter().for_each(|r| println!("{r}")),
            Err(e) => println!("{}", json!({"case": "cb_overhead", "error": e.to_string()})),
        }
    }
    eprintln!("bench_qwen36_decode_ops: {} cases in {:.1} s", cases.len(), t_start.elapsed().as_secs_f64());
}
