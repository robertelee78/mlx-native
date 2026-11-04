//! Reproduction of the legacy D=512 causal-resume chunk overshoot (found
//! during hf2q ADR-063 Q4-GEMMA-KERNEL review; not fixed in that change,
//! which keeps the legacy kernels byte-identical).
//!
//! `flash_attn_prefill_d512_impl` limits the causal chunk loop with
//! `kL_chunks = ceil((iq1 + Q + qL_off) / C)` and clamps only when
//! `kL_chunks * C > kL + C`.  With `kL % 64 == 0` (align_K true) and a
//! partial last Q tile (`qL % 8 != 0`, `qL_off + qL == kL`), the last Q
//! tile's limit lies in `(kL, kL + 8)`, so one chunk starting at kL is
//! visited and its K and V rows `[kL, kL + 64)` are loaded without guards.
//! Causal masking zeroes their scores, but a NaN V row still reaches O
//! through `0 * NaN`.  With `kv_capacity == kL` the same loads run past
//! the logical end of exactly sized K/V buffers; `MTL_SHADER_VALIDATION=1`
//! did not report that on M5 Max (macOS 27), so only the NaN form is a
//! hardware reproduction.
//!
//! The test is ignored: it fails until the legacy clamp is fixed.
//!
//! ```text
//! cargo test --release --test test_flash_attn_prefill_d512_legacy_overshoot -- --ignored --test-threads=1
//! ```

#![cfg(target_os = "macos")]

use half::f16;
use mlx_native::ops::flash_attn_prefill::FlashAttnPrefillResumeParams;
use mlx_native::ops::flash_attn_prefill_d512::dispatch_flash_attn_prefill_f16_d512_resume;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

const H: usize = 4;
const H_KV: usize = 2;
const D: usize = 512;
const KL: usize = 128;
const QL: usize = 33;
const Q_OFF: usize = KL - QL;

fn f16_buffer(device: &MlxDevice, values: &[f32]) -> MlxBuffer {
    let mut buf = device
        .alloc_buffer(values.len() * 2, DType::F16, vec![values.len()])
        .unwrap();
    for (dst, &v) in buf.as_mut_slice::<f16>().unwrap().iter_mut().zip(values) {
        *dst = f16::from_f32(v);
    }
    buf
}

/// `[H_KV, cap, D]` rows: deterministic finite values for `[0, KL)` and
/// `tail` for `[KL, cap)`.
fn cache(cap: usize, salt: f32, tail: f32) -> Vec<f32> {
    let mut out = vec![tail; H_KV * cap * D];
    for h in 0..H_KV {
        for r in 0..KL {
            for d in 0..D {
                let x = ((h * 7919 + r * 104_729 + d * 31) % 997) as f32 / 997.0 - 0.5;
                out[(h * cap + r) * D + d] = x * salt;
            }
        }
    }
    out
}

fn run(cap: usize, tail: f32) -> Vec<f32> {
    let device = MlxDevice::new().unwrap();
    let mut registry = KernelRegistry::new();
    mlx_native::ops::flash_attn_prefill_d512::register(&mut registry);
    let q: Vec<f32> = (0..H * QL * D)
        .map(|i| ((i * 37) % 101) as f32 / 101.0 - 0.5)
        .collect();
    let (q, k, v) = (
        f16_buffer(&device, &q),
        f16_buffer(&device, &cache(cap, 1.0, tail)),
        f16_buffer(&device, &cache(cap, 2.0, tail)),
    );
    let out = f16_buffer(&device, &vec![0.0; H * QL * D]);
    let mut enc = device.command_encoder().unwrap();
    dispatch_flash_attn_prefill_f16_d512_resume(
        &mut enc,
        &device,
        &mut registry,
        &q,
        &k,
        &v,
        &out,
        &FlashAttnPrefillResumeParams {
            n_heads: H as u32,
            n_kv_heads: H_KV as u32,
            head_dim: D as u32,
            seq_len_q: QL as u32,
            seq_len_k: KL as u32,
            batch: 1,
            scale: 1.0 / (D as f32).sqrt(),
            do_causal: true,
            q_offset_in_k: Q_OFF as u32,
            kv_capacity: cap as u32,
        },
    )
    .unwrap();
    enc.commit_and_wait().expect("command buffer");
    out.as_slice::<f16>()
        .unwrap()
        .iter()
        .map(|x| x.to_f32())
        .collect()
}

/// Rows `[kL, cap)` hold NaN (another sequence's or uninitialized cache
/// rows); no O element may depend on them.
#[test]
#[ignore = "legacy d512 causal-resume chunk overshoot: fails until fixed"]
fn nan_rows_past_kl_do_not_reach_o() {
    let poisoned = run(KL + 64, f32::NAN);
    let clean = run(KL + 64, 0.0);
    let nan = poisoned.iter().filter(|x| x.is_nan()).count();
    assert_eq!(nan, 0, "{nan} of {} O elements are NaN", poisoned.len());
    assert_eq!(poisoned, clean);
}
