//! ADR-065 E6 probe 4: existing GPU copy primitives for capture-to-state
//! restore of DeltaNet state, including the parity-aware conv re-index.
//!
//! The authority is the host restore in hf2q (`HybridKvCache::rollback_la_to`):
//!
//! * recurrent: `capture[D_k, D_v, H, n_tokens_max, n_seqs]` token `t` of
//!   slot `s` is copied verbatim into slot `s` of the slot's CURRENT
//!   recurrent buffer (`recurrent` or `recurrent_scratch`, selected by the
//!   slot's ping-pong parity);
//! * conv: `conv_capture[n_seqs, n_tokens_max, K-1, channels]` (channels
//!   innermost) is re-indexed into slot `s` of the CURRENT conv buffer
//!   `[channels, K-1, n_seqs]` (K-1 innermost):
//!   `dst[c * (K-1) + i] = src[i * channels + c]`.
//!
//! GPU candidates, all encoded in one command buffer for a four-slot item
//! list with mixed parity and accepted indices `t = 0..=3` (zero, partial and
//! full acceptance of a 4-token window; `t > 0` and `s > 0` exercise
//! non-zero view offsets):
//!
//! * recurrent: `CommandEncoder::blit_copy_bytes`;
//! * conv: `ops::copy::dispatch_strided_copy_f32` (rows = channels,
//!   cols = K-1, stride_row = 1, stride_col = channels) on slice views, and
//!   `ops::transpose::transpose_2d` (rows = K-1, cols = channels).
//!
//! Payloads cover every u32 class: normal values, ±0, subnormals, ±inf,
//! quiet and signaling NaN payloads. Asserted bitwise: the whole recurrent
//! and conv destination pairs (restored slots, untouched peer slots and the
//! non-current parity buffer) for the blit restore, and the non-NaN classes
//! of the compute-kernel conv restores. Whether the compute kernels preserve
//! NaN payloads is recorded per class (`nan_payload_bitwise`); with
//! `E6_STRICT_NAN=1` it is asserted too.

#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used, clippy::panic, dead_code)]

include!("support/adr065_e6_common.rs");

use mlx_native::ops::copy::{dispatch_strided_copy_f32, StridedCopyParams};
use mlx_native::ops::transpose::transpose_2d;

/// Qwen3.8-27B DeltaNet: conv channels = 2 * 16 * 128 + 48 * 128, width 4.
const CHANNELS: usize = 10_240;
const KM1: usize = 3;
/// Recurrent state per sequence: D_k * D_v * n_v_heads.
const STATE: usize = 128 * 128 * 48;
const SLOTS: usize = 4;
const TOKENS: usize = 4;
const CONV: usize = CHANNELS * KM1;

/// `(slot, accepted index)` restore items; every slot appears once.
const ITEMS: [(usize, usize); SLOTS] = [(2, 1), (0, 3), (3, 0), (1, 2)];
/// Per-slot ping-pong parity (`true` = CURRENT is the scratch buffer).
const FLIPPED: [bool; SLOTS] = [false, true, true, false];

const SPECIALS: [u32; 10] = [
    0x0000_0000, // +0
    0x8000_0000, // -0
    0x0000_0001, // smallest subnormal
    0x807f_ffff, // largest negative subnormal
    0x7f80_0000, // +inf
    0xff80_0000, // -inf
    0x7fc0_0000, // canonical quiet NaN
    0x7fc1_2345, // quiet NaN payload
    0x7f80_0001, // signaling NaN payload
    0xffa5_a5a5, // negative signaling NaN payload
];

fn payload(rng: &mut E6Rng, len: usize) -> Vec<u32> {
    (0..len)
        .map(|index| {
            if index % 7 == 3 {
                SPECIALS[(rng.next_u64() % SPECIALS.len() as u64) as usize]
            } else {
                rng.next_u64() as u32
            }
        })
        .collect()
}

fn is_nan_bits(bits: u32) -> bool {
    (bits & 0x7f80_0000) == 0x7f80_0000 && (bits & 0x007f_ffff) != 0
}

fn bits_buffer(device: &MlxDevice, bits: &[u32]) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(bits.len() * 4, DType::F32, vec![bits.len()])
        .expect("alloc");
    buffer
        .as_mut_slice::<u32>()
        .expect("bits slice")
        .copy_from_slice(bits);
    buffer
}

fn read_bits(buffer: &MlxBuffer) -> Vec<u32> {
    buffer.as_slice::<u32>().expect("read bits").to_vec()
}

/// Host authority: port of `rollback_la_to` for one `(slot, t)`.
fn host_restore(
    capture: &[u32],
    conv_capture: &[u32],
    recurrent_current: &mut [u32],
    conv_current: &mut [u32],
    slot: usize,
    t: usize,
) {
    let src = slot * TOKENS * STATE + t * STATE;
    recurrent_current[slot * STATE..(slot + 1) * STATE].copy_from_slice(&capture[src..src + STATE]);
    let conv_src = slot * TOKENS * CONV + t * CONV;
    let dst = &mut conv_current[slot * CONV..(slot + 1) * CONV];
    for k_i in 0..KM1 {
        for c in 0..CHANNELS {
            dst[c * KM1 + k_i] = conv_capture[conv_src + k_i * CHANNELS + c];
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum ConvPrimitive {
    StridedCopy,
    Transpose2d,
}

struct Fixture {
    capture: Vec<u32>,
    conv_capture: Vec<u32>,
    /// `[plain, scratch]` initial contents.
    recurrent: [Vec<u32>; 2],
    conv: [Vec<u32>; 2],
}

fn fixture() -> Fixture {
    let mut rng = E6Rng::new(0xe6_0c0b);
    Fixture {
        capture: payload(&mut rng, SLOTS * TOKENS * STATE),
        conv_capture: payload(&mut rng, SLOTS * TOKENS * CONV),
        recurrent: [
            payload(&mut rng, SLOTS * STATE),
            payload(&mut rng, SLOTS * STATE),
        ],
        conv: [
            payload(&mut rng, SLOTS * CONV),
            payload(&mut rng, SLOTS * CONV),
        ],
    }
}

fn expected(fixture: &Fixture) -> ([Vec<u32>; 2], [Vec<u32>; 2]) {
    let mut recurrent = fixture.recurrent.clone();
    let mut conv = fixture.conv.clone();
    for &(slot, t) in &ITEMS {
        let current = FLIPPED[slot] as usize;
        let (rec, cv) = (&mut recurrent[current], &mut conv[current]);
        host_restore(&fixture.capture, &fixture.conv_capture, rec, cv, slot, t);
    }
    (recurrent, conv)
}

fn gpu_restore(primitive: ConvPrimitive) {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let fixture = fixture();
    let capture = bits_buffer(&device, &fixture.capture);
    let conv_capture = bits_buffer(&device, &fixture.conv_capture);
    let recurrent = [
        bits_buffer(&device, &fixture.recurrent[0]),
        bits_buffer(&device, &fixture.recurrent[1]),
    ];
    let conv = [
        bits_buffer(&device, &fixture.conv[0]),
        bits_buffer(&device, &fixture.conv[1]),
    ];

    let mut encoder = device.command_encoder().expect("encoder");
    for &(slot, t) in &ITEMS {
        let current = FLIPPED[slot] as usize;
        encoder
            .blit_copy_bytes(
                &capture,
                ((slot * TOKENS + t) * STATE * 4) as u64,
                &recurrent[current],
                (slot * STATE * 4) as u64,
                (STATE * 4) as u64,
            )
            .expect("recurrent blit restore");
        let src = conv_capture.slice_view(((slot * TOKENS + t) * CONV * 4) as u64, CONV);
        let dst = conv[current].slice_view((slot * CONV * 4) as u64, CONV);
        match primitive {
            ConvPrimitive::StridedCopy => dispatch_strided_copy_f32(
                &mut encoder,
                &mut registry,
                device.metal_device(),
                &src,
                &dst,
                &StridedCopyParams {
                    rows: CHANNELS as u32,
                    cols: KM1 as u32,
                    stride_row: 1,
                    stride_col: CHANNELS as u32,
                },
            )
            .expect("strided conv restore"),
            ConvPrimitive::Transpose2d => transpose_2d(
                &mut encoder,
                &mut registry,
                device.metal_device(),
                &src,
                &dst,
                KM1,
                CHANNELS,
                DType::F32,
            )
            .expect("transpose conv restore"),
        }
    }
    encoder.commit_and_wait().expect("restore execution");

    let label = match primitive {
        ConvPrimitive::StridedCopy => "strided_copy_f32",
        ConvPrimitive::Transpose2d => "transpose_2d_f32",
    };
    let origin = e6_pipeline_origin(&registry, label, false);
    let (want_recurrent, want_conv) = expected(&fixture);
    let case = format!("{primitive:?}");

    for parity in 0..2 {
        let got = read_bits(&recurrent[parity]);
        let mismatched = got
            .iter()
            .zip(&want_recurrent[parity])
            .filter(|(a, b)| a != b)
            .count();
        e6_record(
            "capture_restore_copy",
            &format!("{case}/recurrent_blit/parity{parity}"),
            &[
                ("bitwise", (mismatched == 0).to_string()),
                ("elements", got.len().to_string()),
                ("mismatched", mismatched.to_string()),
                ("origin", "blit".to_owned()),
            ],
        );
        assert_eq!(
            mismatched, 0,
            "recurrent parity {parity}: blit restore differs from host restore"
        );
    }

    let strict_nan = std::env::var("E6_STRICT_NAN").as_deref() == Ok("1");
    for parity in 0..2 {
        let got = read_bits(&conv[parity]);
        let want = &want_conv[parity];
        let mut other = 0usize;
        let mut nan = 0usize;
        let mut nan_total = 0usize;
        for (&g, &w) in got.iter().zip(want) {
            if is_nan_bits(w) {
                nan_total += 1;
                nan += (g != w) as usize;
            } else {
                other += (g != w) as usize;
            }
        }
        e6_record(
            "capture_restore_copy",
            &format!("{case}/conv/parity{parity}"),
            &[
                ("bitwise", (other == 0 && nan == 0).to_string()),
                ("elements", got.len().to_string()),
                ("non_nan_mismatched", other.to_string()),
                ("nan_payloads", nan_total.to_string()),
                ("nan_payload_mismatched", nan.to_string()),
                ("nan_payload_bitwise", (nan == 0).to_string()),
                ("origin", origin.clone()),
            ],
        );
        assert_eq!(
            other, 0,
            "{case} conv parity {parity}: non-NaN bits differ from host restore"
        );
        if strict_nan {
            assert_eq!(
                nan, 0,
                "{case} conv parity {parity}: NaN payloads not preserved"
            );
        }
    }
}

#[test]
fn e6_capture_restore_blit_and_strided_copy_match_host_rollback() {
    gpu_restore(ConvPrimitive::StridedCopy);
}

#[test]
fn e6_capture_restore_blit_and_transpose_match_host_rollback() {
    gpu_restore(ConvPrimitive::Transpose2d);
}
