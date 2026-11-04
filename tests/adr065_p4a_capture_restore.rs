//! ADR-065 P4a: qualification of the path A DeltaNet capture restore
//! (`docs/adr065-e6-verify-window-probes.md`, section "P4a").
//!
//! Path A composes two existing primitives per layer and item:
//!
//! * recurrent: `CommandEncoder::blit_copy_bytes` from capture row
//!   `(s·T + t)·E` (logical `[E, T, S]`) into the slot's `E`-element
//!   destination view;
//! * conv re-index: `dispatch_strided_copy_f32` with `rows = C`,
//!   `cols = Km1`, `stride_row = 1`, `stride_col = C` from the capture row
//!   `((s·T + t)·Km1)·C` (logical `[S, T, Km1, C]`), giving
//!   `dst[c·Km1 + k] = src[k·C + c]`.
//!
//! The encoding is the one hf2q ships: every conv strided copy for all
//! layers and items first (one compute pass), then every recurrent blit, in
//! one command buffer. Every comparison is on raw u32 bits for every class
//! (normal, ±0, subnormal, ±inf, quiet and signaling NaN payloads); NaN
//! payload preservation is asserted, not only recorded.
//!
//! Tests: T-P4a-2 acceptance classes, T-P4a-3 compute-first ordering over
//! every ordered subset of four slots, T-P4a-4 anchor images (`T = S = 1`),
//! T-P4a-5 C2 snapshot direction and round trip, T-P4a-6 native refusals and
//! the strided copy's allocation-wide bounds check. T-P4a-1 is
//! `adr065_e6_capture_restore_copy`.

#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used, clippy::panic, dead_code)]

include!("support/adr065_e6_common.rs");
include!("support/adr065_restore_common.rs");

use mlx_native::ops::copy::{dispatch_strided_copy_f32, StridedCopyParams};
use mlx_native::{CommandEncoder, MlxError};

const PROBE: &str = "p4a_capture_restore";
/// Slots in the destination state buffers.
const SLOTS: usize = 4;
/// Per-slot ping-pong parity (`true` = the scratch buffer is current).
const FLIPPED: [bool; SLOTS] = [false, true, true, false];

#[derive(Clone, Copy, Debug)]
struct Geometry {
    name: &'static str,
    /// Recurrent elements per slot (`D_k·D_v·n_v_heads`).
    e: usize,
    /// Conv channels.
    c: usize,
    km1: usize,
    /// Capture token axis.
    t: usize,
    layers: usize,
}

impl Geometry {
    fn conv(&self) -> usize {
        self.c * self.km1
    }
}

/// Qwen3.8-27B dense: 48 value heads, dim 128, conv channels 10240.
const QWEN38: Geometry = Geometry {
    name: "qwen38",
    e: 786_432,
    c: 10_240,
    km1: 3,
    t: 4,
    layers: 1,
};
/// Qwen3.6-35B-A3B MoE: 32 value heads, dim 128, conv channels 8192.
const QWEN36: Geometry = Geometry {
    name: "qwen36",
    e: 524_288,
    c: 8_192,
    km1: 3,
    t: 4,
    layers: 1,
};
const SMALL: Geometry = Geometry {
    name: "small2layer",
    e: 96,
    c: 40,
    km1: 3,
    t: 4,
    layers: 2,
};

/// Capture-layout sources and parity-paired state destinations of a layer.
struct Layer {
    cap: Vec<u32>,
    conv_cap: Vec<u32>,
    cap_buf: MlxBuffer,
    conv_cap_buf: MlxBuffer,
    /// `[plain, scratch]`, each `[S, E]`.
    rec: [MlxBuffer; 2],
    /// `[plain, scratch]`, each `[S, C, Km1]`.
    conv: [MlxBuffer; 2],
    /// Sentinel contents of `rec` / `conv`, restored before every run.
    rec_init: [Vec<u32>; 2],
    conv_init: [Vec<u32>; 2],
}

/// Deterministic sentinel distinct per buffer and per element.
fn sentinel(len: usize, salt: u32) -> Vec<u32> {
    (0..len as u32)
        .map(|index| 0x5a00_0000 ^ salt.rotate_left(13) ^ index.wrapping_mul(0x9e37_79b1))
        .collect()
}

fn layers(device: &MlxDevice, geometry: &Geometry, captured_slots: usize) -> Vec<Layer> {
    let mut rng = E6Rng::new(0x9_4a00 ^ geometry.e as u64);
    (0..geometry.layers)
        .map(|layer| {
            let rows = captured_slots * geometry.t;
            let cap = payload(&mut rng, rows * geometry.e);
            let conv_cap = payload(&mut rng, rows * geometry.conv());
            let salt = (layer as u32) << 4;
            let rec_init = [
                sentinel(SLOTS * geometry.e, salt | 1),
                sentinel(SLOTS * geometry.e, salt | 2),
            ];
            let conv_init = [
                sentinel(SLOTS * geometry.conv(), salt | 3),
                sentinel(SLOTS * geometry.conv(), salt | 4),
            ];
            Layer {
                cap_buf: bits_buffer(device, &cap),
                conv_cap_buf: bits_buffer(device, &conv_cap),
                rec: [
                    bits_buffer(device, &rec_init[0]),
                    bits_buffer(device, &rec_init[1]),
                ],
                conv: [
                    bits_buffer(device, &conv_init[0]),
                    bits_buffer(device, &conv_init[1]),
                ],
                cap,
                conv_cap,
                rec_init,
                conv_init,
            }
        })
        .collect()
}

fn reset(layers: &mut [Layer]) {
    for layer in layers {
        for parity in 0..2 {
            layer.rec[parity]
                .as_mut_slice::<u32>()
                .expect("rec slice")
                .copy_from_slice(&layer.rec_init[parity]);
            layer.conv[parity]
                .as_mut_slice::<u32>()
                .expect("conv slice")
                .copy_from_slice(&layer.conv_init[parity]);
        }
    }
}

/// Path A restore, compute-first: all conv strided copies for every layer
/// and item, then all recurrent blits. `items` are `(slot, t)`; the
/// destination parity is `FLIPPED[slot]`.
fn encode_restore(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    geometry: &Geometry,
    layers: &[Layer],
    items: &[(usize, usize)],
) {
    let (e, conv) = (geometry.e, geometry.conv());
    for layer in layers {
        for &(slot, t) in items {
            let row = slot * geometry.t + t;
            let src = layer.conv_cap_buf.slice_view((row * conv * 4) as u64, conv);
            let dst = layer.conv[FLIPPED[slot] as usize].slice_view((slot * conv * 4) as u64, conv);
            dispatch_strided_copy_f32(
                encoder,
                registry,
                device.metal_device(),
                &src,
                &dst,
                &StridedCopyParams {
                    rows: geometry.c as u32,
                    cols: geometry.km1 as u32,
                    stride_row: 1,
                    stride_col: geometry.c as u32,
                },
            )
            .expect("conv strided restore");
        }
    }
    for layer in layers {
        for &(slot, t) in items {
            let row = slot * geometry.t + t;
            encoder
                .blit_copy_bytes(
                    &layer.cap_buf,
                    (row * e * 4) as u64,
                    &layer.rec[FLIPPED[slot] as usize],
                    (slot * e * 4) as u64,
                    (e * 4) as u64,
                )
                .expect("recurrent blit restore");
        }
    }
}

/// Host authority (port of hf2q `rollback_la_to`) for one capture row.
fn host_conv_row(geometry: &Geometry, conv_cap: &[u32], row: usize) -> Vec<u32> {
    let src = &conv_cap[row * geometry.conv()..(row + 1) * geometry.conv()];
    let mut dst = vec![0u32; geometry.conv()];
    for k in 0..geometry.km1 {
        for c in 0..geometry.c {
            dst[c * geometry.km1 + k] = src[k * geometry.c + c];
        }
    }
    dst
}

fn mismatches(got: &[u32], want: &[u32]) -> usize {
    assert_eq!(got.len(), want.len(), "compare length mismatch");
    got.iter().zip(want).filter(|(g, w)| g != w).count()
}

/// Compare every slot region of both parity buffers of every layer: item
/// regions in the slot's current parity hold the host restore of `(s, t)`;
/// every other region (peer slots, the non-current parity) holds its
/// sentinel. Returns the total mismatching u32 count.
fn check(geometry: &Geometry, layers: &[Layer], items: &[(usize, usize)]) -> usize {
    let (e, conv) = (geometry.e, geometry.conv());
    let mut total = 0;
    for layer in layers {
        for parity in 0..2 {
            let rec = read_bits(&layer.rec[parity]);
            let cv = read_bits(&layer.conv[parity]);
            for slot in 0..SLOTS {
                let restored = items
                    .iter()
                    .find(|&&(s, _)| s == slot && FLIPPED[s] as usize == parity);
                let (rec_got, conv_got) = (
                    &rec[slot * e..(slot + 1) * e],
                    &cv[slot * conv..(slot + 1) * conv],
                );
                total += match restored {
                    Some(&(s, t)) => {
                        let row = s * geometry.t + t;
                        mismatches(rec_got, &layer.cap[row * e..(row + 1) * e])
                            + mismatches(conv_got, &host_conv_row(geometry, &layer.conv_cap, row))
                    }
                    None => {
                        mismatches(rec_got, &layer.rec_init[parity][slot * e..(slot + 1) * e])
                            + mismatches(
                                conv_got,
                                &layer.conv_init[parity][slot * conv..(slot + 1) * conv],
                            )
                    }
                };
            }
        }
    }
    total
}

fn run(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    geometry: &Geometry,
    layers: &mut [Layer],
    items: &[(usize, usize)],
) -> usize {
    reset(layers);
    let mut encoder = device.command_encoder().expect("encoder");
    encode_restore(&mut encoder, registry, device, geometry, layers, items);
    encoder.commit_and_wait().expect("restore execution");
    check(geometry, layers, items)
}

/// Every ordered selection of 1..=4 distinct slots out of four (64 orders).
fn ordered_subsets() -> Vec<Vec<usize>> {
    fn extend(prefix: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if !prefix.is_empty() {
            out.push(prefix.clone());
        }
        for slot in 0..SLOTS {
            if !prefix.contains(&slot) {
                prefix.push(slot);
                extend(prefix, out);
                prefix.pop();
            }
        }
    }
    let mut out = Vec::new();
    extend(&mut Vec::new(), &mut out);
    out
}

/// T-P4a-2: full acceptance (no item: a no-op), partial (`t ∈ {1, 2}`) and
/// zero (`t = 0`) acceptance, for every slot under mixed parity.
#[test]
fn p4a_acceptance_classes_match_host_rollback() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let geometry = QWEN38;
    let mut layers = layers(&device, &geometry, SLOTS);
    let mut classes = vec![("full", Vec::new())];
    for slot in 0..SLOTS {
        classes.push(("zero", vec![(slot, 0)]));
        classes.push(("partial", vec![(slot, 1)]));
        classes.push(("partial", vec![(slot, 2)]));
    }
    let mut origin = String::from("none");
    for (class, items) in &classes {
        let mismatched = run(&device, &mut registry, &geometry, &mut layers, items);
        if !items.is_empty() {
            origin = e6_pipeline_origin(&registry, "strided_copy_f32", false);
        }
        e6_record(
            PROBE,
            &format!("acceptance/{}/{class}/{items:?}", geometry.name),
            &[
                ("bitwise", (mismatched == 0).to_string()),
                ("mismatched", mismatched.to_string()),
                ("origin", origin.clone()),
            ],
        );
        assert_eq!(
            mismatched, 0,
            "{class} {items:?}: restore differs from host"
        );
    }
}

/// T-P4a-3: compute-first encoding (the order hf2q ships) for 1–4 items,
/// every subset of four slots in every order, at both served shapes and a
/// two-layer small shape. Single-item orders are part of the sweep, so each
/// multi-item run is checked against the same host bytes the single-item
/// runs equal.
#[test]
fn p4a_compute_first_order_matches_host_for_every_ordered_slot_subset() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    for geometry in [SMALL, QWEN36, QWEN38] {
        let mut layers = layers(&device, &geometry, SLOTS);
        let mut runs = 0usize;
        let mut failed = 0usize;
        for (index, order) in ordered_subsets().iter().enumerate() {
            let items: Vec<(usize, usize)> = order
                .iter()
                .map(|&slot| (slot, (slot + index) % geometry.t))
                .collect();
            let mismatched = run(&device, &mut registry, &geometry, &mut layers, &items);
            runs += 1;
            if mismatched != 0 {
                failed += 1;
                eprintln!(
                    "{}: items {items:?}: {mismatched} u32 differ",
                    geometry.name
                );
            }
        }
        let origin = e6_pipeline_origin(&registry, "strided_copy_f32", false);
        e6_record(
            PROBE,
            &format!("compute_first/{}", geometry.name),
            &[
                ("bitwise", (failed == 0).to_string()),
                ("runs", runs.to_string()),
                ("failed_runs", failed.to_string()),
                ("layers", geometry.layers.to_string()),
                ("origin", origin),
            ],
        );
        assert_eq!(runs, 64, "ordered subsets of four slots");
        assert_eq!(
            failed, 0,
            "{}: compute-first restore differs",
            geometry.name
        );
    }
}

/// T-P4a-4: an anchor image is a capture-layout pair with `T = S = 1`;
/// restoring `(0, 0)` from it into any slot under either parity is bitwise.
#[test]
fn p4a_anchor_image_restore_matches_host() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let geometry = Geometry { t: 1, ..QWEN38 };
    let mut layers = layers(&device, &geometry, 1);
    let mut cases = 0usize;
    for slot in 0..SLOTS {
        for parity in 0..2 {
            // The image's only row is `(s, t) = (0, 0)`; the destination is
            // `slot`'s region of the named parity buffer.
            reset(&mut layers);
            let layer = &layers[0];
            let current = parity;
            let (e, conv) = (geometry.e, geometry.conv());
            let mut encoder = device.command_encoder().expect("encoder");
            let dst = layer.conv[current].slice_view((slot * conv * 4) as u64, conv);
            dispatch_strided_copy_f32(
                &mut encoder,
                &mut registry,
                device.metal_device(),
                &layer.conv_cap_buf.slice_view(0, conv),
                &dst,
                &StridedCopyParams {
                    rows: geometry.c as u32,
                    cols: geometry.km1 as u32,
                    stride_row: 1,
                    stride_col: geometry.c as u32,
                },
            )
            .expect("anchor conv restore");
            encoder
                .blit_copy_bytes(
                    &layer.cap_buf,
                    0,
                    &layer.rec[current],
                    (slot * e * 4) as u64,
                    (e * 4) as u64,
                )
                .expect("anchor recurrent restore");
            encoder.commit_and_wait().expect("anchor restore execution");

            let mut mismatched = 0usize;
            for p in 0..2 {
                let rec = read_bits(&layer.rec[p]);
                let cv = read_bits(&layer.conv[p]);
                for s in 0..SLOTS {
                    let (rec_got, conv_got) =
                        (&rec[s * e..(s + 1) * e], &cv[s * conv..(s + 1) * conv]);
                    mismatched += if s == slot && p == current {
                        mismatches(rec_got, &layer.cap)
                            + mismatches(conv_got, &host_conv_row(&geometry, &layer.conv_cap, 0))
                    } else {
                        mismatches(rec_got, &layer.rec_init[p][s * e..(s + 1) * e])
                            + mismatches(conv_got, &layer.conv_init[p][s * conv..(s + 1) * conv])
                    };
                }
            }
            let origin = e6_pipeline_origin(&registry, "strided_copy_f32", false);
            e6_record(
                PROBE,
                &format!("anchor_image/{}/slot{slot}/parity{parity}", geometry.name),
                &[
                    ("bitwise", (mismatched == 0).to_string()),
                    ("mismatched", mismatched.to_string()),
                    ("origin", origin),
                ],
            );
            assert_eq!(mismatched, 0, "anchor restore slot {slot} parity {parity}");
            cases += 1;
        }
    }
    assert_eq!(cases, 2 * SLOTS);
}

/// T-P4a-5: C2 snapshot direction. State → image uses the inverse strided
/// copy (`rows = Km1`, `cols = C`, `stride_row = 1`, `stride_col = Km1`,
/// `img[k·C + c] = state[c·Km1 + k]`) and a blit out of the parity-current
/// recurrent region of slot `s > 0`. The image restored back into a
/// sentinel-filled state equals the original state region byte for byte.
#[test]
fn p4a_snapshot_direction_and_round_trip_are_bitwise() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let geometry = Geometry { t: 1, ..QWEN38 };
    let (e, conv) = (geometry.e, geometry.conv());
    let mut rng = E6Rng::new(0xc2_5a95);
    let state_rec = [payload(&mut rng, SLOTS * e), payload(&mut rng, SLOTS * e)];
    let state_conv = [
        payload(&mut rng, SLOTS * conv),
        payload(&mut rng, SLOTS * conv),
    ];
    let src_rec = [
        bits_buffer(&device, &state_rec[0]),
        bits_buffer(&device, &state_rec[1]),
    ];
    let src_conv = [
        bits_buffer(&device, &state_conv[0]),
        bits_buffer(&device, &state_conv[1]),
    ];
    // Image buffers double as the round-trip capture source (`T = S = 1`).
    let mut layers = layers(&device, &geometry, 1);
    for slot in 1..SLOTS {
        for current in 0..2 {
            reset(&mut layers);
            let layer = &layers[0];
            let mut encoder = device.command_encoder().expect("encoder");
            dispatch_strided_copy_f32(
                &mut encoder,
                &mut registry,
                device.metal_device(),
                &src_conv[current].slice_view((slot * conv * 4) as u64, conv),
                &layer.conv_cap_buf,
                &StridedCopyParams {
                    rows: geometry.km1 as u32,
                    cols: geometry.c as u32,
                    stride_row: 1,
                    stride_col: geometry.km1 as u32,
                },
            )
            .expect("snapshot conv");
            encoder
                .blit_copy_bytes(
                    &src_rec[current],
                    (slot * e * 4) as u64,
                    &layer.cap_buf,
                    0,
                    (e * 4) as u64,
                )
                .expect("snapshot recurrent");
            encoder.commit_and_wait().expect("snapshot execution");

            let want_rec = &state_rec[current][slot * e..(slot + 1) * e];
            let want_state_conv = &state_conv[current][slot * conv..(slot + 1) * conv];
            let mut want_img_conv = vec![0u32; conv];
            for k in 0..geometry.km1 {
                for c in 0..geometry.c {
                    want_img_conv[k * geometry.c + c] = want_state_conv[c * geometry.km1 + k];
                }
            }
            let snapshot = mismatches(&read_bits(&layer.cap_buf), want_rec)
                + mismatches(&read_bits(&layer.conv_cap_buf), &want_img_conv);

            // Round trip: image back into the sentinel state, same slot and
            // parity, with the path A restore of the image `(0, 0)`.
            let mut encoder = device.command_encoder().expect("encoder");
            dispatch_strided_copy_f32(
                &mut encoder,
                &mut registry,
                device.metal_device(),
                &layer.conv_cap_buf,
                &layer.conv[current].slice_view((slot * conv * 4) as u64, conv),
                &StridedCopyParams {
                    rows: geometry.c as u32,
                    cols: geometry.km1 as u32,
                    stride_row: 1,
                    stride_col: geometry.c as u32,
                },
            )
            .expect("round-trip conv");
            encoder
                .blit_copy_bytes(
                    &layer.cap_buf,
                    0,
                    &layer.rec[current],
                    (slot * e * 4) as u64,
                    (e * 4) as u64,
                )
                .expect("round-trip recurrent");
            encoder.commit_and_wait().expect("round-trip execution");
            let mut round_trip = 0usize;
            for p in 0..2 {
                let rec = read_bits(&layer.rec[p]);
                let cv = read_bits(&layer.conv[p]);
                for s in 0..SLOTS {
                    let (rec_got, conv_got) =
                        (&rec[s * e..(s + 1) * e], &cv[s * conv..(s + 1) * conv]);
                    round_trip += if s == slot && p == current {
                        mismatches(rec_got, want_rec) + mismatches(conv_got, want_state_conv)
                    } else {
                        mismatches(rec_got, &layer.rec_init[p][s * e..(s + 1) * e])
                            + mismatches(conv_got, &layer.conv_init[p][s * conv..(s + 1) * conv])
                    };
                }
            }
            let nan_payloads = want_rec
                .iter()
                .chain(want_state_conv)
                .filter(|&&b| is_nan_bits(b))
                .count();
            let origin = e6_pipeline_origin(&registry, "strided_copy_f32", false);
            e6_record(
                PROBE,
                &format!("snapshot/{}/slot{slot}/parity{current}", geometry.name),
                &[
                    ("bitwise", (snapshot == 0 && round_trip == 0).to_string()),
                    ("snapshot_mismatched", snapshot.to_string()),
                    ("round_trip_mismatched", round_trip.to_string()),
                    ("nan_payloads", nan_payloads.to_string()),
                    ("origin", origin),
                ],
            );
            assert!(nan_payloads > 0, "payload must carry NaN classes");
            assert_eq!(snapshot, 0, "snapshot slot {slot} parity {current}");
            assert_eq!(round_trip, 0, "round trip slot {slot} parity {current}");
        }
    }
}

/// T-P4a-6: native refusals, plus the strided copy's allocation-wide bounds
/// check (`byte_len()`, not the view's `data_byte_len()`), which is why
/// view-exact range checking belongs to the caller's preflight.
#[test]
fn p4a_native_refusals_and_strided_bounds_caveat() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let bits: Vec<u32> = (0..16).map(|i| 0x7f80_0001 + i).collect();
    let buffer = bits_buffer(&device, &bits);
    let other = bits_buffer(&device, &[0u32; 16]);
    let view = other.slice_view(0, 4);
    let mut encoder = device.command_encoder().expect("encoder");

    let overlap = encoder.blit_copy_bytes(&buffer, 0, &buffer, 4, 16);
    assert!(
        matches!(overlap, Err(MlxError::InvalidArgument(_))),
        "overlapping same-allocation blit must be refused: {overlap:?}"
    );
    let source_past_view = encoder.blit_copy_bytes(&view, 0, &buffer, 0, 32);
    assert!(
        matches!(source_past_view, Err(MlxError::InvalidArgument(_))),
        "blit source past the view must be refused: {source_past_view:?}"
    );
    let destination_past_view = encoder.blit_copy_bytes(&buffer, 0, &view, 0, 32);
    assert!(
        matches!(destination_past_view, Err(MlxError::InvalidArgument(_))),
        "blit destination past the view must be refused: {destination_past_view:?}"
    );
    for (rows, cols) in [(0u32, 3u32), (3, 0)] {
        let zero = dispatch_strided_copy_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &buffer,
            &other,
            &StridedCopyParams {
                rows,
                cols,
                stride_row: 1,
                stride_col: 1,
            },
        );
        assert!(
            matches!(zero, Err(MlxError::InvalidArgument(_))),
            "strided copy with rows={rows} cols={cols} must be refused: {zero:?}"
        );
    }

    // Caveat: an 8-element strided copy into a 4-element view is accepted
    // because the check uses the whole allocation, and it writes past the
    // view.
    dispatch_strided_copy_f32(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &buffer,
        &view,
        &StridedCopyParams {
            rows: 2,
            cols: 4,
            stride_row: 4,
            stride_col: 1,
        },
    )
    .expect("strided copy check is allocation-wide, not view-exact");
    encoder.commit_and_wait().expect("caveat execution");
    let got = read_bits(&other);
    assert_eq!(
        &got[..8],
        &bits[..8],
        "strided copy wrote past the 4-element view"
    );
    assert!(got[8..].iter().all(|&b| b == 0));
    let origin = e6_pipeline_origin(&registry, "strided_copy_f32", false);
    e6_record(
        PROBE,
        "refusals",
        &[
            ("blit_overlap_refused", "true".to_owned()),
            ("blit_past_view_refused", "true".to_owned()),
            ("strided_zero_dims_refused", "true".to_owned()),
            ("strided_bounds", "allocation_byte_len".to_owned()),
            ("origin", origin),
        ],
    );
}
