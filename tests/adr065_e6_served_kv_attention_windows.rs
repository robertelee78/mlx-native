//! ADR-065 E6 probe 2: served-KV attention, verify-window rows versus
//! single-row queries at the same position.
//!
//! A K3 verify window for one slot is `w` consecutive queries at positions
//! `p .. p + w - 1`, each causally attending to `p + i + 1` cached rows. The
//! probe dispatches the window as one call and compares row `i` with a
//! single-row query at position `p + i`, at Qwen3.8-27B full-attention shape
//! (24 query heads, 4 KV heads, head dim 256).
//!
//! Kernels:
//!
//! * TQ-HB batched (`flash_attn_vec_tq_hb_batched`, uniform slots) and banked
//!   (`flash_attn_vec_tq_hb_batched_banked`, unequal slot banks), 8-bit
//!   codebook, `mask_type = 0`, `fuse_fwht_pre = 1` as served by hf2q.
//!   The window's `(nwg, nsg)` bucket is selected from its maximum KV length
//!   (`nwg`: 16 up to 512 rows, 32 above; `nsg = compute_nsg`: 1 up to 1024,
//!   4 above). Windows are placed inside one bucket and across both the
//!   `nwg` 512/513 and the `nsg` 1024/1025 boundaries.
//! * F32 `flash_attn_vec` with `q_seq_len = w`. It has a fixed `nwg = 32`
//!   and no `nsg` axis, so no bucket can be crossed; windows are placed at
//!   the same positions for comparison.
//!
//! Two single-row authorities are recorded for each TQ row:
//!
//! * `served` — the row dispatched alone with its own bucket (what a
//!   one-row-per-slot decode step computes today);
//! * `window_bucket` — the row dispatched alone under the window's bucket.
//!
//! Asserted: every row equals its `window_bucket` authority (rows are
//! independent within a bucket) and, for windows inside one bucket, equals
//! the `served` authority. Rows of bucket-crossing windows, and F32
//! `flash_attn_vec` rows, are recorded (bitwise verdict and max abs diff)
//! and only bounded numerically; their bitwise verdict is the E6 finding.

#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used, clippy::panic, dead_code)]

include!("support/adr065_e6_common.rs");

use mlx_native::ops::flash_attn_vec::{self, FlashAttnVecParams};
use mlx_native::ops::flash_attn_vec_tq_hb::{self, FlashAttnVecTqHbParams};

const NH: u32 = 24;
const NKV: u32 = 4;
const HD: u32 = 256;
const CBITS: u32 = 8;
const CAP: u32 = 1_600;
const N_SLOTS: usize = 4;
const ROW: usize = (NH * HD) as usize;

/// `(case, first position p, window width w)`.
const WINDOWS: &[(&str, u32, u32)] = &[
    ("in_bucket_nwg16_nsg1_w8", 100, 8),
    ("in_bucket_nwg16_nsg1_w4", 300, 4),
    ("cross_nwg16_to_32_w2", 511, 2),
    ("cross_nwg16_to_32_w4", 510, 4),
    ("cross_nwg16_to_32_w8", 508, 8),
    ("in_bucket_nwg32_nsg1_w8", 700, 8),
    ("cross_nsg1_to_4_w4", 1_022, 4),
    ("cross_nsg1_to_4_w8", 1_020, 8),
    ("in_bucket_nwg32_nsg4_w8", 1_500, 8),
];

fn nwg_bucket(kv_seq_len: u32) -> u32 {
    if kv_seq_len > 512 {
        32
    } else {
        16
    }
}

fn bucket(kv_seq_len: u32) -> (u32, u32) {
    (
        nwg_bucket(kv_seq_len),
        flash_attn_vec_tq_hb::compute_nsg(kv_seq_len),
    )
}

fn tq_params(kv_seq_len: u32, nsg: u32, capacity: u32) -> FlashAttnVecTqHbParams {
    FlashAttnVecTqHbParams {
        num_heads: NH,
        num_kv_heads: NKV,
        head_dim: HD,
        kv_seq_len,
        kv_capacity: capacity,
        scale: 1.0 / (HD as f32).sqrt(),
        mask_type: 0,
        sliding_window: 0,
        softcap: 0.0,
        ring_start: 0,
        scale_factor_d512: 1.0,
        codebook_bits: CBITS,
        fuse_fwht_pre: 1,
        nsg,
    }
}

fn assert_env_unset() {
    for name in ["HF2Q_TQ_NWG", "HF2Q_TQ_NSG"] {
        assert!(
            std::env::var_os(name).is_none(),
            "{name} must be unset: the probe measures the default bucket policy"
        );
    }
}

/// One TQ-HB K/V store. `rows` flattened `[kv_head, token]` rows.
struct TqStore {
    k_packed: MlxBuffer,
    k_norms: MlxBuffer,
    v_packed: MlxBuffer,
    v_norms: MlxBuffer,
}

fn tq_store(device: &MlxDevice, rows: usize, seed: u64) -> TqStore {
    let mut rng = E6Rng::new(seed);
    let packed =
        |rng: &mut E6Rng| -> Vec<u8> { (0..rows * HD as usize).map(|_| rng.next_u8()).collect() };
    let norms =
        |rng: &mut E6Rng| -> Vec<f32> { (0..rows).map(|_| 0.25 + rng.unit() * 1.75).collect() };
    let k_packed = packed(&mut rng);
    let k_norms = norms(&mut rng);
    let v_packed = packed(&mut rng);
    let v_norms = norms(&mut rng);
    TqStore {
        k_packed: e6_u8_buffer(device, &k_packed),
        k_norms: e6_f32_buffer(device, &k_norms),
        v_packed: e6_u8_buffer(device, &v_packed),
        v_norms: e6_f32_buffer(device, &v_norms),
    }
}

fn tq_output(device: &MlxDevice, n_q: u32) -> (MlxBuffer, MlxBuffer) {
    let output = device
        .alloc_buffer(n_q as usize * ROW * 4, DType::F32, vec![n_q as usize, ROW])
        .expect("output");
    let tmp_bytes = flash_attn_vec_tq_hb::tmp_buffer_bytes(n_q * NH, HD);
    let tmp = device
        .alloc_buffer(tmp_bytes, DType::F32, vec![tmp_bytes / 4])
        .expect("tmp");
    (output, tmp)
}

/// Uniform-slot batched dispatch. `queries` are `(slot, seq_pos)`.
fn run_uniform(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    store: &TqStore,
    queries: &[(u32, u32)],
    q: &[f32],
    bucket_kv_seq_len: u32,
    nsg: u32,
) -> Vec<f32> {
    let n_q = queries.len() as u32;
    let slots: Vec<u32> = queries.iter().map(|&(slot, _)| slot).collect();
    let positions: Vec<u32> = queries.iter().map(|&(_, position)| position).collect();
    let q_buf = e6_f32_buffer(device, q);
    let slot_buf = e6_u32_buffer(device, &slots);
    let pos_buf = e6_u32_buffer(device, &positions);
    let (output, tmp) = tq_output(device, n_q);
    let mut encoder = device.command_encoder().expect("encoder");
    flash_attn_vec_tq_hb::flash_attn_vec_tq_hb_batched(
        &mut encoder,
        registry,
        device,
        n_q,
        &q_buf,
        &store.k_packed,
        &store.k_norms,
        &store.v_packed,
        &store.v_norms,
        &output,
        &tmp,
        &slot_buf,
        &pos_buf,
        &tq_params(bucket_kv_seq_len, nsg, CAP),
    )
    .expect("TQ-HB batched dispatch");
    encoder.commit_and_wait().expect("TQ-HB batched execution");
    e6_read_f32(&output)
}

/// Unequal-bank layout: slot `s` owns `NKV * BANK_CAPS[s]` rows at
/// `BANK_BASES[s]`.
const BANK_CAPS: [u32; N_SLOTS] = [1_536, 1_600, 1_552, 1_568];

fn bank_bases() -> ([u32; N_SLOTS], u32) {
    let mut bases = [0u32; N_SLOTS];
    // A non-zero leading gap proves base offsets are honored.
    let mut next = 64u32;
    for slot in 0..N_SLOTS {
        bases[slot] = next;
        next += NKV * BANK_CAPS[slot] + 32;
    }
    (bases, next)
}

#[allow(clippy::too_many_arguments)]
fn run_banked(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    store: &TqStore,
    arena_rows: u32,
    queries: &[(u32, u32)],
    q: &[f32],
    bucket_kv_seq_len: u32,
    nsg: u32,
) -> Vec<f32> {
    let (bases, _) = bank_bases();
    let n_q = queries.len() as u32;
    let base_rows: Vec<u32> = queries
        .iter()
        .map(|&(slot, _)| bases[slot as usize])
        .collect();
    let capacities: Vec<u32> = queries
        .iter()
        .map(|&(slot, _)| BANK_CAPS[slot as usize])
        .collect();
    let positions: Vec<u32> = queries.iter().map(|&(_, position)| position).collect();
    let q_buf = e6_f32_buffer(device, q);
    let base_buf = e6_u32_buffer(device, &base_rows);
    let cap_buf = e6_u32_buffer(device, &capacities);
    let pos_buf = e6_u32_buffer(device, &positions);
    let (output, tmp) = tq_output(device, n_q);
    let mut encoder = device.command_encoder().expect("encoder");
    flash_attn_vec_tq_hb::flash_attn_vec_tq_hb_batched_banked(
        &mut encoder,
        registry,
        device,
        n_q,
        &q_buf,
        &store.k_packed,
        &store.k_norms,
        &store.v_packed,
        &store.v_norms,
        &output,
        &tmp,
        &base_buf,
        &cap_buf,
        &pos_buf,
        arena_rows,
        // kv_capacity is a validation bound here; each query's bank
        // capacity comes from `capacities`.
        &tq_params(bucket_kv_seq_len, nsg, *BANK_CAPS.iter().max().unwrap()),
    )
    .expect("TQ-HB banked dispatch");
    encoder.commit_and_wait().expect("TQ-HB banked execution");
    e6_read_f32(&output)
}

fn tq_origin(registry: &mut KernelRegistry, device: &MlxDevice) -> String {
    let label = registry
        .get_pipeline_with_constants(
            "flash_attn_vec_tq_hb_batched_dk256",
            device.metal_device(),
            &[],
            &[(50, CBITS as i32)],
        )
        .expect("TQ-HB batched pipeline")
        .label()
        .to_owned();
    e6_pipeline_origin(registry, &label, true)
}

fn crosses(p: u32, w: u32) -> bool {
    bucket(p + 1) != bucket(p + w)
}

/// Assert or record one window row against its two authorities.
#[allow(clippy::too_many_arguments)]
fn judge_tq_row(
    probe: &str,
    case: &str,
    row: usize,
    position: u32,
    window: &[f32],
    served: &[f32],
    window_bucket: &[f32],
    crossing: bool,
    origin: &str,
) {
    let vs_served = E6Compare::new(served, window);
    let vs_bucket = E6Compare::new(window_bucket, window);
    let mut fields = e6_compare_fields(&vs_served);
    fields.push(("position", position.to_string()));
    fields.push(("row_bucket", format!("{:?}", bucket(position + 1))));
    fields.push(("crossing", crossing.to_string()));
    fields.push(("bitwise_vs_window_bucket", vs_bucket.bitwise().to_string()));
    fields.push(("origin", origin.to_owned()));
    e6_record(probe, &format!("{case}/row{row}"), &fields);
    assert!(
        vs_bucket.bitwise(),
        "{probe} {case} row {row}: differs from the same row alone under the window bucket \
         ({} mismatches, first {:?})",
        vs_bucket.mismatched,
        vs_bucket.first_mismatch
    );
    if crossing {
        assert!(
            window.iter().all(|value| value.is_finite()) && vs_served.max_abs_diff <= 1e-3,
            "{probe} {case} row {row}: bucket-crossing row outside the numeric bound \
             (max abs diff {:e})",
            vs_served.max_abs_diff
        );
    } else {
        assert!(
            vs_served.bitwise(),
            "{probe} {case} row {row}: in-bucket row differs from the served single row \
             ({} mismatches, first {:?})",
            vs_served.mismatched,
            vs_served.first_mismatch
        );
    }
}

fn window_queries(slot: u32, p: u32, w: u32) -> Vec<(u32, u32)> {
    (0..w).map(|i| (slot, p + i)).collect()
}

fn probe_tq(banked: bool) {
    assert_env_unset();
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let (arena_rows, store) = if banked {
        let (_, rows) = bank_bases();
        (rows, tq_store(&device, rows as usize, 0xe6_0b4e))
    } else {
        let rows = N_SLOTS as u32 * NKV * CAP;
        (rows, tq_store(&device, rows as usize, 0xe6_0a11))
    };
    let probe = if banked {
        "tq_hb_banked_window"
    } else {
        "tq_hb_batched_window"
    };
    let run =
        |registry: &mut KernelRegistry, queries: &[(u32, u32)], q: &[f32], kv: u32, nsg: u32| {
            if banked {
                run_banked(&device, registry, &store, arena_rows, queries, q, kv, nsg)
            } else {
                run_uniform(&device, registry, &store, queries, q, kv, nsg)
            }
        };
    let mut rng = E6Rng::new(0xe6_0a77 ^ banked as u64);
    let mut origin = String::new();
    for (index, &(case, p, w)) in WINDOWS.iter().enumerate() {
        let slot = (index % N_SLOTS) as u32;
        let crossing = crosses(p, w);
        assert_eq!(
            crossing,
            case.starts_with("cross"),
            "{case}: bucket placement"
        );
        let queries = window_queries(slot, p, w);
        let q: Vec<f32> = (0..w as usize * ROW).map(|_| rng.signed(1.0)).collect();
        let window_kv = p + w;
        let (_, window_nsg) = bucket(window_kv);
        let window = run(&mut registry, &queries, &q, window_kv, window_nsg);
        if origin.is_empty() {
            origin = tq_origin(&mut registry, &device);
        }
        for i in 0..w as usize {
            let position = p + i as u32;
            let q_row = &q[i * ROW..(i + 1) * ROW];
            let own_kv = position + 1;
            let served = run(
                &mut registry,
                &[(slot, position)],
                q_row,
                own_kv,
                bucket(own_kv).1,
            );
            let window_bucket = run(
                &mut registry,
                &[(slot, position)],
                q_row,
                window_kv,
                window_nsg,
            );
            judge_tq_row(
                probe,
                case,
                i,
                position,
                &window[i * ROW..(i + 1) * ROW],
                &served,
                &window_bucket,
                crossing,
                &origin,
            );
        }
    }

    // Cohort: four slots' windows (mixed widths, non-monotone slot order)
    // in one dispatch inside one bucket, against each window alone.
    let cohort: [(u32, u32, u32); N_SLOTS] = [(2, 40, 4), (0, 200, 3), (3, 333, 1), (1, 120, 4)];
    let mut queries = Vec::new();
    for &(slot, p, w) in &cohort {
        queries.extend(window_queries(slot, p, w));
    }
    let cohort_kv = cohort.iter().map(|&(_, p, w)| p + w).max().unwrap();
    let (_, cohort_nsg) = bucket(cohort_kv);
    assert_eq!(
        bucket(cohort_kv),
        bucket(41),
        "cohort stays inside one bucket"
    );
    let q: Vec<f32> = (0..queries.len() * ROW).map(|_| rng.signed(1.0)).collect();
    let joint = run(&mut registry, &queries, &q, cohort_kv, cohort_nsg);
    let mut offset = 0usize;
    for &(slot, p, w) in &cohort {
        let own = window_queries(slot, p, w);
        let q_own = &q[offset * ROW..(offset + w as usize) * ROW];
        let alone = run(&mut registry, &own, q_own, p + w, bucket(p + w).1);
        let compare = E6Compare::new(&alone, &joint[offset * ROW..(offset + w as usize) * ROW]);
        let mut fields = e6_compare_fields(&compare);
        fields.push(("origin", origin.clone()));
        e6_record(probe, &format!("cohort/slot{slot}_p{p}_w{w}"), &fields);
        assert!(
            compare.bitwise(),
            "{probe} cohort slot {slot} p={p} w={w}: joint rows differ from the window alone \
             ({} mismatches, first {:?})",
            compare.mismatched,
            compare.first_mismatch
        );
        offset += w as usize;
    }
}

#[test]
fn e6_tq_hb_batched_window_rows_match_single_rows() {
    probe_tq(false);
}

#[test]
fn e6_tq_hb_banked_window_rows_match_single_rows() {
    probe_tq(true);
}

fn run_f32_vec(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    k: &MlxBuffer,
    v: &MlxBuffer,
    q_heads_major: &[f32],
    kv_seq_len: u32,
    q_seq_len: u32,
) -> Vec<f32> {
    let q = e6_f32_buffer(device, q_heads_major);
    let output = device
        .alloc_buffer(
            q_seq_len as usize * ROW * 4,
            DType::F32,
            vec![q_seq_len as usize, ROW],
        )
        .expect("output");
    let tmp_bytes = flash_attn_vec::tmp_buffer_bytes_with_qL(NH, HD, q_seq_len);
    let tmp = device
        .alloc_buffer(tmp_bytes, DType::F32, vec![tmp_bytes / 4])
        .expect("tmp");
    let mut encoder = device.command_encoder().expect("encoder");
    flash_attn_vec::flash_attn_vec(
        &mut encoder,
        registry,
        device,
        &q,
        k,
        v,
        &output,
        &tmp,
        &FlashAttnVecParams {
            num_heads: NH,
            num_kv_heads: NKV,
            head_dim: HD,
            kv_seq_len,
            kv_capacity: CAP,
            scale: 1.0 / (HD as f32).sqrt(),
            mask_type: 1,
            sliding_window: 0,
            softcap: 0.0,
            q_seq_len,
        },
    )
    .expect("flash_attn_vec dispatch");
    encoder.commit_and_wait().expect("flash_attn_vec execution");
    // Output rows are `[query, head, dim]`.
    e6_read_f32(&output)
}

#[test]
fn e6_f32_flash_attn_vec_window_rows_match_single_rows() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let mut rng = E6Rng::new(0xe6_0f32);
    let kv_elems = (NKV * CAP * HD) as usize;
    let k: Vec<f32> = (0..kv_elems).map(|_| rng.signed(1.0)).collect();
    let v: Vec<f32> = (0..kv_elems).map(|_| rng.signed(1.0)).collect();
    let k = e6_f32_buffer(&device, &k);
    let v = e6_f32_buffer(&device, &v);
    let mut origin = String::new();
    for &(case, p, w) in WINDOWS {
        // Q rows by query: `[w, NH, HD]`; the kernel reads `[NH, w, HD]`.
        let q_rows: Vec<f32> = (0..w as usize * ROW).map(|_| rng.signed(1.0)).collect();
        let mut q_heads_major = vec![0.0f32; q_rows.len()];
        for i in 0..w as usize {
            for head in 0..NH as usize {
                let src = i * ROW + head * HD as usize;
                let dst = (head * w as usize + i) * HD as usize;
                q_heads_major[dst..dst + HD as usize]
                    .copy_from_slice(&q_rows[src..src + HD as usize]);
            }
        }
        let window = run_f32_vec(&device, &mut registry, &k, &v, &q_heads_major, p + w, w);
        if origin.is_empty() {
            let main = e6_pipeline_origin(&registry, "flash_attn_vec_dk256", false);
            let reduce = e6_pipeline_origin(&registry, "flash_attn_vec_reduce_dk256", false);
            origin = format!("{main},{reduce}");
        }
        for i in 0..w as usize {
            let position = p + i as u32;
            let q_row = &q_rows[i * ROW..(i + 1) * ROW];
            let single = run_f32_vec(&device, &mut registry, &k, &v, q_row, position + 1, 1);
            let compare = E6Compare::new(&single, &window[i * ROW..(i + 1) * ROW]);
            let mut fields = e6_compare_fields(&compare);
            fields.push(("position", position.to_string()));
            fields.push(("nwg", "32".to_owned()));
            fields.push(("origin", origin.clone()));
            e6_record(
                "f32_flash_attn_vec_window",
                &format!("{case}/row{i}"),
                &fields,
            );
            assert!(
                window[i * ROW..(i + 1) * ROW]
                    .iter()
                    .all(|value| value.is_finite())
                    && compare.max_abs_diff <= 1e-3,
                "flash_attn_vec {case} row {i}: outside the numeric bound (max abs diff {:e})",
                compare.max_abs_diff
            );
        }
    }
}
