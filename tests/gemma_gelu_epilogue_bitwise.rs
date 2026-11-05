//! ADR-063 Q4-GEMMA-DECODE T-N6 and T-N3 (F2): the expert-grouped Q4_K
//! gate_up kernel with the GELU-tanh(gate) * up epilogue, at its frozen
//! geometry (R0 = 4), is bitwise equal, on the raw u32 of every F32 output,
//! to the stock chain on the same metallib path:
//! `kernel_mul_mv_id_q4_K_f32` into `[pairs, 2 * n_ff]`, then
//! `moe_swiglu_batch` into `[pairs, n_ff]`.
//!
//! Covered: the Gemma 4 gate_up shape (N = 1408, K = 2816, top-8, 1..=16
//! tokens) with 128 experts; synthetic row tails (n_ff = 7, 515) and a short
//! K; top-1 pairs; all-distinct, all-one-expert, random, pooled, skewed and
//! out-of-range-id patterns (the stock NaN poison through the epilogue); zero
//! and subnormal half scales; ±0, subnormal, large and NaN/∞ activations; a
//! NaN/∞ token row leaves every other pair unchanged; non-zero-offset views
//! and sentinel bytes outside the output view; the encoded kernel and grid.
//!
//! Design §2.4 witness rows: real Q1 routing selections (4 lanes per
//! position) replayed as n_tokens = 4, 8 and 16, from the checked-in sample
//! `tests/fixtures/gemma_routing_witness_sample.json` (always) and from the
//! full witness when present (`GEMMA_DECODE_ROUTING_WITNESS` overrides its
//! path; absent → an explicit SKIPPED line). V2 replay: set
//! `GEMMA_DECODE_V2_OVERLAP` to a file holding the hf2q `[ROUTING_OVERLAP]`
//! line to replay one pattern per observed (width, U).
//!
//! Run on both metallib paths:
//! `MLX_PRECOMPILED_METALLIB=1 cargo test --release --test gemma_gelu_epilogue_bitwise -- --test-threads=1`
//! and the same with `MLX_PRECOMPILED_METALLIB=0`.

#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used, clippy::panic, clippy::too_many_arguments)]

#[path = "support/gemma_expert_data.rs"]
mod gemma_expert_data;

use gemma_expert_data::*;
use mlx_native::ops::moe_dispatch::moe_swiglu_batch_encode;
use mlx_native::{
    quantized_matmul_id_ggml_mv_grouped_geglu, quantized_matmul_id_ggml_mv_with_policy,
    CapturedNode, DType, GgmlRoutingPolicy, GgmlType, KernelRegistry, MlxDevice,
};

/// Gemma 4 gate_up: N = 2 * n_ff, K = hidden, 128 experts.
const GATE_UP_N: usize = 1408;
const GATE_UP_K: usize = 2816;
const GATE_UP_SEED: u64 = 0x9e37_79b9_7f4a_7c15;

struct Harness {
    device: MlxDevice,
    registry: KernelRegistry,
}

impl Harness {
    fn new() -> Self {
        Self {
            device: MlxDevice::new().expect("Metal device"),
            registry: KernelRegistry::new(),
        }
    }

    /// Stock chain: mv_id gate_up, then moe_swiglu_batch. Returns the raw u32
    /// of the whole guarded `[pairs, n/2]` output parent.
    fn reference(
        &mut self,
        w: &Weights,
        n_tokens: usize,
        top_k: usize,
        ids: &[u32],
        input: &[f32],
    ) -> Vec<u32> {
        let pairs = n_tokens * top_k;
        let n_ff = w.n / 2;
        let (_input_parent, input_view) = guarded_f32(&self.device, input);
        let (_ids_parent, ids_view) = guarded_u32(&self.device, ids);
        let (_gate_up_parent, gate_up_view) =
            guarded_f32(&self.device, &vec![f32::from_bits(SENTINEL); pairs * w.n]);
        let (output_parent, output_view) =
            guarded_f32(&self.device, &vec![f32::from_bits(SENTINEL); pairs * n_ff]);

        let mut encoder = self.device.command_encoder().expect("encoder");
        quantized_matmul_id_ggml_mv_with_policy(
            &mut encoder,
            &mut self.registry,
            &self.device,
            &input_view,
            &w.view,
            &ids_view,
            &gate_up_view,
            &params(w, n_tokens, top_k),
            &GgmlRoutingPolicy::default(),
        )
        .expect("stock gate_up");
        encoder.commit_and_wait().expect("stock gate_up GPU");

        let mut encoder = self.device.command_encoder().expect("encoder");
        moe_swiglu_batch_encode(
            &mut encoder,
            &mut self.registry,
            self.device.metal_device(),
            &gate_up_view,
            &output_view,
            n_ff,
            pairs,
        )
        .expect("stock swiglu");
        encoder.commit_and_wait().expect("stock swiglu GPU");
        output_parent.as_slice::<u32>().expect("output").to_vec()
    }

    fn fused(
        &mut self,
        w: &Weights,
        n_tokens: usize,
        top_k: usize,
        ids: &[u32],
        input: &[f32],
    ) -> Vec<u32> {
        let pairs = n_tokens * top_k;
        let (_input_parent, input_view) = guarded_f32(&self.device, input);
        let (_ids_parent, ids_view) = guarded_u32(&self.device, ids);
        let (output_parent, output_view) =
            guarded_f32(&self.device, &vec![f32::from_bits(SENTINEL); pairs * (w.n / 2)]);
        let mut encoder = self.device.command_encoder().expect("encoder");
        quantized_matmul_id_ggml_mv_grouped_geglu(
            &mut encoder,
            &mut self.registry,
            &self.device,
            &input_view,
            &w.view,
            &ids_view,
            &output_view,
            &params(w, n_tokens, top_k),
        )
        .expect("fused gate_up");
        encoder.commit_and_wait().expect("fused GPU");
        output_parent.as_slice::<u32>().expect("output").to_vec()
    }
}

fn check_cell(
    h: &mut Harness,
    w: &Weights,
    n_tokens: usize,
    top_k: usize,
    pattern: Pattern,
    nan_row: bool,
    seed: u64,
    digest: &mut u64,
) {
    let ids = make_ids(pattern, n_tokens, top_k, w.n_experts, seed);
    let label = format!(
        "GEGLU N={} K={} E={} tokens={n_tokens} top_k={top_k} {pattern:?} nan_row={nan_row}",
        w.n, w.k, w.n_experts
    );
    check_ids(h, w, n_tokens, top_k, &ids, nan_row, seed, &label, digest);
}

/// Compares the stock chain with the fused kernel for explicit ids.
fn check_ids(
    h: &mut Harness,
    w: &Weights,
    n_tokens: usize,
    top_k: usize,
    ids: &[u32],
    nan_row: bool,
    seed: u64,
    label: &str,
    digest: &mut u64,
) {
    let n_ff = w.n / 2;
    let pairs = n_tokens * top_k;
    assert_eq!(ids.len(), pairs, "{label}: ids length");
    let mut input = make_input(n_tokens, w.k, seed ^ 0x5eed);
    if nan_row {
        poison_row(&mut input, n_tokens / 2, w.k);
    }
    let stock = h.reference(w, n_tokens, top_k, ids, &input);
    assert_stock_frame(&stock, pairs, n_ff, label);
    for (pair, &id) in ids.iter().enumerate() {
        if id as usize >= w.n_experts {
            let row = &stock[GUARD + pair * n_ff..GUARD + (pair + 1) * n_ff];
            assert!(
                row.iter().all(|&bits| f32::from_bits(bits).is_nan()),
                "{label}: stock poison did not stay NaN through moe_swiglu_batch"
            );
        }
    }
    fnv(digest, &stock);
    let got = h.fused(w, n_tokens, top_k, ids, &input);
    assert_bitwise(&stock, &got, n_ff, ids, label);
}

#[test]
fn geglu_gate_up_is_bitwise_stock_gate_up_then_swiglu_for_every_width_and_pattern() {
    let mut h = Harness::new();
    let w = make_weights(&h.device, GgmlType::Q4_K, GATE_UP_N, GATE_UP_K, 128, GATE_UP_SEED);
    let mut digest = 0xcbf2_9ce4_8422_2325u64;
    for n_tokens in 1..=16 {
        for (index, &pattern) in PATTERNS.iter().enumerate() {
            let nan_row = matches!(pattern, Pattern::Pooled | Pattern::AllOne) && n_tokens > 1;
            check_cell(
                &mut h,
                &w,
                n_tokens,
                8,
                pattern,
                nan_row,
                (n_tokens * 131 + index) as u64 + 1,
                &mut digest,
            );
        }
    }
    eprintln!("geglu gate_up stock digest {digest:#018x}");
}

#[test]
fn geglu_row_tails_short_k_and_top1_are_bitwise_stock() {
    let mut h = Harness::new();
    let mut digest = 0xcbf2_9ce4_8422_2325u64;
    let patterns = [
        Pattern::AllOne,
        Pattern::Pooled,
        Pattern::Random,
        Pattern::InvalidInGroup,
    ];
    for (n, k) in [(14usize, 2816usize), (1030, 2816), (1030, 512)] {
        let w = make_weights(&h.device, GgmlType::Q4_K, n, k, 16, (n * 31 + k) as u64);
        for n_tokens in [1usize, 2, 5, 16] {
            for (index, &pattern) in patterns.iter().enumerate() {
                check_cell(
                    &mut h,
                    &w,
                    n_tokens,
                    8,
                    pattern,
                    n_tokens == 5,
                    (n_tokens * 17 + index) as u64 + 5,
                    &mut digest,
                );
            }
        }
        for pairs in [1usize, 37, 128] {
            for (index, &pattern) in patterns.iter().enumerate() {
                check_cell(
                    &mut h,
                    &w,
                    pairs,
                    1,
                    pattern,
                    pairs == 37,
                    (pairs * 13 + index) as u64 + 9,
                    &mut digest,
                );
            }
        }
    }
    eprintln!("geglu tail stock digest {digest:#018x}");
}

// ---- T-N4 (F2 part): a NaN/∞ token row does not perturb its neighbours ----

#[test]
fn geglu_nan_inf_token_row_does_not_perturb_other_pairs() {
    let mut h = Harness::new();
    let (n_tokens, top_k) = (16usize, 8usize);
    let w = make_weights(&h.device, GgmlType::Q4_K, GATE_UP_N, GATE_UP_K, 32, 0xf2_4a4e);
    let n_ff = w.n / 2;
    for pattern in [Pattern::AllOne, Pattern::Pooled] {
        let ids = make_ids(pattern, n_tokens, top_k, w.n_experts, 0xabc);
        let clean = make_input(n_tokens, w.k, 0xdef);
        let poisoned_token = n_tokens / 2;
        let mut poisoned = clean.clone();
        poison_row(&mut poisoned, poisoned_token, w.k);
        let a = h.fused(&w, n_tokens, top_k, &ids, &clean);
        let b = h.fused(&w, n_tokens, top_k, &ids, &poisoned);
        for pair in 0..n_tokens * top_k {
            let range = GUARD + pair * n_ff..GUARD + (pair + 1) * n_ff;
            if pair / top_k == poisoned_token {
                assert!(
                    b[range].iter().any(|&bits| f32::from_bits(bits).is_nan()),
                    "{pattern:?}: the poisoned token produced no NaN"
                );
            } else {
                assert_eq!(
                    a[range.clone()],
                    b[range],
                    "{pattern:?}: pair {pair} changed when token {poisoned_token} became NaN/inf"
                );
            }
        }
    }
}

// ---- T-N3: design §2.4 witness rows and the V2 replay ----
//
// A witness row is one (layer, position) of the Q1 routing witness: the four
// lanes' top-8 selections become one n_tokens = 4 pattern (token = lane).
// Two consecutive positions give n_tokens = 8 and four give 16. The fused
// gate_up replays the token rows at top-8.

/// The Q1 routing witness (`GEMMA_DECODE_ROUTING_WITNESS` overrides).
fn witness_path() -> std::path::PathBuf {
    if let Some(path) = std::env::var_os("GEMMA_DECODE_ROUTING_WITNESS") {
        return path.into();
    }
    let home = std::env::var_os("HOME").unwrap_or_default();
    std::path::Path::new(&home).join(
        ".local/state/hf2q/campaigns/adr063-preflight-20261023/\
         q1-gemma-routing1511-t0-on-actual-v1/output/routing-witness.jsonl",
    )
}

const WITNESS_LANES: usize = 4;
const WITNESS_LAYERS: u32 = 30;
const WITNESS_POSITIONS: u32 = 144;

/// Four consecutive positions of one layer; `rows[i]` is the 32 ids of
/// position `start + i`, lane-major.
struct WitnessGroup {
    kind: String,
    layer: u32,
    start: u32,
    rows: Vec<Vec<u32>>,
}

/// Every (layer, position) of the full witness, lanes concatenated in lane
/// order. Panics on a malformed or incomplete file.
fn load_witness(path: &std::path::Path) -> std::collections::HashMap<(u32, u32), Vec<u32>> {
    let text = std::fs::read_to_string(path).expect("read routing witness");
    let mut lanes = std::collections::HashMap::<(u32, u32), Vec<Option<Vec<u32>>>>::new();
    for line in text.lines() {
        let record: serde_json::Value = serde_json::from_str(line).expect("witness JSON line");
        if record["kind"] != "routing_row" {
            continue;
        }
        let at = &record["at"];
        let key = (
            at["layer"].as_u64().expect("layer") as u32,
            at["row"]["position"].as_u64().expect("position") as u32,
        );
        let lane = at["row"]["lane"].as_u64().expect("lane") as usize;
        let ids: Vec<u32> = record["selected_ids"]
            .as_array()
            .expect("selected_ids")
            .iter()
            .map(|id| id.as_u64().expect("id") as u32)
            .collect();
        let slot = &mut lanes.entry(key).or_insert_with(|| vec![None; WITNESS_LANES])[lane];
        assert!(slot.is_none(), "duplicate witness row {key:?} lane {lane}");
        *slot = Some(ids);
    }
    assert_eq!(
        lanes.len(),
        (WITNESS_LAYERS * WITNESS_POSITIONS) as usize,
        "witness must hold 30 layers x 144 positions"
    );
    lanes
        .into_iter()
        .map(|(key, lanes)| {
            let rows: Vec<u32> = lanes
                .into_iter()
                .enumerate()
                .flat_map(|(lane, ids)| {
                    ids.unwrap_or_else(|| panic!("witness row {key:?} lacks lane {lane}"))
                })
                .collect();
            (key, rows)
        })
        .collect()
}

/// The checked-in sample (`tests/fixtures/gemma_routing_witness_extract.py`).
fn load_fixture() -> Vec<WitnessGroup> {
    let doc: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/gemma_routing_witness_sample.json"))
            .expect("fixture JSON");
    assert_eq!(doc["lanes"], WITNESS_LANES);
    doc["groups"]
        .as_array()
        .expect("groups")
        .iter()
        .map(|group| {
            let positions = group["positions"].as_array().expect("positions");
            WitnessGroup {
                kind: group["kind"].as_str().expect("kind").to_string(),
                layer: group["layer"].as_u64().expect("layer") as u32,
                start: positions[0].as_u64().expect("position") as u32,
                rows: group["ids"]
                    .as_array()
                    .expect("ids")
                    .iter()
                    .map(|lanes| {
                        lanes
                            .as_array()
                            .expect("lanes")
                            .iter()
                            .flat_map(|ids| ids.as_array().expect("lane ids").iter())
                            .map(|id| id.as_u64().expect("id") as u32)
                            .collect()
                    })
                    .collect(),
            }
        })
        .collect()
}

/// Real routing: 8 distinct in-range experts per token.
fn assert_real_routing(ids: &[u32], label: &str) {
    for token in ids.chunks(8) {
        let mut sorted = token.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert!(
            token.len() == 8 && sorted.len() == 8 && sorted[7] < 128,
            "{label}: not a top-8 selection of 128: {token:?}"
        );
    }
}

/// The n_tokens = 4, 8 and 16 cells of one group: (name, n_tokens, ids).
fn witness_cells(group: &WitnessGroup) -> Vec<(String, usize, Vec<u32>)> {
    assert_eq!(group.rows.len(), 4, "a witness group spans four positions");
    let mut cells = Vec::new();
    for (i, row) in group.rows.iter().enumerate() {
        cells.push((format!("p{}", group.start + i as u32), 4, row.clone()));
    }
    for half in 0..2 {
        let ids = [group.rows[2 * half].as_slice(), &group.rows[2 * half + 1]].concat();
        cells.push((format!("p{}+1", group.start + 2 * half as u32), 8, ids));
    }
    cells.push((format!("p{}+3", group.start), 16, group.rows.concat()));
    cells
}

/// Replays `(label, n_tokens, ids)` cells (top-8 token rows) through the
/// Gemma gate_up shape, stock chain against the fused kernel.
fn replay_top8_cells(source: &str, cells: &[(String, usize, Vec<u32>)]) {
    let mut h = Harness::new();
    let w = make_weights(&h.device, GgmlType::Q4_K, GATE_UP_N, GATE_UP_K, 128, GATE_UP_SEED);
    let mut digest = 0xcbf2_9ce4_8422_2325u64;
    for (index, (label, n_tokens, ids)) in cells.iter().enumerate() {
        assert_real_routing(ids, label);
        let label = format!("{source} {label} GEGLU N={} K={} tokens={n_tokens}", w.n, w.k);
        check_ids(
            &mut h,
            &w,
            *n_tokens,
            8,
            ids,
            false,
            index as u64 * 7919 + 11,
            &label,
            &mut digest,
        );
    }
    eprintln!("{source}: {} cells bitwise stock, digest {digest:#018x}", cells.len());
}

fn group_cells(groups: &[WitnessGroup]) -> Vec<(String, usize, Vec<u32>)> {
    groups
        .iter()
        .flat_map(|group| {
            witness_cells(group).into_iter().map(move |(name, n_tokens, ids)| {
                (format!("{} L{} {name}", group.kind, group.layer), n_tokens, ids)
            })
        })
        .collect()
}

#[test]
fn witness_fixture_rows_replay_bitwise_stock() {
    let groups = load_fixture();
    assert!(
        groups.iter().any(|g| g.start >= 128) && groups.iter().any(|g| g.start < 128),
        "the fixture samples rows 128-143 and 29-127"
    );
    let path = witness_path();
    if path.exists() {
        let witness = load_witness(&path);
        for group in &groups {
            for (i, row) in group.rows.iter().enumerate() {
                let key = (group.layer, group.start + i as u32);
                assert_eq!(&witness[&key], row, "fixture row {key:?} differs from {path:?}");
            }
        }
        eprintln!("fixture rows match the source witness at {}", path.display());
    } else {
        eprintln!(
            "NOTE: source witness absent at {}; fixture provenance not re-checked",
            path.display()
        );
    }
    replay_top8_cells("witness-fixture", &group_cells(&groups));
}

#[test]
fn witness_file_rows_replay_bitwise_stock() {
    let path = witness_path();
    if !path.exists() {
        eprintln!(
            "SKIPPED witness_file_rows_replay_bitwise_stock: routing witness not found at {} \
             (set GEMMA_DECODE_ROUTING_WITNESS); only the checked-in fixture replay ran",
            path.display()
        );
        return;
    }
    let witness = load_witness(&path);
    let group = |kind: &str, layer: u32, start: u32| WitnessGroup {
        kind: kind.to_string(),
        layer,
        start,
        rows: (start..start + 4).map(|p| witness[&(layer, p)].clone()).collect(),
    };
    let mut groups = Vec::new();
    // Rows 128-143 (greedy continuation): every layer, every position.
    for layer in 0..WITNESS_LAYERS {
        for start in [128, 132, 136, 140] {
            groups.push(group("decode", layer, start));
        }
    }
    // Rows 29-127: every third layer, four spread groups.
    for layer in (0..WITNESS_LAYERS).step_by(3) {
        for start in [29, 61, 93, 124] {
            groups.push(group("prefill", layer, start));
        }
    }
    replay_top8_cells("witness-file", &group_cells(&groups));
}

/// Every top-8 `(width, U)` in hf2q `[ROUTING_OVERLAP]` lines (design §7.4),
/// over all layers, from text that holds them (e.g. the worker's stderr).
fn v2_width_u(text: &str) -> std::collections::BTreeSet<(usize, usize)> {
    let reports: Vec<&str> = text
        .lines()
        .filter_map(|line| line.split_once("[ROUTING_OVERLAP]").map(|(_, json)| json.trim()))
        .collect();
    assert!(!reports.is_empty(), "no [ROUTING_OVERLAP] line");
    let mut cells = std::collections::BTreeSet::new();
    for json in reports {
        let report: serde_json::Value = serde_json::from_str(json).expect("[ROUTING_OVERLAP] JSON");
        for entry in report["distinct_experts"].as_array().expect("distinct_experts") {
            assert_eq!(entry["top_k"], 8, "Gemma routes top-8: {entry}");
            let width = entry["width"].as_u64().expect("width") as usize;
            for (u, count) in entry["u_hist"].as_object().expect("u_hist") {
                if count.as_u64().expect("count") > 0 {
                    cells.insert((width, u.parse().expect("U key")));
                }
            }
        }
    }
    cells
}

/// The V2 replay cells: one top-8 pattern with exactly U distinct experts per
/// observed `(width, U)`. Widths beyond the parity matrix (16 tokens) are
/// reported and skipped.
fn v2_cells(text: &str) -> Vec<(String, usize, Vec<u32>)> {
    let mut rng = Rng(0x0000_0000_5eed_00f2);
    let mut cells = Vec::new();
    for (width, u) in v2_width_u(text) {
        if width > 16 {
            eprintln!("V2 width {width} (U {u}) is outside the F2 parity matrix; not replayed");
            continue;
        }
        let ids = pattern_with_u(width, u, &mut rng);
        let mut distinct = ids.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), u, "V2 pattern width {width} must hold exactly U = {u}");
        cells.push((format!("v2 width={width} u={u}"), width, ids));
    }
    cells
}

#[test]
fn v2_overlap_report_builds_patterns_with_each_observed_u() {
    let log = concat!(
        "hf2q worker exit\n",
        "[ROUTING_OVERLAP] {\"steps\":9,\"mixed_bucket_share\":0.0,\"distinct_experts\":[",
        "{\"layer\":0,\"width\":4,\"top_k\":8,\"u_hist\":{\"8\":3,\"21\":5,\"32\":1}},",
        "{\"layer\":1,\"width\":4,\"top_k\":8,\"u_hist\":{\"21\":2,\"26\":0}},",
        "{\"layer\":0,\"width\":1,\"top_k\":8,\"u_hist\":{\"8\":4}},",
        "{\"layer\":0,\"width\":16,\"top_k\":8,\"u_hist\":{\"40\":2,\"128\":1}}],",
        "\"sliding\":{},\"global\":{}}\n",
    );
    let expected: std::collections::BTreeSet<(usize, usize)> =
        [(1, 8), (4, 8), (4, 21), (4, 32), (16, 40), (16, 128)].into();
    assert_eq!(v2_width_u(log), expected);
    let cells = v2_cells(log);
    assert_eq!(cells.len(), expected.len());
    for (label, n_tokens, ids) in &cells {
        assert_eq!(ids.len(), n_tokens * 8, "{label}");
        assert_real_routing(ids, label);
    }
}

#[test]
fn v2_overlap_patterns_replay_bitwise_stock() {
    let Some(path) = std::env::var_os("GEMMA_DECODE_V2_OVERLAP") else {
        eprintln!(
            "SKIPPED v2_overlap_patterns_replay_bitwise_stock: GEMMA_DECODE_V2_OVERLAP is unset \
             (name a file holding the hf2q [ROUTING_OVERLAP] line to replay each observed U)"
        );
        return;
    };
    let text = std::fs::read_to_string(&path).expect("read GEMMA_DECODE_V2_OVERLAP");
    let cells = v2_cells(&text);
    assert!(!cells.is_empty(), "no replayable V2 cells in {path:?}");
    replay_top8_cells("v2", &cells);
}

// ---- launch contract and validation ----

#[test]
fn geglu_encodes_the_frozen_rt4_kernel_on_the_stock_pair_grid() {
    let mut h = Harness::new();
    let w = make_weights(&h.device, GgmlType::Q4_K, GATE_UP_N, GATE_UP_K, 128, GATE_UP_SEED);
    for n_tokens in [1usize, 4, 16] {
        let pairs = n_tokens * 8;
        let input = h
            .device
            .alloc_buffer(n_tokens * w.k * 4, DType::F32, vec![n_tokens * w.k])
            .expect("input");
        let ids = h.device.alloc_buffer(pairs * 4, DType::U32, vec![pairs]).expect("ids");
        let output = h
            .device
            .alloc_buffer(pairs * (w.n / 2) * 4, DType::F32, vec![pairs * (w.n / 2)])
            .expect("output");
        let mut encoder = h.device.command_encoder().expect("encoder");
        encoder.start_capture();
        quantized_matmul_id_ggml_mv_grouped_geglu(
            &mut encoder,
            &mut h.registry,
            &h.device,
            &input,
            &w.view,
            &ids,
            &output,
            &params(&w, n_tokens, 8),
        )
        .expect("capture encode");
        let captured = encoder.take_capture().expect("capture");
        assert_eq!(captured.len(), 1, "expected exactly one dispatch");
        match &captured[0] {
            CapturedNode::Dispatch {
                pipeline,
                threads_per_grid,
                ..
            } => {
                assert_eq!(pipeline.label(), "kernel_mul_mv_id_q4_K_f32_grouped_geglu_rt4");
                // 704 output columns, 8 per threadgroup; one row per pair.
                assert_eq!(
                    [threads_per_grid.width, threads_per_grid.height, threads_per_grid.depth],
                    [GATE_UP_N as u64 / 2 / 8, pairs as u64, 1],
                    "tokens={n_tokens}"
                );
            }
            CapturedNode::Barrier => panic!("expected a dispatch"),
        }
    }
}

#[test]
fn geglu_entry_rejects_odd_widths_and_other_types() {
    let mut h = Harness::new();
    let w = make_weights(&h.device, GgmlType::Q4_K, 8, 256, 2, 3);
    let input = h.device.alloc_buffer(256 * 4, DType::F32, vec![256]).expect("input");
    let ids = h.device.alloc_buffer(4, DType::U32, vec![1]).expect("ids");
    let output = h.device.alloc_buffer(8 * 4, DType::F32, vec![8]).expect("output");
    let mut encoder = h.device.command_encoder().expect("encoder");
    let mut odd = params(&w, 1, 1);
    odd.n = 7;
    let mut q8 = params(&w, 1, 1);
    q8.ggml_type = GgmlType::Q8_0;
    for bad in [odd, q8] {
        assert!(quantized_matmul_id_ggml_mv_grouped_geglu(
            &mut encoder,
            &mut h.registry,
            &h.device,
            &input,
            &w.view,
            &ids,
            &output,
            &bad,
        )
        .is_err());
    }
}
