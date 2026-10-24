//! Ignored bounded primitive control; no model, HF policy, fit or speed claim.
#![cfg(target_vendor = "apple")]
use half::bf16;
use mlx_native::graph::ComputeGraph;
use mlx_native::ops::deepseek_indexer::{
    deepseek_indexer_query_tile_plan, deepseek_indexer_topk_scratch_elements,
    dispatch_deepseek_indexer_into_mma, dispatch_deepseek_indexer_into_mma_tiled,
    DeepSeekIndexerParams,
};
use mlx_native::{CapturedNode, DType, KernelRegistry, MlxBuffer, MlxDevice};
const Q: usize = 257;
const K: usize = 1025;
const STRIDE: usize = 520;
const COLUMN: usize = 4;
const GUARD: usize = 64;
const CANARY: u8 = 0x5a;

struct Guarded {
    root: MlxBuffer,
    data: MlxBuffer,
}
impl Guarded {
    fn new(device: &MlxDevice, dtype: DType, shape: Vec<usize>) -> Self {
        let elements = shape.iter().product::<usize>();
        let bytes = elements * dtype.size_of();
        let mut root = device
            .alloc_buffer(
                bytes + 2 * GUARD,
                dtype,
                vec![(bytes + 2 * GUARD) / dtype.size_of()],
            )
            .unwrap();
        root.as_mut_slice::<u8>().unwrap().fill(CANARY);
        let mut data = root
            .slice_view(GUARD as u64, elements)
            .with_shape(shape)
            .unwrap();
        data.as_mut_slice::<u8>().unwrap().fill(0);
        Self { root, data }
    }
    fn guards(&self) {
        let bytes = self.root.as_slice::<u8>().unwrap();
        assert!(bytes[..GUARD].iter().all(|&x| x == CANARY));
        assert!(bytes[bytes.len() - GUARD..].iter().all(|&x| x == CANARY));
    }
}
fn params(start: u32) -> DeepSeekIndexerParams {
    DeepSeekIndexerParams {
        batch: 1,
        query_len: Q as u32,
        kv_len: K as u32,
        start_pos: start,
        ratio: 4,
        heads: 64,
        head_dim: 128,
        top_k: 512,
        offset: 17,
    }
}
fn key_score(key: usize) -> i32 {
    if key == 1024 {
        32
    } else {
        (key % 16) as i32
    }
}
fn assert_oracle(
    output: &MlxBuffer,
    start: u32,
    q: &MlxBuffer,
    kv: &MlxBuffer,
    weights: &MlxBuffer,
) {
    let q = q.as_slice::<u16>().unwrap();
    let kv = kv.as_slice::<u16>().unwrap();
    let weights = weights.as_slice::<f32>().unwrap();
    // Establish the sparse fixture invariant from actual input bytes, so the
    // scalar dot below accounts for EVERY nonzero feature/head.
    assert!(q
        .iter()
        .enumerate()
        .all(|(i, &x)| i % (64 * 128) == 0 || x == 0));
    assert!(kv.iter().enumerate().all(|(i, &x)| i % 128 == 0 || x == 0));
    assert!(weights
        .iter()
        .enumerate()
        .all(|(i, &x)| i % 64 == 0 || x == 0.0));
    let actual = output.as_slice::<i32>().unwrap();
    // Stream one row of independent scalar candidates. No full F32 q/score copy.
    let mut candidates = Vec::<(f32, i32)>::with_capacity(K);
    for row in 0..Q {
        candidates.clear();
        let valid = K.min((start as usize + row + 1) / 4);
        for key in 0..valid {
            let dot = bf16::from_bits(q[row * 64 * 128]).to_f32()
                * bf16::from_bits(kv[key * 128]).to_f32();
            let score = dot.max(0.0) * weights[row * 64];
            assert!(score.is_finite());
            candidates.push((score, key as i32));
        }
        candidates.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let values = &actual[row * STRIDE..(row + 1) * STRIDE];
        assert!(values[..COLUMN].iter().all(|&v| v == 77));
        assert!(values[COLUMN + 512..].iter().all(|&v| v == 77));
        for rank in 0..512 {
            let expected = candidates.get(rank).map_or(-1, |v| v.1 + 17);
            assert_eq!(
                values[COLUMN + rank],
                expected,
                "row={row} rank={rank} start={start}"
            );
        }
    }
}

fn range(b: &MlxBuffer, offset: usize, bytes: usize) -> (usize, usize) {
    let lo = b.contents_ptr() as usize + b.byte_offset() as usize + offset;
    (lo, lo + bytes)
}
fn check_capture(
    nodes: &[CapturedNode],
    q: &MlxBuffer,
    kv: &MlxBuffer,
    w: &MlxBuffer,
    s: &MlxBuffer,
    top: &MlxBuffer,
    out: &MlxBuffer,
) {
    assert_eq!(nodes.len(), 16); // four dispatches, each followed by a barrier, per tile
    for (tile, (origin, rows)) in [(0, 256), (256, 1)].into_iter().enumerate() {
        let score = range(s, 0, rows * K * 4);
        let a = range(top, 0, rows * 1024 * 4);
        let b = range(top, rows * 1024 * 4, rows * 1024 * 4);
        let expected = [
            (
                vec![
                    range(q, origin * 64 * 128 * 2, rows * 64 * 128 * 2),
                    range(kv, 0, K * 128 * 2),
                    range(w, origin * 64 * 4, rows * 64 * 4),
                ],
                vec![score],
            ),
            (vec![score], vec![a]),
            (vec![score, a], vec![b]),
            (
                vec![b],
                vec![range(out, origin * STRIDE * 4, rows * STRIDE * 4)],
            ),
        ];
        for (stage, (want_reads, want_writes)) in expected.into_iter().enumerate() {
            let index = tile * 8 + stage * 2;
            match &nodes[index] {
                CapturedNode::Dispatch { reads, writes, .. } => {
                    assert_eq!(reads, &want_reads);
                    assert_eq!(writes, &want_writes);
                }
                _ => panic!("expected dispatch"),
            }
            assert!(matches!(nodes[index + 1], CapturedNode::Barrier));
        }
    }
}

#[test]
#[ignore = "bounded Metal control; run alone under external process watchdog"]
fn query_tiles_bounded_mma_matches_untiled_and_streamed_cpu() {
    let device = MlxDevice::new().unwrap();
    let p = params(3843);
    let plan = deepseek_indexer_query_tile_plan(&p, STRIDE, COLUMN).unwrap();
    let mut q = Guarded::new(&device, DType::BF16, vec![1, Q, 64, 128]);
    let mut kv = Guarded::new(&device, DType::BF16, vec![1, K, 128]);
    let mut w = Guarded::new(&device, DType::F32, vec![1, Q, 64]);
    // Direct BF16 initialization: exactly representable one-feature dot products.
    for row in 0..Q {
        q.data.as_mut_slice::<u16>().unwrap()[row * 64 * 128] =
            bf16::from_f32(if row % 3 == 0 { 0.0 } else { 1.0 }).to_bits();
    }
    for key in 0..K {
        kv.data.as_mut_slice::<u16>().unwrap()[key * 128] =
            bf16::from_f32(key_score(key) as f32).to_bits();
    }
    for row in 0..Q {
        w.data.as_mut_slice::<f32>().unwrap()[row * 64] = if row % 3 == 0 { 0.0 } else { 1.0 };
    }
    let full_topk = deepseek_indexer_topk_scratch_elements(1, Q, K).unwrap();
    let score0 = Guarded::new(&device, DType::F32, vec![1, Q, K]);
    let topk0 = Guarded::new(&device, DType::I32, vec![1, Q, full_topk / Q]);
    let score1 = Guarded::new(&device, DType::F32, vec![1, plan.workspace_rows, K]);
    let topk1 = Guarded::new(
        &device,
        DType::I32,
        vec![1, plan.workspace_rows, plan.topk_row_elements],
    );
    let mut out0 = Guarded::new(&device, DType::I32, vec![1, Q, STRIDE]);
    let mut out1 = Guarded::new(&device, DType::I32, vec![1, Q, STRIDE]);
    let roots = [&q, &kv, &w, &score0, &topk0, &score1, &topk1, &out0, &out1];
    let metal_bytes = roots.iter().map(|g| g.root.data_byte_len()).sum::<usize>();
    // All explicit numeric host storage: one K-entry (score,key) row plus
    // fixed params/plans counters. 64 KiB conservative allowance includes stack
    // numeric arrays; opaque driver/compiler/allocator memory is outside this cap.
    let host_bound = K * std::mem::size_of::<(i32, i32)>() + 65536;
    assert_eq!(metal_bytes, 11_913_796 + 9 * 128);
    assert!(metal_bytes + host_bound <= 16 * 1024 * 1024);
    let mut registry = KernelRegistry::new();
    for (case, start, tail_score) in [
        ("causal_winner", 3843, 32.0),
        ("early_causal", 0, 32.0),
        ("cross_block_tie", 3843, 15.0),
    ] {
        // The previous iteration has drained BOTH arms and dropped its graph
        // and command owners before this same-root CPU mutation.
        kv.data.as_mut_slice::<u16>().unwrap()[1024 * 128] = bf16::from_f32(tail_score).to_bits();
        out0.data.as_mut_slice::<i32>().unwrap().fill(77);
        out1.data.as_mut_slice::<i32>().unwrap().fill(77);
        let p = params(start);
        let mut a = device.command_encoder().unwrap();
        let mut b = device.command_encoder().unwrap();
        // Every fallible encode/setup completes before the first submission.
        dispatch_deepseek_indexer_into_mma(
            &mut a,
            &mut registry,
            &device,
            &q.data,
            &kv.data,
            &w.data,
            &score0.data,
            &topk0.data,
            &out0.data,
            STRIDE,
            COLUMN,
            &p,
        )
        .unwrap();
        b.start_capture();
        dispatch_deepseek_indexer_into_mma_tiled(
            &mut b,
            &mut registry,
            &device,
            &q.data,
            &kv.data,
            &w.data,
            &score1.data,
            &topk1.data,
            &out1.data,
            STRIDE,
            COLUMN,
            &p,
        )
        .unwrap();
        let captured = b.take_capture().unwrap();
        check_capture(
            &captured,
            &q.data,
            &kv.data,
            &w.data,
            &score1.data,
            &topk1.data,
            &out1.data,
        );
        // Same checked reorder/replay steps as GraphSession::finish_with_reorder,
        // split from submission so BOTH arms are fully encoded before either commit.
        let mut graph = ComputeGraph::from_nodes(captured);
        assert_eq!(graph.unannotated_dispatch_count(), 0);
        let reordered = graph.reorder();
        let barriers = graph.encode_with_barriers(&mut b);
        a.commit();
        b.commit();
        let done_a = a.wait_until_completed();
        let done_b = b.wait_until_completed();
        // Drain BOTH before any assertion, readback or owner teardown.
        println!(
            "query_tiles_completion case={case} tail_score={tail_score} start={start} untiled={} tiled={}",
            done_a.is_ok(),
            done_b.is_ok()
        );
        done_a.unwrap();
        done_b.unwrap();
        assert!(barriers > 0);
        println!("query_tiles_replay case={case} tail_score={tail_score} start={start} captured=true reordered={reordered} barriers={barriers}");
        assert_eq!(
            out0.data.as_slice::<i32>().unwrap(),
            out1.data.as_slice::<i32>().unwrap()
        );
        assert_oracle(&out1.data, start, &q.data, &kv.data, &w.data);
        if case == "causal_winner" {
            let out = out1.data.as_slice::<i32>().unwrap();
            assert_ne!(out[255 * STRIDE + COLUMN], out[256 * STRIDE + COLUMN]);
            assert_eq!(out[256 * STRIDE + COLUMN], 1041);
        } else if case == "cross_block_tie" {
            let out = out1.data.as_slice::<i32>().unwrap();
            // 64 older score-15 keys (15,31,...,1023) precede key1024.
            assert_eq!(out[256 * STRIDE + COLUMN + 63], 1040);
            assert_eq!(out[256 * STRIDE + COLUMN + 64], 1041);
        } else {
            let out = out1.data.as_slice::<i32>().unwrap();
            assert!(out[COLUMN..COLUMN + 512].iter().all(|&v| v == -1));
        }
        for root in [&q, &kv, &w, &score0, &topk0, &score1, &topk1, &out0, &out1] {
            root.guards();
        }
        println!(
            "query_tiles_oracle case={case} tail_score={tail_score} start={start} cpu=true untiled=true guards=true rows={Q} keys={K}"
        );
        drop(graph);
        drop(a);
        drop(b);
    }
    // Actual buffer mismatch predispatch control, including the last output row.
    let short = out1.data.slice_view(0, Q * STRIDE - 1);
    let mut encoder = device.command_encoder().unwrap();
    encoder.start_capture();
    let result = dispatch_deepseek_indexer_into_mma_tiled(
        &mut encoder,
        &mut registry,
        &device,
        &q.data,
        &kv.data,
        &w.data,
        &score1.data,
        &topk1.data,
        &short,
        STRIDE,
        COLUMN,
        &p,
    );
    assert!(result.is_err());
    assert!(encoder.take_capture().unwrap().is_empty());
    println!("query_tiles_control_complete roots=9 metal_bytes={metal_bytes} numeric_host_bound={host_bound} combined_bound={} undersized_zero_encode=true runtime_admission=false performance_claim=false",metal_bytes+host_bound);
}
