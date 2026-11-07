//! `ops::deepseek_rows`: every row of a rows dispatch is bitwise equal to the
//! existing one-row kernel on that row.
//!
//! For each rows family the reference is the existing one-row dispatch run
//! once per row on that row's buffers:
//! - sparse attention (score + reduce) vs
//!   `dispatch_deepseek_sparse_attention_with_scratch`, widths 128..640,
//!   kv 128..2048, sentinels, out-of-range indices, NaN in q / KV / sinks;
//! - sparse flash vs `dispatch_deepseek_sparse_attention_flash_decode`, one
//!   and several widths (contiguous and permuted flash batches), bad indices;
//! - indexer vs `dispatch_deepseek_indexer_into_mma_tiled`, 1 and 2 top-k
//!   blocks, masked slots (valid < kv), exact score ties, non-finite weights;
//! - compressor vs `dispatch_deepseek_compressor`, ratios 4 (dims 128, 512)
//!   and 128, emit and non-emit steps mixed over consecutive tokens.
//!
//! Cases cover B = 1..8, unequal per-row shapes/positions in one dispatch,
//! entry order differing from query-row order (row permutation), and repeat
//! runs (determinism). Run under both shader paths (latched per process):
//! `cargo test --release --test deepseek_rows` and
//! `MLX_PRECOMPILED_METALLIB=0 cargo test --release --test deepseek_rows`.

#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used, clippy::panic, clippy::too_many_arguments)]

use half::bf16;
use mlx_native::ops::deepseek_compressor::{
    dispatch_deepseek_compressor, DeepSeekCompressorParams,
};
use mlx_native::ops::deepseek_indexer::{
    deepseek_indexer_query_tile_plan, dispatch_deepseek_indexer_into_mma_tiled,
    DeepSeekIndexerParams,
};
use mlx_native::ops::deepseek_rows::{
    deepseek_indexer_rows_workspace_elements, deepseek_rows_kernel_names,
    deepseek_sparse_attention_rows_scratch_elements, deepseek_sparse_flash_rows_contiguous,
    dispatch_deepseek_compressor_rows, dispatch_deepseek_indexer_rows,
    dispatch_deepseek_sparse_attention_flash_rows, dispatch_deepseek_sparse_attention_rows,
    DeepSeekCompressorRow, DeepSeekIndexerRow, DeepSeekSparseAttentionRow,
    DeepSeekSparseFlashRowsWorkspace,
};
use mlx_native::ops::deepseek_sparse_attention::{
    dispatch_deepseek_sparse_attention_flash_decode,
    dispatch_deepseek_sparse_attention_with_scratch, DeepSeekSparseAttentionParams,
};
use mlx_native::{DType, KernelPipelineOrigin, KernelRegistry, MlxBuffer, MlxDevice};

const H: usize = 64;
const D: usize = 512;
const QROW: usize = H * D;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn buffer<T: bytemuck::Pod>(
    device: &MlxDevice,
    dtype: DType,
    values: &[T],
    shape: Vec<usize>,
) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(std::mem::size_of_val(values).max(4), dtype, shape)
        .expect("alloc");
    buffer.as_mut_slice::<T>().expect("slice")[..values.len()].copy_from_slice(values);
    buffer
}

fn bf16_values(rng: &mut Rng, len: usize, scale: f32) -> Vec<u16> {
    (0..len)
        .map(|_| bf16::from_f32(rng.unit() * scale).to_bits())
        .collect()
}

fn bytes(buffer: &MlxBuffer) -> Vec<u8> {
    buffer.as_slice::<u8>().expect("bytes").to_vec()
}

fn run(device: &MlxDevice, encode: impl FnOnce(&mut mlx_native::CommandEncoder)) {
    let mut encoder = device.command_encoder().expect("encoder");
    encode(&mut encoder);
    encoder.commit_and_wait().expect("completion");
}

fn expected_origin() -> KernelPipelineOrigin {
    match std::env::var("MLX_PRECOMPILED_METALLIB").as_deref() {
        Ok("0") | Ok("false") | Ok("off") => KernelPipelineOrigin::RuntimeSource,
        _ => KernelPipelineOrigin::PrecompiledMetallib,
    }
}

fn assert_origins(registry: &KernelRegistry, names: &[&str]) {
    for name in names {
        let identity = registry.pipeline_identity(name).expect("pipeline identity");
        assert_eq!(identity.origin, expected_origin(), "{name} pipeline origin");
    }
}

// ---------------------------------------------------------------------------
// Sparse attention
// ---------------------------------------------------------------------------

/// One row of a sparse case: its KV, indices and query row.
struct SparseCaseRow {
    kv: MlxBuffer,
    kv_len: usize,
    indices: Vec<i32>,
    q_row: usize,
}

#[derive(Clone, Copy)]
enum Fault {
    None,
    BadIndex,
    NanKv,
    NanQHead,
}

fn sparse_rows(
    device: &MlxDevice,
    rng: &mut Rng,
    shapes: &[(usize, usize, Fault)],
    q_rows: &[usize],
    q: &mut [u16],
) -> Vec<SparseCaseRow> {
    shapes
        .iter()
        .zip(q_rows)
        .map(|(&(kv_len, width, fault), &q_row)| {
            let mut kv = bf16_values(rng, kv_len * D, 0.5);
            let mut indices: Vec<i32> = (0..width).map(|_| rng.below(kv_len) as i32).collect();
            // Sentinels and a duplicate in every row.
            indices[rng.below(width)] = -1;
            if width > 2 {
                indices[1] = indices[0].max(0);
            }
            match fault {
                Fault::None => {}
                Fault::BadIndex => indices[width / 2] = kv_len as i32,
                Fault::NanKv => {
                    let selected = indices.iter().copied().find(|&i| i >= 0).unwrap_or(0);
                    kv[selected as usize * D + 7] = bf16::NAN.to_bits();
                }
                Fault::NanQHead => q[q_row * QROW + 5 * D + 3] = bf16::NAN.to_bits(),
            }
            SparseCaseRow {
                kv: buffer(device, DType::BF16, &kv, vec![kv_len, D]),
                kv_len,
                indices,
                q_row,
            }
        })
        .collect()
}

/// Concatenate the rows' indices with a gap so offsets are non-trivial.
fn packed_indices(device: &MlxDevice, rows: &[SparseCaseRow]) -> (MlxBuffer, Vec<usize>) {
    let mut packed = vec![-7i32; 3];
    let mut offsets = Vec::new();
    for row in rows {
        offsets.push(packed.len());
        packed.extend(&row.indices);
        packed.extend([-7, -7]);
    }
    let len = packed.len();
    (buffer(device, DType::I32, &packed, vec![len]), offsets)
}

fn q_row_view(q: &MlxBuffer, row: usize) -> MlxBuffer {
    q.slice_view((row * QROW * 2) as u64, QROW)
        .with_shape(vec![1, 1, H, D])
        .expect("q view")
}

fn params(kv_len: usize, width: usize) -> DeepSeekSparseAttentionParams {
    DeepSeekSparseAttentionParams {
        batch: 1,
        query_len: 1,
        kv_len: kv_len as u32,
        top_k: width as u32,
        heads: H as u32,
        head_dim: D as u32,
        scale: 1.0 / (D as f32).sqrt(),
    }
}

fn reference_sparse(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    q: &MlxBuffer,
    sinks: &MlxBuffer,
    row: &SparseCaseRow,
    flash: bool,
) -> Vec<u8> {
    let width = row.indices.len();
    let q_view = q_row_view(q, row.q_row);
    let kv = row.kv.with_shape(vec![1, row.kv_len, D]).expect("kv view");
    let indices = buffer(device, DType::I32, &row.indices, vec![1, 1, width]);
    let output = device
        .alloc_buffer(QROW * 2, DType::BF16, vec![1, 1, H, D])
        .expect("output");
    if flash {
        let gathered = device
            .alloc_buffer(width * D * 2, DType::BF16, vec![1, 1, width, D])
            .expect("gathered");
        let mask = device
            .alloc_buffer(width * 2, DType::BF16, vec![1, width])
            .expect("mask");
        let global = buffer(device, DType::U32, &[0u32], vec![1]);
        let heads = buffer(device, DType::U32, &[0u32; H], vec![H]);
        run(device, |encoder| {
            dispatch_deepseek_sparse_attention_flash_decode(
                encoder,
                registry,
                device,
                &q_view,
                &kv,
                sinks,
                &indices,
                &gathered,
                &mask,
                &global,
                &heads,
                &output,
                &params(row.kv_len, width),
            )
            .expect("flash decode")
        });
    } else {
        let scratch = device
            .alloc_buffer(H * width * 4, DType::F32, vec![1, 1, H, width])
            .expect("scratch");
        run(device, |encoder| {
            dispatch_deepseek_sparse_attention_with_scratch(
                encoder,
                registry,
                device,
                &q_view,
                &kv,
                sinks,
                &indices,
                &scratch,
                &output,
                &params(row.kv_len, width),
            )
            .expect("sparse")
        });
    }
    bytes(&output)
}

fn check_sparse_case(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    seed: u64,
    shapes: &[(usize, usize, Fault)],
    q_rows: &[usize],
    flash: bool,
    nan_sink: bool,
) {
    let mut rng = Rng(seed);
    let q_count = q_rows.iter().max().unwrap() + 1;
    let mut q_values = bf16_values(&mut rng, q_count * QROW, 1.0);
    let rows = sparse_rows(device, &mut rng, shapes, q_rows, &mut q_values);
    let q = buffer(device, DType::BF16, &q_values, vec![q_count, H, D]);
    let mut sink_values: Vec<f32> = (0..H).map(|_| rng.unit()).collect();
    if nan_sink {
        sink_values[9] = f32::NAN;
    }
    let sinks = buffer(device, DType::F32, &sink_values, vec![H]);
    let (indices, offsets) = packed_indices(device, &rows);
    let specs: Vec<DeepSeekSparseAttentionRow<'_>> = rows
        .iter()
        .zip(&offsets)
        .map(|(row, &index_offset)| DeepSeekSparseAttentionRow {
            kv: &row.kv,
            kv_len: row.kv_len,
            width: row.indices.len(),
            index_offset,
            q_row: row.q_row,
        })
        .collect();
    let n = rows.len();
    let max_width = specs.iter().map(|row| row.width).max().unwrap();
    let total_width: usize = specs.iter().map(|row| row.width).sum();
    let guard = vec![0x7f7fu16; q_count * QROW];
    let mut outputs = Vec::new();
    for _repeat in 0..2 {
        let output = buffer(device, DType::BF16, &guard, vec![q_count, H, D]);
        if flash {
            let gathered = device
                .alloc_buffer(total_width * D * 2, DType::BF16, vec![total_width * D])
                .unwrap();
            let mask = device
                .alloc_buffer(total_width * 2, DType::BF16, vec![total_width])
                .unwrap();
            let global = buffer(device, DType::U32, &vec![0u32; n], vec![n]);
            let heads = buffer(device, DType::U32, &vec![0u32; n * H], vec![n * H]);
            let q_scratch = device
                .alloc_buffer(n * QROW * 2, DType::BF16, vec![n * QROW])
                .unwrap();
            let out_scratch = device
                .alloc_buffer(n * QROW * 2, DType::BF16, vec![n * QROW])
                .unwrap();
            let workspace = DeepSeekSparseFlashRowsWorkspace {
                gathered_kv: &gathered,
                mask: &mask,
                invalid_global: &global,
                invalid_heads: &heads,
                q_scratch: &q_scratch,
                output_scratch: &out_scratch,
            };
            run(device, |encoder| {
                dispatch_deepseek_sparse_attention_flash_rows(
                    encoder,
                    registry,
                    device,
                    &q,
                    &sinks,
                    &indices,
                    &workspace,
                    &output,
                    &specs,
                    1.0 / (D as f32).sqrt(),
                )
                .expect("flash rows")
            });
        } else {
            let elements = deepseek_sparse_attention_rows_scratch_elements(n, max_width);
            let scratch = device
                .alloc_buffer(elements * 4, DType::F32, vec![elements])
                .unwrap();
            run(device, |encoder| {
                dispatch_deepseek_sparse_attention_rows(
                    encoder,
                    registry,
                    device,
                    &q,
                    &sinks,
                    &indices,
                    &scratch,
                    &output,
                    &specs,
                    1.0 / (D as f32).sqrt(),
                )
                .expect("sparse rows")
            });
        }
        outputs.push(bytes(&output));
    }
    assert_eq!(outputs[0], outputs[1], "rows dispatch is not deterministic");
    let output = &outputs[0];
    for (entry, row) in rows.iter().enumerate() {
        let expected = reference_sparse(device, registry, &q, &sinks, row, flash);
        let got = &output[row.q_row * QROW * 2..(row.q_row + 1) * QROW * 2];
        assert!(
            got == expected.as_slice(),
            "seed {seed} flash={flash} entry {entry} (kv {}, width {}, q_row {}) differs from the one-row kernel",
            row.kv_len,
            row.indices.len(),
            row.q_row
        );
    }
    // Query rows no entry names stay untouched.
    for q_row in 0..q_count {
        if !q_rows.contains(&q_row) {
            assert!(output[q_row * QROW * 2..(q_row + 1) * QROW * 2]
                .chunks_exact(2)
                .all(|b| b == [0x7f, 0x7f]));
        }
    }
}

#[test]
fn sparse_attention_rows_equal_one_row_kernel() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    use Fault::*;
    let shapes = [
        (128, 128, None),
        (1280 + 256, 143, None),
        (1280 + 256, 192, BadIndex),
        (128, 117, NanQHead),
        (1280 + 2048, 640, None),
        (2048, 511, NanKv),
        (512, 512, None),
        (2047, 639, None),
    ];
    for b in 1..=8 {
        let q_rows: Vec<usize> = (0..b).collect();
        check_sparse_case(
            &device,
            &mut registry,
            10 + b as u64,
            &shapes[..b],
            &q_rows,
            false,
            false,
        );
    }
    // Row permutation, a subset of query rows, and a NaN sink.
    check_sparse_case(
        &device,
        &mut registry,
        31,
        &shapes[..5],
        &[4, 0, 6, 2, 1],
        false,
        false,
    );
    check_sparse_case(
        &device,
        &mut registry,
        32,
        &shapes[..3],
        &[1, 2, 0],
        false,
        true,
    );
    assert_origins(
        &registry,
        &[
            "deepseek_sparse_attention_score_rows_bf16",
            "deepseek_sparse_attention_reduce_rows_bf16",
        ],
    );
}

#[test]
fn sparse_flash_rows_equal_one_row_flash_decode() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    use Fault::*;
    let one_width = [(1280 + 2048, 640, None); 8];
    for b in 1..=8 {
        let q_rows: Vec<usize> = (0..b).collect();
        check_sparse_case(
            &device,
            &mut registry,
            40 + b as u64,
            &one_width[..b],
            &q_rows,
            true,
            false,
        );
    }
    // Several widths: runs that are contiguous in row order, and interleaved
    // widths that need the permuted flash batches; fail-closed rows.
    let mixed = [
        (1280 + 2048, 640, None),
        (1280 + 2048, 640, BadIndex),
        (1280 + 511, 639, None),
        (1280 + 512, 640, NanQHead),
        (1280 + 400, 528, None),
        (1280 + 511, 639, NanKv),
        (1280 + 300, 428, None),
        (1280 + 2048, 640, None),
    ];
    for b in 2..=8 {
        let q_rows: Vec<usize> = (0..b).collect();
        check_sparse_case(
            &device,
            &mut registry,
            50 + b as u64,
            &mixed[..b],
            &q_rows,
            true,
            false,
        );
    }
    check_sparse_case(
        &device,
        &mut registry,
        61,
        &mixed[..4],
        &[3, 1, 0, 2],
        true,
        true,
    );
    check_sparse_case(
        &device,
        &mut registry,
        62,
        &[
            (1280 + 2048, 640, None),
            (1280 + 2048, 640, None),
            (1280 + 511, 639, None),
        ],
        &[0, 1, 2],
        true,
        false,
    );
    assert_origins(
        &registry,
        &[
            "deepseek_sparse_attention_gather_rows_bf16",
            "deepseek_sparse_attention_permute_q_rows_bf16",
            "deepseek_sparse_attention_sanitize_rows_bf16",
        ],
    );
}

#[test]
fn flash_rows_contiguity_rule() {
    let device = MlxDevice::new().expect("device");
    let kv = device
        .alloc_buffer(640 * D * 2, DType::BF16, vec![640, D])
        .unwrap();
    let row = |width, q_row| DeepSeekSparseAttentionRow {
        kv: &kv,
        kv_len: 640,
        width,
        index_offset: 0,
        q_row,
    };
    assert!(deepseek_sparse_flash_rows_contiguous(&[
        row(640, 0),
        row(640, 1),
        row(639, 2)
    ]));
    assert!(!deepseek_sparse_flash_rows_contiguous(&[
        row(640, 0),
        row(639, 1),
        row(640, 2)
    ]));
    assert!(!deepseek_sparse_flash_rows_contiguous(&[
        row(640, 1),
        row(640, 0)
    ]));
}

// ---------------------------------------------------------------------------
// Indexer
// ---------------------------------------------------------------------------

const IDX_ROW: usize = 64 * 128;
const STRIDE: usize = 1280 + 512 + 9;
const COLUMN: usize = 1280 + 4;

struct IndexerCaseRow {
    kv: MlxBuffer,
    kv_len: usize,
    start_pos: u32,
    offset: i32,
    q_row: usize,
}

fn reference_indexer(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    q: &MlxBuffer,
    weights: &MlxBuffer,
    row: &IndexerCaseRow,
) -> Vec<u8> {
    let params = DeepSeekIndexerParams {
        batch: 1,
        query_len: 1,
        kv_len: row.kv_len as u32,
        start_pos: row.start_pos,
        ratio: 4,
        heads: 64,
        head_dim: 128,
        top_k: 512,
        offset: row.offset,
    };
    let plan = deepseek_indexer_query_tile_plan(&params, STRIDE, COLUMN).expect("plan");
    let q_view = q
        .slice_view((row.q_row * IDX_ROW * 2) as u64, IDX_ROW)
        .with_shape(vec![1, 1, 64, 128])
        .unwrap();
    let w_view = weights
        .slice_view((row.q_row * 64 * 4) as u64, 64)
        .with_shape(vec![1, 1, 64])
        .unwrap();
    let kv = row.kv.with_shape(vec![1, row.kv_len, 128]).unwrap();
    let scores = device
        .alloc_buffer(
            plan.workspace_rows * row.kv_len * 4,
            DType::F32,
            vec![1, plan.workspace_rows, row.kv_len],
        )
        .unwrap();
    let topk = device
        .alloc_buffer(
            plan.workspace_rows * plan.topk_row_elements * 4,
            DType::I32,
            vec![1, plan.workspace_rows, plan.topk_row_elements],
        )
        .unwrap();
    let output = buffer(device, DType::I32, &vec![77i32; STRIDE], vec![1, 1, STRIDE]);
    run(device, |encoder| {
        dispatch_deepseek_indexer_into_mma_tiled(
            encoder, registry, device, &q_view, &kv, &w_view, &scores, &topk, &output, STRIDE,
            COLUMN, &params,
        )
        .expect("indexer")
    });
    bytes(&output)
}

fn check_indexer_case(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    seed: u64,
    shapes: &[(usize, u32, bool)],
    q_rows: &[usize],
) {
    let mut rng = Rng(seed);
    let q_count = q_rows.iter().max().unwrap() + 1;
    let q = buffer(
        device,
        DType::BF16,
        &bf16_values(&mut rng, q_count * IDX_ROW, 1.0),
        vec![q_count, 64, 128],
    );
    let mut weight_values: Vec<f32> = (0..q_count * 64).map(|_| rng.unit()).collect();
    let rows: Vec<IndexerCaseRow> = shapes
        .iter()
        .zip(q_rows)
        .map(|(&(kv_len, start_pos, nan_weight), &q_row)| {
            let mut kv = bf16_values(&mut rng, kv_len * 128, 0.5);
            // Exact score ties: duplicate key rows.
            for dup in 0..kv_len / 7 {
                let (src, dst) = (rng.below(kv_len), rng.below(kv_len));
                let row: Vec<u16> = kv[src * 128..(src + 1) * 128].to_vec();
                kv[dst * 128..(dst + 1) * 128].copy_from_slice(&row);
                let _ = dup;
            }
            if nan_weight {
                weight_values[q_row * 64 + 3] = f32::NAN;
            }
            IndexerCaseRow {
                kv: buffer(device, DType::BF16, &kv, vec![kv_len, 128]),
                kv_len,
                start_pos,
                offset: 1280 + q_row as i32,
                q_row,
            }
        })
        .collect();
    let weights = buffer(device, DType::F32, &weight_values, vec![q_count, 64]);
    let specs: Vec<DeepSeekIndexerRow<'_>> = rows
        .iter()
        .map(|row| DeepSeekIndexerRow {
            kv: &row.kv,
            kv_len: row.kv_len,
            start_pos: row.start_pos,
            offset: row.offset,
            q_row: row.q_row,
            output_offset: row.q_row * STRIDE + COLUMN,
        })
        .collect();
    let max_kv = specs.iter().map(|row| row.kv_len).max().unwrap();
    let (score_elements, topk_elements) =
        deepseek_indexer_rows_workspace_elements(rows.len(), max_kv);
    let mut outputs = Vec::new();
    for _repeat in 0..2 {
        let scores = device
            .alloc_buffer(score_elements * 4, DType::F32, vec![score_elements])
            .unwrap();
        let topk = device
            .alloc_buffer(topk_elements * 4, DType::I32, vec![topk_elements])
            .unwrap();
        let output = buffer(
            device,
            DType::I32,
            &vec![77i32; q_count * STRIDE],
            vec![q_count * STRIDE],
        );
        run(device, |encoder| {
            dispatch_deepseek_indexer_rows(
                encoder, registry, device, &q, &weights, &scores, &topk, &output, &specs,
            )
            .expect("indexer rows")
        });
        outputs.push(bytes(&output));
    }
    assert_eq!(outputs[0], outputs[1], "indexer rows are not deterministic");
    for (entry, row) in rows.iter().enumerate() {
        let expected = reference_indexer(device, registry, &q, &weights, row);
        let got = &outputs[0][row.q_row * STRIDE * 4..(row.q_row + 1) * STRIDE * 4];
        assert!(
            got == expected.as_slice(),
            "seed {seed} entry {entry} (kv {}, pos {}) differs from the one-row indexer",
            row.kv_len,
            row.start_pos
        );
    }
}

#[test]
fn indexer_rows_equal_one_row_tiled_indexer() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    // (kv_len, position, NaN head weight): 1 and 2 top-k blocks, valid < kv
    // (masked slots), fewer than 512 valid keys, a fully non-finite row.
    let shapes = [
        (512, 2047, false),
        (512, 2043, false),
        (2048, 8191, false),
        (189, 757, false),
        (1026, 4103, false),
        (583, 2333, true),
        (2048, 8190, false),
        (2, 7, false),
    ];
    for b in 1..=8 {
        let q_rows: Vec<usize> = (0..b).collect();
        check_indexer_case(&device, &mut registry, 70 + b as u64, &shapes[..b], &q_rows);
    }
    check_indexer_case(&device, &mut registry, 81, &shapes[..5], &[3, 0, 4, 1, 2]);
    check_indexer_case(&device, &mut registry, 82, &shapes[2..4], &[5, 1]);
    assert_origins(
        &registry,
        &[
            "deepseek_indexer_score_mma_rows_bf16",
            "deepseek_indexer_topk_block_rows_i32",
            "deepseek_indexer_topk_merge_rows_i32",
            "deepseek_indexer_topk_finalize_rows_i32",
        ],
    );
}

// ---------------------------------------------------------------------------
// Compressor
// ---------------------------------------------------------------------------

fn check_compressor(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    ratio: u32,
    dim: usize,
    seed: u64,
) {
    let mut rng = Rng(seed);
    let coff = if ratio == 4 { 2 } else { 1 };
    let projected = coff * dim;
    let state = coff * ratio as usize * projected;
    let ape_values: Vec<f32> = (0..ratio as usize * projected)
        .map(|_| rng.unit() * 0.1)
        .collect();
    let ape = buffer(
        device,
        DType::F32,
        &ape_values,
        vec![ratio as usize, projected],
    );
    let norm_values: Vec<f32> = (0..dim).map(|_| 1.0 + 0.1 * rng.unit()).collect();
    let norm = buffer(device, DType::F32, &norm_values, vec![dim]);
    for b in 1..=8usize {
        // Positions in different phases of the compression block.
        let starts: Vec<u32> = (0..b)
            .map(|row| ratio * (3 + row as u32) + (row as u32 * 3 + 1) % ratio)
            .collect();
        let initial: Vec<(Vec<f32>, Vec<f32>)> = (0..b)
            .map(|_| {
                (
                    (0..state).map(|_| rng.unit()).collect(),
                    (0..state).map(|_| rng.unit()).collect(),
                )
            })
            .collect();
        let make_states = || -> Vec<(MlxBuffer, MlxBuffer)> {
            initial
                .iter()
                .map(|(kv, score)| {
                    let shape = vec![1, coff * ratio as usize, projected];
                    (
                        buffer(device, DType::F32, kv, shape.clone()),
                        buffer(device, DType::F32, score, shape),
                    )
                })
                .collect()
        };
        let rows_states = make_states();
        let reference_states = make_states();
        let repeat_states = make_states();
        let steps = (2 * ratio as usize).min(10);
        let ratio_usize = ratio as usize;
        for step in 0..steps {
            let mut inputs: Vec<f32> = (0..b * projected).map(|_| rng.unit()).collect();
            let mut scores: Vec<f32> = (0..b * projected).map(|_| rng.unit()).collect();
            if step == 1 {
                inputs[3] = f32::NAN;
                scores[(b - 1) * projected + 2] = f32::INFINITY;
            }
            let kv = buffer(device, DType::F32, &inputs, vec![b, projected]);
            let score = buffer(device, DType::F32, &scores, vec![b, projected]);
            let positions: Vec<u32> = starts.iter().map(|&s| s + step as u32).collect();
            let mut outputs = Vec::new();
            for states in [&rows_states, &repeat_states] {
                let output = buffer(
                    device,
                    DType::BF16,
                    &vec![0x7f7fu16; b * dim],
                    vec![b, 1, dim],
                );
                let specs: Vec<DeepSeekCompressorRow<'_>> = states
                    .iter()
                    .zip(&positions)
                    .map(
                        |((kv_state, score_state), &start_pos)| DeepSeekCompressorRow {
                            kv_state,
                            score_state,
                            start_pos,
                        },
                    )
                    .collect();
                run(device, |encoder| {
                    dispatch_deepseek_compressor_rows(
                        encoder, registry, device, &kv, &score, &ape, &norm, &output, &specs,
                        ratio, dim as u32, 1e-6,
                    )
                    .expect("compressor rows")
                });
                outputs.push(bytes(&output));
            }
            assert_eq!(
                outputs[0], outputs[1],
                "compressor rows are not deterministic"
            );
            for row in 0..b {
                let kv_row = kv
                    .slice_view((row * projected * 4) as u64, projected)
                    .with_shape(vec![1, 1, projected])
                    .unwrap();
                let score_row = score
                    .slice_view((row * projected * 4) as u64, projected)
                    .with_shape(vec![1, 1, projected])
                    .unwrap();
                let output = device
                    .alloc_buffer(dim * 2, DType::BF16, vec![1, 1, dim])
                    .unwrap();
                let (kv_state, score_state) = &reference_states[row];
                run(device, |encoder| {
                    dispatch_deepseek_compressor(
                        encoder,
                        registry,
                        device,
                        &kv_row,
                        &score_row,
                        &ape,
                        &norm,
                        kv_state,
                        score_state,
                        &output,
                        &output,
                        &DeepSeekCompressorParams {
                            batch: 1,
                            seq_len: 1,
                            start_pos: positions[row],
                            ratio,
                            head_dim: dim as u32,
                            cache_len: positions[row] / ratio + 2,
                            epsilon: 1e-6,
                            write_cache: 0,
                        },
                    )
                    .expect("compressor")
                });
                let emit = (positions[row] as usize + 1) % ratio_usize == 0;
                assert_eq!(
                    &outputs[0][row * dim * 2..(row + 1) * dim * 2],
                    bytes(&output).as_slice(),
                    "ratio {ratio} dim {dim} b {b} step {step} row {row} (emit {emit}) output"
                );
                assert_eq!(
                    bytes(&rows_states[row].0),
                    bytes(kv_state),
                    "ratio {ratio} dim {dim} b {b} step {step} row {row} kv state"
                );
                assert_eq!(
                    bytes(&rows_states[row].1),
                    bytes(score_state),
                    "ratio {ratio} dim {dim} b {b} step {step} row {row} score state"
                );
            }
        }
    }
}

#[test]
fn compressor_rows_equal_one_row_compressor() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    check_compressor(&device, &mut registry, 4, 512, 91);
    check_compressor(&device, &mut registry, 4, 128, 92);
    check_compressor(&device, &mut registry, 128, 512, 93);
    assert_origins(&registry, &["deepseek_compressor_rows_bf16"]);
}

#[test]
fn rows_dispatches_reject_invalid_tables_before_encoding() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let kv = device
        .alloc_buffer(128 * D * 2, DType::BF16, vec![128, D])
        .unwrap();
    let q = device
        .alloc_buffer(9 * QROW * 2, DType::BF16, vec![9, H, D])
        .unwrap();
    let sinks = device.alloc_buffer(H * 4, DType::F32, vec![H]).unwrap();
    let indices = device
        .alloc_buffer(9 * 128 * 4, DType::I32, vec![9 * 128])
        .unwrap();
    let scratch = device
        .alloc_buffer(9 * H * 128 * 4, DType::F32, vec![9 * H * 128])
        .unwrap();
    let row = |q_row| DeepSeekSparseAttentionRow {
        kv: &kv,
        kv_len: 128,
        width: 128,
        index_offset: 0,
        q_row,
    };
    let mut encoder = device.command_encoder().unwrap();
    let nine: Vec<_> = (0..9).map(row).collect();
    for (rows, what) in [
        (&nine[..], "nine rows"),
        (&nine[..0], "zero rows"),
        (&[row(0), row(0)][..], "duplicate q_row"),
    ] {
        assert!(
            dispatch_deepseek_sparse_attention_rows(
                &mut encoder,
                &mut registry,
                &device,
                &q,
                &sinks,
                &indices,
                &scratch,
                &q,
                rows,
                0.1,
            )
            .is_err(),
            "{what} must be rejected"
        );
    }
    let short = DeepSeekSparseAttentionRow {
        kv_len: 129,
        ..row(0)
    };
    assert!(dispatch_deepseek_sparse_attention_rows(
        &mut encoder,
        &mut registry,
        &device,
        &q,
        &sinks,
        &indices,
        &scratch,
        &q,
        &[short],
        0.1,
    )
    .is_err());
    let state = device
        .alloc_buffer(8 * 1024 * 4, DType::F32, vec![8, 1024])
        .unwrap();
    let f32s = device
        .alloc_buffer(1024 * 4, DType::F32, vec![1024])
        .unwrap();
    let out = device
        .alloc_buffer(512 * 2, DType::BF16, vec![1, 1, 512])
        .unwrap();
    let position_zero = DeepSeekCompressorRow {
        kv_state: &state,
        score_state: &f32s,
        start_pos: 0,
    };
    assert!(dispatch_deepseek_compressor_rows(
        &mut encoder,
        &mut registry,
        &device,
        &f32s,
        &f32s,
        &state,
        &f32s,
        &out,
        &[position_zero],
        4,
        512,
        1e-6,
    )
    .is_err());
    let shared = DeepSeekCompressorRow {
        kv_state: &state,
        score_state: &state,
        start_pos: 5,
    };
    assert!(dispatch_deepseek_compressor_rows(
        &mut encoder,
        &mut registry,
        &device,
        &f32s,
        &f32s,
        &state,
        &f32s,
        &out,
        &[shared],
        4,
        512,
        1e-6,
    )
    .is_err());
    encoder.commit_and_wait().unwrap();
    assert_eq!(deepseek_rows_kernel_names().len(), 10);
}
