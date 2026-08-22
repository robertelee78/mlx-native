#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use mlx_native::ops::hadamard_quantize_kv::{
    dispatch_hadamard_quantize_kv_hb, dispatch_hadamard_quantize_kv_hb_seq,
};
use mlx_native::{CapturedNode, DType, DispatchKind, KernelRegistry, MlxBuffer, MlxDevice};

const NKV: u32 = 2;
const CAP: u32 = 11;
const SRC_PREFIX_ELEMENTS: usize = 13;
const SRC_SUFFIX_ELEMENTS: usize = 7;
const PACKED_PREFIX_BYTES: usize = 17;
const PACKED_SUFFIX_BYTES: usize = 19;
const NORMS_PREFIX_ELEMENTS: usize = 5;
const NORMS_SUFFIX_ELEMENTS: usize = 3;

struct SourceView {
    _parent: MlxBuffer,
    view: MlxBuffer,
}

struct OutputViews {
    packed_parent: MlxBuffer,
    packed: MlxBuffer,
    norms_parent: MlxBuffer,
    norms: MlxBuffer,
}

fn source_view(device: &MlxDevice, total_tokens: u32, head_dim: u32) -> SourceView {
    let logical_elements = total_tokens as usize * NKV as usize * head_dim as usize;
    let parent_elements = SRC_PREFIX_ELEMENTS + logical_elements + SRC_SUFFIX_ELEMENTS;
    let mut parent = device
        .alloc_buffer(parent_elements * 4, DType::F32, vec![parent_elements])
        .expect("source parent");
    let values = parent.as_mut_slice::<f32>().expect("source parent values");
    values.fill(f32::from_bits(0x7fc1_2345));
    for i in 0..logical_elements {
        let x = i as f32 * 0.013_671 + head_dim as f32 * 0.000_031;
        values[SRC_PREFIX_ELEMENTS + i] = x.sin() * 1.75 + x.cos() * 0.25;
    }
    let view = parent.slice_view((SRC_PREFIX_ELEMENTS * 4) as u64, logical_elements);
    SourceView {
        _parent: parent,
        view,
    }
}

fn output_views(device: &MlxDevice, head_dim: u32) -> OutputViews {
    let packed_elements = NKV as usize * CAP as usize * head_dim as usize;
    let norms_per_pos = (head_dim / 256).max(1) as usize;
    let norms_elements = NKV as usize * CAP as usize * norms_per_pos;

    let packed_parent_elements = PACKED_PREFIX_BYTES + packed_elements + PACKED_SUFFIX_BYTES;
    let mut packed_parent = device
        .alloc_buffer(
            packed_parent_elements,
            DType::U8,
            vec![packed_parent_elements],
        )
        .expect("packed parent");
    packed_parent
        .as_mut_slice::<u8>()
        .expect("packed parent values")
        .fill(0xa5);
    let packed = packed_parent.slice_view(PACKED_PREFIX_BYTES as u64, packed_elements);

    let norms_parent_elements = NORMS_PREFIX_ELEMENTS + norms_elements + NORMS_SUFFIX_ELEMENTS;
    let mut norms_parent = device
        .alloc_buffer(
            norms_parent_elements * 4,
            DType::F32,
            vec![norms_parent_elements],
        )
        .expect("norms parent");
    norms_parent
        .as_mut_slice::<f32>()
        .expect("norms parent values")
        .fill(f32::from_bits(0x7fc5_4321));
    let norms = norms_parent.slice_view((NORMS_PREFIX_ELEMENTS * 4) as u64, norms_elements);

    OutputViews {
        packed_parent,
        packed,
        norms_parent,
        norms,
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_seq(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    src: &MlxBuffer,
    head_dim: u32,
    codebook_bits: u32,
    write_pos_start: u32,
    n_tokens: u32,
    src_tok_offset: u32,
    is_sliding: bool,
) -> OutputViews {
    let outputs = output_views(device, head_dim);
    let mut encoder = device.command_encoder().expect("sequence encoder");
    dispatch_hadamard_quantize_kv_hb_seq(
        &mut encoder,
        registry,
        device.metal_device(),
        src,
        &outputs.packed,
        &outputs.norms,
        NKV,
        head_dim,
        CAP,
        write_pos_start,
        n_tokens,
        src_tok_offset,
        is_sliding,
        1.0,
        codebook_bits,
    )
    .expect("batched sequence dispatch");
    encoder.commit_and_wait().expect("batched sequence commit");
    outputs
}

#[allow(clippy::too_many_arguments)]
fn encode_scalar_reference(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    src: &MlxBuffer,
    head_dim: u32,
    codebook_bits: u32,
    write_pos_start: u32,
    n_tokens: u32,
    src_tok_offset: u32,
    is_sliding: bool,
) -> OutputViews {
    let outputs = output_views(device, head_dim);
    let row_elements = NKV as usize * head_dim as usize;
    let skipped = if is_sliding {
        n_tokens.saturating_sub(CAP)
    } else {
        0
    };
    let mut encoder = device.command_encoder().expect("scalar encoder");
    for token in skipped..n_tokens {
        let source_token = src_tok_offset + token;
        let row = src.slice_view(source_token as u64 * row_elements as u64 * 4, row_elements);
        let physical_or_raw_pos = if is_sliding {
            ((write_pos_start as u64 + token as u64) % CAP as u64) as u32
        } else {
            write_pos_start + token
        };
        dispatch_hadamard_quantize_kv_hb(
            &mut encoder,
            registry,
            device.metal_device(),
            &row,
            &outputs.packed,
            &outputs.norms,
            NKV,
            head_dim,
            CAP,
            physical_or_raw_pos,
            false,
            1.0,
            codebook_bits,
        )
        .expect("scalar reference dispatch");
    }
    encoder.commit_and_wait().expect("scalar reference commit");
    outputs
}

fn assert_output_parity(label: &str, batched: &OutputViews, scalar: &OutputViews) {
    assert_eq!(
        batched
            .packed_parent
            .as_slice::<u8>()
            .expect("batched packed bytes"),
        scalar
            .packed_parent
            .as_slice::<u8>()
            .expect("scalar packed bytes"),
        "{label}: packed output or surrounding sentinel differs"
    );
    let batched_norms = batched
        .norms_parent
        .as_slice::<f32>()
        .expect("batched norms");
    let scalar_norms = scalar.norms_parent.as_slice::<f32>().expect("scalar norms");
    assert_eq!(batched_norms.len(), scalar_norms.len());
    for (index, (&actual, &expected)) in batched_norms.iter().zip(scalar_norms.iter()).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{label}: norm bit pattern differs at parent element {index}"
        );
    }
}

#[test]
fn contiguous_sequence_is_bit_exact_for_global_ring_and_source_views() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();

    for &head_dim in &[256_u32, 512] {
        let source = source_view(&device, 9, head_dim);
        for &codebook_bits in &[5_u32, 6, 8] {
            for &(label, write_pos, n_tokens, is_sliding) in &[
                ("single", 3_u32, 1_u32, false),
                ("global", 2, 5, false),
                ("ring-wrap", CAP - 2, 5, true),
            ] {
                let batched = encode_seq(
                    &device,
                    &mut registry,
                    &source.view,
                    head_dim,
                    codebook_bits,
                    write_pos,
                    n_tokens,
                    2,
                    is_sliding,
                );
                let scalar = encode_scalar_reference(
                    &device,
                    &mut registry,
                    &source.view,
                    head_dim,
                    codebook_bits,
                    write_pos,
                    n_tokens,
                    2,
                    is_sliding,
                );
                assert_output_parity(
                    &format!("{label}-d{head_dim}-cb{codebook_bits}"),
                    &batched,
                    &scalar,
                );
            }
        }
    }
}

#[test]
fn oversized_sliding_sequence_keeps_the_exact_last_capacity_rows() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let n_tokens = CAP + 4;
    let source = source_view(&device, n_tokens + 3, 256);
    let batched = encode_seq(
        &device,
        &mut registry,
        &source.view,
        256,
        8,
        CAP - 3,
        n_tokens,
        2,
        true,
    );
    let scalar = encode_scalar_reference(
        &device,
        &mut registry,
        &source.view,
        256,
        8,
        CAP - 3,
        n_tokens,
        2,
        true,
    );
    assert_output_parity("oversized-ring", &batched, &scalar);
}

fn simple_buffers(device: &MlxDevice, n_tokens: u32) -> (MlxBuffer, MlxBuffer, MlxBuffer) {
    let src_elements = n_tokens as usize * NKV as usize * 256;
    let packed_elements = NKV as usize * CAP as usize * 256;
    let norms_elements = NKV as usize * CAP as usize;
    (
        device
            .alloc_buffer(src_elements * 4, DType::F32, vec![src_elements])
            .expect("simple src"),
        device
            .alloc_buffer(packed_elements, DType::U8, vec![packed_elements])
            .expect("simple packed"),
        device
            .alloc_buffer(norms_elements * 4, DType::F32, vec![norms_elements])
            .expect("simple norms"),
    )
}

#[test]
fn capture_records_one_tracked_dispatch_with_token_grid() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let n_tokens = 7;
    let (src, packed, norms) = simple_buffers(&device, n_tokens);
    let mut encoder = device.command_encoder().expect("capture encoder");
    encoder.start_capture();
    dispatch_hadamard_quantize_kv_hb_seq(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &src,
        &packed,
        &norms,
        NKV,
        256,
        CAP,
        1,
        n_tokens,
        0,
        false,
        1.0,
        8,
    )
    .expect("capture sequence dispatch");
    let graph = encoder.take_capture().expect("captured graph");
    assert_eq!(graph.len(), 1, "sequence encoder must record one dispatch");
    let CapturedNode::Dispatch {
        pipeline,
        threads_per_grid,
        threads_per_threadgroup,
        dispatch_kind,
        reads,
        writes,
        ..
    } = &graph[0]
    else {
        panic!("expected captured sequence dispatch");
    };
    assert_eq!(pipeline.label(), "hadamard_quantize_kv_hb_d256");
    assert!(matches!(dispatch_kind, DispatchKind::ThreadGroups));
    assert_eq!(threads_per_grid.width, NKV as u64);
    assert_eq!(threads_per_grid.height, n_tokens as u64);
    assert_eq!(threads_per_grid.depth, 1);
    assert_eq!(threads_per_threadgroup.width, 32);
    assert_eq!(reads.len(), 1);
    assert_eq!(writes.len(), 2);
}

#[test]
fn oversized_ring_capture_records_one_capacity_high_dispatch() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let n_tokens = CAP + 4;
    let (src, packed, norms) = simple_buffers(&device, n_tokens);
    let mut encoder = device.command_encoder().expect("capture encoder");
    encoder.start_capture();
    dispatch_hadamard_quantize_kv_hb_seq(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &src,
        &packed,
        &norms,
        NKV,
        256,
        CAP,
        CAP - 3,
        n_tokens,
        0,
        true,
        1.0,
        8,
    )
    .expect("capture oversized ring dispatch");
    let graph = encoder.take_capture().expect("captured graph");
    assert_eq!(graph.len(), 1, "oversized ring must record one dispatch");
    let CapturedNode::Dispatch {
        threads_per_grid, ..
    } = &graph[0]
    else {
        panic!("expected captured oversized-ring dispatch");
    };
    assert_eq!(threads_per_grid.width, NKV as u64);
    assert_eq!(threads_per_grid.height, CAP as u64);
    assert_eq!(threads_per_grid.depth, 1);
}

fn assert_rejected_without_dispatch(
    device: &MlxDevice,
    call: impl FnOnce(&mut mlx_native::CommandEncoder) -> mlx_native::Result<()>,
) -> String {
    let mut encoder = device.command_encoder().expect("rejection encoder");
    encoder.start_capture();
    let error = call(&mut encoder).expect_err("request must fail preflight");
    let graph = encoder.take_capture().expect("rejection capture");
    assert!(graph.is_empty(), "failed preflight must not encode work");
    error.to_string()
}

#[test]
fn validation_is_fail_closed_for_ranges_dtypes_and_aliases() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let (src, packed, norms) = simple_buffers(&device, 2);

    let global_error = assert_rejected_without_dispatch(&device, |encoder| {
        dispatch_hadamard_quantize_kv_hb_seq(
            encoder,
            &mut registry,
            device.metal_device(),
            &src,
            &packed,
            &norms,
            NKV,
            256,
            CAP,
            CAP - 1,
            2,
            0,
            false,
            1.0,
            8,
        )
    });
    assert!(global_error.contains("global cache range"));

    let required_packed_elements = packed.element_count();
    let short_packed_storage = device
        .alloc_buffer(
            required_packed_elements - 1,
            DType::U8,
            vec![required_packed_elements - 1],
        )
        .expect("short packed backing");
    let short_packed = MlxBuffer::from_raw(
        short_packed_storage.metal_buffer().clone(),
        DType::U8,
        vec![required_packed_elements],
    );
    let size_error = assert_rejected_without_dispatch(&device, |encoder| {
        dispatch_hadamard_quantize_kv_hb_seq(
            encoder,
            &mut registry,
            device.metal_device(),
            &src,
            &short_packed,
            &norms,
            NKV,
            256,
            CAP,
            0,
            2,
            0,
            false,
            1.0,
            8,
        )
    });
    assert!(size_error.contains("packed logical buffer"));

    let wrong_src = device
        .alloc_buffer(src.data_byte_len(), DType::U8, vec![src.data_byte_len()])
        .expect("wrong dtype source");
    let dtype_error = assert_rejected_without_dispatch(&device, |encoder| {
        dispatch_hadamard_quantize_kv_hb_seq(
            encoder,
            &mut registry,
            device.metal_device(),
            &wrong_src,
            &packed,
            &norms,
            NKV,
            256,
            CAP,
            0,
            2,
            0,
            false,
            1.0,
            8,
        )
    });
    assert!(dtype_error.contains("src dtype"));

    let tiny_shape_src = MlxBuffer::from_raw(src.metal_buffer().clone(), DType::F32, vec![1]);
    let src_shape_error = assert_rejected_without_dispatch(&device, |encoder| {
        dispatch_hadamard_quantize_kv_hb_seq(
            encoder,
            &mut registry,
            device.metal_device(),
            &tiny_shape_src,
            &packed,
            &norms,
            NKV,
            256,
            CAP,
            0,
            2,
            0,
            false,
            1.0,
            8,
        )
    });
    assert!(src_shape_error.contains("src logical tensor"));

    let tiny_shape_packed = MlxBuffer::from_raw(packed.metal_buffer().clone(), DType::U8, vec![1]);
    let packed_shape_error = assert_rejected_without_dispatch(&device, |encoder| {
        dispatch_hadamard_quantize_kv_hb_seq(
            encoder,
            &mut registry,
            device.metal_device(),
            &src,
            &tiny_shape_packed,
            &norms,
            NKV,
            256,
            CAP,
            0,
            2,
            0,
            false,
            1.0,
            8,
        )
    });
    assert!(packed_shape_error.contains("packed logical tensor"));

    let tiny_shape_norms = MlxBuffer::from_raw(norms.metal_buffer().clone(), DType::F32, vec![1]);
    let norms_shape_error = assert_rejected_without_dispatch(&device, |encoder| {
        dispatch_hadamard_quantize_kv_hb_seq(
            encoder,
            &mut registry,
            device.metal_device(),
            &src,
            &packed,
            &tiny_shape_norms,
            NKV,
            256,
            CAP,
            0,
            2,
            0,
            false,
            1.0,
            8,
        )
    });
    assert!(norms_shape_error.contains("norms logical tensor"));

    let alias_bytes = packed.data_byte_len().max(src.data_byte_len());
    let alias_storage = device
        .alloc_buffer(alias_bytes, DType::U8, vec![alias_bytes])
        .expect("alias storage");
    let alias_src = MlxBuffer::from_raw(
        alias_storage.metal_buffer().clone(),
        DType::F32,
        vec![alias_bytes / 4],
    );
    let alias_error = assert_rejected_without_dispatch(&device, |encoder| {
        dispatch_hadamard_quantize_kv_hb_seq(
            encoder,
            &mut registry,
            device.metal_device(),
            &alias_src,
            &alias_storage,
            &norms,
            NKV,
            256,
            CAP,
            0,
            2,
            0,
            false,
            1.0,
            8,
        )
    });
    assert!(alias_error.contains("must not overlap"));

    let oversized_tokens = CAP + 4;
    let oversized_src_bytes = oversized_tokens as usize * NKV as usize * 256 * 4;
    let skipped_prefix_alias_storage = device
        .alloc_buffer(oversized_src_bytes, DType::U8, vec![oversized_src_bytes])
        .expect("skipped-prefix alias storage");
    let skipped_prefix_alias_src = MlxBuffer::from_raw(
        skipped_prefix_alias_storage.metal_buffer().clone(),
        DType::F32,
        vec![oversized_src_bytes / 4],
    );
    let skipped_prefix_alias_error = assert_rejected_without_dispatch(&device, |encoder| {
        dispatch_hadamard_quantize_kv_hb_seq(
            encoder,
            &mut registry,
            device.metal_device(),
            &skipped_prefix_alias_src,
            &skipped_prefix_alias_storage,
            &norms,
            NKV,
            256,
            CAP,
            CAP - 3,
            oversized_tokens,
            0,
            true,
            1.0,
            8,
        )
    });
    assert!(skipped_prefix_alias_error.contains("must not overlap"));

    let capacity_error = assert_rejected_without_dispatch(&device, |encoder| {
        dispatch_hadamard_quantize_kv_hb_seq(
            encoder,
            &mut registry,
            device.metal_device(),
            &src,
            &packed,
            &norms,
            NKV,
            256,
            0,
            0,
            2,
            0,
            true,
            1.0,
            8,
        )
    });
    assert!(capacity_error.contains("cache_capacity"));

    let mut encoder = device.command_encoder().expect("empty encoder");
    encoder.start_capture();
    dispatch_hadamard_quantize_kv_hb_seq(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &wrong_src,
        &packed,
        &norms,
        0,
        0,
        0,
        0,
        0,
        0,
        false,
        1.0,
        0,
    )
    .expect("empty request remains a no-op");
    assert!(
        encoder.take_capture().expect("empty capture").is_empty(),
        "empty request must not encode work"
    );
}
