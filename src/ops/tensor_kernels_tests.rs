//! The tensor-only kernel families where the Metal tensor API is unavailable.
//!
//! Two hosts have no tensor API (`dense_mm_capability`): a runtime compiler
//! without `<metal_tensor>`, and a device outside the Metal 4 GPU family. The
//! first is reproduced on any host
//! (`KernelRegistry::simulate_missing_tensor_header`, and a probe source that
//! stops the compiler with that diagnostic). The second cannot be reproduced
//! on a device that supports Metal 4, so `every_family_answers_on_this_device`
//! runs the unmodified registry: CI runs it on the hosted `macos-15` and
//! `macos-26` GPUs with `MLX_DENSE_MM_TEST_EXPECT_NO_TENSOR=1`, where every
//! family must answer "unavailable" and every entry point must return the
//! typed error, never the compiler's diagnostic.

#![allow(clippy::expect_used, clippy::panic)]

use super::dense_mm_f16::{dense_matmul_f16_f32_tensor, DenseMmF16F32Params};
use super::dflash2_af4g64_staged::{
    dflash2_af4g64_staged_gate_up_partial_elements, dflash2_af4g64_staged_partial_elements,
    dispatch_dflash2_af4g64_staged, dispatch_dflash2_af4g64_staged_gate_up,
};
use super::dflash2_ring_attention::{
    dflash2_ring_attention_partial_elements, dispatch_dflash2_ring_attention,
    DFlash2RingAttentionBuffers, DFlash2RingAttentionParams, DFLASH2_RING_ATTENTION_CURRENT_VALUES,
    DFLASH2_RING_ATTENTION_HEAD_DIM, DFLASH2_RING_ATTENTION_ROWS, DFLASH2_RING_ATTENTION_TILE,
};
use super::flash_attn_prefill_tensor::flash_attn_prefill_tensor_available;
use super::quantized_matmul_ggml::{
    dispatch_mm_v2_f16, quantized_matmul_mm_tensor_perm021, GgmlQuantizedMatmulParams,
    GgmlQuantizedMatmulPerm021Params, GgmlType,
};
use super::quantized_matmul_id_ggml::{
    moe_tiles_column_tile_exact, prepare_quantized_matmul_id_ggml_grouped_m8,
    GgmlQuantizedMatmulIdParams,
};
use super::repacked_gguf_mm::prepare_repacked_gguf_weight;
use super::tensor_kernels::{
    tensor_kernel_family_available, TensorKernelFamily, TENSOR_API_UNAVAILABLE,
};
use crate::{DType, KernelRegistry, MlxBuffer, MlxDevice, MlxError};

/// The runtime compiler's diagnostic on an SDK without the tensor header.
const MISSING_HEADER_PROBE_SOURCE: &str = "#error fatal error: 'metal_tensor' file not found\n";
/// A source no compiler accepts, for a reason that is not the tensor API.
const BROKEN_PROBE_SOURCE: &str = "#error use of undeclared identifier 'broken'\n";

fn expects_no_tensor() -> bool {
    std::env::var("MLX_DENSE_MM_TEST_EXPECT_NO_TENSOR").as_deref() == Ok("1")
}

fn buffer(device: &MlxDevice, elements: usize, dtype: DType) -> MlxBuffer {
    device
        .alloc_buffer(elements * dtype.size_of(), dtype, vec![elements])
        .expect("buffer")
}

fn assert_unavailable<T>(result: crate::Result<T>, family: TensorKernelFamily, label: &str) {
    match result {
        Err(MlxError::Unavailable { feature, reason }) => {
            assert_eq!(feature, family.name(), "{label}");
            assert_eq!(reason, TENSOR_API_UNAVAILABLE, "{label}");
        }
        Err(other) => panic!("{label}: expected the typed error, got {other}"),
        Ok(_) => panic!("{label}: ran on a host without the tensor API"),
    }
}

/// One valid call of every entry point that owns a tensor-only family, with
/// the smallest shapes each admits. `check` receives the family, the entry
/// point's name and whether it returned [`MlxError::Unavailable`]; any other
/// error fails the test.
fn for_each_entry_point(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    mut check: impl FnMut(TensorKernelFamily, &str, bool),
) {
    let encoded = std::cell::Cell::new(false);
    let mut report =
        |family: TensorKernelFamily, label: &str, result: crate::Result<()>| match result {
            Ok(()) => {
                encoded.set(true);
                check(family, label, false);
            }
            Err(MlxError::Unavailable { feature, reason }) => {
                assert_eq!(feature, family.name(), "{label}");
                assert_eq!(reason, TENSOR_API_UNAVAILABLE, "{label}");
                check(family, label, true);
            }
            Err(other) => panic!("{label}: {other}"),
        };
    let mut encoder = device.command_encoder().expect("encoder");

    // Prepared GGUF matmul: one Q4_K weight of 64 outputs over 256 inputs.
    {
        let (n, k) = (64usize, 256usize);
        let weight = buffer(device, n * 144, DType::U8);
        let params = GgmlQuantizedMatmulParams {
            m: 1,
            n: n as u32,
            k: k as u32,
            ggml_type: GgmlType::Q4_K,
        };
        let prepared =
            prepare_repacked_gguf_weight(&mut encoder, registry, device, &weight, &params);
        report(
            TensorKernelFamily::RepackedGgufMm,
            "prepare_repacked_gguf_weight",
            prepared.map(|_| ()),
        );
    }

    // Staged AF4G64 projection and its gate/up pair: 8 rows, 64 outputs.
    {
        let (m, n, k) = (8u32, 64u32, 64u32);
        let (rows, columns, inputs) = (m as usize, n as usize, k as usize);
        let input = buffer(device, rows * inputs, DType::BF16);
        let weights = buffer(device, columns * inputs / 8, DType::U32);
        let metadata = buffer(device, columns * inputs / 64, DType::U32);
        let partials = buffer(
            device,
            dflash2_af4g64_staged_partial_elements(m, n, k).max(1),
            DType::F32,
        );
        let output = buffer(device, rows * columns, DType::F32);
        report(
            TensorKernelFamily::Dflash2Af4g64Staged,
            "dispatch_dflash2_af4g64_staged",
            dispatch_dflash2_af4g64_staged(
                &mut encoder,
                registry,
                device,
                &input,
                &weights,
                &metadata,
                &partials,
                &output,
                m,
                n,
                k,
            ),
        );
        let pair_partials = buffer(
            device,
            dflash2_af4g64_staged_gate_up_partial_elements(m, n, k).max(1),
            DType::F32,
        );
        let gated = buffer(device, rows * columns, DType::BF16);
        report(
            TensorKernelFamily::Dflash2Af4g64Staged,
            "dispatch_dflash2_af4g64_staged_gate_up",
            dispatch_dflash2_af4g64_staged_gate_up(
                &mut encoder,
                registry,
                device,
                &input,
                &weights,
                &metadata,
                &weights,
                &metadata,
                &pair_partials,
                &gated,
                m,
                n,
                k,
            ),
        );
    }

    // Ring draft attention: one lane, one KV head, one tile of history.
    {
        const ROWS: usize = DFLASH2_RING_ATTENTION_ROWS;
        const DIM: usize = DFLASH2_RING_ATTENTION_HEAD_DIM;
        let (q_heads, kv_heads, window) = (4usize, 1usize, DFLASH2_RING_ATTENTION_TILE);
        let query = buffer(device, ROWS * q_heads * DIM, DType::F32);
        let key = buffer(device, ROWS * kv_heads * DIM, DType::F32);
        let value = buffer(device, ROWS * kv_heads * DIM, DType::F32);
        let ring_keys = buffer(device, kv_heads * window * DIM, DType::BF16);
        let ring_values = buffer(device, kv_heads * window * DIM, DType::BF16);
        let cache_lengths = buffer(device, 1, DType::U32);
        let queries = buffer(device, ROWS * q_heads * DIM, DType::BF16);
        let current_keys = buffer(device, ROWS * kv_heads * DIM, DType::BF16);
        let current_values = buffer(
            device,
            kv_heads * DFLASH2_RING_ATTENTION_CURRENT_VALUES * DIM,
            DType::BF16,
        );
        let partials = buffer(
            device,
            dflash2_ring_attention_partial_elements(1, kv_heads).expect("partials"),
            DType::F32,
        );
        let output = buffer(device, ROWS * q_heads * DIM, DType::BF16);
        report(
            TensorKernelFamily::Dflash2RingAttention,
            "dispatch_dflash2_ring_attention",
            dispatch_dflash2_ring_attention(
                &mut encoder,
                registry,
                device,
                &DFlash2RingAttentionBuffers {
                    query: &query,
                    key: &key,
                    value: &value,
                    ring_keys: &ring_keys,
                    ring_values: &ring_values,
                    cache_lengths: &cache_lengths,
                    queries: &queries,
                    current_keys: &current_keys,
                    current_values: &current_values,
                    partials: &partials,
                    output: &output,
                },
                DFlash2RingAttentionParams {
                    lanes: 1,
                    q_heads: q_heads as u32,
                    kv_heads: kv_heads as u32,
                    window: window as u32,
                },
            ),
        );
    }

    // Prepared expert planes: two Q4_K experts of 64 outputs over 256 inputs.
    {
        let (experts, n, k) = (2usize, 64usize, 256usize);
        let stride = n * 144;
        let weight = buffer(device, experts * stride, DType::U8);
        let params = GgmlQuantizedMatmulIdParams {
            n_tokens: 1,
            top_k: 1,
            n: n as u32,
            k: k as u32,
            n_experts: experts as u32,
            expert_stride: stride as u64,
            ggml_type: GgmlType::Q4_K,
        };
        let prepared = prepare_quantized_matmul_id_ggml_grouped_m8(
            &mut encoder,
            registry,
            device,
            &weight,
            &params,
        );
        report(
            TensorKernelFamily::GroupedM8Experts,
            "prepare_quantized_matmul_id_ggml_grouped_m8",
            prepared.map(|_| ()),
        );
    }

    // Head-major matmul: one Q4_0 weight, two heads of 32.
    {
        let (m, n, k, head_dim) = (32usize, 64usize, 64usize, 32usize);
        let input = buffer(device, m * k, DType::BF16);
        let weight = buffer(device, n * (k / 32) * 18, DType::U8);
        let output = buffer(device, m * n, DType::F32);
        report(
            TensorKernelFamily::MmPerm021,
            "quantized_matmul_mm_tensor_perm021",
            quantized_matmul_mm_tensor_perm021(
                &mut encoder,
                registry,
                device,
                &input,
                &weight,
                &output,
                &GgmlQuantizedMatmulPerm021Params {
                    m: m as u32,
                    n: n as u32,
                    k: k as u32,
                    head_dim: head_dim as u32,
                    ggml_type: GgmlType::Q4_0,
                },
            ),
        );
    }
    // Zero weights and activations: every command writes finite values.
    if encoded.get() {
        encoder.commit_and_wait().expect("commit");
    }

    // The live tiles' register-path probe runs its own command buffers.
    report(
        TensorKernelFamily::MoeExpertTiles,
        "moe_tiles_column_tile_exact",
        moe_tiles_column_tile_exact(device, registry).map(|_| ()),
    );
}

#[test]
fn missing_tensor_header_makes_every_family_unavailable() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    registry
        .simulate_missing_tensor_header()
        .expect("simulate the missing header");
    for family in TensorKernelFamily::ALL {
        for ask in 0..2 {
            assert!(
                !tensor_kernel_family_available(&mut registry, &device, family)
                    .unwrap_or_else(|error| panic!("{family:?}: {error}")),
                "{family:?}, ask {ask}"
            );
        }
    }
    // The families with a query of their own answer the same way.
    for head_dim in [256, 512] {
        assert!(
            !flash_attn_prefill_tensor_available(&mut registry, &device, head_dim)
                .expect("tensor flash attention query")
        );
    }
    let mut entry_points = 0;
    for_each_entry_point(&device, &mut registry, |family, label, unavailable| {
        assert!(
            unavailable,
            "{family:?}: {label} ran without the tensor API"
        );
        entry_points += 1;
    });
    assert_eq!(entry_points, 7);
}

#[test]
fn an_unavailable_probe_is_the_family_answer_and_its_typed_error() {
    let device = MlxDevice::new().expect("device");
    for family in TensorKernelFamily::ALL {
        let mut registry = KernelRegistry::new();
        registry.register_source(family.probe_kernel(), MISSING_HEADER_PROBE_SOURCE);
        assert!(
            !tensor_kernel_family_available(&mut registry, &device, family)
                .unwrap_or_else(|error| panic!("{family:?}: {error}")),
            "{family:?}"
        );
    }
    // The entry point of a family returns the typed error from that answer,
    // before it builds a kernel of its own.
    let mut registry = KernelRegistry::new();
    registry.register_source(
        TensorKernelFamily::RepackedGgufMm.probe_kernel(),
        MISSING_HEADER_PROBE_SOURCE,
    );
    let weight = buffer(&device, 64 * 144, DType::U8);
    let mut encoder = device.command_encoder().expect("encoder");
    assert_unavailable(
        prepare_repacked_gguf_weight(
            &mut encoder,
            &mut registry,
            &device,
            &weight,
            &GgmlQuantizedMatmulParams {
                m: 1,
                n: 64,
                k: 256,
                ggml_type: GgmlType::Q4_K,
            },
        ),
        TensorKernelFamily::RepackedGgufMm,
        "prepare_repacked_gguf_weight",
    );
}

#[test]
fn any_other_probe_failure_is_an_error_on_a_metal4_device() {
    let device = MlxDevice::new().expect("device");
    if !device
        .metal_device()
        .supports_family(metal::MTLGPUFamily::Metal4)
    {
        eprintln!(
            "{}: outside the Metal 4 family, every failed tensor build is \"unavailable\" here",
            device.name()
        );
        return;
    }
    for family in TensorKernelFamily::ALL {
        let mut registry = KernelRegistry::new();
        registry.register_source(family.probe_kernel(), BROKEN_PROBE_SOURCE);
        for ask in 0..2 {
            match tensor_kernel_family_available(&mut registry, &device, family) {
                Err(MlxError::ShaderCompilationError { .. }) => {}
                other => panic!("{family:?}, ask {ask}: expected a compile error, got {other:?}"),
            }
        }
    }
}

#[test]
fn a_changed_source_is_asked_again() {
    let device = MlxDevice::new().expect("device");
    let family = TensorKernelFamily::RepackedGgufMm;
    let mut registry = KernelRegistry::new();
    registry.register_source(family.probe_kernel(), MISSING_HEADER_PROBE_SOURCE);
    assert!(!tensor_kernel_family_available(&mut registry, &device, family).expect("stub"));
    registry.register_source(family.probe_kernel(), super::repacked_gguf_mm::SOURCE);
    let restored = tensor_kernel_family_available(&mut registry, &device, family).expect("source");
    let mut plain = KernelRegistry::new();
    let expected = tensor_kernel_family_available(&mut plain, &device, family).expect("plain");
    assert_eq!(restored, expected);
}

/// Where the tensor F16 matmul is unavailable, `dispatch_mm_v2_f16` encodes
/// the portable F16 matmul and gives its bits.
#[test]
fn f16_matmul_without_the_tensor_api_is_the_portable_kernel() {
    let device = MlxDevice::new().expect("device");
    let (m, n, k) = (33usize, 70usize, 96usize);
    let mut weight = buffer(&device, n * k, DType::F16);
    for (index, value) in weight
        .as_mut_slice::<u16>()
        .expect("weight")
        .iter_mut()
        .enumerate()
    {
        *value = half::f16::from_f32(((index * 37 % 201) as f32 - 100.0) / 128.0).to_bits();
    }
    let mut input = buffer(&device, m * k, DType::F32);
    for (index, value) in input
        .as_mut_slice::<f32>()
        .expect("input")
        .iter_mut()
        .enumerate()
    {
        *value = ((index * 53 % 157) as f32 - 78.0) / 64.0;
    }
    let run = |registry: &mut KernelRegistry, portable_entry: bool| -> (String, Vec<u32>) {
        let output = buffer(&device, m * n, DType::F32);
        let mut encoder = device.command_encoder().expect("encoder");
        encoder.start_encoded_dispatch_receipt(1).expect("receipt");
        if portable_entry {
            dense_matmul_f16_f32_tensor(
                &mut encoder,
                registry,
                &device,
                &weight,
                &input,
                &output,
                &DenseMmF16F32Params {
                    m: m as u32,
                    n: n as u32,
                    k: k as u32,
                    src0_batch: 1,
                    src1_batch: 1,
                },
            )
            .expect("portable F16 matmul");
        } else {
            dispatch_mm_v2_f16(
                &mut encoder,
                registry,
                &device,
                &weight,
                &input,
                &output,
                m as u32,
                n as u32,
                k as u32,
            )
            .expect("dispatch_mm_v2_f16");
        }
        let receipt = encoder.take_encoded_dispatch_receipt().expect("receipt");
        encoder.commit_and_wait().expect("commit");
        let kernel = receipt[0]
            .pipeline_label
            .split('|')
            .next()
            .expect("kernel")
            .to_string();
        let bits = output
            .as_slice::<f32>()
            .expect("output")
            .iter()
            .map(|value| value.to_bits())
            .collect();
        (kernel, bits)
    };
    let mut unavailable = KernelRegistry::new();
    unavailable
        .simulate_missing_tensor_header()
        .expect("simulate the missing header");
    let (kernel, bits) = run(&mut unavailable, false);
    assert_eq!(kernel, "hf2q_dense_mm_f16_f32_fallback");
    let mut portable = KernelRegistry::new();
    portable
        .simulate_missing_tensor_header()
        .expect("simulate the missing header");
    let (portable_kernel, portable_bits) = run(&mut portable, true);
    assert_eq!(portable_kernel, "hf2q_dense_mm_f16_f32_fallback");
    assert!(bits == portable_bits, "differs from the portable kernel");
    assert!(
        bits.iter().any(|&word| word != 0),
        "the product is not zero"
    );

    // The unmodified registry encodes whichever kernel this device has.
    let mut auto = KernelRegistry::new();
    let (auto_kernel, auto_bits) = run(&mut auto, false);
    if auto_kernel == "hf2q_dense_mm_f16_f32_fallback" {
        assert!(
            auto_bits == portable_bits,
            "differs from the portable kernel"
        );
    } else {
        assert_eq!(auto_kernel, "hf2q_mul_mm_tensor_v2_f16");
        assert!(!expects_no_tensor(), "{}: tensor kernel", device.name());
    }
}

#[test]
fn every_family_answers_on_this_device() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let mut available = 0;
    for family in TensorKernelFamily::ALL {
        let answer = tensor_kernel_family_available(&mut registry, &device, family)
            .unwrap_or_else(|error| panic!("{family:?}: {error}"));
        available += usize::from(answer);
    }
    let families = TensorKernelFamily::ALL.len();
    let expected = if expects_no_tensor() {
        available == 0
    } else {
        available == 0 || available == families
    };
    assert!(
        expected,
        "{}: {available} of {families} tensor-only families available",
        device.name()
    );
    // Every entry point agrees with its family's answer: it runs where the
    // family is available and returns the typed error where it is not.
    for_each_entry_point(&device, &mut registry, |family, label, unavailable| {
        assert_eq!(unavailable, available == 0, "{family:?}: {label}");
    });
    eprintln!(
        "{}: Metal 4 family {}, tensor-only kernel families: {}",
        device.name(),
        device
            .metal_device()
            .supports_family(metal::MTLGPUFamily::Metal4),
        if available == 0 {
            "unavailable"
        } else {
            "available"
        }
    );
}
