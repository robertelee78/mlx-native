//! Typed evidence for the concrete Metal dispatches encoded by one GGML call.
//!
//! Capability describes the structurally eligible route. A resolved trace is
//! stronger: it is collected around the exact explicit-policy entrypoint and
//! names every pipeline and launch geometry that was actually encoded.

use serde::{Deserialize, Serialize};

use crate::device::MlxDevice;
use crate::encoder::{CommandEncoder, DispatchKind, EncodedKernelDispatch};
use crate::error::{MlxError, Result};
use crate::ggml_capability::{
    ggml_capability, GgmlCapability, GgmlCapabilityRequest, GgmlInvocation, GgmlKernelRoute,
    GgmlRoutingPolicy, GgmlWorkloadClass, GGML_CAPABILITY_SCHEMA_VERSION,
};
use crate::kernel_registry::{KernelPipelineIdentity, KernelRegistry};
use crate::ops::quantized_matmul_ggml::GgmlType;

pub const GGML_RESOLVED_DISPATCH_TRACE_SCHEMA_VERSION: u32 = 2;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GgmlDeviceIdentity {
    pub name: String,
    pub registry_id: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GgmlKernelDispatchReceipt {
    pub encoded: EncodedKernelDispatch,
    pub pipeline: KernelPipelineIdentity,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum GgmlResolvedKernelRoute {
    DenseMv,
    DenseMvNr2,
    DenseQ4kWidthMn,
    DenseQ6kWidthMn,
    DenseWidthMvExt,
    DenseMmSimdgroup,
    DenseMmTensorV1,
    DenseMmTensorV2,
    DenseMmTensorQ4_64x32,
    DenseGateUpSilu,
}

/// Exact typed request, pure capability decision, physical device, and the
/// concrete Metal dispatches encoded by one successful public GGML call.
///
/// This proves host-side encoding only. It does not prove command-buffer
/// submission/completion, numerical correctness, or latency. Performance
/// admission must join it to a separately committed/completed matched
/// benchmark receipt; timing the traced call itself includes evidence overhead.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GgmlResolvedDispatchTrace {
    pub schema_version: u32,
    pub mlx_native_version: String,
    pub request: GgmlCapabilityRequest,
    pub capability: GgmlCapability,
    pub resolved_route: GgmlResolvedKernelRoute,
    pub device: GgmlDeviceIdentity,
    pub dispatches: Vec<GgmlKernelDispatchReceipt>,
}

pub(crate) fn trace_ggml_operation<F>(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    request: GgmlCapabilityRequest,
    operation: F,
) -> Result<GgmlResolvedDispatchTrace>
where
    F: FnOnce(&mut CommandEncoder, &mut KernelRegistry) -> Result<()>,
{
    let capability = ggml_capability(request);
    if !capability.executable {
        return Err(MlxError::InvalidArgument(format!(
            "GGML capability rejected traced request: {}",
            capability.diagnostic
        )));
    }

    let expected_dispatches = usize::try_from(capability.dispatches).map_err(|_| {
        MlxError::InvalidArgument("GGML capability dispatch count does not fit usize".into())
    })?;
    let encoder_registry_id = encoder.device_registry_id();
    let device_registry_id = device.registry_id();
    if encoder_registry_id != device_registry_id {
        return Err(MlxError::InvalidArgument(format!(
            "command encoder belongs to Metal device {encoder_registry_id}, not {device_registry_id}"
        )));
    }
    encoder.start_encoded_dispatch_receipt(expected_dispatches)?;
    let operation_result = operation(encoder, registry);
    // Always drain the scoped recorder before propagating an operation error,
    // so a failed call cannot poison or merge with a later receipt.
    let encoded_result = encoder.take_encoded_dispatch_receipt();
    operation_result?;
    let encoded = encoded_result?;

    if encoded.len() != expected_dispatches {
        return Err(MlxError::InvalidArgument(format!(
            "GGML capability predicted {expected_dispatches} dispatches but execution encoded {}",
            encoded.len()
        )));
    }

    let mut dispatches = Vec::with_capacity(encoded.len());
    for encoded in encoded {
        let pipeline = registry.pipeline_identity(&encoded.pipeline_label)?;
        if pipeline.pipeline_label != encoded.pipeline_label {
            return Err(MlxError::InvalidArgument(format!(
                "pipeline identity label {} does not match encoded label {}",
                pipeline.pipeline_label, encoded.pipeline_label
            )));
        }
        dispatches.push(GgmlKernelDispatchReceipt { encoded, pipeline });
    }
    let resolved_route = validate_dense_dispatches(&request, &capability, &dispatches)?;

    Ok(GgmlResolvedDispatchTrace {
        schema_version: GGML_RESOLVED_DISPATCH_TRACE_SCHEMA_VERSION,
        mlx_native_version: env!("CARGO_PKG_VERSION").to_string(),
        request,
        capability,
        resolved_route,
        device: GgmlDeviceIdentity {
            name: device.name(),
            registry_id: device_registry_id,
        },
        dispatches,
    })
}

fn div_ceil_u64(value: u32, divisor: u64) -> u64 {
    (u64::from(value) + divisor - 1) / divisor
}

fn require_dispatch(
    dispatch: &GgmlKernelDispatchReceipt,
    kernel_name: &str,
    pipeline_label: &str,
    grid: [u64; 3],
    threads: [u64; 3],
    threadgroup_memory: &[(u64, u64)],
) -> Result<()> {
    if dispatch.pipeline.kernel_name != kernel_name
        || dispatch.pipeline.pipeline_label != pipeline_label
        || dispatch.encoded.pipeline_label != pipeline_label
        || dispatch.encoded.dispatch_kind != DispatchKind::ThreadGroups
        || dispatch.encoded.grid != grid
        || dispatch.encoded.threads_per_threadgroup != threads
        || dispatch.encoded.threadgroup_memory != threadgroup_memory
    {
        return Err(MlxError::InvalidArgument(format!(
            "resolved dispatch does not match {kernel_name}: got {:?}",
            dispatch.encoded
        )));
    }
    Ok(())
}

fn single_batch_pipeline_label(kernel_name: &str) -> String {
    format!("{kernel_name}|700:i1|701:i1|702:i1")
}

fn require_one<'a>(
    dispatches: &'a [GgmlKernelDispatchReceipt],
) -> Result<&'a GgmlKernelDispatchReceipt> {
    if dispatches.len() != 1 {
        return Err(MlxError::InvalidArgument(format!(
            "resolved route requires one dispatch, got {}",
            dispatches.len()
        )));
    }
    Ok(&dispatches[0])
}

fn dense_dimensions(request: &GgmlCapabilityRequest) -> Result<(u32, u32, u32)> {
    match request.invocation {
        GgmlInvocation::DenseAuto { m, n, k } | GgmlInvocation::DenseGateUpSiluPair { m, n, k } => {
            Ok((m, n, k))
        }
        _ => Err(MlxError::InvalidArgument(
            "dense trace validator received a non-dense invocation".into(),
        )),
    }
}

fn validate_dense_dispatches(
    request: &GgmlCapabilityRequest,
    capability: &GgmlCapability,
    dispatches: &[GgmlKernelDispatchReceipt],
) -> Result<GgmlResolvedKernelRoute> {
    let (m, n, k) = dense_dimensions(request)?;
    let structural_route = capability.route.ok_or_else(|| {
        MlxError::InvalidArgument("executable GGML capability omitted its route".into())
    })?;
    match structural_route {
        GgmlKernelRoute::DenseMv | GgmlKernelRoute::DenseMvNr2 => {
            let dispatch = require_one(dispatches)?;
            let nr2 = structural_route == GgmlKernelRoute::DenseMvNr2;
            let (kernel, align, threads, shmem) = match (request.ggml_type, nr2) {
                (GgmlType::Q6_K, true) => ("kernel_mul_mv_q6_K_f32_nr2", 4, [2, 32, 1], Vec::new()),
                (GgmlType::Q8_0, true) => {
                    ("kernel_mul_mv_q8_0_f32_nr2", 2, [32, 4, 1], vec![(0, 256)])
                }
                (kind, false) => {
                    let (align, threads) = match kind {
                        GgmlType::Q4_0
                        | GgmlType::Q8_0
                        | GgmlType::Q5_1
                        | GgmlType::IQ4_NL
                        | GgmlType::IQ4_XS => (8, [8, 8, 1]),
                        GgmlType::Q2_K => (8, [2, 32, 1]),
                        GgmlType::Q3_K => (4, [2, 32, 1]),
                        GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K => (2, [2, 32, 1]),
                        _ => {
                            return Err(MlxError::InvalidArgument(
                                "unsupported dense matvec type in trace".into(),
                            ))
                        }
                    };
                    (kind.kernel_name(), align, threads, Vec::new())
                }
                _ => {
                    return Err(MlxError::InvalidArgument(
                        "capability selected an invalid dense NR2 type".into(),
                    ))
                }
            };
            require_dispatch(
                dispatch,
                kernel,
                &single_batch_pipeline_label(kernel),
                [div_ceil_u64(n, align), u64::from(m), 1],
                threads,
                &shmem,
            )?;
            Ok(if nr2 {
                GgmlResolvedKernelRoute::DenseMvNr2
            } else {
                GgmlResolvedKernelRoute::DenseMv
            })
        }
        GgmlKernelRoute::DenseQ4kWidthMn | GgmlKernelRoute::DenseQ6kWidthMn => {
            let is_q4 = structural_route == GgmlKernelRoute::DenseQ4kWidthMn;
            let type_name = if is_q4 { "Q4_K" } else { "Q6_K" };
            let kernel_type = if is_q4 { "q4_K" } else { "q6_K" };
            let widths: &[u32] = match m {
                2 => &[2],
                3 => &[3],
                4 => &[4],
                5 => &[5],
                6 => &[3, 3],
                7 => &[4, 3],
                8 => &[4, 4],
                _ => {
                    return Err(MlxError::InvalidArgument(format!(
                        "{type_name} width trace requires M in 2..=8"
                    )))
                }
            };
            if dispatches.len() != widths.len() {
                return Err(MlxError::InvalidArgument(format!(
                    "{type_name} width trace requires {} tiles, got {}",
                    widths.len(),
                    dispatches.len()
                )));
            }
            for (dispatch, width) in dispatches.iter().zip(widths.iter()) {
                let kernel = format!("kernel_mul_mv_{kernel_type}_f32_mN_r1_{width}");
                require_dispatch(
                    dispatch,
                    &kernel,
                    &single_batch_pipeline_label(&kernel),
                    [div_ceil_u64(n, if is_q4 { 2 } else { 4 }), 1, 1],
                    [2, 32, 1],
                    &[],
                )?;
            }
            Ok(if is_q4 {
                GgmlResolvedKernelRoute::DenseQ4kWidthMn
            } else {
                GgmlResolvedKernelRoute::DenseQ6kWidthMn
            })
        }
        GgmlKernelRoute::DenseWidthMvExt => {
            let dispatch = require_one(dispatches)?;
            let r1 = match m {
                2..=5 => m,
                6 => 3,
                7 | 8 => 4,
                _ => {
                    return Err(MlxError::InvalidArgument(
                        "mul_mv_ext trace requires M in 2..=8".into(),
                    ))
                }
            };
            let type_name = match request.ggml_type {
                GgmlType::Q4_0 => "q4_0",
                GgmlType::Q8_0 => "q8_0",
                GgmlType::Q4_K => "q4_K",
                GgmlType::Q5_K => "q5_K",
                GgmlType::Q6_K => "q6_K",
                GgmlType::Q5_1 => "q5_1",
                GgmlType::IQ4_NL => "iq4_nl",
                _ => {
                    return Err(MlxError::InvalidArgument(
                        "unsupported mul_mv_ext type in trace".into(),
                    ))
                }
            };
            let nx = if k % 256 == 0 && m < 3 {
                16
            } else if k % 128 == 0 {
                8
            } else {
                4
            };
            let r0 = (32 / nx) * 2;
            let kernel = format!("kernel_mul_mv_ext_{type_name}_f32_r1_{r1}");
            require_dispatch(
                dispatch,
                &kernel,
                &format!("{kernel}|600:i2|601:i{nx}"),
                [
                    div_ceil_u64(n, r0 as u64),
                    div_ceil_u64(m, u64::from(r1)),
                    1,
                ],
                [32, 2, 1],
                &[],
            )?;
            Ok(GgmlResolvedKernelRoute::DenseWidthMvExt)
        }
        GgmlKernelRoute::DenseMmSimdgroup | GgmlKernelRoute::DenseMmDeviceSelected => {
            let dispatch = require_one(dispatches)?;
            let (resolved, kernel, grid, shmem) = if dispatch.pipeline.kernel_name
                == request.ggml_type.mm_kernel_name()
            {
                (
                    GgmlResolvedKernelRoute::DenseMmSimdgroup,
                    request.ggml_type.mm_kernel_name(),
                    [div_ceil_u64(m, 32), div_ceil_u64(n, 64), 1],
                    vec![(0, 8192)],
                )
            } else if dispatch.pipeline.kernel_name == request.ggml_type.mm_tensor_kernel_name() {
                if request.routing.allow_dense_large_tile_mm {
                    return Err(MlxError::InvalidArgument(
                        "tensor-v1 dispatch violates large-tile routing policy".into(),
                    ));
                }
                (
                    GgmlResolvedKernelRoute::DenseMmTensorV1,
                    request.ggml_type.mm_tensor_kernel_name(),
                    [div_ceil_u64(m, 32), div_ceil_u64(n, 64), 1],
                    vec![(0, 8192)],
                )
            } else if request.ggml_type == GgmlType::Q4_0
                && dispatch.pipeline.kernel_name == "kernel_mul_mm_q4_0_tensor_64x32_f32"
            {
                if !request.routing.allow_dense_large_tile_mm {
                    return Err(MlxError::InvalidArgument(
                        "Q4 exact-plan tensor dispatch violates large-tile routing policy".into(),
                    ));
                }
                (
                    GgmlResolvedKernelRoute::DenseMmTensorQ4_64x32,
                    "kernel_mul_mm_q4_0_tensor_64x32_f32",
                    [div_ceil_u64(m, 32), div_ceil_u64(n, 64), 1],
                    vec![(0, 4096)],
                )
            } else if dispatch.pipeline.kernel_name == request.ggml_type.mm_tensor_v2_kernel_name()
            {
                if !request.routing.allow_dense_large_tile_mm {
                    return Err(MlxError::InvalidArgument(
                        "large-tile tensor dispatch violates routing policy".into(),
                    ));
                }
                (
                    GgmlResolvedKernelRoute::DenseMmTensorV2,
                    request.ggml_type.mm_tensor_v2_kernel_name(),
                    [div_ceil_u64(m, 128), div_ceil_u64(n, 64), 1],
                    vec![(0, 4096)],
                )
            } else {
                return Err(MlxError::InvalidArgument(format!(
                    "pipeline {} is not a legal dense MM route for {:?}",
                    dispatch.pipeline.kernel_name, request.ggml_type
                )));
            };
            if structural_route == GgmlKernelRoute::DenseMmSimdgroup
                && resolved != GgmlResolvedKernelRoute::DenseMmSimdgroup
            {
                return Err(MlxError::InvalidArgument(
                    "ForceSimd capability executed a tensor-MM pipeline".into(),
                ));
            }
            require_dispatch(
                dispatch,
                kernel,
                &single_batch_pipeline_label(kernel),
                grid,
                [128, 1, 1],
                &shmem,
            )?;
            Ok(resolved)
        }
        GgmlKernelRoute::FusedGateUpSilu => {
            let dispatch = require_one(dispatches)?;
            let (kernel, output_rows_per_group, threads, shmem) = match request.ggml_type {
                GgmlType::Q4_K => (
                    "kernel_fused_gate_up_silu_q4_K_f32",
                    2,
                    [32, 2, 1],
                    Vec::new(),
                ),
                GgmlType::Q5_K => (
                    "kernel_fused_gate_up_silu_q5_K_f32",
                    2,
                    [32, 2, 1],
                    Vec::new(),
                ),
                GgmlType::Q6_K => (
                    "kernel_fused_gate_up_silu_q6_K_f32",
                    2,
                    [2, 32, 1],
                    Vec::new(),
                ),
                GgmlType::Q8_0 => (
                    "kernel_fused_gate_up_silu_q8_0_f32",
                    2,
                    [32, 4, 1],
                    vec![(0, 512)],
                ),
                GgmlType::IQ4_NL => (
                    "kernel_fused_gate_up_silu_iq4_nl_f32",
                    8,
                    [8, 8, 1],
                    Vec::new(),
                ),
                _ => {
                    return Err(MlxError::InvalidArgument(
                        "unsupported fused gate/up type in traced wrapper".into(),
                    ))
                }
            };
            require_dispatch(
                dispatch,
                kernel,
                &single_batch_pipeline_label(kernel),
                [div_ceil_u64(n, output_rows_per_group), u64::from(m), 1],
                threads,
                &shmem,
            )?;
            Ok(GgmlResolvedKernelRoute::DenseGateUpSilu)
        }
        other => Err(MlxError::InvalidArgument(format!(
            "resolved-dispatch tracing does not admit route {other:?}"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn trace_dense_gate_up_silu_operation<F>(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    ggml_type: GgmlType,
    m: u32,
    n: u32,
    k: u32,
    routing: &GgmlRoutingPolicy,
    workload: GgmlWorkloadClass,
    operation: F,
) -> Result<GgmlResolvedDispatchTrace>
where
    F: FnOnce(&mut CommandEncoder, &mut KernelRegistry) -> Result<()>,
{
    trace_ggml_operation(
        encoder,
        registry,
        device,
        GgmlCapabilityRequest {
            schema_version: GGML_CAPABILITY_SCHEMA_VERSION,
            invocation: GgmlInvocation::DenseGateUpSiluPair { m, n, k },
            ggml_type,
            workload,
            routing: *routing,
        },
        operation,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_registry::{KernelPipelineOrigin, KERNEL_PIPELINE_IDENTITY_SCHEMA_VERSION};

    fn fake_dispatch(
        kernel_name: &str,
        pipeline_label: &str,
        grid: [u64; 3],
        threads: [u64; 3],
        threadgroup_memory: Vec<(u64, u64)>,
    ) -> GgmlKernelDispatchReceipt {
        GgmlKernelDispatchReceipt {
            encoded: EncodedKernelDispatch {
                pipeline_label: pipeline_label.to_string(),
                dispatch_kind: DispatchKind::ThreadGroups,
                grid,
                threads_per_threadgroup: threads,
                threadgroup_memory,
            },
            pipeline: KernelPipelineIdentity {
                schema_version: KERNEL_PIPELINE_IDENTITY_SCHEMA_VERSION,
                pipeline_label: pipeline_label.to_string(),
                kernel_name: kernel_name.to_string(),
                origin: KernelPipelineOrigin::RuntimeSource,
                runtime_source_sha256: Some("0".repeat(64)),
                embedded_metallib_sha256: None,
                precise_fp32_math: false,
                threadgroup_size_multiple_hint: false,
            },
        }
    }

    fn request(
        invocation: GgmlInvocation,
        ggml_type: GgmlType,
        workload: GgmlWorkloadClass,
        routing: GgmlRoutingPolicy,
    ) -> GgmlCapabilityRequest {
        GgmlCapabilityRequest {
            schema_version: GGML_CAPABILITY_SCHEMA_VERSION,
            invocation,
            ggml_type,
            workload,
            routing,
        }
    }

    #[test]
    fn q8_nr2_trace_requires_exact_pipeline_constants_and_geometry() {
        let small_tile_request = request(
            GgmlInvocation::DenseAuto {
                m: 1,
                n: 64,
                k: 256,
            },
            GgmlType::Q8_0,
            GgmlWorkloadClass::DecodeSingle,
            GgmlRoutingPolicy::default(),
        );
        let capability = ggml_capability(small_tile_request);
        let kernel = "kernel_mul_mv_q8_0_f32_nr2";
        let label = single_batch_pipeline_label(kernel);
        let valid = fake_dispatch(kernel, &label, [32, 1, 1], [32, 4, 1], vec![(0, 256)]);
        assert_eq!(
            validate_dense_dispatches(&small_tile_request, &capability, &[valid.clone()]).unwrap(),
            GgmlResolvedKernelRoute::DenseMvNr2
        );

        let wrong_constants = fake_dispatch(
            kernel,
            &format!("{kernel}|700:i2|701:i1|702:i1"),
            [32, 1, 1],
            [32, 4, 1],
            vec![(0, 256)],
        );
        assert!(
            validate_dense_dispatches(&small_tile_request, &capability, &[wrong_constants])
                .is_err()
        );

        let wrong_geometry = fake_dispatch(kernel, &label, [31, 1, 1], [32, 4, 1], vec![(0, 256)]);
        assert!(
            validate_dense_dispatches(&small_tile_request, &capability, &[wrong_geometry]).is_err()
        );
    }

    #[test]
    fn q6_width_trace_requires_exact_ordered_tiles() {
        let width_request = request(
            GgmlInvocation::DenseAuto {
                m: 7,
                n: 65,
                k: 256,
            },
            GgmlType::Q6_K,
            GgmlWorkloadClass::ContinuousWidth,
            GgmlRoutingPolicy::default(),
        );
        let capability = ggml_capability(width_request);
        let dispatch = |width| {
            let kernel = format!("kernel_mul_mv_q6_K_f32_mN_r1_{width}");
            fake_dispatch(
                &kernel,
                &single_batch_pipeline_label(&kernel),
                [17, 1, 1],
                [2, 32, 1],
                Vec::new(),
            )
        };
        let valid = vec![dispatch(4), dispatch(3)];
        assert_eq!(
            validate_dense_dispatches(&width_request, &capability, &valid).unwrap(),
            GgmlResolvedKernelRoute::DenseQ6kWidthMn
        );
        assert!(validate_dense_dispatches(
            &width_request,
            &capability,
            &[dispatch(3), dispatch(4)]
        )
        .is_err());
    }

    #[test]
    fn q4_width_trace_requires_exact_ordered_tiles_and_geometry() {
        let width_request = request(
            GgmlInvocation::DenseAuto {
                m: 7,
                n: 65,
                k: 256,
            },
            GgmlType::Q4_K,
            GgmlWorkloadClass::ContinuousWidth,
            GgmlRoutingPolicy::default(),
        );
        let capability = ggml_capability(width_request);
        let dispatch = |width| {
            let kernel = format!("kernel_mul_mv_q4_K_f32_mN_r1_{width}");
            fake_dispatch(
                &kernel,
                &single_batch_pipeline_label(&kernel),
                [33, 1, 1],
                [2, 32, 1],
                Vec::new(),
            )
        };
        let valid = vec![dispatch(4), dispatch(3)];
        assert_eq!(
            validate_dense_dispatches(&width_request, &capability, &valid).unwrap(),
            GgmlResolvedKernelRoute::DenseQ4kWidthMn
        );
        assert!(validate_dense_dispatches(
            &width_request,
            &capability,
            &[dispatch(3), dispatch(4)]
        )
        .is_err());
        let kernel = "kernel_mul_mv_q4_K_f32_mN_r1_4";
        let wrong_geometry = fake_dispatch(
            kernel,
            &single_batch_pipeline_label(kernel),
            [17, 1, 1],
            [2, 32, 1],
            Vec::new(),
        );
        assert!(validate_dense_dispatches(
            &width_request,
            &capability,
            &[wrong_geometry, dispatch(3)]
        )
        .is_err());
    }

    #[test]
    fn device_selected_mm_rejects_a_disallowed_large_tile_pipeline() {
        let routing = GgmlRoutingPolicy {
            allow_dense_large_tile_mm: false,
            ..GgmlRoutingPolicy::default()
        };
        let small_tile_request = request(
            GgmlInvocation::DenseAuto {
                m: 9,
                n: 64,
                k: 256,
            },
            GgmlType::Q8_0,
            GgmlWorkloadClass::Prompt,
            routing,
        );
        let capability = ggml_capability(small_tile_request);
        let kernel = GgmlType::Q8_0.mm_tensor_v2_kernel_name();
        let dispatch = fake_dispatch(
            kernel,
            &single_batch_pipeline_label(kernel),
            [1, 1, 1],
            [128, 1, 1],
            vec![(0, 4096)],
        );
        assert!(validate_dense_dispatches(&small_tile_request, &capability, &[dispatch]).is_err());

        let large_tile_request = request(
            GgmlInvocation::DenseAuto {
                m: 9,
                n: 64,
                k: 256,
            },
            GgmlType::Q8_0,
            GgmlWorkloadClass::Prompt,
            GgmlRoutingPolicy::default(),
        );
        let large_tile_capability = ggml_capability(large_tile_request);
        let v1_kernel = GgmlType::Q8_0.mm_tensor_kernel_name();
        let v1_dispatch = fake_dispatch(
            v1_kernel,
            &single_batch_pipeline_label(v1_kernel),
            [1, 1, 1],
            [128, 1, 1],
            vec![(0, 8192)],
        );
        assert!(validate_dense_dispatches(
            &large_tile_request,
            &large_tile_capability,
            &[v1_dispatch]
        )
        .is_err());
    }

    #[test]
    fn fused_iq4_trace_binds_the_distinct_kernel_geometry() {
        let request = request(
            GgmlInvocation::DenseGateUpSiluPair {
                m: 1,
                n: 65,
                k: 256,
            },
            GgmlType::IQ4_NL,
            GgmlWorkloadClass::DecodeSingle,
            GgmlRoutingPolicy::default(),
        );
        let capability = ggml_capability(request);
        let kernel = "kernel_fused_gate_up_silu_iq4_nl_f32";
        let dispatch = fake_dispatch(
            kernel,
            &single_batch_pipeline_label(kernel),
            [9, 1, 1],
            [8, 8, 1],
            Vec::new(),
        );
        assert_eq!(
            validate_dense_dispatches(&request, &capability, &[dispatch]).unwrap(),
            GgmlResolvedKernelRoute::DenseGateUpSilu
        );
    }
}
