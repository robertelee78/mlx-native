//! Ownership of immutable expert planes, independent of the selected arithmetic.
use super::{
    as_bytes, checked_byte_extent, require_tensor_kernel_family, CommandEncoder, DType,
    GgmlQuantizedMatmulIdParams, GgmlType, GroupedM8PrepareGpuParams, KernelArg, KernelRegistry,
    MlxBuffer, MlxDevice, MlxError, Result, TensorKernelFamily,
};
use crate::device::metal_device_registry_id;
use std::ops::Deref;
use std::sync::{Arc, Mutex};

/// Immutable Q4_K/Q6_K planes borrowed by expert consumers.
///
/// Fields and construction remain private: borrowing an asynchronously prepared
/// view does not certify that its producer has completed. Producer ordering is
/// still required when using the source-retaining asynchronous handle.
pub struct GroupedM8ExpertPlanes {
    pub(super) bytes: MlxBuffer,
    pub(super) n_experts: u32,
    pub(super) n: u32,
    pub(super) k: u32,
    pub(super) ggml_type: GgmlType,
    pub(super) prepared_expert_stride: u64,
    pub(super) plane1_offset: u32,
    pub(super) meta_offset: u32,
}

struct PendingExpertPlanes {
    _source: MlxBuffer,
    planes: GroupedM8ExpertPlanes,
}

/// Source-retaining result of an encode-only preparation operation.
///
/// Keep this owner alive until preparation completes. Cloning shares both the
/// source and planes without copying buffer-shape metadata.
#[derive(Clone)]
pub struct PreparedGroupedM8Experts {
    inner: Arc<PendingExpertPlanes>,
}

impl PreparedGroupedM8Experts {
    pub(super) fn pending(source: MlxBuffer, planes: GroupedM8ExpertPlanes) -> Self {
        Self { inner: Arc::new(PendingExpertPlanes { _source: source, planes }) }
    }
}

impl Deref for PreparedGroupedM8Experts {
    type Target = GroupedM8ExpertPlanes;

    fn deref(&self) -> &Self::Target {
        &self.inner.planes
    }
}

/// Successfully completed preparation, owning planes but no source buffer.
///
/// Only the native-owned batch completion boundary constructs this handle.
/// Completion does not prove that other source owners or residency memberships
/// have been retired by the caller.
#[derive(Clone)]
pub struct CompletedGroupedM8Experts {
    planes: Arc<GroupedM8ExpertPlanes>,
}

impl Deref for CompletedGroupedM8Experts {
    type Target = GroupedM8ExpertPlanes;

    fn deref(&self) -> &Self::Target {
        &self.planes
    }
}

/// An owned source and its immutable expert-stack geometry for batch preparation.
pub struct GroupedM8Preparation {
    pub source: MlxBuffer,
    pub params: GgmlQuantizedMatmulIdParams,
}

struct PreparationPlan {
    gpu: GroupedM8PrepareGpuParams,
    ggml_type: GgmlType,
    bytes: usize,
    threads: u64,
}

impl PreparationPlan {
    fn new(
        device: &MlxDevice,
        source: &MlxBuffer,
        params: &GgmlQuantizedMatmulIdParams,
    ) -> Result<Self> {
        let invalid = |message: &str| MlxError::InvalidArgument(message.into());
        if source.dtype() != DType::U8 {
            return Err(invalid("expert preparation requires U8 source storage"));
        }
        let format = match params.ggml_type {
            GgmlType::Q4_K => 0,
            GgmlType::Q6_K => 1,
            _ => return Err(invalid("expert preparation requires Q4_K or Q6_K")),
        };
        if params.n_experts == 0 || params.n == 0 || params.n % 64 != 0
            || params.k == 0 || params.k % 256 != 0
        {
            return Err(invalid("expert preparation requires experts > 0, N % 64 == 0 and K % 256 == 0"));
        }
        if metal_device_registry_id(source.metal_buffer().device()) != device.registry_id() {
            return Err(invalid("expert preparation source belongs to another device"));
        }
        // The producer reads half metadata from every raw block. A private
        // Metal allocation is legitimate; CPU addressability is not required.
        if source.byte_offset() % 2 != 0 || params.expert_stride % 2 != 0 {
            return Err(invalid("expert preparation source metadata is not half-aligned"));
        }
        let physical_end = source.byte_offset().checked_add(source.data_byte_len() as u64)
            .ok_or_else(|| invalid("expert preparation source window overflow"))?;
        if physical_end > source.byte_len() as u64 {
            return Err(invalid("expert preparation source window exceeds its allocation"));
        }
        let matrix = checked_byte_extent("expert preparation source matrix", &[
            params.n as usize, (params.k / params.ggml_type.block_values()) as usize,
            params.ggml_type.block_bytes() as usize,
        ])?;
        if params.expert_stride < matrix as u64 {
            return Err(invalid("expert preparation stride is smaller than its matrix"));
        }
        let required = u64::from(params.n_experts - 1).checked_mul(params.expert_stride)
            .and_then(|offset| offset.checked_add(matrix as u64))
            .ok_or_else(|| invalid("expert preparation source extent overflow"))?;
        if required > source.data_byte_len() as u64 {
            return Err(invalid("expert preparation source extent is too small"));
        }
        let threads = u64::from(params.n_experts).checked_mul(u64::from(params.n))
            .and_then(|value| value.checked_mul(u64::from(params.k / 32)))
            .filter(|value| *value <= u64::from(u32::MAX))
            .ok_or_else(|| invalid("expert preparation exceeds its shader's u32 index range"))?;
        let rows = params.n as usize;
        let groups = (params.k / 32) as usize;
        let low = checked_byte_extent("expert preparation low plane", &[rows, groups, 16])?;
        let high = if format == 1 {
            checked_byte_extent("expert preparation high plane", &[rows, groups, 8])?
        } else { 0 };
        let metadata = checked_byte_extent("expert preparation metadata", &[
            rows, groups / 8, if format == 1 { 20 } else { 16 },
        ])?;
        let stride = low.checked_add(high).and_then(|value| value.checked_add(metadata))
            .ok_or_else(|| invalid("expert preparation plane stride overflow"))?;
        let bytes = stride.checked_mul(params.n_experts as usize)
            .ok_or_else(|| invalid("expert preparation total extent overflow"))?;
        let gpu = GroupedM8PrepareGpuParams {
            n_experts: params.n_experts, n: params.n, k: params.k, format,
            source_expert_stride: params.expert_stride,
            prepared_expert_stride: stride as u64,
            plane1_offset: u32::try_from(low)
                .map_err(|_| invalid("expert preparation high-plane offset exceeds u32"))?,
            meta_offset: u32::try_from(low + high)
                .map_err(|_| invalid("expert preparation metadata offset exceeds u32"))?,
            source_row_bytes: u32::try_from(matrix / rows)
                .map_err(|_| invalid("expert preparation row bytes exceed u32"))?,
            _pad: 0,
        };
        Ok(Self { gpu, ggml_type: params.ggml_type, bytes, threads })
    }

    fn allocate(&self, device: &MlxDevice) -> Result<GroupedM8ExpertPlanes> {
        Ok(GroupedM8ExpertPlanes {
            bytes: device.alloc_buffer(self.bytes, DType::U8, vec![self.bytes])?,
            n_experts: self.gpu.n_experts, n: self.gpu.n, k: self.gpu.k,
            ggml_type: self.ggml_type, prepared_expert_stride: self.gpu.prepared_expert_stride,
            plane1_offset: self.gpu.plane1_offset, meta_offset: self.gpu.meta_offset,
        })
    }

    fn encode(
        &self,
        encoder: &mut CommandEncoder,
        pipeline: &metal::ComputePipelineStateRef,
        source: &MlxBuffer,
        planes: &GroupedM8ExpertPlanes,
    ) {
        encoder.encode_with_args(pipeline, &[
            (0, KernelArg::Buffer(source)),
            (1, KernelArg::Buffer(&planes.bytes)),
            (2, KernelArg::Bytes(as_bytes(&self.gpu))),
        ], metal::MTLSize::new(self.threads, 1, 1),
            metal::MTLSize::new(256.min(self.threads), 1, 1));
    }
}

pub(super) fn prepare_expert_planes(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    source: &MlxBuffer,
    params: &GgmlQuantizedMatmulIdParams,
) -> Result<GroupedM8ExpertPlanes> {
    let plan = PreparationPlan::new(device, source, params)?;
    require_tensor_kernel_family(registry, device, TensorKernelFamily::GroupedM8Experts)?;
    let pipeline = registry.get_pipeline("grouped_m8_prepare_q4k_q6k", device.metal_device())?;
    let planes = plan.allocate(device)?;
    plan.encode(encoder, pipeline, source, &planes);
    encoder.memory_barrier();
    Ok(planes)
}

struct BatchFlight {
    _sources: Vec<GroupedM8Preparation>,
    _destinations: Vec<Arc<GroupedM8ExpertPlanes>>,
}

/// Consume source owners and complete one batch of expert-plane preparations.
///
/// All requests are validated and all destinations allocated before encoding.
/// The batch has one preparation submission and checked wait; the existing
/// capability probe may perform its own independent initialization work.
/// No partially completed handles escape. A command-attached resource owner
/// covers both retained and unretained command buffers, including an unexpected
/// nonterminal wait result. No caller-provided completion assertion is accepted.
///
/// Successful completion does not retire aliases held elsewhere or flush
/// removals staged on a source's originating residency set.
pub fn prepare_quantized_matmul_id_ggml_grouped_m8_batch(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    requests: Vec<GroupedM8Preparation>,
) -> Result<Vec<CompletedGroupedM8Experts>> {
    if requests.is_empty() {
        return Ok(Vec::new());
    }
    let plans = requests.iter().map(|request| {
        PreparationPlan::new(device, &request.source, &request.params)
    }).collect::<Result<Vec<_>>>()?;
    require_tensor_kernel_family(registry, device, TensorKernelFamily::GroupedM8Experts)?;
    let pipeline = registry.get_pipeline("grouped_m8_prepare_q4k_q6k", device.metal_device())?;
    let planes = plans.iter().map(|plan| {
        plan.allocate(device).map(Arc::new)
    }).collect::<Result<Vec<_>>>()?;
    // Declare the encoder last: unsubmitted encoding is discarded before its
    // buffers on every early return or ordinary Rust unwind.
    let mut encoder = device.command_encoder()?;
    for ((plan, request), destination) in plans.iter().zip(&requests).zip(&planes) {
        plan.encode(&mut encoder, pipeline, &request.source, destination);
    }
    encoder.memory_barrier();
    let flight = Mutex::new(Some(BatchFlight {
        _sources: requests,
        _destinations: planes.clone(),
    }));
    let callback = move |_command: &metal::CommandBufferRef| {
        let owned = {
            let mut slot = flight.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            slot.take()
        };
        // Dispose outside the lock. Buffer drops stage residency removals.
        drop(owned);
    };
    fn require_send_sync<T: Send + Sync>(_: &T) {}
    require_send_sync(&callback);
    let handler = block::ConcreteBlock::new(callback).copy();
    encoder.add_completed_handler(&handler);
    let result = encoder.commit_and_wait_labeled("expert_planes.prepare_batch");
    // Retained command bindings must end too, before publishing completed
    // handles or reporting release of this batch's input ownership.
    drop(encoder);
    drop(handler);
    result?;
    Ok(planes.into_iter().map(|planes| CompletedGroupedM8Experts { planes }).collect())
}
