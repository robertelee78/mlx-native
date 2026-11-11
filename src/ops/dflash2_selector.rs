//! Fused DFlash2 top-16, pairwise edge scoring, and coherent path selection.

use crate::{
    buffer::MlxBuffer,
    device::MlxDevice,
    dtypes::DType,
    encoder::CommandEncoder,
    error::{MlxError, Result},
    kernel_registry::KernelRegistry,
};
use metal::MTLSize;

use super::encode_helpers::{as_bytes, encode_threadgroups_with_args, encode_with_args, KernelArg};

pub const DFLASH2_TOP16_KERNEL: &str = "dflash2_select_top16";
pub const DFLASH2_EDGES_KERNEL: &str = "dflash2_select_edges";
pub const DFLASH2_PATH_KERNEL: &str = "dflash2_select_path";
pub const DFLASH2_VERIFY_ROWS_KERNEL: &str = "dflash2_select_verify_rows";
pub const DFLASH2_COMPOSE_VERIFY_INPUT_KERNEL: &str = "dflash2_compose_verify_input";
pub static DFLASH2_SELECTOR_SOURCE: &str = include_str!("../shaders/dflash2_selector.metal");

pub const DFLASH2_MAX_LANES: usize = 4;
pub const DFLASH2_SHARDS: usize = 8;
pub const DFLASH2_CANDIDATES: usize = 16;
pub const DFLASH2_SELECTOR_RANK: usize = 256;
pub const DFLASH2_MAX_POSITIONS: usize = 15;
pub const DFLASH2_VERIFY_COSTS: usize = DFLASH2_MAX_LANES * (DFLASH2_MAX_POSITIONS + 1) + 1;

pub fn register(registry: &mut KernelRegistry) {
    for name in [
        DFLASH2_TOP16_KERNEL,
        DFLASH2_EDGES_KERNEL,
        DFLASH2_PATH_KERNEL,
        DFLASH2_VERIFY_ROWS_KERNEL,
        DFLASH2_COMPOSE_VERIFY_INPUT_KERNEL,
    ] {
        registry.register_source(name, DFLASH2_SELECTOR_SOURCE);
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DFlash2SelectorParams {
    pub anchor: [u32; DFLASH2_MAX_LANES],
    pub temperature: [f32; DFLASH2_MAX_LANES],
    pub lanes: u32,
    pub rows_per_lane: u32,
    pub vocabulary: u32,
    pub sampling_mask: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DFlash2VerifyRowsParams {
    pub slope: f32,
    pub intercept: [f32; DFLASH2_MAX_POSITIONS],
    /// Costs 0..64. The final value is a separate field because this
    /// bytemuck version implements Pod for arrays only through length 64.
    pub cost: [f32; DFLASH2_VERIFY_COSTS - 1],
    pub cost_64: f32,
    pub lanes: u32,
    pub rows_per_lane: u32,
    pub every_row: u32,
    pub maximum: [u32; DFLASH2_MAX_LANES],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DFlash2ComposeVerifyInputParams {
    pub anchor: [u32; DFLASH2_MAX_LANES],
    pub physical_lane: [u32; DFLASH2_MAX_LANES],
    pub rows: [u32; DFLASH2_MAX_LANES],
    pub physical_lanes: u32,
    pub lanes: u32,
    pub rows_per_lane: u32,
    pub width: u32,
}

pub struct DFlash2SelectorBuffers<'a> {
    pub logits: &'a MlxBuffer,
    pub selector_hidden: &'a MlxBuffer,
    pub predecessor_codebook: &'a MlxBuffer,
    pub successor_codebook: &'a MlxBuffer,
    pub uniforms: &'a MlxBuffer,
    pub partial_ids: &'a MlxBuffer,
    pub partial_values: &'a MlxBuffer,
    pub candidates: &'a MlxBuffer,
    pub unary: &'a MlxBuffer,
    pub edges: &'a MlxBuffer,
    pub tokens: &'a MlxBuffer,
    pub proposal_probabilities: &'a MlxBuffer,
    pub confidence: &'a MlxBuffer,
}

fn product(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1usize, |count, dim| {
        count
            .checked_mul(*dim)
            .ok_or_else(|| MlxError::InvalidArgument("DFlash2 selector shape overflow".into()))
    })
}

fn require(buffer: &MlxBuffer, dtype: DType, shape: &[usize], label: &str) -> Result<()> {
    if buffer.dtype() != dtype || buffer.shape() != shape {
        return Err(MlxError::InvalidArgument(format!(
            "DFlash2 selector {label} must be {dtype} {shape:?}, got {} {:?}",
            buffer.dtype(),
            buffer.shape()
        )));
    }
    let bytes = product(shape)?
        .checked_mul(dtype.size_of())
        .ok_or_else(|| MlxError::InvalidArgument("DFlash2 selector byte overflow".into()))?;
    if buffer.data_byte_len() != bytes {
        return Err(MlxError::InvalidArgument(format!(
            "DFlash2 selector {label} has {} data bytes, expected {bytes}",
            buffer.data_byte_len()
        )));
    }
    Ok(())
}

fn physical_range(buffer: &MlxBuffer) -> Result<(usize, usize)> {
    let start = (buffer.contents_ptr() as usize)
        .checked_add(
            usize::try_from(buffer.byte_offset())
                .map_err(|_| MlxError::InvalidArgument("DFlash2 buffer offset overflow".into()))?,
        )
        .ok_or_else(|| MlxError::InvalidArgument("DFlash2 buffer start overflow".into()))?;
    let end = start
        .checked_add(buffer.data_byte_len())
        .ok_or_else(|| MlxError::InvalidArgument("DFlash2 buffer end overflow".into()))?;
    Ok((start, end))
}

fn validate_disjoint(reads: &[&MlxBuffer], writes: &[&MlxBuffer]) -> Result<()> {
    let read_ranges = reads
        .iter()
        .map(|buffer| physical_range(buffer))
        .collect::<Result<Vec<_>>>()?;
    let write_ranges = writes
        .iter()
        .map(|buffer| physical_range(buffer))
        .collect::<Result<Vec<_>>>()?;
    for (write_index, write) in write_ranges.iter().enumerate() {
        for read in &read_ranges {
            if write.0 < read.1 && read.0 < write.1 {
                return Err(MlxError::InvalidArgument(
                    "DFlash2 selector writable buffers must not alias inputs".into(),
                ));
            }
        }
        for other in write_ranges.iter().skip(write_index + 1) {
            if write.0 < other.1 && other.0 < write.1 {
                return Err(MlxError::InvalidArgument(
                    "DFlash2 selector writable buffers must not alias each other".into(),
                ));
            }
        }
    }
    Ok(())
}

/// Select one coherent DFlash2 proposal path per lane. All workspace buffers
/// are caller-owned and reusable; this function performs no allocation.
pub fn dispatch_dflash2_selector(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    buffers: &DFlash2SelectorBuffers<'_>,
    params: DFlash2SelectorParams,
) -> Result<()> {
    let lanes = params.lanes as usize;
    let rows = params.rows_per_lane as usize;
    let vocabulary = params.vocabulary as usize;
    if lanes == 0
        || lanes > DFLASH2_MAX_LANES
        || !(2..=16).contains(&rows)
        || vocabulary < DFLASH2_CANDIDATES
        || params.sampling_mask >> lanes != 0
    {
        return Err(MlxError::InvalidArgument(format!(
            "invalid DFlash2 selector geometry: lanes={lanes}, rows={rows}, vocabulary={vocabulary}, sampling_mask={:#x}",
            params.sampling_mask
        )));
    }
    for lane in 0..lanes {
        if params.anchor[lane] as usize >= vocabulary {
            return Err(MlxError::InvalidArgument(format!(
                "DFlash2 anchor {} for lane {lane} exceeds vocabulary {vocabulary}",
                params.anchor[lane]
            )));
        }
        if params.sampling_mask & (1 << lane) != 0
            && (!params.temperature[lane].is_finite() || params.temperature[lane] <= 0.0)
        {
            return Err(MlxError::InvalidArgument(format!(
                "DFlash2 sampling temperature for lane {lane} must be finite and positive"
            )));
        }
    }
    let positions = rows - 1;
    require(
        buffers.logits,
        DType::F32,
        &[lanes, rows, vocabulary],
        "logits",
    )?;
    require(
        buffers.selector_hidden,
        DType::BF16,
        &[lanes, rows, DFLASH2_SELECTOR_RANK],
        "selector hidden",
    )?;
    for (buffer, label) in [
        (buffers.predecessor_codebook, "predecessor codebook"),
        (buffers.successor_codebook, "successor codebook"),
    ] {
        require(
            buffer,
            DType::BF16,
            &[vocabulary, DFLASH2_SELECTOR_RANK],
            label,
        )?;
    }
    require(
        buffers.uniforms,
        DType::F32,
        &[lanes, positions],
        "uniforms",
    )?;
    require(
        buffers.partial_ids,
        DType::U32,
        &[lanes, positions, DFLASH2_SHARDS, DFLASH2_CANDIDATES],
        "partial ids",
    )?;
    require(
        buffers.partial_values,
        DType::F32,
        &[lanes, positions, DFLASH2_SHARDS, DFLASH2_CANDIDATES],
        "partial values",
    )?;
    require(
        buffers.candidates,
        DType::U32,
        &[lanes, positions, DFLASH2_CANDIDATES],
        "candidates",
    )?;
    require(
        buffers.unary,
        DType::F32,
        &[lanes, positions, DFLASH2_CANDIDATES],
        "unary scores",
    )?;
    require(
        buffers.edges,
        DType::F32,
        &[lanes, positions, DFLASH2_CANDIDATES, DFLASH2_CANDIDATES],
        "edge scores",
    )?;
    require(buffers.tokens, DType::U32, &[lanes, positions], "tokens")?;
    require(
        buffers.proposal_probabilities,
        DType::F32,
        &[lanes, positions, DFLASH2_CANDIDATES],
        "proposal probabilities",
    )?;
    require(
        buffers.confidence,
        DType::F32,
        &[lanes, positions],
        "confidence",
    )?;

    let reads = [
        buffers.logits,
        buffers.selector_hidden,
        buffers.predecessor_codebook,
        buffers.successor_codebook,
        buffers.uniforms,
    ];
    let writes = [
        buffers.partial_ids,
        buffers.partial_values,
        buffers.candidates,
        buffers.unary,
        buffers.edges,
        buffers.tokens,
        buffers.proposal_probabilities,
        buffers.confidence,
    ];
    validate_disjoint(&reads, &writes)?;
    if encoder.is_capturing() {
        encoder.set_pending_buffer_ranges(
            reads
                .iter()
                .map(|buffer| physical_range(buffer))
                .collect::<Result<Vec<_>>>()?,
            writes
                .iter()
                .map(|buffer| physical_range(buffer))
                .collect::<Result<Vec<_>>>()?,
        );
    }

    let top16 = registry.get_pipeline(DFLASH2_TOP16_KERNEL, device.metal_device())?;
    encode_threadgroups_with_args(
        encoder,
        top16,
        &[
            (0, KernelArg::Buffer(buffers.logits)),
            (1, KernelArg::Buffer(buffers.partial_ids)),
            (2, KernelArg::Buffer(buffers.partial_values)),
            (3, KernelArg::Bytes(as_bytes(&params))),
        ],
        MTLSize::new((lanes * positions * DFLASH2_SHARDS) as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    encoder.memory_barrier();
    let edges = registry.get_pipeline(DFLASH2_EDGES_KERNEL, device.metal_device())?;
    encode_threadgroups_with_args(
        encoder,
        edges,
        &[
            (0, KernelArg::Buffer(buffers.partial_ids)),
            (1, KernelArg::Buffer(buffers.partial_values)),
            (2, KernelArg::Buffer(buffers.candidates)),
            (3, KernelArg::Buffer(buffers.unary)),
            (4, KernelArg::Buffer(buffers.edges)),
            (5, KernelArg::Buffer(buffers.selector_hidden)),
            (6, KernelArg::Buffer(buffers.predecessor_codebook)),
            (7, KernelArg::Buffer(buffers.successor_codebook)),
            (8, KernelArg::Bytes(as_bytes(&params))),
        ],
        MTLSize::new((lanes * positions) as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    encoder.memory_barrier();
    let path = registry.get_pipeline(DFLASH2_PATH_KERNEL, device.metal_device())?;
    encode_with_args(
        encoder,
        path,
        &[
            (0, KernelArg::Buffer(buffers.candidates)),
            (1, KernelArg::Buffer(buffers.unary)),
            (2, KernelArg::Buffer(buffers.edges)),
            (3, KernelArg::Buffer(buffers.uniforms)),
            (4, KernelArg::Buffer(buffers.tokens)),
            (5, KernelArg::Buffer(buffers.proposal_probabilities)),
            (6, KernelArg::Buffer(buffers.confidence)),
            (7, KernelArg::Bytes(as_bytes(&params))),
        ],
        MTLSize::new(lanes as u64, 1, 1),
        MTLSize::new(1, 1, 1),
    );
    Ok(())
}

/// Choose the target verification prefix per lane from selector confidence.
/// `every_row=1` is the correctness-first path and ignores the empirical model.
pub fn dispatch_dflash2_verify_rows(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    confidence: &MlxBuffer,
    verify_rows: &MlxBuffer,
    params: DFlash2VerifyRowsParams,
) -> Result<()> {
    let lanes = params.lanes as usize;
    let rows = params.rows_per_lane as usize;
    if lanes == 0
        || lanes > DFLASH2_MAX_LANES
        || !(2..=DFLASH2_MAX_POSITIONS + 1).contains(&rows)
        || params.every_row > 1
    {
        return Err(MlxError::InvalidArgument(format!(
            "invalid DFlash2 verify-row geometry: lanes={lanes}, rows={rows}, every_row={}",
            params.every_row
        )));
    }
    let mut maximum_total = 0usize;
    for lane in 0..lanes {
        let maximum = params.maximum[lane] as usize;
        if maximum == 0 || maximum > rows {
            return Err(MlxError::InvalidArgument(format!(
                "DFlash2 maximum verify rows {} for lane {lane} must be in 1..={rows}",
                params.maximum[lane]
            )));
        }
        maximum_total = maximum_total
            .checked_add(maximum)
            .ok_or_else(|| MlxError::InvalidArgument("DFlash2 verify-row total overflow".into()))?;
    }
    if params.every_row == 0 {
        let cost = |index: usize| {
            if index < DFLASH2_VERIFY_COSTS - 1 {
                params.cost[index]
            } else {
                params.cost_64
            }
        };
        if !params.slope.is_finite()
            || params.intercept[..rows - 1]
                .iter()
                .any(|value| !value.is_finite())
            || (lanes..=maximum_total)
                .map(cost)
                .any(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(MlxError::InvalidArgument(
                "DFlash2 verify-row model must have finite slope/intercepts and finite positive reachable costs"
                    .into(),
            ));
        }
    }
    require(confidence, DType::F32, &[lanes, rows - 1], "confidence")?;
    require(verify_rows, DType::U32, &[lanes], "verify rows")?;
    validate_disjoint(&[confidence], &[verify_rows])?;
    if encoder.is_capturing() {
        encoder.set_pending_buffer_ranges(
            vec![physical_range(confidence)?],
            vec![physical_range(verify_rows)?],
        );
    }
    let pipeline = registry.get_pipeline(DFLASH2_VERIFY_ROWS_KERNEL, device.metal_device())?;
    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(confidence)),
            (1, KernelArg::Buffer(verify_rows)),
            (2, KernelArg::Bytes(as_bytes(&params))),
        ],
        MTLSize::new(1, 1, 1),
        MTLSize::new(1, 1, 1),
    );
    Ok(())
}

/// Compose the fixed-width target token tile directly from GPU-selected
/// proposals. Logical lanes may name any distinct physical draft lanes;
/// unused target rows replay logical lane zero's anchor.
pub fn dispatch_dflash2_compose_verify_input(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    proposals: &MlxBuffer,
    target_tokens: &MlxBuffer,
    params: DFlash2ComposeVerifyInputParams,
) -> Result<()> {
    let physical_lanes = params.physical_lanes as usize;
    let lanes = params.lanes as usize;
    let rows_per_lane = params.rows_per_lane as usize;
    let width = params.width as usize;
    if physical_lanes == 0
        || physical_lanes > DFLASH2_MAX_LANES
        || lanes == 0
        || lanes > physical_lanes
        || !(2..=DFLASH2_MAX_POSITIONS + 1).contains(&rows_per_lane)
        || width == 0
        || width > DFLASH2_MAX_LANES * (DFLASH2_MAX_POSITIONS + 1)
    {
        return Err(MlxError::InvalidArgument(format!(
            "invalid DFlash2 verify-input geometry: physical_lanes={physical_lanes}, lanes={lanes}, rows_per_lane={rows_per_lane}, width={width}"
        )));
    }
    let mut total = 0usize;
    for lane in 0..lanes {
        let physical = params.physical_lane[lane] as usize;
        let rows = params.rows[lane] as usize;
        if physical >= physical_lanes
            || params.physical_lane[..lane].contains(&params.physical_lane[lane])
            || rows == 0
            || rows > rows_per_lane
        {
            return Err(MlxError::InvalidArgument(format!(
                "invalid DFlash2 logical lane {lane}: physical={physical}, rows={rows}"
            )));
        }
        total = total.checked_add(rows).ok_or_else(|| {
            MlxError::InvalidArgument("DFlash2 verify-input row total overflow".into())
        })?;
    }
    if total > width {
        return Err(MlxError::InvalidArgument(format!(
            "DFlash2 verify-input rows {total} exceed target width {width}"
        )));
    }
    require(
        proposals,
        DType::U32,
        &[physical_lanes, rows_per_lane - 1],
        "proposals",
    )?;
    require(target_tokens, DType::U32, &[width], "target tokens")?;
    validate_disjoint(&[proposals], &[target_tokens])?;
    if encoder.is_capturing() {
        encoder.set_pending_buffer_ranges(
            vec![physical_range(proposals)?],
            vec![physical_range(target_tokens)?],
        );
    }
    let pipeline =
        registry.get_pipeline(DFLASH2_COMPOSE_VERIFY_INPUT_KERNEL, device.metal_device())?;
    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(proposals)),
            (1, KernelArg::Buffer(target_tokens)),
            (2, KernelArg::Bytes(as_bytes(&params))),
        ],
        MTLSize::new(width as u64, 1, 1),
        MTLSize::new(1, 1, 1),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_path(
        candidates: &[u32],
        unary: &[f32],
        edges: &[f32],
        positions: usize,
    ) -> (Vec<u32>, Vec<f32>) {
        let mut predecessor = 0;
        let mut tokens = Vec::with_capacity(positions);
        let mut confidence = Vec::with_capacity(positions);
        for position in 0..positions {
            let mut scores = [0.0; DFLASH2_CANDIDATES];
            let mut best = f32::NEG_INFINITY;
            let mut selected = 0;
            for candidate in 0..DFLASH2_CANDIDATES {
                scores[candidate] = unary[position * DFLASH2_CANDIDATES + candidate]
                    + edges[(position * DFLASH2_CANDIDATES + predecessor) * DFLASH2_CANDIDATES
                        + candidate];
                best = best.max(scores[candidate]);
                if scores[candidate] > scores[selected] {
                    selected = candidate;
                }
            }
            confidence.push(
                1.0 / scores
                    .iter()
                    .map(|score| (*score - best).exp())
                    .sum::<f32>(),
            );
            predecessor = selected;
            tokens.push(candidates[position * DFLASH2_CANDIDATES + selected]);
        }
        (tokens, confidence)
    }

    fn alloc(device: &MlxDevice, dtype: DType, shape: &[usize]) -> MlxBuffer {
        let elements: usize = shape.iter().product();
        device
            .alloc_buffer(elements * dtype.size_of(), dtype, shape.to_vec())
            .unwrap()
    }

    fn bf16(value: f32) -> u16 {
        (value.to_bits() >> 16) as u16
    }

    #[test]
    fn parameter_abis_match_metal_layouts() {
        assert_eq!(std::mem::size_of::<DFlash2SelectorParams>(), 48);
        assert_eq!(std::mem::size_of::<DFlash2VerifyRowsParams>(), 352);
        assert_eq!(std::mem::size_of::<DFlash2ComposeVerifyInputParams>(), 64);
    }

    #[test]
    fn coherent_path_uses_selected_predecessor_edges() {
        let positions = 2;
        let candidates: Vec<u32> = (0..positions * DFLASH2_CANDIDATES)
            .map(|index| index as u32)
            .collect();
        let mut unary = vec![0.0; positions * DFLASH2_CANDIDATES];
        unary[3] = 4.0;
        unary[DFLASH2_CANDIDATES + 2] = 3.0;
        let mut edges = vec![0.0; positions * DFLASH2_CANDIDATES.pow(2)];
        edges[(DFLASH2_CANDIDATES + 3) * DFLASH2_CANDIDATES + 7] = 10.0;
        let (tokens, confidence) = reference_path(&candidates, &unary, &edges, positions);
        assert_eq!(tokens, [3, 23]);
        assert!(confidence[0] > 0.75);
        assert!(confidence[1] > 0.99);
    }

    #[test]
    fn gpu_selector_walks_the_scored_path() {
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        register(&mut registry);
        let lanes = 1usize;
        let rows = 3usize;
        let positions = rows - 1;
        let vocabulary = 32usize;
        let mut logits = alloc(&device, DType::F32, &[lanes, rows, vocabulary]);
        logits.as_mut_slice::<f32>().unwrap().fill(-10.0);
        logits.as_mut_slice::<f32>().unwrap()[vocabulary + 3] = 4.0;
        logits.as_mut_slice::<f32>().unwrap()[2 * vocabulary + 2] = 3.0;
        logits.as_mut_slice::<f32>().unwrap()[2 * vocabulary + 7] = 2.0;
        let mut hidden = alloc(&device, DType::BF16, &[lanes, rows, DFLASH2_SELECTOR_RANK]);
        hidden.as_mut_slice::<u16>().unwrap()[2 * DFLASH2_SELECTOR_RANK] = bf16(1.0);
        let mut predecessor = alloc(&device, DType::BF16, &[vocabulary, DFLASH2_SELECTOR_RANK]);
        predecessor.as_mut_slice::<u16>().unwrap()[3 * DFLASH2_SELECTOR_RANK] = bf16(1.0);
        let mut successor = alloc(&device, DType::BF16, &[vocabulary, DFLASH2_SELECTOR_RANK]);
        successor.as_mut_slice::<u16>().unwrap()[7 * DFLASH2_SELECTOR_RANK] = bf16(10.0);
        let buffers = DFlash2SelectorBuffers {
            logits: &logits,
            selector_hidden: &hidden,
            predecessor_codebook: &predecessor,
            successor_codebook: &successor,
            uniforms: &alloc(&device, DType::F32, &[lanes, positions]),
            partial_ids: &alloc(
                &device,
                DType::U32,
                &[lanes, positions, DFLASH2_SHARDS, DFLASH2_CANDIDATES],
            ),
            partial_values: &alloc(
                &device,
                DType::F32,
                &[lanes, positions, DFLASH2_SHARDS, DFLASH2_CANDIDATES],
            ),
            candidates: &alloc(&device, DType::U32, &[lanes, positions, DFLASH2_CANDIDATES]),
            unary: &alloc(&device, DType::F32, &[lanes, positions, DFLASH2_CANDIDATES]),
            edges: &alloc(
                &device,
                DType::F32,
                &[lanes, positions, DFLASH2_CANDIDATES, DFLASH2_CANDIDATES],
            ),
            tokens: &alloc(&device, DType::U32, &[lanes, positions]),
            proposal_probabilities: &alloc(
                &device,
                DType::F32,
                &[lanes, positions, DFLASH2_CANDIDATES],
            ),
            confidence: &alloc(&device, DType::F32, &[lanes, positions]),
        };
        let mut encoder = device.command_encoder().unwrap();
        dispatch_dflash2_selector(
            &mut encoder,
            &mut registry,
            &device,
            &buffers,
            DFlash2SelectorParams {
                anchor: [0; DFLASH2_MAX_LANES],
                temperature: [1.0; DFLASH2_MAX_LANES],
                lanes: lanes as u32,
                rows_per_lane: rows as u32,
                vocabulary: vocabulary as u32,
                sampling_mask: 0,
            },
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();
        assert_eq!(buffers.tokens.as_slice::<u32>().unwrap(), [3, 7]);
        let confidence = buffers.confidence.as_slice::<f32>().unwrap();
        assert!(confidence[0] > 0.99);
        assert!(confidence[1] > 0.99);
    }

    #[test]
    fn gpu_top16_covers_vectorized_shards_and_token_ties() {
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        register(&mut registry);
        let (lanes, rows, vocabulary) = (1usize, 2usize, 1_031usize);
        let positions = rows - 1;
        let mut logits = alloc(&device, DType::F32, &[lanes, rows, vocabulary]);
        logits.as_mut_slice::<f32>().unwrap().fill(-100.0);
        let row = &mut logits.as_mut_slice::<f32>().unwrap()[vocabulary..];
        for (token, value) in [
            (1_000, 50.0),
            (7, 50.0),
            (129, 49.0),
            (257, 48.0),
            (385, 47.0),
            (513, 46.0),
            (641, 45.0),
            (769, 44.0),
            (897, 43.0),
            (1_029, 42.0),
            (3, 41.0),
            (131, 40.0),
            (259, 39.0),
            (387, 38.0),
            (515, 37.0),
            (643, 36.0),
            (771, 35.0),
        ] {
            row[token] = value;
        }
        let mut expected = (0..vocabulary as u32).collect::<Vec<_>>();
        expected.sort_unstable_by(|left, right| {
            row[*right as usize]
                .total_cmp(&row[*left as usize])
                .then_with(|| left.cmp(right))
        });
        expected.truncate(DFLASH2_CANDIDATES);

        let hidden = alloc(&device, DType::BF16, &[lanes, rows, DFLASH2_SELECTOR_RANK]);
        let predecessor = alloc(&device, DType::BF16, &[vocabulary, DFLASH2_SELECTOR_RANK]);
        let successor = alloc(&device, DType::BF16, &[vocabulary, DFLASH2_SELECTOR_RANK]);
        let uniforms = alloc(&device, DType::F32, &[lanes, positions]);
        let partial_ids = alloc(
            &device,
            DType::U32,
            &[lanes, positions, DFLASH2_SHARDS, DFLASH2_CANDIDATES],
        );
        let partial_values = alloc(
            &device,
            DType::F32,
            &[lanes, positions, DFLASH2_SHARDS, DFLASH2_CANDIDATES],
        );
        let candidates = alloc(&device, DType::U32, &[lanes, positions, DFLASH2_CANDIDATES]);
        let unary = alloc(&device, DType::F32, &[lanes, positions, DFLASH2_CANDIDATES]);
        let edges = alloc(
            &device,
            DType::F32,
            &[lanes, positions, DFLASH2_CANDIDATES, DFLASH2_CANDIDATES],
        );
        let tokens = alloc(&device, DType::U32, &[lanes, positions]);
        let proposal_probabilities =
            alloc(&device, DType::F32, &[lanes, positions, DFLASH2_CANDIDATES]);
        let confidence = alloc(&device, DType::F32, &[lanes, positions]);
        let buffers = DFlash2SelectorBuffers {
            logits: &logits,
            selector_hidden: &hidden,
            predecessor_codebook: &predecessor,
            successor_codebook: &successor,
            uniforms: &uniforms,
            partial_ids: &partial_ids,
            partial_values: &partial_values,
            candidates: &candidates,
            unary: &unary,
            edges: &edges,
            tokens: &tokens,
            proposal_probabilities: &proposal_probabilities,
            confidence: &confidence,
        };
        let mut encoder = device.command_encoder().unwrap();
        dispatch_dflash2_selector(
            &mut encoder,
            &mut registry,
            &device,
            &buffers,
            DFlash2SelectorParams {
                anchor: [0; DFLASH2_MAX_LANES],
                temperature: [1.0; DFLASH2_MAX_LANES],
                lanes: lanes as u32,
                rows_per_lane: rows as u32,
                vocabulary: vocabulary as u32,
                sampling_mask: 0,
            },
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();
        assert_eq!(candidates.as_slice::<u32>().unwrap(), expected);
        assert_eq!(tokens.as_slice::<u32>().unwrap(), [7]);
    }

    #[test]
    fn gpu_verify_rows_every_row_returns_declared_maximum() {
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        register(&mut registry);
        let confidence = alloc(&device, DType::F32, &[2, 7]);
        let verify_rows = alloc(&device, DType::U32, &[2]);
        let mut maximum = [0; DFLASH2_MAX_LANES];
        maximum[..2].copy_from_slice(&[8, 5]);
        let mut encoder = device.command_encoder().unwrap();
        dispatch_dflash2_verify_rows(
            &mut encoder,
            &mut registry,
            &device,
            &confidence,
            &verify_rows,
            DFlash2VerifyRowsParams {
                slope: 0.0,
                intercept: [0.0; DFLASH2_MAX_POSITIONS],
                cost: [0.0; DFLASH2_VERIFY_COSTS - 1],
                cost_64: 0.0,
                lanes: 2,
                rows_per_lane: 8,
                every_row: 1,
                maximum,
            },
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();
        assert_eq!(verify_rows.as_slice::<u32>().unwrap(), [8, 5]);
    }

    #[test]
    fn gpu_verify_input_compacts_physical_lanes_and_pads_with_anchor() {
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        register(&mut registry);
        let mut proposals = alloc(&device, DType::U32, &[4, 7]);
        proposals
            .as_mut_slice::<u32>()
            .unwrap()
            .copy_from_slice(&(100..128).collect::<Vec<_>>());
        let target_tokens = alloc(&device, DType::U32, &[12]);
        let mut encoder = device.command_encoder().unwrap();
        dispatch_dflash2_compose_verify_input(
            &mut encoder,
            &mut registry,
            &device,
            &proposals,
            &target_tokens,
            DFlash2ComposeVerifyInputParams {
                anchor: [11, 22, 0, 0],
                physical_lane: [2, 0, 0, 0],
                rows: [4, 2, 0, 0],
                physical_lanes: 4,
                lanes: 2,
                rows_per_lane: 8,
                width: 12,
            },
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();
        assert_eq!(
            target_tokens.as_slice::<u32>().unwrap(),
            [11, 114, 115, 116, 22, 100, 11, 11, 11, 11, 11, 11]
        );
    }
}
