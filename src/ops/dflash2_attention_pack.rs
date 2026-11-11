//! DFlash2 sliding-window attention packing and transactional KV-ring writes.

use crate::{
    buffer::MlxBuffer,
    device::MlxDevice,
    dtypes::DType,
    encoder::CommandEncoder,
    error::{MlxError, Result},
    kernel_registry::KernelRegistry,
};
use metal::MTLSize;

use super::encode_helpers::{as_bytes, encode_with_args, KernelArg};

pub const PACK_KERNEL: &str = "dflash2_pack_attention_bf16";
pub const MASK_KERNEL: &str = "dflash2_fill_attention_mask_bf16";
pub const COMMIT_KERNEL: &str = "dflash2_commit_kv_bf16";
pub const MASK_ROWS_KERNEL: &str = "dflash2_set_mask_rows_f32";
pub const MASK_MOE_ROUTES_KERNEL: &str = "dflash2_mask_moe_routes";
pub static SOURCE: &str = include_str!("../shaders/dflash2_attention_pack.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source(PACK_KERNEL, SOURCE);
    registry.register_source(MASK_KERNEL, SOURCE);
    registry.register_source(COMMIT_KERNEL, SOURCE);
    registry.register_source(MASK_ROWS_KERNEL, SOURCE);
    registry.register_source(MASK_MOE_ROUTES_KERNEL, SOURCE);
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DFlash2AttentionPackParams {
    pub lanes: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub rows: u32,
    pub window: u32,
    pub packed_len: u32,
    pub max_commit: u32,
}

impl DFlash2AttentionPackParams {
    pub fn validate(self) -> Result<Self> {
        if self.lanes == 0
            || self.lanes > 4
            || self.q_heads == 0
            || self.kv_heads == 0
            || self.q_heads % self.kv_heads != 0
            || !matches!(self.head_dim, 128 | 256)
            || self.rows < 2
            || self.rows > 16
            || self.window <= self.rows
            || self.packed_len != self.window - 1 + self.rows
            || self.max_commit == 0
            || self.max_commit > self.rows
        {
            return Err(MlxError::InvalidArgument(format!(
                "invalid DFlash2 attention geometry: {self:?}"
            )));
        }
        Ok(self)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DFlash2MaskRowsParams {
    pub lanes: u32,
    pub rows: u32,
    pub hidden: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DFlash2MoeRouteMaskParams {
    /// Physical draft-ring lane for each packed target lane.
    pub physical_lane: [u32; 4],
    pub lanes: u32,
    pub rows_per_lane: u32,
    pub top_k: u32,
    pub n_experts: u32,
    pub width: u32,
}

fn count(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |n, d| {
        n.checked_mul(*d)
            .ok_or_else(|| MlxError::InvalidArgument("DFlash2 attention shape overflow".into()))
    })
}

fn require(buffer: &MlxBuffer, dtype: DType, shape: &[usize], label: &str) -> Result<()> {
    let bytes = count(shape)?
        .checked_mul(dtype.size_of())
        .ok_or_else(|| MlxError::InvalidArgument("DFlash2 attention byte overflow".into()))?;
    if buffer.dtype() != dtype || buffer.shape() != shape || buffer.data_byte_len() != bytes {
        return Err(MlxError::InvalidArgument(format!(
            "DFlash2 attention {label} must be {dtype} {shape:?} ({bytes} bytes), got {} {:?} ({} bytes)",
            buffer.dtype(),
            buffer.shape(),
            buffer.data_byte_len()
        )));
    }
    Ok(())
}

fn geometry(p: DFlash2AttentionPackParams) -> (usize, usize, usize, usize, usize, usize, usize) {
    (
        p.lanes as usize,
        p.q_heads as usize,
        p.kv_heads as usize,
        p.head_dim as usize,
        p.rows as usize,
        p.window as usize,
        p.packed_len as usize,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn dispatch_dflash2_pack_attention_bf16(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    ring_keys: &MlxBuffer,
    ring_values: &MlxBuffer,
    current_queries: &MlxBuffer,
    current_keys: &MlxBuffer,
    current_values: &MlxBuffer,
    cache_lengths: &MlxBuffer,
    write_positions: &MlxBuffer,
    packed_queries: &MlxBuffer,
    packed_keys: &MlxBuffer,
    packed_values: &MlxBuffer,
    params: DFlash2AttentionPackParams,
) -> Result<()> {
    let p = params.validate()?;
    let (b, h, hk, d, r, w, k) = geometry(p);
    require(ring_keys, DType::BF16, &[b, hk, w, d], "ring keys")?;
    require(ring_values, DType::BF16, &[b, hk, w, d], "ring values")?;
    require(
        current_queries,
        DType::F32,
        &[b, r, h, d],
        "current queries",
    )?;
    require(current_keys, DType::F32, &[b, r, hk, d], "current keys")?;
    require(current_values, DType::F32, &[b, r, hk, d], "current values")?;
    require(cache_lengths, DType::U32, &[b], "cache lengths")?;
    require(write_positions, DType::U32, &[b], "write positions")?;
    require(packed_queries, DType::BF16, &[b, h, r, d], "packed queries")?;
    require(packed_keys, DType::BF16, &[b, hk, k, d], "packed keys")?;
    require(packed_values, DType::BF16, &[b, hk, k, d], "packed values")?;
    let elements = count(&[b, hk, k, d])?.max(count(&[b, h, r, d])?);
    let pipeline = registry.get_pipeline(PACK_KERNEL, device.metal_device())?;
    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(ring_keys)),
            (1, KernelArg::Buffer(ring_values)),
            (2, KernelArg::Buffer(current_queries)),
            (3, KernelArg::Buffer(current_keys)),
            (4, KernelArg::Buffer(current_values)),
            (5, KernelArg::Buffer(cache_lengths)),
            (6, KernelArg::Buffer(write_positions)),
            (7, KernelArg::Buffer(packed_queries)),
            (8, KernelArg::Buffer(packed_keys)),
            (9, KernelArg::Buffer(packed_values)),
            (10, KernelArg::Bytes(as_bytes(&p))),
        ],
        MTLSize::new(elements as u64, 1, 1),
        MTLSize::new(256.min(elements) as u64, 1, 1),
    );
    Ok(())
}

pub fn dispatch_dflash2_attention_mask_bf16(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    cache_lengths: &MlxBuffer,
    mask: &MlxBuffer,
    params: DFlash2AttentionPackParams,
) -> Result<()> {
    let p = params.validate()?;
    let (b, h, _, _, r, _, k) = geometry(p);
    require(cache_lengths, DType::U32, &[b], "cache lengths")?;
    require(mask, DType::BF16, &[b, h, r, k], "mask")?;
    let elements = count(&[b, h, r, k])?;
    let pipeline = registry.get_pipeline(MASK_KERNEL, device.metal_device())?;
    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(cache_lengths)),
            (1, KernelArg::Buffer(mask)),
            (2, KernelArg::Bytes(as_bytes(&p))),
        ],
        MTLSize::new(elements as u64, 1, 1),
        MTLSize::new(256.min(elements) as u64, 1, 1),
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn dispatch_dflash2_commit_kv_bf16(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    new_keys: &MlxBuffer,
    new_values: &MlxBuffer,
    commit_counts: &MlxBuffer,
    write_positions: &MlxBuffer,
    ring_keys: &MlxBuffer,
    ring_values: &MlxBuffer,
    params: DFlash2AttentionPackParams,
) -> Result<()> {
    let p = params.validate()?;
    let (b, _, hk, d, _, w, _) = geometry(p);
    let c = p.max_commit as usize;
    require(new_keys, DType::F32, &[b, c, hk, d], "new keys")?;
    require(new_values, DType::F32, &[b, c, hk, d], "new values")?;
    require(commit_counts, DType::U32, &[b], "commit counts")?;
    require(write_positions, DType::U32, &[b], "write positions")?;
    require(ring_keys, DType::BF16, &[b, hk, w, d], "ring keys")?;
    require(ring_values, DType::BF16, &[b, hk, w, d], "ring values")?;
    let elements = count(&[b, c, hk, d])?;
    let pipeline = registry.get_pipeline(COMMIT_KERNEL, device.metal_device())?;
    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(new_keys)),
            (1, KernelArg::Buffer(new_values)),
            (2, KernelArg::Buffer(commit_counts)),
            (3, KernelArg::Buffer(write_positions)),
            (4, KernelArg::Buffer(ring_keys)),
            (5, KernelArg::Buffer(ring_values)),
            (6, KernelArg::Bytes(as_bytes(&p))),
        ],
        MTLSize::new(elements as u64, 1, 1),
        MTLSize::new(256.min(elements) as u64, 1, 1),
    );
    Ok(())
}

pub fn dispatch_dflash2_mask_rows_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    mask_embedding: &MlxBuffer,
    hidden: &MlxBuffer,
    params: DFlash2MaskRowsParams,
) -> Result<()> {
    if params.lanes == 0
        || params.lanes > 4
        || params.rows < 2
        || params.rows > 16
        || params.hidden == 0
    {
        return Err(MlxError::InvalidArgument(format!(
            "invalid DFlash2 mask-row geometry: {params:?}"
        )));
    }
    let lanes = params.lanes as usize;
    let rows = params.rows as usize;
    let width = params.hidden as usize;
    require(mask_embedding, DType::BF16, &[width], "mask embedding")?;
    require(hidden, DType::F32, &[lanes * rows, width], "draft input")?;
    let elements = count(&[lanes, rows - 1, width])?;
    let pipeline = registry.get_pipeline(MASK_ROWS_KERNEL, device.metal_device())?;
    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(mask_embedding)),
            (1, KernelArg::Buffer(hidden)),
            (2, KernelArg::Bytes(as_bytes(&params))),
        ],
        MTLSize::new(elements as u64, 1, 1),
        MTLSize::new(256.min(elements) as u64, 1, 1),
    );
    Ok(())
}

pub fn dispatch_dflash2_mask_moe_routes(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    verify_rows: &MlxBuffer,
    ids: &MlxBuffer,
    routing_weights: &MlxBuffer,
    params: DFlash2MoeRouteMaskParams,
) -> Result<()> {
    if params.lanes == 0
        || params.lanes > 4
        || params.rows_per_lane == 0
        || params.rows_per_lane > 16
        || params.top_k == 0
        || params.n_experts == 0
        || params.width < params.lanes * params.rows_per_lane
        || params.width > 32
    {
        return Err(MlxError::InvalidArgument(format!(
            "invalid DFlash2 MoE route-mask geometry: {params:?}"
        )));
    }
    let physical_lanes = verify_rows.element_count();
    if verify_rows.dtype() != DType::U32
        || verify_rows.shape() != [physical_lanes]
        || params.physical_lane[..params.lanes as usize]
            .iter()
            .any(|&lane| lane as usize >= physical_lanes)
    {
        return Err(MlxError::InvalidArgument(format!(
            "DFlash2 MoE verify rows must be a U32 vector covering physical lanes {:?}, got {} {:?}",
            &params.physical_lane[..params.lanes as usize],
            verify_rows.dtype(),
            verify_rows.shape()
        )));
    }
    let pairs = count(&[params.width as usize, params.top_k as usize])?;
    require(ids, DType::U32, &[pairs], "MoE expert ids")?;
    require(routing_weights, DType::F32, &[pairs], "MoE routing weights")?;
    let pipeline = registry.get_pipeline(MASK_MOE_ROUTES_KERNEL, device.metal_device())?;
    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(verify_rows)),
            (1, KernelArg::Buffer(ids)),
            (2, KernelArg::Buffer(routing_weights)),
            (3, KernelArg::Bytes(as_bytes(&params))),
        ],
        MTLSize::new(pairs as u64, 1, 1),
        MTLSize::new(256.min(pairs) as u64, 1, 1),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_fixed_packed_length() {
        let p = DFlash2AttentionPackParams {
            lanes: 4,
            q_heads: 32,
            kv_heads: 8,
            head_dim: 128,
            rows: 8,
            window: 2048,
            packed_len: 2055,
            max_commit: 8,
        };
        assert!(p.validate().is_ok());
        assert!(DFlash2AttentionPackParams {
            packed_len: 2048,
            ..p
        }
        .validate()
        .is_err());
    }

    fn bf16(value: f32) -> u16 {
        (value.to_bits() >> 16) as u16
    }

    #[test]
    fn gpu_pack_mask_and_commit_match_ring_semantics() {
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        register(&mut registry);
        let p = DFlash2AttentionPackParams {
            lanes: 1,
            q_heads: 1,
            kv_heads: 1,
            head_dim: 128,
            rows: 2,
            window: 4,
            packed_len: 5,
            max_commit: 2,
        };
        let mut ring_k = device
            .alloc_buffer(4 * 128 * 2, DType::BF16, vec![1, 1, 4, 128])
            .unwrap();
        let mut ring_v = device
            .alloc_buffer(4 * 128 * 2, DType::BF16, vec![1, 1, 4, 128])
            .unwrap();
        for slot in 0..4 {
            ring_k.as_mut_slice::<u16>().unwrap()[slot * 128] = bf16((slot + 1) as f32);
            ring_v.as_mut_slice::<u16>().unwrap()[slot * 128] = bf16((slot + 11) as f32);
        }
        let mut current_q = device
            .alloc_buffer(2 * 128 * 4, DType::F32, vec![1, 2, 1, 128])
            .unwrap();
        let mut current_k = device
            .alloc_buffer(2 * 128 * 4, DType::F32, vec![1, 2, 1, 128])
            .unwrap();
        let mut current_v = device
            .alloc_buffer(2 * 128 * 4, DType::F32, vec![1, 2, 1, 128])
            .unwrap();
        current_q.as_mut_slice::<f32>().unwrap()[0] = 40.0;
        current_k.as_mut_slice::<f32>().unwrap()[0] = 20.0;
        current_k.as_mut_slice::<f32>().unwrap()[128] = 21.0;
        current_v.as_mut_slice::<f32>().unwrap()[0] = 30.0;
        current_v.as_mut_slice::<f32>().unwrap()[128] = 31.0;
        let mut lengths = device.alloc_buffer(4, DType::U32, vec![1]).unwrap();
        lengths.as_mut_slice::<u32>().unwrap()[0] = 3;
        let mut positions = device.alloc_buffer(4, DType::U32, vec![1]).unwrap();
        positions.as_mut_slice::<u32>().unwrap()[0] = 1;
        let packed_q = device
            .alloc_buffer(2 * 128 * 2, DType::BF16, vec![1, 1, 2, 128])
            .unwrap();
        let packed_k = device
            .alloc_buffer(5 * 128 * 2, DType::BF16, vec![1, 1, 5, 128])
            .unwrap();
        let packed_v = device
            .alloc_buffer(5 * 128 * 2, DType::BF16, vec![1, 1, 5, 128])
            .unwrap();
        let mask = device
            .alloc_buffer(2 * 5 * 2, DType::BF16, vec![1, 1, 2, 5])
            .unwrap();
        let mut encoder = device.command_encoder().unwrap();
        dispatch_dflash2_pack_attention_bf16(
            &mut encoder,
            &mut registry,
            &device,
            &ring_k,
            &ring_v,
            &current_q,
            &current_k,
            &current_v,
            &lengths,
            &positions,
            &packed_q,
            &packed_k,
            &packed_v,
            p,
        )
        .unwrap();
        encoder.memory_barrier();
        dispatch_dflash2_attention_mask_bf16(
            &mut encoder,
            &mut registry,
            &device,
            &lengths,
            &mask,
            p,
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();
        assert_eq!(packed_q.as_slice::<u16>().unwrap()[0], bf16(40.0));
        let keys = packed_k.as_slice::<u16>().unwrap();
        assert_eq!(
            [keys[0], keys[128], keys[256], keys[384], keys[512]],
            [bf16(20.0), bf16(21.0), bf16(1.0), bf16(4.0), bf16(3.0)]
        );
        let mask = mask.as_slice::<u16>().unwrap();
        assert!(mask[..9].iter().all(|value| *value == bf16(0.0)));
        assert_eq!(mask[9], bf16(f32::NEG_INFINITY));

        let mut counts = device.alloc_buffer(4, DType::U32, vec![1]).unwrap();
        counts.as_mut_slice::<u32>().unwrap()[0] = 2;
        let mut encoder = device.command_encoder().unwrap();
        dispatch_dflash2_commit_kv_bf16(
            &mut encoder,
            &mut registry,
            &device,
            &current_k,
            &current_v,
            &counts,
            &positions,
            &ring_k,
            &ring_v,
            p,
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();
        let ring = ring_k.as_slice::<u16>().unwrap();
        assert_eq!(ring[128], bf16(20.0));
        assert_eq!(ring[256], bf16(21.0));
    }

    #[test]
    fn gpu_mask_rows_preserve_anchors() {
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        register(&mut registry);
        let mut mask = device.alloc_buffer(6, DType::BF16, vec![3]).unwrap();
        mask.as_mut_slice::<u16>()
            .unwrap()
            .copy_from_slice(&[bf16(2.0), bf16(3.0), bf16(4.0)]);
        let mut hidden = device.alloc_buffer(24, DType::F32, vec![2, 3]).unwrap();
        hidden
            .as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(&[7.0, 8.0, 9.0, 0.0, 0.0, 0.0]);
        let mut encoder = device.command_encoder().unwrap();
        dispatch_dflash2_mask_rows_f32(
            &mut encoder,
            &mut registry,
            &device,
            &mask,
            &hidden,
            DFlash2MaskRowsParams {
                lanes: 1,
                rows: 2,
                hidden: 3,
            },
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();
        assert_eq!(
            hidden.as_slice::<f32>().unwrap(),
            &[7.0, 8.0, 9.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn gpu_moe_route_mask_uses_physical_lane_budgets() {
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        register(&mut registry);
        let mut verify_rows = device.alloc_buffer(12, DType::U32, vec![3]).unwrap();
        verify_rows
            .as_mut_slice::<u32>()
            .unwrap()
            .copy_from_slice(&[4, 2, 3]);
        let pairs = 2 * 4 * 2;
        let mut ids = device
            .alloc_buffer(pairs * 4, DType::U32, vec![pairs])
            .unwrap();
        ids.as_mut_slice::<u32>().unwrap().fill(7);
        let mut weights = device
            .alloc_buffer(pairs * 4, DType::F32, vec![pairs])
            .unwrap();
        weights.as_mut_slice::<f32>().unwrap().fill(0.25);
        let mut encoder = device.command_encoder().unwrap();
        dispatch_dflash2_mask_moe_routes(
            &mut encoder,
            &mut registry,
            &device,
            &verify_rows,
            &ids,
            &weights,
            DFlash2MoeRouteMaskParams {
                physical_lane: [2, 1, 0, 0],
                lanes: 2,
                rows_per_lane: 4,
                top_k: 2,
                n_experts: 64,
                width: 8,
            },
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();
        let ids = ids.as_slice::<u32>().unwrap();
        let weights = weights.as_slice::<f32>().unwrap();
        for pair in 0..pairs {
            let row = (pair / 2) % 4;
            let lane = (pair / 2) / 4;
            let live = if lane == 0 { row < 3 } else { row < 2 };
            assert_eq!(ids[pair], if live { 7 } else { 64 });
            assert_eq!(weights[pair], if live { 0.25 } else { 0.0 });
        }
    }
}
