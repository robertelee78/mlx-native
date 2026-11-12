//! DFlash2 draft attention over the persistent K/V rings, read in place.
//!
//! The packed route copies every lane's whole ring window into a linear K/V
//! tile and fills a dense mask before each layer's attention. This route
//! multiplies the ring where it lives: one threadgroup per (KV head, lane,
//! split) walks its share of the live 128-slot ring tiles with an online
//! softmax, the last split adds the eight current rows, and a reduce combines
//! the splits in split order.
//!
//! Portions derived from Splash commit 031a25a6 (Apache-2.0). See
//! `src/shaders/dflash2_ring_attention.metal` for full attribution.
//!
//! Numerics contract: an attention row's bits depend on its own query, its
//! lane's ring and current rows and its lane's context length. They do not
//! depend on other query rows, other lanes, or the lane's slot.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;
use crate::ops::encode_helpers::{as_bytes, encode_with_args, KernelArg};

pub const PACK_KERNEL: &str = "dflash2_ring_attention_pack";
pub const SPLIT_KERNEL: &str = "dflash2_ring_attention_split";
pub const REDUCE_KERNEL: &str = "dflash2_ring_attention_reduce";
pub static SOURCE: &str = include_str!("../shaders/dflash2_ring_attention.metal");

/// Query rows per lane, head dimension, query heads per KV head, ring slots
/// per tile and splits the kernels are compiled for.
pub const DFLASH2_RING_ATTENTION_ROWS: usize = 8;
pub const DFLASH2_RING_ATTENTION_HEAD_DIM: usize = 128;
pub const DFLASH2_RING_ATTENTION_GROUP_HEADS: usize = 4;
pub const DFLASH2_RING_ATTENTION_TILE: usize = 128;
pub const DFLASH2_RING_ATTENTION_SPLITS: usize = 4;
/// Current value rows per (lane, KV head): the eight rows and eight zero rows.
pub const DFLASH2_RING_ATTENTION_CURRENT_VALUES: usize = 16;
/// Lanes one dispatch takes, as the packed attention does.
pub const DFLASH2_RING_ATTENTION_MAX_LANES: usize = 4;
const ATTENTION_ROWS: usize = DFLASH2_RING_ATTENTION_GROUP_HEADS * DFLASH2_RING_ATTENTION_ROWS;
const PARTIAL_FLOATS: usize = ATTENTION_ROWS * DFLASH2_RING_ATTENTION_HEAD_DIM + 2 * ATTENTION_ROWS;

pub fn register(registry: &mut KernelRegistry) {
    for name in [PACK_KERNEL, SPLIT_KERNEL, REDUCE_KERNEL] {
        registry.register_source(name, SOURCE);
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DFlash2RingAttentionParams {
    pub lanes: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub window: u32,
}

/// Whether the compiled kernels take an attention of this shape.
pub fn dflash2_ring_attention_supports(
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rows: usize,
    window: usize,
) -> bool {
    kv_heads > 0
        && q_heads == kv_heads * DFLASH2_RING_ATTENTION_GROUP_HEADS
        && head_dim == DFLASH2_RING_ATTENTION_HEAD_DIM
        && rows == DFLASH2_RING_ATTENTION_ROWS
        && window >= DFLASH2_RING_ATTENTION_TILE
        && window % DFLASH2_RING_ATTENTION_TILE == 0
}

fn elements(dims: &[usize]) -> Result<usize> {
    dims.iter()
        .try_fold(1_usize, |count, dim| count.checked_mul(*dim))
        .ok_or_else(|| {
            MlxError::InvalidArgument(format!("DFlash2 ring attention size {dims:?} overflows"))
        })
}

/// F32 elements of split partials for `lanes` lanes of `kv_heads` KV heads.
pub fn dflash2_ring_attention_partial_elements(lanes: usize, kv_heads: usize) -> Result<usize> {
    elements(&[
        lanes,
        kv_heads,
        DFLASH2_RING_ATTENTION_SPLITS,
        PARTIAL_FLOATS,
    ])
}

pub struct DFlash2RingAttentionBuffers<'a> {
    /// F32 `[lanes * 8][q_heads][128]` rotated queries.
    pub query: &'a MlxBuffer,
    /// F32 `[lanes * 8][kv_heads][128]` rotated current keys.
    pub key: &'a MlxBuffer,
    /// F32 `[lanes * 8][kv_heads][128]` current values.
    pub value: &'a MlxBuffer,
    /// BF16 `[lanes][kv_heads][window][128]`.
    pub ring_keys: &'a MlxBuffer,
    /// BF16 `[lanes][kv_heads][window][128]`.
    pub ring_values: &'a MlxBuffer,
    /// U32 `[lanes]` rows each lane's ring holds.
    pub cache_lengths: &'a MlxBuffer,
    /// BF16 scratch `[lanes][q_heads][8][128]`.
    pub queries: &'a MlxBuffer,
    /// BF16 scratch `[lanes][kv_heads][8][128]`.
    pub current_keys: &'a MlxBuffer,
    /// BF16 scratch `[lanes][kv_heads][16][128]`.
    pub current_values: &'a MlxBuffer,
    /// F32 scratch of [`dflash2_ring_attention_partial_elements`] elements.
    pub partials: &'a MlxBuffer,
    /// BF16 `[lanes * 8][q_heads * 128]`.
    pub output: &'a MlxBuffer,
}

fn require(buffer: &MlxBuffer, dtype: DType, elements: usize, label: &str) -> Result<()> {
    let need = elements
        .checked_mul(dtype.size_of())
        .ok_or_else(|| MlxError::InvalidArgument("DFlash2 ring attention size overflow".into()))?;
    if buffer.dtype() != dtype || buffer.data_byte_len() < need {
        return Err(MlxError::InvalidArgument(format!(
            "DFlash2 ring attention {label} must be {dtype} with at least {need} bytes, got {} with {}",
            buffer.dtype(),
            buffer.data_byte_len()
        )));
    }
    Ok(())
}

/// Encode the attention of every lane's eight query rows over its ring and
/// its current rows. The caller places the barriers around the operation;
/// the two inside it are encoded here.
pub fn dispatch_dflash2_ring_attention(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    buffers: &DFlash2RingAttentionBuffers<'_>,
    params: DFlash2RingAttentionParams,
) -> Result<()> {
    let (lanes, q_heads, kv_heads, window) = (
        params.lanes as usize,
        params.q_heads as usize,
        params.kv_heads as usize,
        params.window as usize,
    );
    const ROWS: usize = DFLASH2_RING_ATTENTION_ROWS;
    const DIM: usize = DFLASH2_RING_ATTENTION_HEAD_DIM;
    if lanes == 0
        || lanes > DFLASH2_RING_ATTENTION_MAX_LANES
        || !dflash2_ring_attention_supports(q_heads, kv_heads, DIM, ROWS, window)
    {
        return Err(MlxError::InvalidArgument(format!(
            "unsupported DFlash2 ring attention geometry: lanes={lanes}, q_heads={q_heads}, kv_heads={kv_heads}, window={window}"
        )));
    }
    let query_elements = elements(&[lanes, ROWS, q_heads, DIM])?;
    let key_elements = elements(&[lanes, ROWS, kv_heads, DIM])?;
    let value_elements = elements(&[lanes, kv_heads, DFLASH2_RING_ATTENTION_CURRENT_VALUES, DIM])?;
    let ring_elements = elements(&[lanes, kv_heads, window, DIM])?;
    // The pack kernel counts its rows' elements in 32 bits.
    if u32::try_from(query_elements.max(key_elements).max(value_elements)).is_err() {
        return Err(MlxError::InvalidArgument(format!(
            "DFlash2 ring attention of {lanes} lanes and {q_heads} query heads exceeds 32-bit element counts"
        )));
    }
    require(buffers.query, DType::F32, query_elements, "query")?;
    require(buffers.key, DType::F32, key_elements, "key")?;
    require(buffers.value, DType::F32, key_elements, "value")?;
    require(buffers.ring_keys, DType::BF16, ring_elements, "key ring")?;
    require(
        buffers.ring_values,
        DType::BF16,
        ring_elements,
        "value ring",
    )?;
    require(buffers.cache_lengths, DType::U32, lanes, "cache lengths")?;
    require(
        buffers.queries,
        DType::BF16,
        query_elements,
        "query scratch",
    )?;
    require(
        buffers.current_keys,
        DType::BF16,
        key_elements,
        "current key scratch",
    )?;
    require(
        buffers.current_values,
        DType::BF16,
        value_elements,
        "current value scratch",
    )?;
    require(
        buffers.partials,
        DType::F32,
        dflash2_ring_attention_partial_elements(lanes, kv_heads)?,
        "partials",
    )?;
    require(buffers.output, DType::BF16, query_elements, "output")?;

    // A kernel's static threadgroup memory is fixed when it is compiled (and
    // doubled by shader validation); refuse before anything is encoded.
    let limit = device.metal_device().max_threadgroup_memory_length();
    let used = registry
        .get_pipeline(SPLIT_KERNEL, device.metal_device())?
        .static_threadgroup_memory_length();
    if used > limit {
        return Err(MlxError::InvalidArgument(format!(
            "{SPLIT_KERNEL} needs {used} bytes of threadgroup memory; the device allows {limit}"
        )));
    }
    let pack = registry.get_pipeline(PACK_KERNEL, device.metal_device())?;
    encode_with_args(
        encoder,
        pack,
        &[
            (0, KernelArg::Buffer(buffers.query)),
            (1, KernelArg::Buffer(buffers.key)),
            (2, KernelArg::Buffer(buffers.value)),
            (3, KernelArg::Buffer(buffers.queries)),
            (4, KernelArg::Buffer(buffers.current_keys)),
            (5, KernelArg::Buffer(buffers.current_values)),
            (6, KernelArg::Bytes(as_bytes(&params))),
        ],
        MTLSize::new(query_elements.max(value_elements) as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    encoder.memory_barrier();
    let split = registry.get_pipeline(SPLIT_KERNEL, device.metal_device())?;
    encoder.encode_threadgroups_with_args(
        split,
        &[
            (0, KernelArg::Buffer(buffers.queries)),
            (1, KernelArg::Buffer(buffers.ring_keys)),
            (2, KernelArg::Buffer(buffers.ring_values)),
            (3, KernelArg::Buffer(buffers.current_keys)),
            (4, KernelArg::Buffer(buffers.current_values)),
            (5, KernelArg::Buffer(buffers.cache_lengths)),
            (6, KernelArg::Buffer(buffers.partials)),
            (7, KernelArg::Bytes(as_bytes(&params))),
        ],
        MTLSize::new(
            kv_heads as u64,
            lanes as u64,
            DFLASH2_RING_ATTENTION_SPLITS as u64,
        ),
        MTLSize::new(256, 1, 1),
    );
    encoder.memory_barrier();
    let reduce = registry.get_pipeline(REDUCE_KERNEL, device.metal_device())?;
    encoder.encode_threadgroups_with_args(
        reduce,
        &[
            (0, KernelArg::Buffer(buffers.partials)),
            (1, KernelArg::Buffer(buffers.output)),
            (2, KernelArg::Bytes(as_bytes(&params))),
        ],
        MTLSize::new(kv_heads as u64, lanes as u64, 1),
        MTLSize::new(256, 1, 1),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;

    const ROWS: usize = DFLASH2_RING_ATTENTION_ROWS;
    const DIM: usize = DFLASH2_RING_ATTENTION_HEAD_DIM;

    /// Query heads, KV heads and ring slots of one test attention.
    #[derive(Clone, Copy, Debug)]
    struct Shape {
        q_heads: usize,
        kv_heads: usize,
        window: usize,
    }

    /// The served draft attention.
    const SERVED: Shape = Shape {
        q_heads: 32,
        kv_heads: 8,
        window: 2_048,
    };
    const SMALL: Shape = Shape {
        q_heads: 8,
        kv_heads: 2,
        window: 2_048,
    };
    const SHORT_RING: Shape = Shape {
        q_heads: 8,
        kv_heads: 2,
        window: 256,
    };

    impl Shape {
        fn lane_elements(self) -> usize {
            ROWS * self.q_heads * DIM
        }

        fn params(self, lanes: usize) -> DFlash2RingAttentionParams {
            DFlash2RingAttentionParams {
                lanes: lanes as u32,
                q_heads: self.q_heads as u32,
                kv_heads: self.kv_heads as u32,
                window: self.window as u32,
            }
        }
    }

    #[test]
    fn only_the_compiled_shape_is_supported() {
        assert!(dflash2_ring_attention_supports(32, 8, 128, 8, 2_048));
        assert!(dflash2_ring_attention_supports(8, 2, 128, 8, 256));
        assert!(!dflash2_ring_attention_supports(32, 16, 128, 8, 2_048));
        assert!(!dflash2_ring_attention_supports(32, 8, 256, 8, 2_048));
        assert!(!dflash2_ring_attention_supports(32, 8, 128, 4, 2_048));
        assert!(!dflash2_ring_attention_supports(32, 8, 128, 8, 2_000));
        assert!(!dflash2_ring_attention_supports(0, 0, 128, 8, 2_048));
    }

    #[test]
    fn sizes_are_checked_before_anything_is_encoded() {
        assert_eq!(
            dflash2_ring_attention_partial_elements(4, 8).unwrap(),
            4 * 8 * 4 * (32 * 128 + 64)
        );
        assert!(dflash2_ring_attention_partial_elements(usize::MAX, 2).is_err());

        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        let word = |dtype: DType| {
            device
                .alloc_buffer(4, dtype, vec![4 / dtype.size_of()])
                .unwrap()
        };
        let (f32_word, bf16_word, u32_word) =
            (word(DType::F32), word(DType::BF16), word(DType::U32));
        let buffers = DFlash2RingAttentionBuffers {
            query: &f32_word,
            key: &f32_word,
            value: &f32_word,
            ring_keys: &bf16_word,
            ring_values: &bf16_word,
            cache_lengths: &u32_word,
            queries: &bf16_word,
            current_keys: &bf16_word,
            current_values: &bf16_word,
            partials: &f32_word,
            output: &bf16_word,
        };
        let mut refuse = |params: DFlash2RingAttentionParams, expected: &str| {
            let mut encoder = device.command_encoder().unwrap();
            let error = dispatch_dflash2_ring_attention(
                &mut encoder,
                &mut registry,
                &device,
                &buffers,
                params,
            )
            .unwrap_err();
            assert!(error.to_string().contains(expected), "{params:?}: {error}");
        };
        // More lanes than one dispatch takes.
        refuse(
            SERVED.params(DFLASH2_RING_ATTENTION_MAX_LANES + 1),
            "geometry",
        );
        refuse(SERVED.params(0), "geometry");
        // Element counts the pack kernel cannot index in 32 bits.
        refuse(
            DFlash2RingAttentionParams {
                lanes: 4,
                q_heads: 1 << 22,
                kv_heads: 1 << 20,
                window: 2_048,
            },
            "32-bit",
        );
        // Buffers smaller than the geometry.
        refuse(SERVED.params(4), "at least");
    }

    /// Shader validation doubles a kernel's static threadgroup memory, and a
    /// kernel over the device limit aborts under the Metal debug layer.
    #[cfg(mlx_native_has_metal_tensor_artifact)]
    #[test]
    fn split_kernel_threadgroup_memory_fits_under_shader_validation() {
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        let limit = device.metal_device().max_threadgroup_memory_length();
        let used = registry
            .get_pipeline(SPLIT_KERNEL, device.metal_device())
            .unwrap()
            .static_threadgroup_memory_length();
        assert!(used <= limit, "{SPLIT_KERNEL}: {used} bytes exceed {limit}");
        if std::env::var_os("MTL_SHADER_VALIDATION").is_none() {
            assert!(
                2 * used <= limit,
                "{SPLIT_KERNEL}: {used} bytes leave no room for shader validation under {limit}"
            );
        }
    }

    fn random(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((*state >> 40) as f32) / ((1_u64 << 24) as f32) - 0.5
    }

    /// One lane's inputs, every float already a BF16 value.
    #[derive(Clone)]
    struct Lane {
        length: u32,
        /// `[ROWS][q_heads][DIM]`
        query: Vec<f32>,
        /// `[ROWS][kv_heads][DIM]`
        key: Vec<f32>,
        value: Vec<f32>,
        /// `[kv_heads][window][DIM]`
        ring_keys: Vec<f32>,
        ring_values: Vec<f32>,
    }

    fn lane(shape: Shape, length: u32, seed: u64) -> Lane {
        let mut state = seed;
        let mut values = |count: usize, scale: f32| -> Vec<f32> {
            (0..count)
                .map(|_| bf16::from_f32(scale * random(&mut state)).to_f32())
                .collect()
        };
        Lane {
            length,
            query: values(ROWS * shape.q_heads * DIM, 4.0),
            key: values(ROWS * shape.kv_heads * DIM, 4.0),
            value: values(ROWS * shape.kv_heads * DIM, 2.0),
            // Every slot holds data, so a slot outside the live window that is
            // not masked changes the result.
            ring_keys: values(shape.kv_heads * shape.window * DIM, 4.0),
            ring_values: values(shape.kv_heads * shape.window * DIM, 2.0),
        }
    }

    fn f32_buffer(device: &MlxDevice, values: &[f32]) -> MlxBuffer {
        let mut buffer = device
            .alloc_buffer(values.len() * 4, DType::F32, vec![values.len()])
            .unwrap();
        buffer
            .as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(values);
        buffer
    }

    fn bf16_buffer(device: &MlxDevice, values: &[f32]) -> MlxBuffer {
        let mut buffer = device
            .alloc_buffer(values.len() * 2, DType::BF16, vec![values.len()])
            .unwrap();
        for (bits, value) in buffer
            .as_mut_slice::<bf16>()
            .unwrap()
            .iter_mut()
            .zip(values)
        {
            *bits = bf16::from_f32(*value);
        }
        buffer
    }

    /// Scratch that holds NaN, as a reused workspace may: a result that reads
    /// an element its command did not write is not finite.
    fn nan_scratch(device: &MlxDevice, elements: usize, dtype: DType) -> MlxBuffer {
        let mut buffer = device
            .alloc_buffer(elements * dtype.size_of(), dtype, vec![elements])
            .unwrap();
        match dtype {
            DType::F32 => buffer.as_mut_slice::<f32>().unwrap().fill(f32::NAN),
            DType::BF16 => buffer.as_mut_slice::<bf16>().unwrap().fill(bf16::NAN),
            other => panic!("no NaN scratch of {other}"),
        }
        buffer
    }

    /// `[lanes][ROWS][q_heads][DIM]` BF16 results as F32.
    fn attend(
        device: &MlxDevice,
        registry: &mut KernelRegistry,
        shape: Shape,
        lanes: &[&Lane],
    ) -> Vec<f32> {
        let count = lanes.len();
        let concat = |field: fn(&Lane) -> &Vec<f32>| -> Vec<f32> {
            lanes
                .iter()
                .flat_map(|lane| field(lane).iter().copied())
                .collect()
        };
        let query = f32_buffer(device, &concat(|lane| &lane.query));
        let key = f32_buffer(device, &concat(|lane| &lane.key));
        let value = f32_buffer(device, &concat(|lane| &lane.value));
        let ring_keys = bf16_buffer(device, &concat(|lane| &lane.ring_keys));
        let ring_values = bf16_buffer(device, &concat(|lane| &lane.ring_values));
        let mut cache_lengths = device
            .alloc_buffer(count * 4, DType::U32, vec![count])
            .unwrap();
        for (slot, lane) in cache_lengths
            .as_mut_slice::<u32>()
            .unwrap()
            .iter_mut()
            .zip(lanes)
        {
            *slot = lane.length;
        }
        let query_elements = count * shape.lane_elements();
        let queries = nan_scratch(device, query_elements, DType::BF16);
        let current_keys = nan_scratch(device, count * shape.kv_heads * ROWS * DIM, DType::BF16);
        let current_values = nan_scratch(
            device,
            count * shape.kv_heads * DFLASH2_RING_ATTENTION_CURRENT_VALUES * DIM,
            DType::BF16,
        );
        let partials = nan_scratch(
            device,
            dflash2_ring_attention_partial_elements(count, shape.kv_heads).unwrap(),
            DType::F32,
        );
        let output = nan_scratch(device, query_elements, DType::BF16);
        let mut encoder = device.command_encoder().unwrap();
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
            shape.params(count),
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();
        let result: Vec<f32> = output
            .as_slice::<bf16>()
            .unwrap()
            .iter()
            .map(|value| value.to_f32())
            .collect();
        assert!(
            result.iter().all(|value| value.is_finite()),
            "{shape:?}: a result read scratch its command did not write"
        );
        result
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    /// F64 attention of one lane and, per output element, the bound on a
    /// kernel result's distance from it. Scores are 128 exact BF16 products
    /// added in F32 and scaled once, so a score is off by at most
    /// `score_error`; a softmax whose scores move by that much moves every
    /// probability by a factor within `exp(+-2 score_error)`; the `keys`
    /// weighted values are added in F32; `exp` is good to a few ulp; and the
    /// result is rounded once to BF16.
    fn reference(shape: Shape, lane: &Lane) -> (Vec<f64>, Vec<f64>) {
        let Shape {
            q_heads,
            kv_heads,
            window,
        } = shape;
        let scale = 1.0 / (DIM as f64).sqrt();
        let length = lane.length as usize;
        let mut expected = vec![0.0_f64; shape.lane_elements()];
        let mut bound = vec![0.0_f64; shape.lane_elements()];
        for row in 0..ROWS {
            let first = (length + row).saturating_sub(window - 1);
            for head in 0..q_heads {
                let kv_head = head / DFLASH2_RING_ATTENTION_GROUP_HEADS;
                let query = &lane.query[(row * q_heads + head) * DIM..][..DIM];
                let ring = |position: usize| (kv_head * window + position % window) * DIM;
                let current = |index: usize| (index * kv_heads + kv_head) * DIM;
                let keys: Vec<(&[f32], &[f32])> = (first..length)
                    .map(|position| {
                        (
                            &lane.ring_keys[ring(position)..][..DIM],
                            &lane.ring_values[ring(position)..][..DIM],
                        )
                    })
                    .chain((0..ROWS).map(|index| {
                        (
                            &lane.key[current(index)..][..DIM],
                            &lane.value[current(index)..][..DIM],
                        )
                    }))
                    .collect();
                let mut scores = Vec::with_capacity(keys.len());
                let mut score_magnitude = 0.0_f64;
                for (key, _) in &keys {
                    let mut score = 0.0_f64;
                    let mut magnitude = 0.0_f64;
                    for (q, k) in query.iter().zip(*key) {
                        let product = f64::from(*q) * f64::from(*k);
                        score += product;
                        magnitude += product.abs();
                    }
                    scores.push(score * scale);
                    score_magnitude = score_magnitude.max(magnitude * scale);
                }
                let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = scores.iter().map(|s| (s - maximum).exp()).collect();
                let total: f64 = weights.iter().sum();
                let score_error = (DIM as f64 + 2.0) * 2.0_f64.powi(-24) * score_magnitude;
                let relative =
                    (2.0 * score_error).exp_m1() + (keys.len() as f64 + 64.0) * 2.0_f64.powi(-23);
                let mut sums = [0.0_f64; DIM];
                let mut magnitudes = [0.0_f64; DIM];
                for (weight, (_, value)) in weights.iter().zip(&keys) {
                    for dim in 0..DIM {
                        let value = f64::from(value[dim]);
                        sums[dim] += weight * value;
                        magnitudes[dim] += weight * value.abs();
                    }
                }
                for dim in 0..DIM {
                    let at = (row * q_heads + head) * DIM + dim;
                    expected[at] = sums[dim] / total;
                    bound[at] = expected[at].abs() * 2.0_f64.powi(-8)
                        + relative * magnitudes[dim] / total
                        + 2.0_f64.powi(-120);
                }
            }
        }
        (expected, bound)
    }

    /// Ring lengths per four-lane batch: empty, partial, one short of the
    /// window, wrapped, and far past it.
    const LENGTHS: [[u32; 4]; 2] = [[0, 5, 300, 2_047], [2_048, 2_825, 10_000, 129]];
    /// The same at the served shape, with every tile and window boundary.
    const SERVED_LENGTHS: [[u32; 4]; 6] = [
        [0, 1, 7, 8],
        [127, 128, 129, 300],
        [2_039, 2_040, 2_046, 2_047],
        [2_048, 2_049, 2_055, 2_056],
        [2_175, 4_095, 4_096, 4_097],
        [6_143, 10_000, 1_000_003, 5],
    ];

    fn assert_within_reference(
        device: &MlxDevice,
        registry: &mut KernelRegistry,
        shape: Shape,
        lanes: &[Lane],
    ) -> Vec<f32> {
        let actual = attend(device, registry, shape, &lanes.iter().collect::<Vec<_>>());
        let per_lane = shape.lane_elements();
        let mut worst = 0.0_f64;
        for (index, lane) in lanes.iter().enumerate() {
            let (expected, bound) = reference(shape, lane);
            for at in 0..per_lane {
                let error = (f64::from(actual[index * per_lane + at]) - expected[at]).abs();
                assert!(
                    error <= bound[at],
                    "{shape:?} length {} element {at}: error {error} exceeds {}",
                    lane.length,
                    bound[at]
                );
                worst = worst.max(error / bound[at]);
            }
        }
        assert!(worst > 0.0, "{shape:?}: reference comparison is vacuous");
        actual
    }

    #[cfg(mlx_native_has_metal_tensor_artifact)]
    #[test]
    fn attention_is_within_the_bound_of_the_f64_reference() {
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        for (shape, divisor) in [(SMALL, 1_u32), (SHORT_RING, 8)] {
            for (batch, lengths) in LENGTHS.iter().enumerate() {
                let lanes: Vec<Lane> = lengths
                    .iter()
                    .enumerate()
                    .map(|(index, length)| {
                        lane(shape, length / divisor, (batch * 4 + index) as u64 + 7)
                    })
                    .collect();
                assert_within_reference(&device, &mut registry, shape, &lanes);
            }
        }
    }

    /// The served 32-over-8-head attention on a 2,048-slot ring: reference,
    /// repeat, and every lane alone and in two-, three- and four-lane
    /// cohorts at other slots.
    #[cfg(mlx_native_has_metal_tensor_artifact)]
    #[test]
    fn served_attention_matches_the_reference_alone_and_in_every_cohort() {
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        let per_lane = SERVED.lane_elements();
        for (set, lengths) in SERVED_LENGTHS.iter().enumerate() {
            let lanes: Vec<Lane> = lengths
                .iter()
                .enumerate()
                .map(|(index, length)| lane(SERVED, *length, (set * 4 + index) as u64 + 901))
                .collect();
            let base = assert_within_reference(&device, &mut registry, SERVED, &lanes);
            let all: Vec<&Lane> = lanes.iter().collect();
            assert_eq!(
                bits(&attend(&device, &mut registry, SERVED, &all)),
                bits(&base),
                "set {set}: repeat differs"
            );
            let orders: [&[usize]; 10] = [
                &[0],
                &[1],
                &[2],
                &[3],
                &[1, 0],
                &[3, 2],
                &[3, 0, 2],
                &[1, 3, 0],
                &[2, 3, 0, 1],
                &[3, 2, 1, 0],
            ];
            for order in orders {
                let cohort: Vec<&Lane> = order.iter().map(|index| &lanes[*index]).collect();
                let moved = attend(&device, &mut registry, SERVED, &cohort);
                for (slot, index) in order.iter().enumerate() {
                    assert_eq!(
                        bits(&moved[slot * per_lane..(slot + 1) * per_lane]),
                        bits(&base[index * per_lane..(index + 1) * per_lane]),
                        "set {set} lane {index} (length {}) differs at slot {slot} of {order:?}",
                        lanes[*index].length
                    );
                }
            }
        }
    }

    #[cfg(mlx_native_has_metal_tensor_artifact)]
    #[test]
    fn attention_rows_are_deterministic_and_independent_of_their_cohort() {
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        let shape = SMALL;
        let per_lane = shape.lane_elements();
        let row_elements = shape.q_heads * DIM;
        let lanes: Vec<Lane> = [300_u32, 2_825, 0, 2_047]
            .iter()
            .enumerate()
            .map(|(index, length)| lane(shape, *length, index as u64 + 101))
            .collect();
        let all: Vec<&Lane> = lanes.iter().collect();
        let base = attend(&device, &mut registry, shape, &all);
        assert_eq!(
            bits(&base),
            bits(&attend(&device, &mut registry, shape, &all)),
            "repeat differs"
        );

        // Every lane alone, and in another slot beside other lanes.
        let others: Vec<Lane> = (0..4)
            .map(|other| lane(shape, 77 * other as u32 + 1_900, other as u64 + 211))
            .collect();
        for (index, subject) in lanes.iter().enumerate() {
            let alone = attend(&device, &mut registry, shape, &[subject]);
            assert_eq!(
                bits(&alone),
                bits(&base[index * per_lane..(index + 1) * per_lane]),
                "lane {index} depends on its batch"
            );
            let slot = 3 - index;
            let mut cohort: Vec<&Lane> = others.iter().collect();
            cohort[slot] = subject;
            let moved = attend(&device, &mut registry, shape, &cohort);
            assert_eq!(
                bits(&moved[slot * per_lane..(slot + 1) * per_lane]),
                bits(&base[index * per_lane..(index + 1) * per_lane]),
                "lane {index} depends on its slot or neighbours"
            );
        }

        // Another query row's query does not move a row.
        let mut perturbed = lanes.clone();
        for value in &mut perturbed[1].query[3 * row_elements..4 * row_elements] {
            *value = -*value;
        }
        let changed = attend(
            &device,
            &mut registry,
            shape,
            &perturbed.iter().collect::<Vec<_>>(),
        );
        for row in (0..ROWS).filter(|row| *row != 3) {
            let range = per_lane + row * row_elements..per_lane + (row + 1) * row_elements;
            assert_eq!(
                bits(&changed[range.clone()]),
                bits(&base[range]),
                "row {row} depends on row 3's query"
            );
        }
        let row_three = per_lane + 3 * row_elements..per_lane + 4 * row_elements;
        assert_ne!(bits(&changed[row_three.clone()]), bits(&base[row_three]));
    }
}
