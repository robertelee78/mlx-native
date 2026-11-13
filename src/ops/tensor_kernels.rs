//! Availability of the kernel families that exist only on the Metal tensor
//! API.
//!
//! The dense, quantized and expert matmuls each have a portable twin and
//! choose it by themselves where the tensor API is unavailable
//! (`dense_mm_capability`). The tensor flash attention and the tiled verify
//! attention have availability queries of their own. The families here had
//! neither. With one exception named on its variant they have no portable
//! twin: their layout, tiling or schedule is the route. For them the query
//! is the contract. A caller asks [`tensor_kernel_family_available`] before
//! it chooses the route, and a dispatch on a host that cannot run it returns
//! [`MlxError::Unavailable`] instead of the shader compiler's diagnostic.
//!
//! The classifier is the one every tensor probe uses
//! (`dense_mm_capability::is_unavailable_tensor_api`): a runtime compiler
//! without `<metal_tensor>`, or a failed pipeline build on a device outside
//! the Metal 4 GPU family, is "unavailable"; any other failure on a Metal 4
//! device is returned as the error it is and not cached.
//!
//! `MLX_NATIVE_DISABLE_METAL_TENSOR=1` makes every family here answer
//! "unavailable" ([`TENSOR_KERNELS_DISABLED`]). The variable does not make a
//! process behave as a host without the tensor API: it also selects the
//! portable dense BF16/F16/F32 matmuls, but the quantized and expert matmul
//! probes, the tensor flash attention and the tiled verify attention do not
//! read it and keep their tensor kernels. A registry that behaves as such a
//! host in every probe is
//! [`KernelRegistry::simulate_missing_tensor_header`].

use crate::device::MlxDevice;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;
use crate::ops::dense_mm_capability::{is_unavailable_tensor_api, tensor_disabled_from_env};

/// Why a tensor-only family is unavailable on a host without the tensor API.
pub const TENSOR_API_UNAVAILABLE: &str = "the Metal tensor API is unavailable (a runtime \
    compiler without <metal_tensor>, or a GPU outside the Metal 4 family)";

/// Why a tensor-only family is unavailable when the operator disabled it.
pub const TENSOR_KERNELS_DISABLED: &str =
    "MLX_NATIVE_DISABLE_METAL_TENSOR=1 disables the tensor-only kernel families";

/// Why [`tensor_kernel_family_available`] answers `false` in this process:
/// [`TENSOR_KERNELS_DISABLED`] under `MLX_NATIVE_DISABLE_METAL_TENSOR=1`,
/// [`TENSOR_API_UNAVAILABLE`] otherwise. The `reason` of the
/// [`MlxError::Unavailable`] a family's entry point returns.
pub fn tensor_kernel_family_unavailable_reason() -> &'static str {
    if tensor_disabled_from_env() {
        TENSOR_KERNELS_DISABLED
    } else {
        TENSOR_API_UNAVAILABLE
    }
}

/// A kernel family built on the Metal tensor API whose entry points had no
/// availability probe. All but [`Self::MmV2F16`] have no portable twin in
/// this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TensorKernelFamily {
    /// [`crate::ops::repacked_gguf_mm`]: the prepared Q4_K / Q6_K row-tile
    /// matmul.
    RepackedGgufMm,
    /// [`crate::ops::dflash2_af4g64_staged`]: the staged AF4G64 projection
    /// tile.
    Dflash2Af4g64Staged,
    /// [`crate::ops::dflash2_ring_attention`]: draft attention over the
    /// context rings in place.
    Dflash2RingAttention,
    /// `quantized_matmul_id_ggml_grouped_m8_*`: prepared expert planes on
    /// eight-row tiles.
    GroupedM8Experts,
    /// `moe_tiles_*` and `moe_tiles_fused`: the MoE expert live tiles.
    MoeExpertTiles,
    /// `quantized_matmul_mm_tensor_perm021*`: the head-major BF16-input
    /// matmul.
    MmPerm021,
    /// `dispatch_mm_v2_f16`: the large-tile F16-weight matmul. The one
    /// family here with a portable twin: where it is unavailable the
    /// dispatch runs the portable F16 matmul (`dense_mm_f16`) by itself, and
    /// the query says which of the two a dispatch encodes.
    MmV2F16,
}

/// One probe pipeline: its kernel and integer function constants.
type Probe = (&'static str, &'static [(usize, i32)]);

/// The batch constants every dispatch of the quantized tensor matmul source
/// specialises its pipeline with.
const MM_TENSOR_CONSTANTS: &[(usize, i32)] = &[(700, 1), (701, 1), (702, 1)];

const _: () = assert!(TensorKernelFamily::ALL.len() == TensorKernelFamily::COUNT);

impl TensorKernelFamily {
    /// Every family.
    pub const ALL: &'static [Self] = &[
        Self::RepackedGgufMm,
        Self::Dflash2Af4g64Staged,
        Self::Dflash2RingAttention,
        Self::GroupedM8Experts,
        Self::MoeExpertTiles,
        Self::MmPerm021,
        Self::MmV2F16,
    ];

    /// Families, as the registry's answer table is sized.
    pub(crate) const COUNT: usize = 7;

    pub(crate) const fn index(self) -> usize {
        self as usize
    }

    /// The family as it is named in [`MlxError::Unavailable`].
    pub const fn name(self) -> &'static str {
        match self {
            Self::RepackedGgufMm => "the prepared GGUF matmul (repacked_gguf_mm)",
            Self::Dflash2Af4g64Staged => "the staged AF4G64 projection (dflash2_af4g64_staged)",
            Self::Dflash2RingAttention => "the ring draft attention (dflash2_ring_attention)",
            Self::GroupedM8Experts => "the prepared expert planes (grouped_m8)",
            Self::MoeExpertTiles => "the MoE expert live tiles (moe_tiles)",
            Self::MmPerm021 => "the head-major tensor matmul (mm_tensor_perm021)",
            Self::MmV2F16 => "the large-tile F16 tensor matmul (mm_tensor_v2_f16)",
        }
    }

    /// The pipeline whose build answers for the family: one of its kernels
    /// that multiplies through the tensor API. Every kernel of a family is
    /// compiled from one source, so the missing header stops them all, and a
    /// device outside the Metal 4 family builds none that holds a
    /// cooperative tensor.
    const fn probe(self) -> Probe {
        match self {
            Self::RepackedGgufMm => (crate::ops::repacked_gguf_mm::Q4K_MM_BF16_INPUT_KERNEL, &[]),
            Self::Dflash2Af4g64Staged => (
                crate::ops::dflash2_af4g64_staged::PROJECTION_KERNELS[0],
                &[],
            ),
            Self::Dflash2RingAttention => (crate::ops::dflash2_ring_attention::SPLIT_KERNEL, &[]),
            Self::GroupedM8Experts => ("grouped_m8_id_q4_K_f32", &[]),
            Self::MoeExpertTiles => ("moe_tiles_gate_up_q4_K", &[]),
            Self::MmPerm021 => (
                "kernel_mul_mm_q4_0_tensor_bf16_perm021",
                MM_TENSOR_CONSTANTS,
            ),
            Self::MmV2F16 => ("hf2q_mul_mm_tensor_v2_f16", MM_TENSOR_CONSTANTS),
        }
    }

    /// The kernel the availability query builds, for a test that replaces
    /// its source (`register_source`). Not part of the supported surface.
    #[doc(hidden)]
    pub const fn probe_kernel(self) -> &'static str {
        self.probe().0
    }
}

/// Whether `family` can run on `device`: `Ok(false)` where the Metal tensor
/// API is unavailable or the families are disabled (see the module
/// documentation; [`tensor_kernel_family_unavailable_reason`] says which),
/// an error for any other failure to build the family's probe pipeline.
///
/// The first call for a registry builds one pipeline; the answer is cached
/// in the registry, which is bound to one device, and later calls neither
/// compile nor allocate.
pub fn tensor_kernel_family_available(
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    family: TensorKernelFamily,
) -> Result<bool> {
    let registry_id = device.registry_id();
    if let Some(available) = registry.tensor_kernel_family(family, registry_id) {
        return Ok(available);
    }
    let available = !tensor_disabled_from_env() && {
        let (kernel, constants) = family.probe();
        let unavailable = |error: &MlxError| is_unavailable_tensor_api(device, error);
        let probe = if constants.is_empty() {
            registry.probe_optional_pipeline(
                kernel,
                device.metal_device(),
                registry_id,
                unavailable,
            )?
        } else {
            registry.probe_optional_pipeline_with_constants(
                kernel,
                device.metal_device(),
                registry_id,
                &[],
                constants,
                unavailable,
            )?
        };
        probe.available
    };
    registry.set_tensor_kernel_family(device.metal_device(), family, available)?;
    Ok(available)
}

/// `Ok` when `family` can run on `device`, [`MlxError::Unavailable`] when it
/// cannot. Every entry point of a family calls it once
/// its arguments are valid, before it allocates or encodes anything.
pub(crate) fn require_tensor_kernel_family(
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    family: TensorKernelFamily,
) -> Result<()> {
    if tensor_kernel_family_available(registry, device, family)? {
        Ok(())
    } else {
        Err(MlxError::Unavailable {
            feature: family.name(),
            reason: tensor_kernel_family_unavailable_reason(),
        })
    }
}

/// The error of a failed pipeline build of a tensor kernel that has an
/// availability query of its own (`flash_attn_prefill_tensor_available`,
/// `tq_hb_verify_tile_available`): [`MlxError::Unavailable`] where the
/// tensor API is unavailable, `error` itself otherwise.
pub(crate) fn typed_unavailable(
    device: &MlxDevice,
    feature: &'static str,
    error: MlxError,
) -> MlxError {
    if is_unavailable_tensor_api(device, &error) {
        MlxError::Unavailable {
            feature,
            reason: TENSOR_API_UNAVAILABLE,
        }
    } else {
        error
    }
}
