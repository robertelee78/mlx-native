//! MoE two-pass mm_id pre-pass — sorts tokens by expert assignment.
//!
//! ADR-033 §Pi next-iter arc (iter A) — port of llama.cpp's
//! `kernel_mul_mm_id_map0`. See
//! `project_adr033_pi_next_iter_arc_moe_q4_0_kernel_port_2026_05_23.md`.
//!
//! Given top-k routing decisions for N tokens (`src2` of shape
//! `[n_tokens, ne20]` where `ne20 = n_expert_used`), produce per-expert
//! token-slot lists that iter B's main mm_id kernel will consume.
//!
//! Output layout:
//! - `htpe[ide]`              — token count for expert `ide` (uint32, len `n_experts`)
//! - `hids[ide][0..htpe[ide]]` — packed `(token_idx * ne20 + slot)` indices
//!   for tokens that named expert `ide` in their top-k (int32, len
//!   `n_experts * n_tokens`)

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

/// Supported `ne20` (n_expert_used) specializations. Mirrors llama.cpp's
/// `kernel_mul_mm_id_map0_ne20_{N}` host_name set at
/// `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:9780-9788`.
pub const SUPPORTED_NE20: &[u32] = &[1, 2, 4, 5, 6, 8, 10, 16, 22];

/// Parameter layout — bytewise identical to the `MoeMmIdMap0Params` struct
/// in `shaders/moe_mm_id_map0.metal`, which itself mirrors
/// `ggml_metal_kargs_mul_mm_id_map0` at
/// `ggml-metal-impl.h:496-505`. Keep field order unchanged.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct MoeMmIdMap0Params {
    pub ne02: i32,
    pub ne10: i32,
    pub ne11: i32,
    pub nb11: u64,
    pub nb12: u64,
    pub ne21: i32, // n_tokens
    pub ne20: i32, // n_expert_used
    pub nb21: u64, // byte stride between successive top-k rows
}

/// Returns the kernel host_name for the given `ne20`. Returns
/// `MlxError::InvalidArgument` if the value is unsupported.
fn kernel_name_for(ne20: u32) -> Result<&'static str> {
    match ne20 {
        1 => Ok("moe_mm_id_map0_ne20_1"),
        2 => Ok("moe_mm_id_map0_ne20_2"),
        4 => Ok("moe_mm_id_map0_ne20_4"),
        5 => Ok("moe_mm_id_map0_ne20_5"),
        6 => Ok("moe_mm_id_map0_ne20_6"),
        8 => Ok("moe_mm_id_map0_ne20_8"),
        10 => Ok("moe_mm_id_map0_ne20_10"),
        16 => Ok("moe_mm_id_map0_ne20_16"),
        22 => Ok("moe_mm_id_map0_ne20_22"),
        _ => Err(MlxError::InvalidArgument(format!(
            "moe_mm_id_map0: ne20={} not in supported set {:?}",
            ne20, SUPPORTED_NE20
        ))),
    }
}

/// Dispatch the map0 pre-pass on the GPU.
///
/// # Arguments
/// - `src2`      — top-k routing indices `[n_tokens, ne20]` (i32)
/// - `htpe`      — output token-counts `[n_experts]` (u32, written)
/// - `hids`      — output indices `[n_experts, n_tokens]` (i32, written)
/// - `params_buf`— pre-populated `MoeMmIdMap0Params` upload
/// - `n_experts` — `args.ne02`; sets threadgroup width
/// - `ne20`      — `args.ne20`; selects template specialization
pub fn dispatch_moe_mm_id_map0(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    src2: &MlxBuffer,
    htpe: &MlxBuffer,
    hids: &MlxBuffer,
    params_buf: &MlxBuffer,
    n_experts: u32,
    ne20: u32,
) -> Result<()> {
    if n_experts == 0 {
        return Err(MlxError::InvalidArgument(
            "moe_mm_id_map0: n_experts must be > 0".into(),
        ));
    }
    if htpe.element_count() < n_experts as usize {
        return Err(MlxError::InvalidArgument(format!(
            "moe_mm_id_map0: htpe element_count={} < n_experts {}",
            htpe.element_count(),
            n_experts
        )));
    }

    let kernel_name = kernel_name_for(ne20)?;
    let pipeline = registry.get_pipeline(kernel_name, device)?;

    // One thread per expert. Cap at 1024 (Metal SIMD limit) — if n_experts
    // exceeds 1024 we'd need a multi-pass design, but Qwen3.6 (128 experts)
    // and larger MoE models (DeepSeek-V3 256 experts) stay under the cap.
    if n_experts > 1024 {
        return Err(MlxError::InvalidArgument(format!(
            "moe_mm_id_map0: n_experts={} exceeds single-threadgroup cap 1024",
            n_experts
        )));
    }
    let tg_size = n_experts as u64;

    // Threadgroup shmem: each thread loads ONE token's top-k list into a
    // private striped slot. shmem = ntg * ne20 * sizeof(uint16_t).
    let shmem_bytes = tg_size * ne20 as u64 * 2;

    encoder.encode_threadgroups_with_shared(
        pipeline,
        &[
            (0, params_buf),
            (1, src2),
            (2, htpe),
            (3, hids),
        ],
        &[(0, shmem_bytes)],
        MTLSize::new(1, 1, 1),       // single threadgroup
        MTLSize::new(tg_size, 1, 1), // one thread per expert
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CPU reference for the map0 pre-pass — used to validate the GPU
    /// kernel produces byte-identical (tpe, hids) outputs.
    fn cpu_reference(
        src2: &[i32],
        n_tokens: usize,
        ne20: usize,
        n_experts: usize,
    ) -> (Vec<u32>, Vec<i32>) {
        let mut tpe = vec![0u32; n_experts];
        let mut hids = vec![0i32; n_experts * n_tokens];

        for ide in 0..n_experts {
            let mut n_all = 0u32;
            for i21 in 0..n_tokens {
                let mut sel = 0i32;
                for i20 in 0..ne20 {
                    if src2[i21 * ne20 + i20] == ide as i32 {
                        sel = (i20 + 1) as i32;
                        break;
                    }
                }
                if sel > 0 {
                    hids[ide * n_tokens + n_all as usize] =
                        (i21 * ne20) as i32 + sel - 1;
                    n_all += 1;
                }
            }
            tpe[ide] = n_all;
        }

        (tpe, hids)
    }

    #[test]
    fn cpu_reference_basic_top_1() {
        // 4 tokens, 4 experts, top-1: each token names exactly one expert.
        let src2 = vec![
            0i32, // tok 0 -> expert 0
            1,    // tok 1 -> expert 1
            0,    // tok 2 -> expert 0
            2,    // tok 3 -> expert 2
        ];
        let (tpe, hids) = cpu_reference(&src2, 4, 1, 4);
        // Expert 0 was named by tokens 0 and 2 in slot 0 -> packed = 0*1+0=0 and 2*1+0=2.
        assert_eq!(tpe, vec![2, 1, 1, 0]);
        // hids layout: [ide=0:[0,2,_,_], ide=1:[1,_,_,_], ide=2:[3,_,_,_], ide=3:[_,_,_,_]]
        assert_eq!(&hids[0..2], &[0, 2]);
        assert_eq!(&hids[4..5], &[1]);
        assert_eq!(&hids[8..9], &[3]);
        assert_eq!(tpe[3], 0);
    }

    #[test]
    fn cpu_reference_top_k_packed_slot_index() {
        // 2 tokens, 4 experts, top-2: slot index must encode which of the
        // top-2 positions the expert occupied.
        let src2 = vec![
            0i32, 1, // tok 0 -> [expert 0 (slot 0), expert 1 (slot 1)]
            1,    2, // tok 1 -> [expert 1 (slot 0), expert 2 (slot 1)]
        ];
        let (tpe, hids) = cpu_reference(&src2, 2, 2, 4);
        // Expert 0: tok 0 slot 0 -> packed = 0*2 + 0 = 0
        assert_eq!(tpe[0], 1);
        assert_eq!(hids[0 * 2], 0);
        // Expert 1: tok 0 slot 1 (packed=0*2+1=1) + tok 1 slot 0 (packed=1*2+0=2)
        assert_eq!(tpe[1], 2);
        assert_eq!(&hids[1 * 2..1 * 2 + 2], &[1, 2]);
        // Expert 2: tok 1 slot 1 -> packed = 1*2 + 1 = 3
        assert_eq!(tpe[2], 1);
        assert_eq!(hids[2 * 2], 3);
        // Expert 3: never named
        assert_eq!(tpe[3], 0);
    }

    #[test]
    fn supported_ne20_set_matches_llamacpp() {
        // Pinned set from /opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal
        // lines 9780-9788. Drift here means the iter B main-kernel port
        // also needs to track new specializations.
        assert_eq!(SUPPORTED_NE20, &[1u32, 2, 4, 5, 6, 8, 10, 16, 22]);
    }

    #[test]
    fn kernel_name_dispatch_table_complete() {
        for &ne20 in SUPPORTED_NE20 {
            let name = kernel_name_for(ne20).expect("supported ne20 should resolve");
            assert!(name.starts_with("moe_mm_id_map0_ne20_"));
        }
        assert!(kernel_name_for(3).is_err());
        assert!(kernel_name_for(0).is_err());
    }

    /// GPU parity test at Qwen3.6 35B-A3B production shape — verifies
    /// `dispatch_moe_mm_id_map0` produces byte-identical (tpe, hids)
    /// outputs vs the CPU reference at (n_tokens=576, ne20=8,
    /// n_experts=128). Iter A close-out.
    #[test]
    fn gpu_parity_qwen36_shape() {
        use crate::dtypes::DType;
        use crate::device::MlxDevice;

        let n_tokens = 576usize;
        let ne20 = 8usize;
        let n_experts = 128usize;

        // Deterministic LCG routing — each token picks 8 distinct experts.
        let mut src2 = vec![0i32; n_tokens * ne20];
        let mut rng_state: u64 = 0xDEADBEEF_CAFEBABE;
        for i21 in 0..n_tokens {
            let mut chosen = vec![false; n_experts];
            let mut slot = 0usize;
            while slot < ne20 {
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let pick = ((rng_state >> 32) as u32 % n_experts as u32) as usize;
                if !chosen[pick] {
                    chosen[pick] = true;
                    src2[i21 * ne20 + slot] = pick as i32;
                    slot += 1;
                }
            }
        }

        let (cpu_tpe, cpu_hids) = cpu_reference(&src2, n_tokens, ne20, n_experts);

        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return, // no Metal device (non-Apple CI)
        };
        let mut registry = KernelRegistry::default();

        // src2: int32 [n_tokens, ne20]
        let src2_bytes = n_tokens * ne20 * std::mem::size_of::<i32>();
        let mut src2_buf = device
            .alloc_buffer(src2_bytes, DType::I32, vec![n_tokens, ne20])
            .expect("alloc src2");
        src2_buf.as_mut_slice::<i32>().expect("src2 slice").copy_from_slice(&src2);

        // params: MoeMmIdMap0Params (40 bytes)
        let nb21 = (ne20 * std::mem::size_of::<i32>()) as u64;
        let params = MoeMmIdMap0Params {
            ne02: n_experts as i32,
            ne10: ne20 as i32,
            ne11: ne20 as i32,
            nb11: 0,
            nb12: 0,
            ne21: n_tokens as i32,
            ne20: ne20 as i32,
            nb21,
        };
        let params_bytes_len = std::mem::size_of::<MoeMmIdMap0Params>();
        let mut params_buf = device
            .alloc_buffer(params_bytes_len, DType::U8, vec![params_bytes_len])
            .expect("alloc params");
        unsafe {
            let dst = params_buf.as_mut_slice::<u8>().expect("params slice").as_mut_ptr();
            std::ptr::copy_nonoverlapping(
                &params as *const MoeMmIdMap0Params as *const u8,
                dst,
                params_bytes_len,
            );
        }

        // outputs
        let htpe_buf = device
            .alloc_buffer(n_experts * 4, DType::U32, vec![n_experts])
            .expect("alloc htpe");
        let hids_buf = device
            .alloc_buffer(
                n_experts * n_tokens * 4,
                DType::I32,
                vec![n_experts, n_tokens],
            )
            .expect("alloc hids");

        let mut enc = device.command_encoder().expect("encoder");
        dispatch_moe_mm_id_map0(
            &mut enc,
            &mut registry,
            device.metal_device(),
            &src2_buf,
            &htpe_buf,
            &hids_buf,
            &params_buf,
            n_experts as u32,
            ne20 as u32,
        )
        .expect("dispatch");
        enc.commit_and_wait().expect("commit_and_wait");

        let gpu_tpe: &[u32] = htpe_buf.as_slice().expect("read htpe");
        let gpu_hids: &[i32] = hids_buf.as_slice().expect("read hids");

        assert_eq!(
            gpu_tpe, cpu_tpe.as_slice(),
            "tpe mismatch (n_experts={n_experts})"
        );
        // Tail beyond tpe[ide] is uninitialized — only compare populated prefix.
        for ide in 0..n_experts {
            let len = cpu_tpe[ide] as usize;
            if len == 0 {
                continue;
            }
            let cpu_slice = &cpu_hids[ide * n_tokens..ide * n_tokens + len];
            let gpu_slice = &gpu_hids[ide * n_tokens..ide * n_tokens + len];
            assert_eq!(
                gpu_slice, cpu_slice,
                "hids mismatch at expert {ide} (tpe={len})"
            );
        }
    }
}
