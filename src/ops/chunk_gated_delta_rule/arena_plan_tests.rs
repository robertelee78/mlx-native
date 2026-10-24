use super::*;
use crate::ops::chunk_gated_delta_rule::chunk_cumsum_param_words;
use crate::ops::chunk_gated_delta_rule_tri_solve_invert::{
    chunk_tri_solve_invert_param_words, ChunkTriSolveInvertParams,
};
use crate::ops::gated_delta_net_chunk::{
    gated_delta_net_chunk_param_words, GatedDeltaNetChunkParams,
};
use crate::ops::gated_delta_net_chunk_o::{
    gated_delta_net_chunk_o_param_words, GatedDeltaNetChunkOParams,
};
use crate::ops::gated_delta_net_kkt::{gated_delta_net_kkt_param_words, GatedDeltaNetKktParams};
use crate::ops::gated_delta_net_recompute_wu::{
    gated_delta_net_recompute_wu_param_words, GatedDeltaNetRecomputeWuParams,
};

#[test]
fn qwen38_selected_endpoint_has_all_thirteen_exact_flat_roots() {
    let plan = ChunkInternalArenaPlan::for_shape(1, 2048, 48, 48, 128, 128, 64).unwrap();
    use ChunkArenaRole::*;
    let roles = [
        Cumsum,
        Strict,
        Inverse,
        W,
        U,
        Hidden,
        VNew,
        CumsumParams,
        KktParams,
        InvertParams,
        RecomputeWuParams,
        ChunkParams,
        ChunkOParams,
    ];
    let bytes = [
        393216, 25165824, 25165824, 25165824, 25165824, 50331648, 25165824, 20, 32, 16, 32, 32, 44,
    ];
    for (index, root) in plan.roots().iter().enumerate() {
        let dtype = if index < 3 {
            DType::F32
        } else if index < 7 {
            DType::BF16
        } else {
            DType::U32
        };
        assert_eq!(root.role(), roles[index]);
        assert_eq!(root.dtype(), dtype);
        assert_eq!(root.requested_bytes(), bytes[index]);
        assert_eq!(root.shape(), [bytes[index] / dtype.size_of()]);
        assert_eq!(plan.root(roles[index]), root);
    }
    assert_eq!(plan.requested_bytes(), 176_554_160);
    assert_eq!(
        plan.roots()[7..]
            .iter()
            .map(|r| r.requested_bytes())
            .sum::<usize>(),
        176
    );
}
#[test]
fn shared_parameter_serialization_keeps_padding_constants_and_scale_bits() {
    let (b, t, hg, h, k, v, bt) = (1, 2048, 48, 48, 128, 128, 64);
    assert_eq!(chunk_cumsum_param_words(b, t, h, bt), [1, 2048, 48, 64, 32]);
    assert_eq!(
        gated_delta_net_kkt_param_words(GatedDeltaNetKktParams { b, t, hg, h, k, bt }),
        [1, 2048, 48, 48, 128, 64, 32, 64]
    );
    assert_eq!(
        chunk_tri_solve_invert_param_words(ChunkTriSolveInvertParams { b, t, h, bt }),
        [1, 2048, 48, 64]
    );
    assert_eq!(
        gated_delta_net_recompute_wu_param_words(GatedDeltaNetRecomputeWuParams {
            b,
            t,
            hg,
            h,
            k,
            v,
            bt
        }),
        [1, 2048, 48, 48, 128, 128, 64, 32]
    );
    assert_eq!(
        gated_delta_net_chunk_param_words(GatedDeltaNetChunkParams {
            b,
            t,
            hg,
            h,
            k,
            v,
            bt
        }),
        [1, 2048, 48, 48, 128, 128, 64, 32]
    );
    assert_eq!(
        gated_delta_net_chunk_o_param_words(GatedDeltaNetChunkOParams {
            b,
            t,
            hg,
            h,
            k,
            v,
            bt,
            scale: 1.0
        }),
        [1, 2048, 48, 48, 128, 128, 64, 32, 32, 32, 1.0f32.to_bits()]
    );
    let scaled = gated_delta_net_chunk_o_param_words(GatedDeltaNetChunkOParams {
        b,
        t,
        hg,
        h,
        k,
        v,
        bt,
        scale: 0.25,
    });
    assert_eq!(scaled[10], 0.25f32.to_bits());
}
#[test]
fn minimum_multibatch_grouped_heads_and_max_v_preserve_constructor_scope() {
    let a = ChunkInternalArenaPlan::for_shape(1, 64, 1, 2, 128, 256, 64).unwrap();
    let b = ChunkInternalArenaPlan::for_shape(2, 64, 1, 2, 128, 256, 64).unwrap();
    for (small, large) in a.roots()[..7].iter().zip(&b.roots()[..7]) {
        assert_eq!(large.requested_bytes(), 2 * small.requested_bytes());
    }
    assert_eq!(a.roots()[7..], b.roots()[7..]);
    // Existing constructor accepts V1: this plan is allocation eligibility,
    // not a claim all downstream subkernels support that V without rejection.
    assert!(ChunkInternalArenaPlan::for_shape(1, 64, 1, 1, 128, 1, 64).is_ok());
}
#[test]
fn constraints_fail_in_existing_order_before_any_allocation() {
    for dims in [
        (0, 0, 0, 0, 0, 0, 0),
        (1, 64, 2, 3, 128, 128, 64),
        (1, 64, 1, 1, 256, 128, 64),
        (1, 64, 1, 1, 128, 257, 64),
        (1, 64, 1, 1, 128, 128, 32),
        (1, 65, 1, 1, 128, 128, 64),
    ] {
        let (b, t, hg, h, k, v, bt) = dims;
        assert!(ChunkInternalArenaPlan::for_shape(b, t, hg, h, k, v, bt).is_err());
    }
    let error = ChunkInternalArenaPlan::for_shape(1, 64, 2, 3, 256, 257, 32)
        .unwrap_err()
        .to_string();
    assert!(error.contains("multiple of Hg"));
}
#[test]
fn old_u32_product_overflow_is_explicit_error_not_wrapping_or_widening() {
    let error = ChunkInternalArenaPlan::for_shape(u32::MAX, 64, 1, 1, 128, 128, 64)
        .unwrap_err()
        .to_string();
    assert!(error.contains("Cumsum element overflow"));
    // g fits, but the larger A root's original u32 product does not.
    let error = ChunkInternalArenaPlan::for_shape(1, 64, 1, 1_048_576, 128, 128, 64)
        .unwrap_err()
        .to_string();
    assert!(error.contains("Strict element overflow"));
}

/// Actual constructor/buffer readback only. No kernel, encoder or dispatch.
#[test]
#[ignore = "13 tiny Metal allocations/CPU readback; explicit payload115120B, no GPU dispatch"]
fn actual_constructor_matches_thirteen_root_plan_and_parameter_contents() {
    use crate::ops::chunk_gated_delta_rule::ChunkInternalArena;
    use metal::foreign_types::ForeignType;
    use ChunkArenaRole::*;
    let device = crate::MlxDevice::new().expect("mandatory Metal device, no skip");
    let plan = ChunkInternalArenaPlan::for_shape(1, 64, 1, 1, 128, 128, 64).unwrap();
    let arena = ChunkInternalArena::new(&device, 1, 64, 1, 1, 128, 128, 64).unwrap();
    let actual = [
        (Cumsum, &arena.g_cumsum_buf),
        (Strict, &arena.a_strict_buf),
        (Inverse, &arena.a_inv_buf),
        (W, &arena.w_buf),
        (U, &arena.u_buf),
        (Hidden, &arena.h_buf),
        (VNew, &arena.v_new_buf),
        (CumsumParams, &arena.cumsum_params_buf),
        (KktParams, &arena.kkt_params_buf),
        (InvertParams, &arena.invert_params_buf),
        (RecomputeWuParams, &arena.recompute_wu_params_buf),
        (ChunkParams, &arena.chunk_params_buf),
        (ChunkOParams, &arena.chunk_o_params_buf),
    ];
    let mut bytes = 0usize;
    for (index, (role, buffer)) in actual.iter().enumerate() {
        let descriptor = plan.root(*role);
        assert_eq!(descriptor.role(), plan.roots()[index].role());
        assert_eq!(buffer.dtype(), descriptor.dtype());
        assert_eq!(buffer.shape(), descriptor.shape().as_slice());
        assert_eq!(buffer.byte_len(), descriptor.requested_bytes());
        assert_eq!(buffer.data_byte_len(), descriptor.requested_bytes());
        bytes += buffer.byte_len();
        if index < 7 {
            // Actual allocator zero-initialization across the WHOLE large root,
            // not just a chosen logical prefix or an assumed fresh heap page.
            assert!(buffer
                .as_slice::<u8>()
                .unwrap()
                .iter()
                .all(|byte| *byte == 0));
        }
        for (_, earlier) in &actual[..index] {
            assert_ne!(
                buffer.metal_buffer().as_ptr(),
                earlier.metal_buffer().as_ptr(),
                "two constructor fields alias the same supposed owned root"
            );
        }
    }
    // Fixed independent expected parameter CONTENTS, not parameter helper output.
    let expected: &[(ChunkArenaRole, &[u32])] = &[
        (CumsumParams, &[1, 64, 1, 64, 1]),
        (KktParams, &[1, 64, 1, 1, 128, 64, 1, 64]),
        (InvertParams, &[1, 64, 1, 64]),
        (RecomputeWuParams, &[1, 64, 1, 1, 128, 128, 64, 1]),
        (ChunkParams, &[1, 64, 1, 1, 128, 128, 64, 1]),
        (
            ChunkOParams,
            &[1, 64, 1, 1, 128, 128, 64, 1, 32, 32, 0x3f80_0000],
        ),
    ];
    for (role, words) in expected {
        let buffer = actual
            .iter()
            .find(|(candidate, _)| candidate == role)
            .unwrap()
            .1;
        assert_eq!(
            buffer.as_slice::<u32>().unwrap(),
            *words,
            "{role:?} contents"
        );
    }
    assert_eq!(bytes, 115_120);
    assert_eq!(plan.requested_bytes(), bytes as u64);
    // Bounded explicit buffer payload + finite borrowed/static test descriptors;
    // device/residency/Arc/allocator metadata is separately Unknown.
    assert!(bytes + 4096 <= 16 * 1024 * 1024);
    eprintln!("CHUNK_ARENA_ALLOCATION completed roots=13 requested_payload_bytes={bytes} zeroed_large_roots=7 parameter_roots=6 gpu_dispatch=false model_loaded=false runtime_admission=false");
}
