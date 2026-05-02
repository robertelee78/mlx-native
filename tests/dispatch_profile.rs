//! Per-dispatch GPU counter sampling end-to-end test (ADR-015 iter63
//! Phase A.4).
//!
//! Verifies that when `MLX_PROFILE_DISPATCH=1` is set:
//!
//! * If the device supports `MTLCounterSamplingPoint::AtDispatchBoundary`
//!   (post-2026 Apple Silicon, AMD/Intel discrete, simulators), per-CB
//!   `commit_and_wait_labeled` produces non-empty `dump_dispatches`
//!   with monotone end ≥ start timestamps and the right `op_kind` /
//!   `dispatch_index` ordering.
//!
//! * If the device does NOT support it (current Apple Silicon —
//!   verified runtime: AGXG17XFamilyComputeContext only supports
//!   `AtStageBoundary`), `dump_dispatches` is empty and a one-shot
//!   stderr warning is emitted.  The kit gracefully degrades; no
//!   panics, no assertion failures.  Per-CB profiling (separate env
//!   gate `MLX_PROFILE_CB=1`) is unaffected.
//!
//! ## Cross-validation with per-CB ground truth (Risk R3)
//!
//! When the device supports per-dispatch sampling, the
//! `with_barrier:true` semantics serialize the encoder under
//! `MTLDispatchTypeConcurrent`; the sum-of-per-dispatch `gpu_ns`
//! therefore represents an upper-bound serialized cost.  We assert
//! that the per-CB total is *at least as large* as the sum (since
//! per-CB also includes `apply_labels` + commit overhead).  We do
//! NOT pin a tight ratio in CI — the 5% drift gate lives in the bench
//! script.
//!
//! ## Env-cache caveat
//!
//! `is_dispatch_enabled()` and `is_enabled()` cache their env-var
//! decision in an AtomicI8 on first read.  We set both vars in a
//! `static OnceLock` init guard at module scope so they precede any
//! `is_dispatch_enabled` call; cargo runs each `tests/*.rs` in its
//! own binary, so the caches are virgin per `cargo test --test
//! dispatch_profile`.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::sync::OnceLock;

use mlx_native::{DType, KernelRegistry, MlxDevice};

/// One-shot env-var setter; runs the first time any test in this file
/// calls [`enable_dispatch_profile`].  Must complete before any
/// `kernel_profile::is_dispatch_enabled` call to avoid the AtomicI8
/// cache going cold-OFF.
fn enable_dispatch_profile() {
    static GUARD: OnceLock<()> = OnceLock::new();
    GUARD.get_or_init(|| {
        // SAFETY: set_var on env vars before any other code in this
        // process reads them.  Tests in this binary are the only
        // consumers of the cached flags.
        unsafe {
            std::env::set_var("MLX_PROFILE_DISPATCH", "1");
            std::env::set_var("MLX_PROFILE_CB", "1");
        }
    });
}

/// Whether the device supports `AtDispatchBoundary` counter sampling.
/// On Apple Silicon (M-series) this is `false` — only `AtStageBoundary`
/// is supported, which is incompatible with the persistent-encoder
/// pattern.  On other platforms (AMD/Intel discrete, simulators) it
/// is `true` and the per-dispatch entries will populate.
fn supports_dispatch_boundary_sampling(device: &MlxDevice) -> bool {
    use metal::MTLCounterSamplingPoint;
    device
        .metal_device()
        .supports_counter_sampling(MTLCounterSamplingPoint::AtDispatchBoundary)
}

/// Compile a tiny "increment by 1" Metal kernel and return (registry,
/// pipeline-name) ready for repeated dispatches.  The kernel writes
/// `out[id] = in[id] + 1.0` for an N-element f32 buffer.
fn make_inc_kernel() -> (KernelRegistry, &'static str) {
    let mut registry = KernelRegistry::new();
    registry.register_source(
        "inc_one",
        r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void inc_one(
            device const float *in   [[buffer(0)]],
            device       float *out  [[buffer(1)]],
            uint id [[thread_position_in_grid]]
        ) {
            out[id] = in[id] + 1.0f;
        }
        "#,
    );
    (registry, "inc_one")
}

#[test]
fn per_dispatch_sampling_records_nonzero_ns() {
    enable_dispatch_profile();
    mlx_native::kernel_profile::reset();

    let device = MlxDevice::new().expect("MlxDevice::new");
    let supported = supports_dispatch_boundary_sampling(&device);
    let (mut registry, kernel) = make_inc_kernel();
    let pipeline = registry
        .get_pipeline(kernel, device.metal_device())
        .expect("get_pipeline")
        .clone();

    // Two F32 buffers, 256 elements each.
    let n: usize = 256;
    let byte_len = n * std::mem::size_of::<f32>();
    let mut in_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![n])
        .expect("alloc in");
    let out_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![n])
        .expect("alloc out");
    {
        let s: &mut [f32] = in_buf.as_mut_slice().expect("in slice");
        for (i, v) in s.iter_mut().enumerate() {
            *v = i as f32;
        }
    }

    // Encode TWO dispatches into the same CB.  We use
    // encode_threadgroups_with_args so we hit one of the production
    // dispatch paths.
    let mut enc = device.command_encoder().expect("encoder");
    use mlx_native::KernelArg;
    use metal::MTLSize;
    let bindings: Vec<(u64, KernelArg<'_>)> = vec![
        (0, KernelArg::Buffer(&in_buf)),
        (1, KernelArg::Buffer(&out_buf)),
    ];
    let tg = MTLSize { width: (n as u64 + 31) / 32, height: 1, depth: 1 };
    let tg_size = MTLSize { width: 32, height: 1, depth: 1 };
    enc.encode_threadgroups_with_args(&pipeline, &bindings, tg, tg_size);
    enc.memory_barrier();
    enc.encode_threadgroups_with_args(&pipeline, &bindings, tg, tg_size);
    enc.commit_and_wait_labeled("test.profile.cb")
        .expect("commit_and_wait_labeled");

    let dumps = mlx_native::kernel_profile::dump_dispatches();
    if !supported {
        // Apple-Silicon path (AtDispatchBoundary unsupported): per-
        // dispatch table must be empty (graceful degrade per
        // ensure_sample_buffer's stderr warn).  Per-CB table still
        // populates because MLX_PROFILE_CB=1 path is independent.
        eprintln!(
            "[test] device does NOT support AtDispatchBoundary; \
             verifying graceful degrade"
        );
        assert!(
            dumps.is_empty(),
            "device unsupported → dump_dispatches must be empty; got {:?}",
            dumps
        );
        let cb_dump = mlx_native::kernel_profile::dump();
        assert!(
            cb_dump.iter().any(|(l, _)| l == "test.profile.cb"),
            "MLX_PROFILE_CB=1 should still produce a per-CB entry on \
             Apple Silicon (independent code path); cb_dump={:?}",
            cb_dump
        );
        mlx_native::kernel_profile::reset();
        return;
    }

    // Supported path: assert real per-dispatch entries.
    assert!(
        !dumps.is_empty(),
        "device supports AtDispatchBoundary AND \
         MLX_PROFILE_DISPATCH=1 set + commit_and_wait_labeled called \
         → expected at least one DispatchEntry; dumps={:?}",
        dumps
    );
    let (label, entries) = dumps
        .iter()
        .find(|(l, _)| l == "test.profile.cb")
        .expect("test.profile.cb dispatches present");
    assert_eq!(label, "test.profile.cb");
    assert_eq!(
        entries.len(),
        2,
        "exactly 2 dispatches encoded; got {} entries",
        entries.len()
    );
    for (i, e) in entries.iter().enumerate() {
        assert_eq!(
            e.dispatch_index as usize, i,
            "dispatch_index must equal insertion order; entry={:?}",
            e
        );
        assert!(
            e.end_gpu_ns >= e.start_gpu_ns,
            "end timestamp must follow start; entry={:?}",
            e
        );
        assert_eq!(e.op_kind, "Other", "default op kind = Other; got {:?}", e);
    }

    // Cross-validation per Risk R3.
    let cb_dump = mlx_native::kernel_profile::dump();
    let cb_entry = cb_dump
        .iter()
        .find(|(l, _)| l == "test.profile.cb")
        .expect("MLX_PROFILE_CB=1 should produce a per-CB entry");
    let cb_total = cb_entry.1.total_ns;
    let dispatch_sum: u64 = entries.iter().map(|e| e.gpu_ns).sum();
    assert!(
        cb_total + 1_000_000 >= dispatch_sum,
        "per-CB total ({} ns) must be >= sum of per-dispatch ({} ns) \
         within ±1ms slop; cb_entry={:?}, entries={:?}",
        cb_total,
        dispatch_sum,
        cb_entry,
        entries
    );

    mlx_native::kernel_profile::reset();
}

#[test]
fn per_dispatch_dump_is_grouped_by_cb_label() {
    enable_dispatch_profile();
    mlx_native::kernel_profile::reset();

    let device = MlxDevice::new().expect("MlxDevice::new");
    let supported = supports_dispatch_boundary_sampling(&device);
    let (mut registry, kernel) = make_inc_kernel();
    let pipeline = registry
        .get_pipeline(kernel, device.metal_device())
        .expect("get_pipeline")
        .clone();
    let n: usize = 64;
    let byte_len = n * std::mem::size_of::<f32>();
    let in_buf = device.alloc_buffer(byte_len, DType::F32, vec![n]).unwrap();
    let out_buf = device.alloc_buffer(byte_len, DType::F32, vec![n]).unwrap();

    use mlx_native::KernelArg;
    use metal::MTLSize;
    let bindings: Vec<(u64, KernelArg<'_>)> = vec![
        (0, KernelArg::Buffer(&in_buf)),
        (1, KernelArg::Buffer(&out_buf)),
    ];
    let tg = MTLSize { width: (n as u64 + 31) / 32, height: 1, depth: 1 };
    let tg_size = MTLSize { width: 32, height: 1, depth: 1 };

    {
        let mut enc = device.command_encoder().unwrap();
        enc.encode_threadgroups_with_args(&pipeline, &bindings, tg, tg_size);
        enc.commit_and_wait_labeled("test.cb.alpha").unwrap();
    }
    {
        let mut enc = device.command_encoder().unwrap();
        enc.encode_threadgroups_with_args(&pipeline, &bindings, tg, tg_size);
        enc.commit_and_wait_labeled("test.cb.beta").unwrap();
    }

    let dumps = mlx_native::kernel_profile::dump_dispatches();
    if !supported {
        assert!(dumps.is_empty(), "unsupported → empty dispatches; got {:?}", dumps);
        mlx_native::kernel_profile::reset();
        return;
    }
    let alpha = dumps
        .iter()
        .find(|(l, _)| l == "test.cb.alpha")
        .expect("alpha label");
    let beta = dumps
        .iter()
        .find(|(l, _)| l == "test.cb.beta")
        .expect("beta label");
    assert_eq!(alpha.1.len(), 1, "alpha has 1 dispatch");
    assert_eq!(beta.1.len(), 1, "beta has 1 dispatch");
    assert_eq!(alpha.1[0].dispatch_index, 0);
    assert_eq!(beta.1[0].dispatch_index, 0);

    mlx_native::kernel_profile::reset();
}

#[test]
fn op_kind_label_propagates_into_dispatch_entry() {
    enable_dispatch_profile();
    mlx_native::kernel_profile::reset();

    use mlx_native::CapturedOpKind;
    let device = MlxDevice::new().expect("device");
    let supported = supports_dispatch_boundary_sampling(&device);
    let (mut registry, kernel) = make_inc_kernel();
    let pipeline = registry
        .get_pipeline(kernel, device.metal_device())
        .expect("pipeline")
        .clone();
    let n: usize = 64;
    let byte_len = n * std::mem::size_of::<f32>();
    let in_buf = device.alloc_buffer(byte_len, DType::F32, vec![n]).unwrap();
    let out_buf = device.alloc_buffer(byte_len, DType::F32, vec![n]).unwrap();

    let mut enc = device.command_encoder().unwrap();
    enc.set_op_kind(CapturedOpKind::Sdpa);
    use mlx_native::KernelArg;
    use metal::MTLSize;
    let bindings: Vec<(u64, KernelArg<'_>)> = vec![
        (0, KernelArg::Buffer(&in_buf)),
        (1, KernelArg::Buffer(&out_buf)),
    ];
    let tg = MTLSize { width: (n as u64 + 31) / 32, height: 1, depth: 1 };
    let tg_size = MTLSize { width: 32, height: 1, depth: 1 };
    enc.encode_threadgroups_with_args(&pipeline, &bindings, tg, tg_size);
    enc.commit_and_wait_labeled("test.opkind.cb").unwrap();

    let dumps = mlx_native::kernel_profile::dump_dispatches();
    if !supported {
        assert!(dumps.is_empty(), "unsupported → empty; got {:?}", dumps);
        mlx_native::kernel_profile::reset();
        return;
    }
    let (_, entries) = dumps
        .iter()
        .find(|(l, _)| l == "test.opkind.cb")
        .expect("opkind cb present");
    assert_eq!(entries.len(), 1);
    assert_eq!(
        entries[0].op_kind, "Sdpa",
        "set_op_kind(Sdpa) must propagate into DispatchEntry; got {:?}",
        entries[0]
    );

    mlx_native::kernel_profile::reset();
}

#[test]
fn supports_counter_sampling_capability_diagnostic() {
    // Diagnostic test — emits the device + supported sampling points
    // to stderr so operators reading the bench log can confirm which
    // path the kit took on this hardware.  Always passes; never
    // gates CI.
    let device = MlxDevice::new().expect("device");
    let md = device.metal_device();
    use metal::MTLCounterSamplingPoint;
    eprintln!(
        "[mlx-native diag] device={:?} \
         supports_counter_sampling: \
         AtStageBoundary={} AtDispatchBoundary={} \
         AtBlitBoundary={} AtTileDispatchBoundary={} AtDrawBoundary={}",
        device.name(),
        md.supports_counter_sampling(MTLCounterSamplingPoint::AtStageBoundary),
        md.supports_counter_sampling(MTLCounterSamplingPoint::AtDispatchBoundary),
        md.supports_counter_sampling(MTLCounterSamplingPoint::AtBlitBoundary),
        md.supports_counter_sampling(MTLCounterSamplingPoint::AtTileDispatchBoundary),
        md.supports_counter_sampling(MTLCounterSamplingPoint::AtDrawBoundary),
    );
}
