//! One-shot recorded GPU-only reuse; independent from quiescent CPU control.
use super::*;
use crate::{DType, GraphExecutor, KernelRegistry, MlxBuffer, MlxDevice};

const N: usize = 16;
const BYTES: usize = N * 4;
fn root_id(buffer: &MlxBuffer) -> usize {
    &**buffer.metal_buffer() as *const metal::BufferRef as usize
}
fn range(buffer: &MlxBuffer) -> (usize, usize) {
    let start = buffer.contents_ptr() as usize + buffer.byte_offset() as usize;
    (start, start + buffer.byte_len())
}
fn initialized(device: &MlxDevice, value: f32) -> MlxBuffer {
    let mut buffer = device.alloc_buffer(BYTES, DType::F32, vec![N]).unwrap();
    buffer.as_mut_slice::<f32>().unwrap().fill(value);
    buffer
}
fn exact(actual: &[f32], expected: &[f32]) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(a, e)| a.is_finite() && a == e)
}

/// Fresh isolated process. Ten roots/640 requested bytes including nine direct
/// fixture roots outside this authority, plus eight 4-byte inline args. No model.
/// Driver/registry/capture/allocator object overhead is explicitly unknown.
#[test]
#[ignore = "tiny recorded GPU control; exact fresh-process supervised execution only"]
fn actual_ordered_gpu_alias_reuse_and_standalone_lineage() {
    for key in [
        "MLX_UNRETAINED_REFS",
        "MLX_METAL_CAPTURE",
        "METAL_CAPTURE_ENABLED",
        "MLX_PROFILE_DISPATCH",
        "MLX_PROFILE_CB",
        "MLX_PROFILE_BARRIERS",
        "HF2Q_PROFILE_GPU_TS",
        "HF2Q_DECODE_PROFILE",
        "HF2Q_REORDER_DUMP",
    ] {
        assert!(std::env::var_os(key).is_none(), "forbidden override {key}");
    }
    objc::rc::autoreleasepool(|| {
        let device = MlxDevice::new().expect("Metal required; absence is not completion");
        let mut registry = KernelRegistry::new();
        let quota = GrowthQuota {
            bytes: BYTES,
            roots: 1,
            attempts: 1,
        };
        let authority = LegacyPoolGrowthAuthority::install(LegacyGrowthLimits {
            quota,
            max_pools: 1,
            max_buckets_per_pool: 1,
            max_rank: 1,
            max_batch: 1,
        })
        .unwrap();
        let mut pool = authority
            .bind(
                &device,
                GrowthPoolKey {
                    kind: 1,
                    worker_generation: 1,
                },
                quota,
                &[GrowthBucket {
                    bytes: BYTES,
                    quota,
                }],
            )
            .unwrap();
        let mut x = initialized(&device, 0.0);
        let y = initialized(&device, 3.0);
        let z = initialized(&device, 2.0);
        for (i, value) in x.as_mut_slice::<f32>().unwrap().iter_mut().enumerate() {
            *value = i as f32 - 8.0;
        }
        let c = initialized(&device, 123456.0);
        let d = initialized(&device, -65432.0);
        let a_s = initialized(&device, 7777.0);
        let b_s = initialized(&device, 8888.0);
        let c_s = initialized(&device, 123456.0);
        let d_s = initialized(&device, -65432.0);
        let a = pool.alloc(&device, BYTES, DType::F32, vec![N]).unwrap();
        let old_identity = root_id(&a);
        let old_range = range(&a);
        let old_offset = a.byte_offset();
        // All ten simultaneously existing roots are distinct before the alias.
        let mut identities = [&x, &y, &z, &c, &d, &a_s, &b_s, &c_s, &d_s, &a].map(root_id);
        identities.sort_unstable();
        assert!(identities.windows(2).all(|w| w[0] != w[1]));
        let executor = GraphExecutor::new(device.clone());
        let mut session = executor.begin_recorded().unwrap();
        // Complete physical range annotations are mandatory for reorder. Plain
        // elementwise wrappers delegate kernels but do not supply these ranges.
        session.barrier_between(&[&x, &y], &[&a]);
        session
            .elementwise_add(
                &mut registry,
                device.metal_device(),
                &x,
                &y,
                &a,
                N,
                DType::F32,
            )
            .unwrap();
        session.barrier_between(&[&a, &z], &[&c]);
        session
            .elementwise_mul(
                &mut registry,
                device.metal_device(),
                &a,
                &z,
                &c,
                N,
                DType::F32,
            )
            .unwrap();
        session.barrier();
        drop(a); // Native captured bindings retain the outgoing root.
                 // ONE unsubmitted one-shot session; all old accesses encoded. Complete
                 // alias ranges preserve read-A -> write-B WAR through actual reorder.
        unsafe {
            pool.reset_for_ordered_gpu_reuse(&device).unwrap();
        }
        let b = pool
            .alloc_uninitialized(&device, BYTES, DType::F32, vec![N])
            .unwrap();
        assert_eq!(root_id(&b), old_identity);
        assert_eq!(range(&b), old_range);
        assert_eq!(b.byte_offset(), old_offset);
        assert_eq!(b.byte_len(), BYTES);
        assert_eq!(b.metal_buffer().length() as usize, BYTES);
        assert_eq!(
            pool.snapshot().unwrap().pool,
            GrowthCounts {
                bytes: BYTES,
                roots: 1,
                attempts: 1
            }
        );
        // No CPU initialization/read of B: its GPU producer covers all 64 bytes.
        session.barrier_between(&[&x, &y], &[&b]);
        session
            .elementwise_mul(
                &mut registry,
                device.metal_device(),
                &x,
                &y,
                &b,
                N,
                DType::F32,
            )
            .unwrap();
        session.barrier_between(&[&b, &z], &[&d]);
        session
            .elementwise_add(
                &mut registry,
                device.metal_device(),
                &b,
                &z,
                &d,
                N,
                DType::F32,
            )
            .unwrap();
        // All fallible encoding precedes the first submit. This consumes capture
        // exactly once and waits before any output access or assertion on results.
        let pooled_finish = session.finish_with_reorder();
        if pooled_finish.is_err() {
            authority.quarantine_generation(1).unwrap();
        }
        let (pooled_moved, pooled_barriers) = pooled_finish.unwrap();
        assert_eq!(pooled_moved, 0, "alias WAR/WAW forbids moving B before C");
        assert_eq!(pooled_barriers, 3, "derived RAW/WAR/RAW boundaries");

        let mut session = executor.begin_recorded().unwrap();
        session.barrier_between(&[&x, &y], &[&a_s]);
        session
            .elementwise_add(
                &mut registry,
                device.metal_device(),
                &x,
                &y,
                &a_s,
                N,
                DType::F32,
            )
            .unwrap();
        session.barrier_between(&[&a_s, &z], &[&c_s]);
        session
            .elementwise_mul(
                &mut registry,
                device.metal_device(),
                &a_s,
                &z,
                &c_s,
                N,
                DType::F32,
            )
            .unwrap();
        session.barrier();
        session.barrier_between(&[&x, &y], &[&b_s]);
        session
            .elementwise_mul(
                &mut registry,
                device.metal_device(),
                &x,
                &y,
                &b_s,
                N,
                DType::F32,
            )
            .unwrap();
        session.barrier_between(&[&b_s, &z], &[&d_s]);
        session
            .elementwise_add(
                &mut registry,
                device.metal_device(),
                &b_s,
                &z,
                &d_s,
                N,
                DType::F32,
            )
            .unwrap();
        let standalone_finish = session.finish_with_reorder();
        if standalone_finish.is_err() {
            authority.quarantine_generation(1).unwrap();
        }
        let (standalone_moved, standalone_barriers) = standalone_finish.unwrap();
        assert!(
            standalone_moved > 0,
            "independent B_s actually moves before C_s"
        );
        assert!(standalone_barriers < pooled_barriers);
        let expected_c: [f32; N] = std::array::from_fn(|i| ((i as f32 - 8.0) + 3.0) * 2.0);
        let expected_d: [f32; N] = std::array::from_fn(|i| (i as f32 - 8.0) * 3.0 + 2.0);
        let overwritten_c: [f32; N] = std::array::from_fn(|i| ((i as f32 - 8.0) * 3.0) * 2.0);
        assert!(
            !exact(&overwritten_c, &expected_c),
            "deterministic wrong-lineage negative; no GPU race"
        );
        let cv = c.as_slice::<f32>().unwrap();
        let dv = d.as_slice::<f32>().unwrap();
        let csv = c_s.as_slice::<f32>().unwrap();
        let dsv = d_s.as_slice::<f32>().unwrap();
        assert!(exact(cv, &expected_c) && exact(csv, &expected_c));
        assert!(exact(dv, &expected_d) && exact(dsv, &expected_d));
        assert!(exact(cv, csv) && exact(dv, dsv));
        drop(b);
        // Both sessions consumed and completed, all issued Rust handles dropped.
        unsafe {
            pool.reset(&device).unwrap();
        }
        println!("LEGACY_GROWTH_ORDERED_GPU_COMPLETE requested_bytes=640 roots=10 pooled_roots=1 pooled_moved={pooled_moved} pooled_barriers={pooled_barriers} standalone_moved={standalone_moved} standalone_barriers={standalone_barriers} cpu_oracle=exact standalone_parity=exact wrong_oracle_rejected=true");
    });
}
