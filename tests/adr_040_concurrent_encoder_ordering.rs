//! Smoke coverage for dependent dispatches through the retained concurrent encoder.
//!
//! The encoder must remain owned across autorelease-pool drains, and dependent
//! dispatches must preserve writer→barrier→reader ordering.
//!
//! This isolated, fully synchronized test does not reproduce the timing-sensitive
//! encoder-ownership failure. The production model-parity oracle is
//! `hf2q slot_aware_n8_per_slot_parity_vs_serial` with
//! `MLX_PRECOMPILED_METALLIB=0`; this test only covers the functional GPU path.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use mlx_native::{DType, GraphExecutor, KernelRegistry, MlxDevice};

/// Functional SMOKE test of the fixed writer→barrier→reader path.
///
/// Writer (`tmp = a + b`) → `barrier()` → reader (`out = tmp * c`) on the
/// persistent concurrent compute encoder, ×16, asserting `out == (a+b)*c`.
///
/// HONEST SCOPE — this is NOT a fail-direction repro of the ownership bug.
/// Verified: with the retain temporarily reverted (commented out), this test
/// still PASSES. An isolated elementwise add→mul edge with a per-iteration
/// `finish()` full-sync does not open the race window — the fast producer wins
/// by timing regardless of retain, exactly as the earlier deleted timing test
/// did. So this test guards only that the happy path stays correct; it does
/// NOT discriminate the ownership fix. The production model-parity oracle is
/// `hf2q slot_aware_n8_per_slot_parity_vs_serial` with
/// `MLX_PRECOMPILED_METALLIB=0`, which exercises the slow-producer workload.
/// Kept as a cheap smoke of dispatch ordering, not as an ownership regression gate.
#[test]
fn writer_barrier_reader_orders_through_retained_encoder() {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();

    let n = 4096usize;
    let byte_len = n * std::mem::size_of::<f32>();

    // Deterministic, non-degenerate inputs (no zeros so a stale tmp read can't
    // accidentally match the correct product).
    let a_data: Vec<f32> = (0..n).map(|i| 1.0 + (i % 97) as f32 * 0.5).collect();
    let b_data: Vec<f32> = (0..n).map(|i| 2.0 + (i % 53) as f32 * 0.25).collect();
    let c_data: Vec<f32> = (0..n).map(|i| 0.5 + (i % 31) as f32 * 0.125).collect();
    let expected: Vec<f32> = (0..n).map(|i| (a_data[i] + b_data[i]) * c_data[i]).collect();

    let mut a_buf = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("a");
    let mut b_buf = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("b");
    let mut c_buf = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("c");
    a_buf.as_mut_slice::<f32>().expect("a").copy_from_slice(&a_data);
    b_buf.as_mut_slice::<f32>().expect("b").copy_from_slice(&b_data);
    c_buf.as_mut_slice::<f32>().expect("c").copy_from_slice(&c_data);

    let tmp = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("tmp");
    let out = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("out");

    let executor = GraphExecutor::new(MlxDevice::new().expect("device2"));

    // Repeat the writer→barrier→reader edge several times. Each iteration opens
    // the persistent concurrent encoder, encodes the dependent pair across a
    // drained autorelease pool, and full-syncs. Under the pre-fix unretained
    // encoder this is the shape that lost barrier ordering; under the fix every
    // iteration must produce the byte-correct product.
    for iter in 0..16 {
        {
            let mut session = executor.begin().expect("begin");

            // Writer: tmp = a + b
            session
                .elementwise_add(
                    &mut registry,
                    device.metal_device(),
                    &a_buf,
                    &b_buf,
                    &tmp,
                    n,
                    DType::F32,
                )
                .expect("graph add (writer)");

            // RAW dependency: tmp written by add, read by mul. The barrier must
            // order the reader after the writer on the concurrent encoder. This
            // is the same primitive (`memoryBarrierWithScope:Buffers` on a
            // concurrent encoder) that lost ordering pre-fix when the encoder
            // was held unretained across the per-`begin()` command-buffer cycle.
            session.barrier();

            // Reader: out = tmp * c
            session
                .elementwise_mul(
                    &mut registry,
                    device.metal_device(),
                    &tmp,
                    &c_buf,
                    &out,
                    n,
                    DType::F32,
                )
                .expect("graph mul (reader)");

            session.finish().expect("finish");
        }

        let result: Vec<f32> = out.as_slice::<f32>().expect("read out").to_vec();
        for i in 0..n {
            let diff = (result[i] - expected[i]).abs();
            assert!(
                diff < 1e-4,
                "iter {iter}, element {i}: reader observed a stale `tmp` — the \
                 barrier did not order the writer before the reader on the \
                 concurrent encoder (ordering regression; ADR-040 §0.21c-track2 \
                 retain root fix). expected={}, got={}, diff={diff}",
                expected[i],
                result[i]
            );
        }
    }
}
