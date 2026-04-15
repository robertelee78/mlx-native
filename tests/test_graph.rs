//! Tests for the GraphExecutor single-encoder dispatch pattern.
//!
//! Proves that encoding multiple ops through a single GraphSession (one
//! CommandEncoder, one commit_and_wait) produces the same results as
//! dispatching each op with its own encoder, AND is measurably faster.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::{DType, GraphExecutor, KernelRegistry, MlxDevice};

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

fn pseudo_random_f32(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let frac = ((state >> 33) as f32) / (u32::MAX as f32) - 0.5;
            frac
        })
        .collect()
}

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let registry = KernelRegistry::new();
    (device, registry)
}

// --------------------------------------------------------------------------
// Test 1: Single op through GraphSession matches direct dispatch
// --------------------------------------------------------------------------

#[test]
fn test_graph_single_op_elementwise_add() {
    let (device, mut registry) = setup();

    let n = 1024;
    let byte_len = n * std::mem::size_of::<f32>();
    let a_data = pseudo_random_f32(42, n);
    let b_data = pseudo_random_f32(99, n);

    // -- Direct dispatch (baseline) --
    let mut a_buf = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("a");
    let mut b_buf = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("b");
    let out_direct = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("out");

    a_buf.as_mut_slice::<f32>().expect("a").copy_from_slice(&a_data);
    b_buf.as_mut_slice::<f32>().expect("b").copy_from_slice(&b_data);

    {
        let mut encoder = device.command_encoder().expect("enc");
        mlx_native::ops::elementwise::elementwise_add(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &a_buf,
            &b_buf,
            &out_direct,
            n,
            DType::F32,
        )
        .expect("direct add");
        encoder.commit_and_wait().expect("commit");
    }

    let direct_result: Vec<f32> = out_direct.as_slice::<f32>().expect("read").to_vec();

    // -- GraphSession dispatch --
    let out_graph = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("out");

    let executor = GraphExecutor::new(MlxDevice::new().expect("device2"));
    {
        let mut session = executor.begin().expect("begin");
        session
            .elementwise_add(
                &mut registry,
                device.metal_device(),
                &a_buf,
                &b_buf,
                &out_graph,
                n,
                DType::F32,
            )
            .expect("graph add");
        session.finish().expect("finish");
    }

    let graph_result: Vec<f32> = out_graph.as_slice::<f32>().expect("read").to_vec();

    // -- Compare --
    for i in 0..n {
        let diff = (direct_result[i] - graph_result[i]).abs();
        assert!(
            diff < 1e-6,
            "Mismatch at {i}: direct={}, graph={}, diff={diff}",
            direct_result[i],
            graph_result[i]
        );
    }
}

// --------------------------------------------------------------------------
// Test 2: Multi-op sequence through single GraphSession
// --------------------------------------------------------------------------

#[test]
fn test_graph_multi_op_sequence() {
    let (device, mut registry) = setup();

    let n = 512;
    let byte_len = n * std::mem::size_of::<f32>();
    let a_data = pseudo_random_f32(100, n);
    let b_data = pseudo_random_f32(200, n);
    let c_data = pseudo_random_f32(300, n);

    let mut a_buf = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("a");
    let mut b_buf = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("b");
    let mut c_buf = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("c");

    a_buf.as_mut_slice::<f32>().expect("a").copy_from_slice(&a_data);
    b_buf.as_mut_slice::<f32>().expect("b").copy_from_slice(&b_data);
    c_buf.as_mut_slice::<f32>().expect("c").copy_from_slice(&c_data);

    // -- Sequential dispatch: each op gets own encoder + commit_and_wait --
    let tmp_seq = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("tmp");
    let out_seq = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("out");

    // Step 1: tmp = a + b
    {
        let mut enc = device.command_encoder().expect("enc");
        mlx_native::ops::elementwise::elementwise_add(
            &mut enc,
            &mut registry,
            device.metal_device(),
            &a_buf,
            &b_buf,
            &tmp_seq,
            n,
            DType::F32,
        )
        .expect("add");
        enc.commit_and_wait().expect("commit");
    }

    // Step 2: out = tmp * c
    {
        let mut enc = device.command_encoder().expect("enc");
        mlx_native::ops::elementwise::elementwise_mul(
            &mut enc,
            &mut registry,
            device.metal_device(),
            &tmp_seq,
            &c_buf,
            &out_seq,
            n,
            DType::F32,
        )
        .expect("mul");
        enc.commit_and_wait().expect("commit");
    }

    let seq_result: Vec<f32> = out_seq.as_slice::<f32>().expect("read").to_vec();

    // -- Batched dispatch: all ops in one GraphSession --
    let tmp_graph = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("tmp");
    let out_graph = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("out");

    let executor = GraphExecutor::new(MlxDevice::new().expect("device2"));
    {
        let mut session = executor.begin().expect("begin");

        // Step 1: tmp = a + b (same encoder)
        session
            .elementwise_add(
                &mut registry,
                device.metal_device(),
                &a_buf,
                &b_buf,
                &tmp_graph,
                n,
                DType::F32,
            )
            .expect("graph add");

        // Barrier: tmp_graph was written by add, will be read by mul.
        session.barrier();

        // Step 2: out = tmp * c (same encoder, NO intermediate commit_and_wait)
        session
            .elementwise_mul(
                &mut registry,
                device.metal_device(),
                &tmp_graph,
                &c_buf,
                &out_graph,
                n,
                DType::F32,
            )
            .expect("graph mul");

        // Single sync point
        session.finish().expect("finish");
    }

    let graph_result: Vec<f32> = out_graph.as_slice::<f32>().expect("read").to_vec();

    // -- Compare: results must match --
    for i in 0..n {
        let diff = (seq_result[i] - graph_result[i]).abs();
        assert!(
            diff < 1e-5,
            "Mismatch at {i}: sequential={}, graph={}, diff={diff}",
            seq_result[i],
            graph_result[i]
        );
    }
}

// --------------------------------------------------------------------------
// Test 3: Microbench — sequential vs batched dispatch speed
// --------------------------------------------------------------------------

#[test]
fn test_graph_vs_sequential_speed() {
    let (device, mut registry) = setup();

    // Simulate 30 decoder layers of elementwise add (cheap ops to isolate
    // dispatch overhead). Using hf2q's actual hidden size: 4096.
    let dim = 4096;
    let n_layers = 30;
    let n_ops_per_layer = 4; // add, mul, add, mul — simulating residual + scaling
    let total_ops = n_layers * n_ops_per_layer;
    let byte_len = dim * std::mem::size_of::<f32>();
    let iterations = 100;

    // Pre-allocate all buffers.
    let mut bufs = Vec::new();
    for i in 0..3 {
        let mut buf = device
            .alloc_buffer(byte_len, DType::F32, vec![dim])
            .expect("alloc");
        let data = pseudo_random_f32(i as u64, dim);
        buf.as_mut_slice::<f32>().expect("write").copy_from_slice(&data);
        bufs.push(buf);
    }

    // Warm up both paths.
    for _ in 0..3 {
        // Sequential warm-up
        for _ in 0..total_ops {
            let mut enc = device.command_encoder().expect("enc");
            mlx_native::ops::elementwise::elementwise_add(
                &mut enc,
                &mut registry,
                device.metal_device(),
                &bufs[0],
                &bufs[1],
                &bufs[2],
                dim,
                DType::F32,
            )
            .expect("add");
            enc.commit_and_wait().expect("commit");
        }

        // Batched warm-up
        let executor = GraphExecutor::new(MlxDevice::new().expect("dev"));
        let mut session = executor.begin().expect("begin");
        for _ in 0..total_ops {
            session
                .elementwise_add(
                    &mut registry,
                    device.metal_device(),
                    &bufs[0],
                    &bufs[1],
                    &bufs[2],
                    dim,
                    DType::F32,
                )
                .expect("add");
        }
        session.finish().expect("finish");
    }

    // -- Measure sequential --
    let mut seq_times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = std::time::Instant::now();
        for _ in 0..total_ops {
            let mut enc = device.command_encoder().expect("enc");
            mlx_native::ops::elementwise::elementwise_add(
                &mut enc,
                &mut registry,
                device.metal_device(),
                &bufs[0],
                &bufs[1],
                &bufs[2],
                dim,
                DType::F32,
            )
            .expect("add");
            enc.commit_and_wait().expect("commit");
        }
        seq_times.push(start.elapsed());
    }

    // -- Measure batched --
    let mut batch_times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let executor = GraphExecutor::new(MlxDevice::new().expect("dev"));
        let start = std::time::Instant::now();
        let mut session = executor.begin().expect("begin");
        for _ in 0..total_ops {
            session
                .elementwise_add(
                    &mut registry,
                    device.metal_device(),
                    &bufs[0],
                    &bufs[1],
                    &bufs[2],
                    dim,
                    DType::F32,
                )
                .expect("add");
        }
        session.finish().expect("finish");
        batch_times.push(start.elapsed());
    }

    // Sort and take median.
    seq_times.sort();
    batch_times.sort();
    let seq_median = seq_times[iterations / 2];
    let batch_median = batch_times[iterations / 2];

    let seq_us = seq_median.as_micros();
    let batch_us = batch_median.as_micros();
    let speedup = seq_us as f64 / batch_us as f64;

    println!("=== GraphExecutor Microbench ===");
    println!("  Ops per iteration: {total_ops} (simulating {n_layers} decoder layers)");
    println!("  Iterations: {iterations}");
    println!("  Sequential (per-op encoder+commit): {seq_us} us (median)");
    println!("  Batched (single encoder+commit):    {batch_us} us (median)");
    println!("  Speedup: {speedup:.2}x");
    println!("================================");

    // The batched path should be faster. We assert at least 1.5x speedup to
    // avoid flaky failures on loaded CI, but real hardware typically shows 5-20x.
    assert!(
        speedup > 1.5,
        "Expected batched dispatch to be at least 1.5x faster, got {speedup:.2}x \
         (seq={seq_us}us, batch={batch_us}us)"
    );
}

// --------------------------------------------------------------------------
// Test 4: Session drop without commit does not crash
// --------------------------------------------------------------------------

#[test]
fn test_graph_session_drop_without_commit() {
    let (device, mut registry) = setup();

    let n = 256;
    let byte_len = n * std::mem::size_of::<f32>();
    let mut a_buf = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("a");
    let mut b_buf = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("b");
    let out_buf = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("out");

    a_buf
        .as_mut_slice::<f32>()
        .expect("a")
        .copy_from_slice(&pseudo_random_f32(1, n));
    b_buf
        .as_mut_slice::<f32>()
        .expect("b")
        .copy_from_slice(&pseudo_random_f32(2, n));

    let executor = GraphExecutor::new(MlxDevice::new().expect("dev"));

    // Encode ops but intentionally drop the session without calling finish().
    // This should not panic or leave the GPU in a bad state.
    {
        let mut session = executor.begin().expect("begin");
        session
            .elementwise_add(
                &mut registry,
                device.metal_device(),
                &a_buf,
                &b_buf,
                &out_buf,
                n,
                DType::F32,
            )
            .expect("add");
        // Intentional: no session.finish() — dropped here.
    }

    // Verify the GPU is still usable after the abandoned session.
    let mut enc = device.command_encoder().expect("enc after drop");
    enc.commit_and_wait().expect("commit after drop");
}

// --------------------------------------------------------------------------
// Test 5: encoder_mut() escape hatch works
// --------------------------------------------------------------------------

#[test]
fn test_graph_encoder_mut_escape_hatch() {
    let (device, mut registry) = setup();

    let n = 128;
    let byte_len = n * std::mem::size_of::<f32>();
    let mut a_buf = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("a");
    let mut b_buf = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("b");
    let out_buf = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("out");

    let a_data = pseudo_random_f32(77, n);
    let b_data = pseudo_random_f32(88, n);
    a_buf.as_mut_slice::<f32>().expect("a").copy_from_slice(&a_data);
    b_buf.as_mut_slice::<f32>().expect("b").copy_from_slice(&b_data);

    let executor = GraphExecutor::new(MlxDevice::new().expect("dev"));
    let mut session = executor.begin().expect("begin");

    // Use encoder_mut() to call the op directly.
    mlx_native::ops::elementwise::elementwise_add(
        session.encoder_mut(),
        &mut registry,
        device.metal_device(),
        &a_buf,
        &b_buf,
        &out_buf,
        n,
        DType::F32,
    )
    .expect("direct add via encoder_mut");

    session.finish().expect("finish");

    let result: Vec<f32> = out_buf.as_slice::<f32>().expect("read").to_vec();
    for i in 0..n {
        let expected = a_data[i] + b_data[i];
        let diff = (result[i] - expected).abs();
        assert!(
            diff < 1e-6,
            "Mismatch at {i}: expected={expected}, got={}, diff={diff}",
            result[i]
        );
    }
}
