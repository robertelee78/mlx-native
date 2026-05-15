//! ADR-029 iter-175 Step 1ap — H-O: microbench GraphSession::barrier_between cost.
//!
//! Step 1ak attributed 452 ns/dispatch to forward_mlx.rs orchestration.
//! Step 1al ruled out pipeline lookups (~0.4% wall).
//! Step 1am/1an ruled in env-var reads (~0.2% wall, partially fixed).
//!
//! barrier_between is called ~866 times/token (Step 1y), invoking
//! conflicts_reason (O(reads+writes × ranges_in_group) linear scan).
//! This bench measures the per-call cost.

use std::time::Instant;

use mlx_native::{DType, MlxBuffer, MlxDevice, GraphSession, GraphExecutor};

const ITERATIONS: usize = 100_000;
const WARMUP: usize = 1_000;

#[test]
fn h_o_barrier_between_cost() {
    let device = MlxDevice::new().expect("MlxDevice::new");

    // Allocate ~6 small buffers to simulate typical dispatch inputs.
    let bufs: Vec<MlxBuffer> = (0..6)
        .map(|_| device.alloc_buffer(64, DType::F32, vec![16]).expect("alloc"))
        .collect();

    let exec = GraphExecutor::new(device);
    let mut session: GraphSession = exec.begin().expect("session");

    // Pattern: each "dispatch" reads 2 bufs, writes 1. Different read/write
    // sets to match typical forward path. This is the steady-state worst case
    // for conflicts_reason because the tracker grows until next barrier.
    let reads_a = [&bufs[0], &bufs[1]];
    let writes_a = [&bufs[2]];
    let reads_b = [&bufs[2], &bufs[3]];
    let writes_b = [&bufs[4]];

    // Warmup
    for _ in 0..WARMUP {
        session.barrier_between(&reads_a, &writes_a);
        session.barrier_between(&reads_b, &writes_b);
    }

    let t0 = Instant::now();
    for _ in 0..ITERATIONS {
        session.barrier_between(&reads_a, &writes_a);
        session.barrier_between(&reads_b, &writes_b);
    }
    let dur = t0.elapsed();
    let per_call_ns = dur.as_secs_f64() * 1e9 / (ITERATIONS * 2) as f64;

    eprintln!("\n[H-O] GraphSession::barrier_between cost:");
    eprintln!("  {} calls in {:.2}ms = {:.1} ns/call", ITERATIONS*2, dur.as_secs_f64()*1000.0, per_call_ns);

    let calls_per_tok = 866;
    let per_tok_us = per_call_ns * calls_per_tok as f64 / 1000.0;
    let wall_pct = per_tok_us / 9.73 / 10.0;
    eprintln!("\n[H-O] Extrapolated per-token impact:");
    eprintln!("  {} calls × {:.1} ns = {:.1} µs/token = {:.2}% wall", calls_per_tok, per_call_ns, per_tok_us, wall_pct);
}
