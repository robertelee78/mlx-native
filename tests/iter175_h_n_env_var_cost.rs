//! ADR-029 iter-175 Step 1am — H-N: microbench std::env::var cost.
//!
//! Step 1ak attributed 452 ns/dispatch to forward_mlx.rs orchestration.
//! Step 1al ruled out pipeline lookups (40 µs/tok = 0.4% wall).
//!
//! Reading src/env_flags.rs reveals env_default_true() calls
//! std::env::var(name) UNCACHED on every dispatch. dispatch_id_mv()
//! makes 2 such calls per invocation (HF2Q_Q6K_ID_MV_NR2 +
//! HF2Q_Q8_0_ID_MV_NR2).  Other dispatch sites do similar.
//!
//! This bench measures env::var cost in isolation.

use std::time::Instant;

const ITERATIONS: usize = 100_000;
const WARMUP: usize = 1_000;

#[test]
fn h_n_env_var_cost() {
    // Set a real env var to test the hit path (more expensive than miss).
    std::env::set_var("MLX_TEST_ENV_VAR", "value");

    // Warmup
    for _ in 0..WARMUP {
        let _ = std::env::var("MLX_TEST_ENV_VAR");
    }

    // ARM A: env::var on a set variable (hit path).
    let t0 = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = std::env::var("MLX_TEST_ENV_VAR");
    }
    let dur_hit = t0.elapsed();
    let per_hit = dur_hit.as_secs_f64() * 1e9 / ITERATIONS as f64;
    eprintln!("\n[H-N] std::env::var (set var):");
    eprintln!("  {} iterations in {:.2}ms = {:.1} ns/call", ITERATIONS, dur_hit.as_secs_f64()*1000.0, per_hit);

    // ARM B: env::var on an UNSET variable (miss path — typical production case).
    for _ in 0..WARMUP {
        let _ = std::env::var("MLX_TEST_ENV_VAR_NOT_SET");
    }
    let t0 = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = std::env::var("MLX_TEST_ENV_VAR_NOT_SET");
    }
    let dur_miss = t0.elapsed();
    let per_miss = dur_miss.as_secs_f64() * 1e9 / ITERATIONS as f64;
    eprintln!("\n[H-N] std::env::var (unset var — typical production):");
    eprintln!("  {} iterations in {:.2}ms = {:.1} ns/call", ITERATIONS, dur_miss.as_secs_f64()*1000.0, per_miss);

    // ARM C: env_default_true wrapper (what mlx-native's hot path uses).
    // Recreate inline since env_default_true is pub(crate):
    fn env_default_true_inline(name: &str) -> bool {
        match std::env::var(name).ok().as_deref() {
            None => true,
            Some(v) if v.eq_ignore_ascii_case("0") || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("off") => false,
            Some(_) => true,
        }
    }
    for _ in 0..WARMUP {
        let _ = env_default_true_inline("HF2Q_Q6K_ID_MV_NR2");
    }
    let t0 = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = env_default_true_inline("HF2Q_Q6K_ID_MV_NR2");
    }
    let dur_edt = t0.elapsed();
    let per_edt = dur_edt.as_secs_f64() * 1e9 / ITERATIONS as f64;
    eprintln!("\n[H-N] env_default_true_inline (unset, default-true semantics):");
    eprintln!("  {} iterations in {:.2}ms = {:.1} ns/call", ITERATIONS, dur_edt.as_secs_f64()*1000.0, per_edt);

    let dispatches_per_tok = 866.0;
    let env_calls_per_dispatch = 2.0;  // dispatch_id_mv hits 2 env vars; many other sites hit 1-3
    let env_overhead_per_tok_us = (per_edt * env_calls_per_dispatch * dispatches_per_tok) / 1000.0;
    let wall_pct = env_overhead_per_tok_us / 9.73 / 10.0;  // 9.73 ms wall

    eprintln!("\n[H-N] Extrapolated impact (2 env calls × 866 dispatches × {:.1} ns):", per_edt);
    eprintln!("  Per-token env-read overhead: {:.1} µs = {:.2}% wall", env_overhead_per_tok_us, wall_pct);
}
