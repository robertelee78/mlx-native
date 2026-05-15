//! ADR-029 iter-175 Step 1al — H-M: microbench KernelRegistry lookup cost.
//!
//! Step 1ak attributed 452 ns/dispatch to forward_mlx.rs orchestration.
//! Likely sources include `registry.get_pipeline()` calls per dispatch
//! (HashMap lookup × 2) and `registry.get_pipeline_with_constants()` (with
//! cache_key string allocation).
//!
//! This microbench measures the cost of those lookups in isolation, so we
//! can decide whether they're worth optimizing (and how much wall benefit
//! to expect).
//!
//! Run: `cargo test --release --test iter175_h_m_pipeline_lookup_cost -- --nocapture`

use std::time::Instant;

use mlx_native::{KernelRegistry, MlxDevice};

const ITERATIONS: usize = 100_000;
const WARMUP: usize = 1_000;

#[test]
fn h_m_pipeline_lookup_cost() {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();
    let metal_dev = device.metal_device();

    // -------- ARM A: get_pipeline (no FCs) — first-call compile + N warm calls --------
    // Use down_exps kernel (no FC variant).
    let name_a = "kernel_mul_mv_id_q8_0_f32";

    // Trigger compile (cold path)
    let _ = registry.get_pipeline(name_a, metal_dev).expect("compile");

    // Warmup
    for _ in 0..WARMUP {
        let _ = registry.get_pipeline(name_a, metal_dev).expect("warmup");
    }

    let t0 = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = registry.get_pipeline(name_a, metal_dev).expect("hot");
    }
    let dur_a = t0.elapsed();
    let per_call_a = dur_a.as_secs_f64() * 1e9 / ITERATIONS as f64;
    eprintln!("\n[H-M] ARM A get_pipeline (no FCs, 'kernel_mul_mv_id_q8_0_f32'):");
    eprintln!("  {} iterations in {:.2}ms = {:.1} ns/call", ITERATIONS, dur_a.as_secs_f64()*1000.0, per_call_a);

    // -------- ARM B: get_pipeline_with_constants (FCs + cache_key string alloc) --------
    let name_b = "kernel_mul_mv_q6_K_f32_nr2";
    let bool_consts: &[(usize, bool)] = &[];
    let int_consts: &[(usize, i32)] = &[(700, 1), (701, 1), (702, 1)];

    let _ = registry.get_pipeline_with_constants(name_b, metal_dev, bool_consts, int_consts).expect("compile FC");

    for _ in 0..WARMUP {
        let _ = registry.get_pipeline_with_constants(name_b, metal_dev, bool_consts, int_consts).expect("warmup FC");
    }

    let t0 = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = registry.get_pipeline_with_constants(name_b, metal_dev, bool_consts, int_consts).expect("hot FC");
    }
    let dur_b = t0.elapsed();
    let per_call_b = dur_b.as_secs_f64() * 1e9 / ITERATIONS as f64;
    eprintln!("\n[H-M] ARM B get_pipeline_with_constants (3 i32 FCs, 'kernel_mul_mv_q6_K_f32_nr2'):");
    eprintln!("  {} iterations in {:.2}ms = {:.1} ns/call", ITERATIONS, dur_b.as_secs_f64()*1000.0, per_call_b);

    eprintln!("\n[H-M] Per-call cost summary:");
    eprintln!("  get_pipeline (no FCs)         : {:.1} ns", per_call_a);
    eprintln!("  get_pipeline_with_constants    : {:.1} ns  (Δ +{:.1} ns from cache_key alloc + larger hash)", per_call_b, per_call_b - per_call_a);

    let f_call_count_per_tok = 150;  // rough estimate for FC-using kernels
    let nofc_call_count = 866 - f_call_count_per_tok;
    let total_lookup_ns = per_call_a * nofc_call_count as f64 + per_call_b * f_call_count_per_tok as f64;
    let wall_pct = 100.0 * total_lookup_ns / 1e6 / 9.73;  // 9.73 ms wall per Step 1ah

    eprintln!("\n[H-M] Per-token lookup cost (extrapolated):");
    eprintln!("  ~{} get_pipeline calls × {:.1} ns = {:.1} µs", nofc_call_count, per_call_a, per_call_a * nofc_call_count as f64 / 1000.0);
    eprintln!("  ~{} get_pipeline_with_constants × {:.1} ns = {:.1} µs", f_call_count_per_tok, per_call_b, per_call_b * f_call_count_per_tok as f64 / 1000.0);
    eprintln!("  Total per-token lookup overhead = {:.1} µs = ~{:.2}% of 9.73 ms wall", total_lookup_ns / 1000.0, wall_pct);

    eprintln!("\n[H-M] Verdict for optimization yield:");
    if wall_pct > 1.0 {
        eprintln!("  Worth optimizing: closure potential {:.1}% wall", wall_pct);
    } else {
        eprintln!("  Marginal: closure potential {:.1}% wall — focus elsewhere", wall_pct);
    }
}
