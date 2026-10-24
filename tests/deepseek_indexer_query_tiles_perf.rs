//! Fixed legal long-prefix operator tradeoff. No model/admission/HF adoption.
#![cfg(target_os = "macos")]
// objc 0.2 macros check a legacy cargo-clippy feature at the crate call site.
#![allow(unexpected_cfgs)]
#[path = "native54_perf/fixture.rs"]
mod fixture;
#[path = "native54_perf/observer.rs"]
mod observer;
#[path = "native54_perf/runner.rs"]
mod runner;
#[test]
#[ignore = "1.231GB explicit packet; root-supervised quiet window and 600s process-group watchdog required"]
fn query_tiles_legal_endpoint_dense_operator_tradeoff() {
    runner::run();
}
