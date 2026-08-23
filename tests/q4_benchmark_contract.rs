#[allow(dead_code)]
#[path = "../benches/bench_q4_mm_tensor_64x32.rs"]
mod candidate_bench;

#[test]
fn unmatched_shape_filter_is_a_hard_error() {
    let error = candidate_bench::selected_shape_count(Some("definitely-not-a-real-shape"))
        .expect_err("an unmatched filter must not produce a successful zero-work benchmark");
    assert!(error.contains("matched no shapes"));
}
