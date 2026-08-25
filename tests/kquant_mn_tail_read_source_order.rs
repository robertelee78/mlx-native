//! Static proof that padded output rows cannot form or dereference packed
//! weight pointers in the scalar authorities or exact multi-column kernels.
//!
//! Output parity alone cannot detect an out-of-bounds read whose result is
//! discarded by a later store guard. Keep these source-order assertions next
//! to the odd-N hardware gates.

const SHADER: &str = include_str!("../src/shaders/quantized_matmul_ggml.metal");

fn kernel_body(start: &str, end: &str) -> &'static str {
    let start_index = SHADER
        .find(start)
        .unwrap_or_else(|| panic!("missing kernel start: {start}"));
    let tail = &SHADER[start_index..];
    let end_index = tail
        .find(end)
        .unwrap_or_else(|| panic!("missing kernel end after {start}: {end}"));
    assert!(end_index > start.len(), "empty kernel body for {start}");
    &tail[..end_index]
}

fn position(body: &str, needle: &str) -> usize {
    body.find(needle)
        .unwrap_or_else(|| panic!("missing `{needle}` in kernel body"))
}

fn assert_single_row_guard(body: &str, row: &str, guard: &str, x: &str) {
    let row_pos = position(body, row);
    let guard_pos = position(body, guard);
    let offset_pos = position(body, "const uint offset0");
    let x_pos = position(body, x);

    assert!(
        row_pos < guard_pos,
        "tail guard must follow row calculation"
    );
    assert!(
        guard_pos < offset_pos,
        "tail guard must precede offset formation"
    );
    assert!(
        offset_pos < x_pos,
        "test did not identify the packed-row pointer after offset0"
    );

    let between = &body[row_pos + row.len()..guard_pos];
    assert!(
        !between.contains("const "),
        "another value was formed before the tail guard: {between}"
    );
    assert!(
        !between.contains("device "),
        "a pointer was formed before the tail guard: {between}"
    );
}

fn assert_two_row_guard(body: &str) {
    let first_row = "const int first_row";
    let early_guard = "if (first_row >= p.ne01)";
    let first_pos = position(body, first_row);
    let early_pos = position(body, early_guard);
    let offset_pos = position(body, "const uint offset0");
    let base_pos = position(body, "device const block_q6_K * x_base");
    assert!(first_pos < early_pos && early_pos < offset_pos && offset_pos < base_pos);

    let before_early = &body[first_pos + first_row.len()..early_pos];
    assert!(
        !before_early.contains("const "),
        "value formed before group tail guard: {before_early}"
    );
    assert!(
        !before_early.contains("device "),
        "pointer formed before group tail guard: {before_early}"
    );

    let row_loop = position(body, "for (int row = 0; row < nr0; ++row)");
    let partial_guard = body[row_loop..]
        .find("if (first_row + row >= p.ne01)")
        .map(|offset| row_loop + offset)
        .expect("missing partial-group row guard");
    let row_pointer = body[row_loop..]
        .find("device const block_q6_K * xr")
        .map(|offset| row_loop + offset)
        .expect("missing per-row packed pointer");
    assert!(row_loop < partial_guard && partial_guard < row_pointer);
}

#[test]
fn scalar_and_exact_mn_kquant_tail_guards_precede_packed_row_access() {
    assert_single_row_guard(
        kernel_body(
            "kernel void kernel_mul_mv_q4_K_f32(",
            "// ---- Q4_K mat-vec kernel, mN",
        ),
        "const int row = 2 * (int)r0 + (int)sgitg;",
        "if (row >= (int)p.ne01)",
        "device const block_q4_K * x",
    );
    assert_single_row_guard(
        kernel_body(
            "kernel void hf2q_mul_mv_q4_K_f32_mN_impl(",
            "template [[host_name(\"kernel_mul_mv_q4_K",
        ),
        "const int row = 2 * (int)r0 + (int)sgitg;",
        "if (row >= (int)p.ne01)",
        "device const block_q4_K * x",
    );
    assert_single_row_guard(
        kernel_body(
            "kernel void kernel_mul_mv_q5_K_f32(",
            "// ---- Q5_K mat-vec kernel, mN",
        ),
        "const int row = 2 * (int)r0 + (int)sgitg;",
        "if (row >= (int)p.ne01)",
        "device const block_q5_K * x",
    );
    assert_single_row_guard(
        kernel_body(
            "kernel void hf2q_mul_mv_q5_K_f32_mN_impl(",
            "template [[host_name(\"kernel_mul_mv_q5_K",
        ),
        "const int row = 2 * (int)r0 + (int)sgitg;",
        "if (row >= (int)p.ne01)",
        "device const block_q5_K * x",
    );
    assert_single_row_guard(
        kernel_body(
            "kernel void kernel_mul_mv_q6_K_f32(",
            "// ---- Q6_K mat-vec kernel, nr0=2",
        ),
        "const int row = 2 * r0 + sgitg;",
        "if (row >= p.ne01)",
        "device const block_q6_K * x",
    );

    assert_two_row_guard(kernel_body(
        "kernel void kernel_mul_mv_q6_K_f32_nr2(",
        "// ---- Q6_K mat-vec kernel, mN",
    ));
    assert_two_row_guard(kernel_body(
        "kernel void hf2q_mul_mv_q6_K_f32_mN_impl(",
        "// Explicit physical R1",
    ));
}
