//! ADR-040 §0.21c — BYTE-IDENTITY spike for `kernel_mul_mv_q6_K_f32_mN_r1_{R1}`.
//!
//! The mN kernel amortizes the Q6_K weight read + dequant across R1 src1
//! COLUMNS (the batched-decode m axis), vs plain `kernel_mul_mv_q6_K_f32` which
//! re-reads the weight once per column.  Because the per-column accumulation is
//! a LITERAL clone of plain mv's `sums[4]`/`sc`/`dall`/`simd_sum` tree (only the
//! per-column src1 pointer and dst index differ), the result MUST be BIT-EQUAL
//! — not merely fp-tolerant — to running plain mv once per column.
//!
//! This is the GATE for the ADR-040 §0.21c lever: it must pass via `.to_bits()`
//! u32 compare for ALL R1 ∈ {2..8}.  If any bit differs, the source shape is
//! wrong and the kernel may NOT ship default-on.
//!
//! Both compile paths: this test runs under the runtime source-compile path
//! always.  Set `MLX_PRECOMPILED_METALLIB=1` (with a Metal-toolchain build that
//! actually populated default.metallib) to additionally exercise the
//! precompiled path — same registered source + host_name instantiation, so the
//! entry point resolves identically.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::{
    dispatch_mv_q6k_mn, dispatch_mv_q6k_mn_adaptive, DType, GgmlQuantizedMatmulParams, GgmlType,
    KernelRegistry, MlxDevice,
};

// PRNG matching adr_028_iter309_q6k_mv_nr2_parity.rs.
fn pseudo_random_f32(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32) / (u32::MAX as f32) - 0.5
        })
        .collect()
}

// Q6_K pack — identical to adr_028_iter309_q6k_mv_nr2_parity.rs.
fn pack_q6_k(values: &[f32]) -> Vec<u8> {
    assert!(values.len() % 256 == 0);
    let mut buf = Vec::new();
    for block in values.chunks(256) {
        let mut sub_scales = [0.0f32; 16];
        let mut sub_scale_int = [0i8; 16];
        let mut max_scale: f32 = 0.0;

        for (s, sub) in block.chunks(16).enumerate() {
            let amax = sub.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            sub_scales[s] = amax;
            if amax > max_scale {
                max_scale = amax;
            }
        }

        let d = max_scale / (32.0 * 127.0);
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        for s in 0..16 {
            sub_scale_int[s] = if sub_scales[s] != 0.0 {
                (sub_scales[s] * id / 32.0).round().clamp(-128.0, 127.0) as i8
            } else {
                0
            };
        }

        let mut q6 = [0u8; 256];
        for (s, sub) in block.chunks(16).enumerate() {
            let sc = sub_scale_int[s] as f32;
            let sub_d = d * sc;
            let sub_id = if sub_d != 0.0 { 1.0 / sub_d } else { 0.0 };
            for (i, &v) in sub.iter().enumerate() {
                let q = (v * sub_id + 32.0).round().clamp(0.0, 63.0) as u8;
                q6[s * 16 + i] = q;
            }
        }

        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];

        for l0_base in (0..32usize).step_by(4) {
            for l in 0..4usize {
                let ql_idx = l0_base + l;
                let v0 = q6[l0_base + l];
                let v2 = q6[l0_base + l + 64];
                ql[ql_idx] = (v0 & 0x0F) | ((v2 & 0x0F) << 4);

                let v1 = q6[l0_base + l + 32];
                let v3 = q6[l0_base + l + 96];
                ql[ql_idx + 32] = (v1 & 0x0F) | ((v3 & 0x0F) << 4);

                let h0 = (v0 >> 4) & 0x03;
                let h1 = (v1 >> 4) & 0x03;
                let h2 = (v2 >> 4) & 0x03;
                let h3 = (v3 >> 4) & 0x03;
                qh[ql_idx] = h0 | (h1 << 2) | (h2 << 4) | (h3 << 6);
            }
        }

        for l0_base in (0..32usize).step_by(4) {
            for l in 0..4usize {
                let ql_idx = 64 + l0_base + l;
                let qh_idx = 32 + l0_base + l;
                let v0 = q6[128 + l0_base + l];
                let v2 = q6[128 + l0_base + l + 64];
                ql[ql_idx] = (v0 & 0x0F) | ((v2 & 0x0F) << 4);

                let v1 = q6[128 + l0_base + l + 32];
                let v3 = q6[128 + l0_base + l + 96];
                ql[ql_idx + 32] = (v1 & 0x0F) | ((v3 & 0x0F) << 4);

                let h0 = (v0 >> 4) & 0x03;
                let h1 = (v1 >> 4) & 0x03;
                let h2 = (v2 >> 4) & 0x03;
                let h3 = (v3 >> 4) & 0x03;
                qh[qh_idx] = h0 | (h1 << 2) | (h2 << 4) | (h3 << 6);
            }
        }

        buf.extend_from_slice(&ql);
        buf.extend_from_slice(&qh);
        buf.extend_from_slice(&sub_scale_int.iter().map(|&s| s as u8).collect::<Vec<_>>());
        let d_f16 = half::f16::from_f32(d);
        buf.extend_from_slice(&d_f16.to_le_bytes());
    }
    buf
}

/// Run the model's DEFAULT serial decode kernel for a single column: NR2 is
/// default-ON in the gemma4 model, so the serial m=1 reference the parity test
/// compares against uses `kernel_mul_mv_q6_K_f32_nr2`, NOT plain mv. This helper
/// runs with NR2 ON so the spike reference matches what the model actually does.
fn run_nr2_single_col(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    n: usize,
    k: usize,
    weight_bytes: &[u8],
    col_input: &[f32],
) -> Vec<f32> {
    std::env::set_var("HF2Q_Q6K_MV_NR2", "1");
    std::env::remove_var("HF2Q_DECODE_MVN");
    std::env::remove_var("HF2Q_DECODE_MV_EXT");

    let mut input_buf = device
        .alloc_buffer(k * 4, DType::F32, vec![1, k])
        .expect("alloc input");
    input_buf
        .as_mut_slice::<f32>()
        .expect("input mut")
        .copy_from_slice(col_input);
    let mut weight_buf = device
        .alloc_buffer(weight_bytes.len(), DType::U8, vec![weight_bytes.len()])
        .expect("alloc weight");
    weight_buf
        .as_mut_slice::<u8>()
        .expect("weight mut")
        .copy_from_slice(weight_bytes);
    let mut output_buf = device
        .alloc_buffer(n * 4, DType::F32, vec![1, n])
        .expect("alloc output");
    for v in output_buf.as_mut_slice::<f32>().expect("out mut").iter_mut() {
        *v = 0.0;
    }
    let params = GgmlQuantizedMatmulParams {
        m: 1,
        n: n as u32,
        k: k as u32,
        ggml_type: GgmlType::Q6_K,
    };
    let mut encoder = device.command_encoder().expect("encoder");
    mlx_native::quantized_matmul_ggml(
        &mut encoder,
        registry,
        device,
        &input_buf,
        &weight_buf,
        &mut output_buf,
        &params,
    )
    .expect("nr2 dispatch");
    encoder.commit_and_wait().expect("gpu");
    std::env::remove_var("HF2Q_Q6K_MV_NR2");
    output_buf.as_slice::<f32>().expect("read out").to_vec()
}

/// Run plain mv (`HF2Q_DECODE_MVN` unset, NR2 forced off so the route lands on
/// the baseline `kernel_mul_mv_q6_K_f32`) once for a SINGLE column of input.
/// We invoke it column-by-column so the reference is exactly "plain mv per
/// column" — the thing the mN kernel must reproduce bit-for-bit.
fn run_plain_mv_single_col(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    n: usize,
    k: usize,
    weight_bytes: &[u8],
    col_input: &[f32], // length k
) -> Vec<f32> {
    // Force the true baseline kernel (not NR2, not MVN, not mv_ext).
    std::env::set_var("HF2Q_Q6K_MV_NR2", "0");
    std::env::remove_var("HF2Q_DECODE_MVN");
    std::env::remove_var("HF2Q_DECODE_MV_EXT");

    let mut input_buf = device
        .alloc_buffer(k * 4, DType::F32, vec![1, k])
        .expect("alloc input");
    input_buf
        .as_mut_slice::<f32>()
        .expect("input mut")
        .copy_from_slice(col_input);

    let mut weight_buf = device
        .alloc_buffer(weight_bytes.len(), DType::U8, vec![weight_bytes.len()])
        .expect("alloc weight");
    weight_buf
        .as_mut_slice::<u8>()
        .expect("weight mut")
        .copy_from_slice(weight_bytes);

    let mut output_buf = device
        .alloc_buffer(n * 4, DType::F32, vec![1, n])
        .expect("alloc output");
    for v in output_buf.as_mut_slice::<f32>().expect("out mut").iter_mut() {
        *v = 0.0;
    }

    let params = GgmlQuantizedMatmulParams {
        m: 1,
        n: n as u32,
        k: k as u32,
        ggml_type: GgmlType::Q6_K,
    };

    let mut encoder = device.command_encoder().expect("encoder");
    mlx_native::quantized_matmul_ggml(
        &mut encoder,
        registry,
        device,
        &input_buf,
        &weight_buf,
        &mut output_buf,
        &params,
    )
    .expect("plain mv dispatch");
    encoder.commit_and_wait().expect("gpu");

    std::env::remove_var("HF2Q_Q6K_MV_NR2");
    output_buf.as_slice::<f32>().expect("read out").to_vec()
}

/// Run the mN kernel for the full [m, k] input at the given R1 width; returns
/// the [m, n] output (row-major: out[col*n + row]).
fn run_mN(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    r1: usize,
    m: usize,
    n: usize,
    k: usize,
    weight_bytes: &[u8],
    input: &[f32], // [m, k]
) -> Vec<f32> {
    let mut input_buf = device
        .alloc_buffer(m * k * 4, DType::F32, vec![m, k])
        .expect("alloc input");
    input_buf
        .as_mut_slice::<f32>()
        .expect("input mut")
        .copy_from_slice(input);

    let mut weight_buf = device
        .alloc_buffer(weight_bytes.len(), DType::U8, vec![weight_bytes.len()])
        .expect("alloc weight");
    weight_buf
        .as_mut_slice::<u8>()
        .expect("weight mut")
        .copy_from_slice(weight_bytes);

    let mut output_buf = device
        .alloc_buffer(m * n * 4, DType::F32, vec![m, n])
        .expect("alloc output");
    for v in output_buf.as_mut_slice::<f32>().expect("out mut").iter_mut() {
        *v = 0.0;
    }

    let params = GgmlQuantizedMatmulParams {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        ggml_type: GgmlType::Q6_K,
    };

    let mut encoder = device.command_encoder().expect("encoder");
    dispatch_mv_q6k_mn(
        &mut encoder,
        registry,
        device,
        &input_buf,
        &weight_buf,
        &output_buf,
        &params,
        r1,
    )
    .expect("mN dispatch");
    encoder.commit_and_wait().expect("gpu");

    output_buf.as_slice::<f32>().expect("read out").to_vec()
}

/// Run the ADAPTIVE (column-tiled, register-safe) dispatch for the full [m,k]
/// input; returns [m,n] output (out[col*n+row]). This exercises the
/// BufferWithOffset chunk path that production routing uses.
fn run_mN_adaptive(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    m: usize,
    n: usize,
    k: usize,
    weight_bytes: &[u8],
    input: &[f32],
) -> Vec<f32> {
    let mut input_buf = device
        .alloc_buffer(m * k * 4, DType::F32, vec![m, k])
        .expect("alloc input");
    input_buf.as_mut_slice::<f32>().expect("in mut").copy_from_slice(input);
    let mut weight_buf = device
        .alloc_buffer(weight_bytes.len(), DType::U8, vec![weight_bytes.len()])
        .expect("alloc weight");
    weight_buf.as_mut_slice::<u8>().expect("w mut").copy_from_slice(weight_bytes);
    let mut output_buf = device
        .alloc_buffer(m * n * 4, DType::F32, vec![m, n])
        .expect("alloc output");
    for v in output_buf.as_mut_slice::<f32>().expect("out mut").iter_mut() {
        *v = 0.0;
    }
    let params = GgmlQuantizedMatmulParams {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        ggml_type: GgmlType::Q6_K,
    };
    let mut encoder = device.command_encoder().expect("encoder");
    dispatch_mv_q6k_mn_adaptive(&mut encoder, registry, device, &input_buf, &weight_buf, &output_buf, &params)
        .expect("mN adaptive dispatch");
    encoder.commit_and_wait().expect("gpu");
    output_buf.as_slice::<f32>().expect("read out").to_vec()
}

/// The ADAPTIVE path (column-tiled with BufferWithOffset) must be byte-equal to
/// NR2 at every real gemma4 Q6_K shape, for every m∈{2..8}. m∈{6,7,8} actually
/// exercise the multi-tile offset binding (6=3+3, 7=4+3, 8=4+4).
#[test]
fn q6k_mN_adaptive_vs_nr2_real_shapes_all_m() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let k = 2816usize;
    for &n in &[1024usize, 2048, 2112, 4096, 8192] {
        let weight_f32 = pseudo_random_f32(0xc0ffee ^ n as u64, n * k);
        let weight_bytes = pack_q6_k(&weight_f32);
        for m in 2..=8usize {
            let input = pseudo_random_f32(0xbeef ^ (m as u64) ^ ((n as u64) << 16), m * k);
            let mut nr2_ref: Vec<Vec<f32>> = Vec::with_capacity(m);
            for col in 0..m {
                let ci = &input[col * k..(col + 1) * k];
                nr2_ref.push(run_nr2_single_col(&device, &mut registry, n, k, &weight_bytes, ci));
            }
            let out = run_mN_adaptive(&device, &mut registry, m, n, k, &weight_bytes, &input);
            let mut mis = 0usize;
            let mut first: Option<(usize, usize, u32, u32)> = None;
            for col in 0..m {
                for row in 0..n {
                    let got = out[col * n + row].to_bits();
                    let want = nr2_ref[col][row].to_bits();
                    if got != want {
                        mis += 1;
                        if first.is_none() {
                            first = Some((col, row, want, got));
                        }
                    }
                }
            }
            assert_eq!(
                mis, 0,
                "adaptive mN(m={m}) vs NR2 at n={n} k={k}: {mis} mismatches; first \
                 col={:?} row={:?} nr2=0x{:08x} mN=0x{:08x}",
                first.map(|f| f.0), first.map(|f| f.1),
                first.map(|f| f.2).unwrap_or(0), first.map(|f| f.3).unwrap_or(0),
            );
        }
        eprintln!("[mvN-adaptive] n={n} k={k}: adaptive mN == NR2 bit-exact for all m=2..8");
    }
}

fn check_byte_identity(label: &str, m: usize, n: usize, k: usize) {
    assert!(k % 256 == 0, "k must be a multiple of 256");
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();

    let weight_f32 = pseudo_random_f32(0xc0ffee, n * k);
    let weight_bytes = pack_q6_k(&weight_f32);
    let input = pseudo_random_f32(0xbeef, m * k); // [m, k]

    // Reference: NR2 (the model's default serial decode kernel), run once per
    // column → [m][n]. mN is built to be byte-equal to NR2, because NR2 is what
    // the gemma4 m=1 serial parity reference uses (NR2 is NOT byte-equal to
    // plain mv in general — see q6k_mN_vs_nr2_real_shapes_all_m).
    let mut reference: Vec<Vec<f32>> = Vec::with_capacity(m);
    for col in 0..m {
        let col_input = &input[col * k..(col + 1) * k];
        reference.push(run_nr2_single_col(
            &device,
            &mut registry,
            n,
            k,
            &weight_bytes,
            col_input,
        ));
    }

    // Exercise the production ADAPTIVE path (single-tile for m≤5, column-tiled
    // for m≥6). mN output layout is [m, n]: out[col*n + row].
    let out = run_mN_adaptive(&device, &mut registry, m, n, k, &weight_bytes, &input);
    let mut mismatches = 0usize;
    let mut first: Option<(usize, usize, u32, u32)> = None;
    for col in 0..m {
        for row in 0..n {
            let got = out[col * n + row];
            let want = reference[col][row];
            if got.to_bits() != want.to_bits() {
                mismatches += 1;
                if first.is_none() {
                    first = Some((col, row, want.to_bits(), got.to_bits()));
                }
            }
        }
    }
    assert_eq!(
        mismatches, 0,
        "[{label}] m={m}: {mismatches} bit-mismatches (of {} elems); first \
         col={:?} row={:?} nr2.bits=0x{:08x} mN.bits=0x{:08x}",
        m * n,
        first.map(|f| f.0),
        first.map(|f| f.1),
        first.map(|f| f.2).unwrap_or(0),
        first.map(|f| f.3).unwrap_or(0),
    );
    eprintln!("[mvN-byte-parity] {label} m={m} N={n} K={k}: BYTE-EQUAL vs NR2 ({} elems)", m * n);
}

/// CRUX: the gemma4 model runs Q6_K decode through NR2 (default-on). The serial
/// m=1 parity reference therefore uses NR2, so for the model parity to stay
/// bit-exact, mN must equal NR2 — NOT (only) plain mv. This test pins whether
/// NR2 is itself bit-equal to plain mv, and whether mN equals NR2.
#[test]
fn q6k_nr2_vs_plain_and_mN_vs_nr2_bit_exact() {
    // lm_head-like shape (k=2816, the gemma4 Q6_K lm_head/embd K). Use a larger
    // representative n to exercise the multi-block-per-row sum at the real K.
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let (m, n, k) = (8usize, 512usize, 2816usize);

    let weight_f32 = pseudo_random_f32(0xc0ffee, n * k);
    let weight_bytes = pack_q6_k(&weight_f32);
    let input = pseudo_random_f32(0xbeef, m * k);

    // Per-column references from BOTH kernels.
    let mut plain_ref: Vec<Vec<f32>> = Vec::with_capacity(m);
    let mut nr2_ref: Vec<Vec<f32>> = Vec::with_capacity(m);
    for col in 0..m {
        let ci = &input[col * k..(col + 1) * k];
        plain_ref.push(run_plain_mv_single_col(&device, &mut registry, n, k, &weight_bytes, ci));
        nr2_ref.push(run_nr2_single_col(&device, &mut registry, n, k, &weight_bytes, ci));
    }

    // (1) NR2 vs plain mv, bit-exact.
    let mut nr2_vs_plain = 0usize;
    for col in 0..m {
        for row in 0..n {
            if nr2_ref[col][row].to_bits() != plain_ref[col][row].to_bits() {
                nr2_vs_plain += 1;
            }
        }
    }
    eprintln!("[mvN-crux] NR2 vs plain-mv bit-mismatches: {nr2_vs_plain} / {}", m * n);

    // (2) mN (R1=m) vs NR2, bit-exact — this is what the model needs.
    let out = run_mN(&device, &mut registry, m, m, n, k, &weight_bytes, &input);
    let mut mN_vs_nr2 = 0usize;
    let mut mN_vs_plain = 0usize;
    for col in 0..m {
        for row in 0..n {
            let got = out[col * n + row].to_bits();
            if got != nr2_ref[col][row].to_bits() {
                mN_vs_nr2 += 1;
            }
            if got != plain_ref[col][row].to_bits() {
                mN_vs_plain += 1;
            }
        }
    }
    eprintln!("[mvN-crux] mN vs NR2 bit-mismatches:      {mN_vs_nr2} / {}", m * n);
    eprintln!("[mvN-crux] mN vs plain-mv bit-mismatches: {mN_vs_plain} / {}", m * n);
    // Diagnostic only — assertions live in the dedicated tests below.
}

/// Exhaustive bit-exact check of mN(m) vs NR2(m=1)-per-column at EVERY real
/// gemma4 Q6_K shape (n ∈ observed set, k=2816) for EVERY m∈{2..8}. This is the
/// shape set the model's batched decode body actually routes to mN; if any one
/// diverges from the NR2 serial reference, the model parity test fails.
#[test]
fn q6k_mN_vs_nr2_real_shapes_all_m() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let k = 2816usize;
    let shapes_n = [1024usize, 2048, 2112, 4096, 8192];

    for &n in &shapes_n {
        let weight_f32 = pseudo_random_f32(0xc0ffee ^ n as u64, n * k);
        let weight_bytes = pack_q6_k(&weight_f32);
        for m in 2..=8usize {
            let input = pseudo_random_f32(0xbeef ^ (m as u64) ^ ((n as u64) << 16), m * k);
            // NR2 per-column reference (what the m=1 serial replay uses).
            let mut nr2_ref: Vec<Vec<f32>> = Vec::with_capacity(m);
            for col in 0..m {
                let ci = &input[col * k..(col + 1) * k];
                nr2_ref.push(run_nr2_single_col(&device, &mut registry, n, k, &weight_bytes, ci));
            }
            let out = run_mN(&device, &mut registry, m, m, n, k, &weight_bytes, &input);
            let mut mis = 0usize;
            let mut first: Option<(usize, usize, u32, u32)> = None;
            for col in 0..m {
                for row in 0..n {
                    let got = out[col * n + row].to_bits();
                    let want = nr2_ref[col][row].to_bits();
                    if got != want {
                        mis += 1;
                        if first.is_none() {
                            first = Some((col, row, want, got));
                        }
                    }
                }
            }
            assert_eq!(
                mis, 0,
                "mN(m={m}) vs NR2 at n={n} k={k}: {mis} bit-mismatches; first \
                 col={:?} row={:?} nr2.bits=0x{:08x} mN.bits=0x{:08x}",
                first.map(|f| f.0),
                first.map(|f| f.1),
                first.map(|f| f.2).unwrap_or(0),
                first.map(|f| f.3).unwrap_or(0),
            );
        }
        eprintln!("[mvN-real] n={n} k={k}: mN == NR2 bit-exact for all m=2..8");
    }
}

#[test]
fn q6k_mvN_byte_parity_m8_n64_k256() {
    // N=64 (>32 rows, exercises multi-TG in N), K=256 (one block/row), m=8.
    check_byte_identity("m8 N=64 K=256", 8, 64, 256);
}

#[test]
fn q6k_mvN_byte_parity_m8_n128_k512() {
    // 2 blocks/row, N=128.
    check_byte_identity("m8 N=128 K=512", 8, 128, 512);
}

#[test]
fn q6k_mvN_byte_parity_lmhead_shape() {
    // gemma4 lm_head-like: large N, K=2816 (11 blocks/row). N=256 keeps it fast
    // while exercising the multi-block per-row sum identically.
    check_byte_identity("lmhead-mini N=256 K=2816", 8, 256, 2816);
}

#[test]
fn q6k_mvN_byte_parity_m_each_2_to_8() {
    // Exercise every m so the single-tile R1=m routing the model uses is
    // covered, and so the boundary store guard (r1_base+c < ne1) is hit when
    // m is not a multiple of R1 for the wider sweeps.
    for m in 2..=8usize {
        check_byte_identity(&format!("m={m} N=96 K=512"), m, 96, 512);
    }
}
