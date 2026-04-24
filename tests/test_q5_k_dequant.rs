//! Tests for Q5_K and I16 f32 dequantization (ADR-013 P5 completion).
//!
//! Q5_K spec (derived from ggml-quants.c::dequantize_row_q5_K; no code
//! copied):
//!
//! ```text
//! Block (176 bytes, 256 values):
//!   d    : f16 super-block scale
//!   dmin : f16 super-block min
//!   scales[12] : packed 6-bit scales/mins (shared with Q4_K get_scale_min_k4)
//!   qh[32]     : one high bit per value (bit selector u1/u2 shifts per pair)
//!   qs[128]    : low 4 bits per value (low nibble for sub-block is, high for is+1)
//!
//! Per pair (is, is+1) of 64 values:
//!   u1 = 1 << (2*pair_idx),   u2 = 2 << (2*pair_idx)
//!   (sc1, m1) = get_scale_min_k4(is,   scales);   d1 = d*sc1;   min1 = dmin*m1
//!   (sc2, m2) = get_scale_min_k4(is+1, scales);   d2 = d*sc2;   min2 = dmin*m2
//!   For l in 0..32:
//!     out[is*32 + l]     = d1 * ((qs[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - min1
//!     out[(is+1)*32 + l] = d2 * ((qs[l] >> 4)  + ((qh[l] & u2) ? 16 : 0)) - min2
//! ```

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use half::f16;

/// Mirror of the private `get_scale_min_k4` in mlx-native — we re-derive
/// here from the spec rather than pub-expose the internal helper.
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (sc, m)
    }
}

/// Build a synthetic Q5_K block with all fields controllable.
fn build_q5_k_block(d: f32, dmin: f32, scales: &[u8; 12], qh: &[u8; 32], qs: &[u8; 128]) -> [u8; 176] {
    let mut b = [0u8; 176];
    let d_bits = f16::from_f32(d).to_le_bytes();
    let dm_bits = f16::from_f32(dmin).to_le_bytes();
    b[0] = d_bits[0];
    b[1] = d_bits[1];
    b[2] = dm_bits[0];
    b[3] = dm_bits[1];
    b[4..16].copy_from_slice(scales);
    b[16..48].copy_from_slice(qh);
    b[48..176].copy_from_slice(qs);
    b
}

/// Re-implementation of the dequant formula in the test, to cross-validate
/// against the mlx-native implementation.
fn cpu_dequant_q5_k(block: &[u8; 176]) -> [f32; 256] {
    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
    let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
    let scales: &[u8] = &block[4..16];
    let qh: &[u8] = &block[16..48];
    let qs: &[u8] = &block[48..176];

    let mut out = [0.0f32; 256];
    let mut is = 0usize;
    let mut u1: u8 = 1;
    let mut u2: u8 = 2;
    let mut y = 0usize;
    let mut ql_off = 0usize;

    while ql_off < 128 {
        let ql = &qs[ql_off..ql_off + 32];
        let (sc1, m1) = get_scale_min_k4(is, scales);
        let (sc2, m2) = get_scale_min_k4(is + 1, scales);
        let d1 = d * (sc1 as f32);
        let m1 = dmin * (m1 as f32);
        let d2 = d * (sc2 as f32);
        let m2 = dmin * (m2 as f32);

        for l in 0..32 {
            let low = (ql[l] & 0x0F) as u32;
            let high = if (qh[l] & u1) != 0 { 16 } else { 0 };
            out[y] = d1 * (low + high) as f32 - m1;
            y += 1;
        }
        for l in 0..32 {
            let low = (ql[l] >> 4) as u32;
            let high = if (qh[l] & u2) != 0 { 16 } else { 0 };
            out[y] = d2 * (low + high) as f32 - m2;
            y += 1;
        }

        is += 2;
        ql_off += 32;
        u1 <<= 2;
        u2 <<= 2;
    }
    out
}

/// Call mlx-native's (private) `dequantize_to_f32` via the public `load_tensor_f32`.
/// Since neither is publicly exposed for arbitrary bytes, we invoke it through the
/// public `GgufFile::open_from_bytes`-like path. That doesn't exist either — so
/// we test via the round-trip: build a minimal GGUF in-memory with one Q5_K
/// tensor, open it, call `load_tensor_f32`, and compare.
///
/// However that's a lot of scaffolding. Simpler path: re-derive by construction
/// (same as cpu_dequant_q5_k above) and treat this test as a spec-asserting
/// harness. The mlx-native implementation is cross-validated via the apex
/// GGUF `load_tensor_f32` integration test (below) plus a fixed-block numeric
/// test that checks specific output indices.
///
/// Actually — we *can* test the real implementation by going through
/// `load_tensor_f32` with a synthetic GGUF. But to keep scope tight this
/// iter, the cpu_dequant_q5_k function is our authoritative specification
/// and we verify it against hand-computed values for specific configurations.
/// The apex integration test (in hf2q) then cross-validates against the real
/// implementation.

// ===================================================================
// Hand-computed values
// ===================================================================

/// All-zero block: scales=0, qs=0, qh=0. Output should be all zeros.
#[test]
fn q5_k_all_zeros() {
    let block = build_q5_k_block(0.0, 0.0, &[0; 12], &[0; 32], &[0; 128]);
    let out = cpu_dequant_q5_k(&block);
    for (i, v) in out.iter().enumerate() {
        assert!(v.abs() < 1e-6, "expected 0 at {}, got {}", i, v);
    }
}

/// Unit scales, no min, all 5-bit 0: output is 0.
#[test]
fn q5_k_zero_quant_nonzero_scale() {
    let mut scales = [0u8; 12];
    for s in &mut scales {
        *s = 1; // all sub-block scales = 1, mins = 0 (verified via get_scale_min_k4)
    }
    // scales[0..4]: low bits hold scale (1), high bits hold min (0 upper).
    // scales[4..8]: low bits hold min (1 in the & 63 mask? — but we set 1 which is
    //  sc=1 for j<4 and m=scales[4+j]&63=1 → non-zero min.
    // To get min=0 with all-equal bytes, we need scales[4..8]=0 and scales[0..4]=1.
    let mut scales = [0u8; 12];
    for i in 0..4 {
        scales[i] = 1;
    }
    let block = build_q5_k_block(1.0, 0.0, &scales, &[0; 32], &[0; 128]);
    let out = cpu_dequant_q5_k(&block);
    for (i, v) in out.iter().enumerate() {
        assert!(v.abs() < 1e-6, "expected 0 at {} (zero-quant), got {}", i, v);
    }
}

/// Set qs[0] = 0xF (low nibble = 15), qh[0] bit 0 = 1 → q=31, d1=1, min=0.
/// First output value should be 31.
#[test]
fn q5_k_first_value_with_high_bit() {
    let mut scales = [0u8; 12];
    scales[0] = 1; // sub-block 0: scale=1, min=0
    let mut qs = [0u8; 128];
    qs[0] = 0x0F; // low nibble 15
    let mut qh = [0u8; 32];
    qh[0] = 0x01; // u1=1 selects bit 0 for pair 0's first sub-block
    let block = build_q5_k_block(1.0, 0.0, &scales, &qh, &qs);
    let out = cpu_dequant_q5_k(&block);
    assert!(
        (out[0] - 31.0).abs() < 1e-6,
        "expected 31, got {}",
        out[0]
    );
    // Second value: qs[1] low nibble 0, qh[1] 0 → 0.
    assert!(
        out[1].abs() < 1e-6,
        "expected 0, got {}",
        out[1]
    );
}

/// Sub-block 1 (high nibble + u2): qh[0] bit 1 = 1 → q=31, output d2*31.
/// With d=1, sc2=2, m2=0 → output = 62.
#[test]
fn q5_k_second_subblock_uses_high_nibble_and_u2() {
    let mut scales = [0u8; 12];
    scales[0] = 1; // sub-block 0: sc=1, m=0
    scales[1] = 2; // sub-block 1: sc=2, m=0
    let mut qs = [0u8; 128];
    qs[0] = 0xF0; // high nibble = 15
    let mut qh = [0u8; 32];
    qh[0] = 0x02; // u2 = 2 → high bit set for sub-block 1
    let block = build_q5_k_block(1.0, 0.0, &scales, &qh, &qs);
    let out = cpu_dequant_q5_k(&block);
    // Sub-block 0, position 0: low nibble 0 + high bit 0 → 0.
    assert!(out[0].abs() < 1e-6);
    // Sub-block 1, position 0: at index 32. high nibble 15 + high bit 16 → 31. d2=2 → 62.
    assert!(
        (out[32] - 62.0).abs() < 1e-6,
        "expected 62, got {}",
        out[32]
    );
}

/// Third pair uses u1=16, u2=32. qh[0]=0x10 → sub-block 4 gets high bit.
#[test]
fn q5_k_third_pair_uses_shifted_u1() {
    let mut scales = [0u8; 12];
    // Sub-block 4 scale = 3 (stored in upper bits of scales[4+4-4]=scales[4] upper 2
    // + scales[4+4]=scales[8] lower 4 bits. Per get_scale_min_k4(j=4):
    //   sc = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
    //   sc = (scales[8] & 0xF) | ((scales[0] >> 6) << 4)
    // For sc=3: (scales[8] & 0xF) = 3, scales[0]>>6 = 0 → scales[8] = 3, scales[0] = 0.
    scales[8] = 3;
    // Sub-block 4 min = 0 → scales[8] >> 4 = 0 (OK) and scales[0] >> 6 = 0 (OK).
    // So: scales[0]=0, scales[8]=3 gives (sc=3, m=0) for j=4.

    let mut qs = [0u8; 128];
    // Pair 2 (is=4, is+1=5) uses qs bytes 64..96. Low nibble 0x5 at qs[64].
    qs[64] = 0x05;
    let mut qh = [0u8; 32];
    // u1 = 1 << 4 = 0x10 at pair 2. Set bit 4 at qh[0].
    qh[0] = 0x10;

    let block = build_q5_k_block(1.0, 0.0, &scales, &qh, &qs);
    let out = cpu_dequant_q5_k(&block);
    // Output position: pair 2 starts at ys=128. l=0.
    // low nibble 5 + high bit 16 = 21. d1=3 → 63.
    assert!(
        (out[128] - 63.0).abs() < 1e-6,
        "expected 63 at idx 128, got {}",
        out[128]
    );
}

/// Non-zero min: scale=1, min=1, d=2, dmin=3 → output = 2*q - 3*1 = 2q - 3.
/// Low nibble 0xF, high bit set → q=31, output = 2*31 - 3 = 59.
#[test]
fn q5_k_with_nonzero_min() {
    let mut scales = [0u8; 12];
    scales[0] = 1; // sc=1, m=0 (j<4: sc = scales[0]&63 = 1)
    scales[4] = 1; // j<4: m = scales[4]&63 = 1. So (sc, m) = (1, 1). ✓
    let mut qs = [0u8; 128];
    qs[0] = 0x0F;
    let mut qh = [0u8; 32];
    qh[0] = 0x01;
    let block = build_q5_k_block(2.0, 3.0, &scales, &qh, &qs);
    let out = cpu_dequant_q5_k(&block);
    // d1 = 2*1 = 2; m1 = 3*1 = 3; q = 15+16 = 31; output = 2*31 - 3 = 59.
    assert!(
        (out[0] - 59.0).abs() < 1e-6,
        "expected 59, got {}",
        out[0]
    );
}

// ===================================================================
// Cross-validation against mlx-native's dequant via a synthetic GGUF
// ===================================================================
//
// Build a minimal GGUF in memory with one Q5_K tensor, write it to a
// tempfile, open via mlx-native, load_tensor_f32, and compare against
// cpu_dequant_q5_k.

use mlx_native::{gguf::GgufFile, MlxDevice};
use std::io::Write;

/// Write a minimal GGUF file to `path` containing one tensor of the given
/// data, all metadata trimmed to the required fields.
fn write_minimal_gguf(path: &std::path::Path, tensor_name: &str, tensor_data: &[u8]) {
    // Minimal GGUF v3 layout:
    //   u32 magic ("GGUF")
    //   u32 version = 3
    //   u64 n_tensors = 1
    //   u64 n_kv = 0
    //   TensorInfo: name (str), n_dims (u32), dims[] (u64), type (u32 = 13 for Q5_K),
    //                offset (u64)
    //   Padding to alignment (32-byte default).
    //   Tensor data.
    //
    // One Q5_K block = 176 bytes = 256 values. So 1D shape [256].
    let mut buf = Vec::new();
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&1u64.to_le_bytes()); // n_tensors
    buf.extend_from_slice(&0u64.to_le_bytes()); // n_kv

    // Tensor info: name length + name.
    buf.extend_from_slice(&(tensor_name.len() as u64).to_le_bytes());
    buf.extend_from_slice(tensor_name.as_bytes());
    // n_dims = 1.
    buf.extend_from_slice(&1u32.to_le_bytes());
    // dims[0] = 256.
    buf.extend_from_slice(&256u64.to_le_bytes());
    // type = 13 (Q5_K).
    buf.extend_from_slice(&13u32.to_le_bytes());
    // tensor offset (relative to alignment block start) = 0.
    buf.extend_from_slice(&0u64.to_le_bytes());

    // Align to 32 bytes.
    while buf.len() % 32 != 0 {
        buf.push(0);
    }
    buf.extend_from_slice(tensor_data);

    let mut f = std::fs::File::create(path).expect("create tmp gguf");
    f.write_all(&buf).expect("write");
    f.flush().expect("flush");
}

#[test]
fn q5_k_mlx_native_load_tensor_f32_matches_cpu_reference() {
    // Build a non-trivial block.
    let mut scales = [0u8; 12];
    scales[0] = 5;
    scales[1] = 7;
    scales[2] = 11;
    scales[3] = 13;
    scales[4] = 2; // mins 0..3
    scales[5] = 3;
    scales[6] = 5;
    scales[7] = 7;
    scales[8] = 0x21;
    scales[9] = 0x43;
    scales[10] = 0x65;
    scales[11] = 0x17;

    let mut qh = [0u8; 32];
    let mut qs = [0u8; 128];
    // Deterministic bit pattern.
    for i in 0..32 {
        qh[i] = (i as u8).wrapping_mul(37);
    }
    for i in 0..128 {
        qs[i] = (i as u8).wrapping_mul(97).wrapping_add(3);
    }

    let block = build_q5_k_block(0.125, 0.0625, &scales, &qh, &qs);
    let expected = cpu_dequant_q5_k(&block);

    // Write to tempfile and load via mlx-native.
    let tmp = std::env::temp_dir().join(format!(
        "mlx_q5k_test_{}.gguf",
        std::process::id()
    ));
    write_minimal_gguf(&tmp, "test_tensor", &block);

    let device = MlxDevice::new().expect("device");
    let gguf = GgufFile::open(&tmp).expect("open mini gguf");
    let buf = gguf
        .load_tensor_f32("test_tensor", &device)
        .expect("load_tensor_f32");

    let got: &[f32] = buf.as_slice().expect("as slice");
    assert_eq!(got.len(), 256);
    for i in 0..256 {
        let d = (got[i] - expected[i]).abs();
        assert!(
            d < 1e-5,
            "mismatch at {}: got {}, expected {}, diff {}",
            i, got[i], expected[i], d
        );
    }

    std::fs::remove_file(&tmp).ok();
}

// ===================================================================
// I16 dequant
// ===================================================================

#[test]
fn i16_dequant_via_mlx_native_matches_simple_cast() {
    // i16 values: [0, 1, -1, 32767, -32768].
    let values: [i16; 5] = [0, 1, -1, 32767, -32768];
    let mut bytes = [0u8; 10];
    for (i, v) in values.iter().enumerate() {
        bytes[i * 2..i * 2 + 2].copy_from_slice(&v.to_le_bytes());
    }

    // Build minimal GGUF with type=17 (I16), shape=[5].
    let mut buf = Vec::new();
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&1u64.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes());
    let name = "i16_tensor";
    buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
    buf.extend_from_slice(name.as_bytes());
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&5u64.to_le_bytes());
    buf.extend_from_slice(&17u32.to_le_bytes()); // I16
    buf.extend_from_slice(&0u64.to_le_bytes());
    while buf.len() % 32 != 0 {
        buf.push(0);
    }
    buf.extend_from_slice(&bytes);

    let tmp = std::env::temp_dir().join(format!(
        "mlx_i16_test_{}.gguf",
        std::process::id()
    ));
    std::fs::write(&tmp, &buf).expect("write");

    let device = MlxDevice::new().expect("device");
    let gguf = GgufFile::open(&tmp).expect("open mini gguf i16");
    let mbuf = gguf
        .load_tensor_f32("i16_tensor", &device)
        .expect("load_tensor_f32 i16");
    let got: &[f32] = mbuf.as_slice().expect("slice");
    assert_eq!(got.len(), 5);
    for (i, v) in values.iter().enumerate() {
        assert_eq!(got[i], *v as f32, "I16 cast at {}: got {}", i, got[i]);
    }

    std::fs::remove_file(&tmp).ok();
}
