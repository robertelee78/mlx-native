//! ADR-040 iter-G determinism spike (codex-recommended, 2026-06-25).
//!
//! GATING QUESTION: the reverted "mount-trick" batched slot-aware prefill was
//! non-deterministic run-to-run at slot-view `cache_capacity = 32768` (the
//! iter-F-kvcap per-slot Full-layer capacity). §0.17 pinned the suspect to the
//! TQ-HB-V KV path at cap=32768; codex's review narrowed it to the KV-WRITE
//! (`dispatch_hadamard_quantize_kv_hb_seq`, which addresses packed/norms by
//! `params.cache_capacity`). This spike ISOLATES that write at the global-layer
//! shape that actually reaches cap=32768 (head_dim=512, is_sliding=false; the
//! sliding layers stay at cap=1024 and are NOT the suspect).
//!
//! Two decisive checks, both at cap=32768:
//!   (A) DETERMINISM: identical src + identical zero-init output, two dispatches
//!       → written region must be byte-identical (else the kernel itself races).
//!   (B) RMW-INDEPENDENCE: identical src, but output buffers pre-filled with
//!       DIFFERENT sentinel patterns (0x00 vs 0xFF) before each dispatch → the
//!       WRITTEN region [0..n_tokens) must be identical regardless of the
//!       pre-fill. If it differs, the kernel reads its own uninitialized output
//!       (read-modify-write) → that is exactly the cross-process non-determinism
//!       (fresh allocations carry different garbage each launch).
//!
//! A control run at cap=1024 (small) vs cap=32768 (large) localizes any
//! capacity-sensitivity.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::ops::hadamard_quantize_kv;
use mlx_native::{DType, KernelRegistry, MlxDevice};

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();
    hadamard_quantize_kv::register(&mut registry);
    (device, registry)
}

fn norms_per_pos(head_dim: u32) -> usize {
    ((head_dim / 256) as usize).max(1)
}

/// Deterministic src V data: [n_tokens * num_kv_heads * head_dim] f32.
fn make_src(n_tokens: u32, num_kv_heads: u32, head_dim: u32) -> Vec<f32> {
    let n = (n_tokens * num_kv_heads * head_dim) as usize;
    (0..n)
        .map(|i| {
            // bounded, varied, fully deterministic
            let x = (i as f32) * 0.000_173_f32;
            (x.sin() * 1.5) - 0.3 + ((i % 7) as f32) * 0.01
        })
        .collect()
}

/// Run the HB-seq V quantize-write once. `prefill_byte` seeds packed/norms
/// output buffers before the dispatch. Returns (packed_full, norms_full).
#[allow(clippy::too_many_arguments)]
fn run_hb_seq_write(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    src_flat: &[f32],
    num_kv_heads: u32,
    head_dim: u32,
    cache_capacity: u32,
    n_tokens: u32,
    is_sliding: bool,
    cb_bits: u32,
    prefill_byte: u8,
) -> (Vec<u8>, Vec<f32>) {
    // HB = byte-packed (1 byte/element), packed = nkv * cap * head_dim bytes.
    let packed_bytes = (num_kv_heads as usize) * (cache_capacity as usize) * (head_dim as usize);
    let npp = norms_per_pos(head_dim);
    let norms_elems = (num_kv_heads as usize) * (cache_capacity as usize) * npp;

    let mut src_buf = device
        .alloc_buffer(src_flat.len() * 4, DType::F32, vec![src_flat.len()])
        .expect("alloc src");
    src_buf
        .as_mut_slice::<f32>()
        .expect("write src")
        .copy_from_slice(src_flat);

    let mut packed_buf = device
        .alloc_buffer(packed_bytes, DType::U8, vec![packed_bytes])
        .expect("alloc packed");
    for b in packed_buf.as_mut_slice::<u8>().expect("seed packed").iter_mut() {
        *b = prefill_byte;
    }

    let mut norms_buf = device
        .alloc_buffer(norms_elems * 4, DType::F32, vec![norms_elems])
        .expect("alloc norms");
    {
        // Seed norms with a sentinel derived from prefill_byte so an RMW read
        // of norms would also surface. 0x00->0.0, 0xFF->a large sentinel.
        let seed_f = if prefill_byte == 0 { 0.0_f32 } else { 12345.678_f32 };
        for v in norms_buf.as_mut_slice::<f32>().expect("seed norms").iter_mut() {
            *v = seed_f;
        }
    }

    let mut encoder = device.command_encoder().expect("command_encoder");
    hadamard_quantize_kv::dispatch_hadamard_quantize_kv_hb_seq(
        &mut encoder,
        registry,
        device.metal_device(),
        &src_buf,
        &packed_buf,
        &norms_buf,
        num_kv_heads,
        head_dim,
        cache_capacity,
        0, // write_pos_start
        n_tokens,
        0, // src_tok_offset
        is_sliding,
        1.0, // scale_factor_d512
        cb_bits,
    )
    .expect("dispatch_hadamard_quantize_kv_hb_seq");
    encoder.commit_and_wait().expect("commit_and_wait");

    let packed_out = packed_buf.as_slice::<u8>().expect("read packed").to_vec();
    let norms_out = norms_buf.as_slice::<f32>().expect("read norms").to_vec();
    (packed_out, norms_out)
}

/// Compare only the WRITTEN region: per-head, positions [0..n_tokens).
/// Returns (packed_mismatches, norm_mismatches).
fn diff_written_region(
    a_packed: &[u8],
    b_packed: &[u8],
    a_norms: &[f32],
    b_norms: &[f32],
    num_kv_heads: u32,
    head_dim: u32,
    cache_capacity: u32,
    n_tokens: u32,
) -> (usize, usize) {
    let npp = norms_per_pos(head_dim);
    let packed_stride = head_dim as usize; // HB byte-packed: head_dim bytes/slot
    let mut packed_mismatch = 0usize;
    let mut norm_mismatch = 0usize;
    for h in 0..num_kv_heads as usize {
        for pos in 0..n_tokens as usize {
            let base = h * (cache_capacity as usize) * packed_stride + pos * packed_stride;
            for k in 0..packed_stride {
                if a_packed[base + k] != b_packed[base + k] {
                    packed_mismatch += 1;
                }
            }
            let nbase = h * (cache_capacity as usize) * npp + pos * npp;
            for k in 0..npp {
                if a_norms[nbase + k].to_bits() != b_norms[nbase + k].to_bits() {
                    norm_mismatch += 1;
                }
            }
        }
    }
    (packed_mismatch, norm_mismatch)
}

/// CHECK (A)+(B) at cap=32768, global-layer shape (head_dim=512, non-sliding).
#[test]
fn iter_g_hb_seq_write_deterministic_and_rmw_independent_cap32768() {
    let (device, mut registry) = setup();

    let num_kv_heads: u32 = 8;
    let head_dim: u32 = 512; // gemma4 global/Full layers
    let cache_capacity: u32 = 32768; // iter-F-kvcap per-slot Full capacity
    let n_tokens: u32 = 512; // representative prompt chunk
    let is_sliding = false; // Full layers are NOT sliding
    let cb_bits: u32 = 8; // production TQ-HB 8-bit

    let src = make_src(n_tokens, num_kv_heads, head_dim);

    // Run 1: zero-seeded output.
    let (p0, n0) = run_hb_seq_write(
        &device, &mut registry, &src, num_kv_heads, head_dim, cache_capacity, n_tokens,
        is_sliding, cb_bits, 0x00,
    );
    // Run 2: identical src, zero-seeded output → DETERMINISM check.
    let (p0b, n0b) = run_hb_seq_write(
        &device, &mut registry, &src, num_kv_heads, head_dim, cache_capacity, n_tokens,
        is_sliding, cb_bits, 0x00,
    );
    // Run 3: identical src, 0xFF-seeded output → RMW-INDEPENDENCE check.
    let (pf, nf) = run_hb_seq_write(
        &device, &mut registry, &src, num_kv_heads, head_dim, cache_capacity, n_tokens,
        is_sliding, cb_bits, 0xFF,
    );

    let (det_p, det_n) = diff_written_region(
        &p0, &p0b, &n0, &n0b, num_kv_heads, head_dim, cache_capacity, n_tokens,
    );
    let (rmw_p, rmw_n) = diff_written_region(
        &p0, &pf, &n0, &nf, num_kv_heads, head_dim, cache_capacity, n_tokens,
    );

    println!(
        "[iter-G spike cap=32768] determinism: packed_mismatch={} norm_mismatch={} | \
         rmw-independence: packed_mismatch={} norm_mismatch={}",
        det_p, det_n, rmw_p, rmw_n
    );

    assert_eq!(det_p, 0, "DETERMINISM FAIL: packed differs run-to-run (kernel races)");
    assert_eq!(det_n, 0, "DETERMINISM FAIL: norms differ run-to-run (non-deterministic reduction)");
    assert_eq!(
        rmw_p, 0,
        "RMW FAIL: packed written region depends on pre-fill pattern → kernel reads \
         uninitialized output (this IS the cross-process non-determinism source)"
    );
    assert_eq!(
        rmw_n, 0,
        "RMW FAIL: norms written region depends on pre-fill pattern → kernel reads \
         uninitialized output"
    );
}

/// Control: same checks at cap=1024 (small). If cap=32768 fails but this passes,
/// the bug is capacity-sensitive (the iter-F-kvcap large-capacity regime).
#[test]
fn iter_g_hb_seq_write_deterministic_and_rmw_independent_cap1024_control() {
    let (device, mut registry) = setup();

    let num_kv_heads: u32 = 8;
    let head_dim: u32 = 512;
    let cache_capacity: u32 = 1024;
    let n_tokens: u32 = 512;
    let is_sliding = false;
    let cb_bits: u32 = 8;

    let src = make_src(n_tokens, num_kv_heads, head_dim);

    let (p0, n0) = run_hb_seq_write(
        &device, &mut registry, &src, num_kv_heads, head_dim, cache_capacity, n_tokens,
        is_sliding, cb_bits, 0x00,
    );
    let (p0b, n0b) = run_hb_seq_write(
        &device, &mut registry, &src, num_kv_heads, head_dim, cache_capacity, n_tokens,
        is_sliding, cb_bits, 0x00,
    );
    let (pf, nf) = run_hb_seq_write(
        &device, &mut registry, &src, num_kv_heads, head_dim, cache_capacity, n_tokens,
        is_sliding, cb_bits, 0xFF,
    );

    let (det_p, det_n) = diff_written_region(
        &p0, &p0b, &n0, &n0b, num_kv_heads, head_dim, cache_capacity, n_tokens,
    );
    let (rmw_p, rmw_n) = diff_written_region(
        &p0, &pf, &n0, &nf, num_kv_heads, head_dim, cache_capacity, n_tokens,
    );

    println!(
        "[iter-G spike cap=1024 control] determinism: packed_mismatch={} norm_mismatch={} | \
         rmw-independence: packed_mismatch={} norm_mismatch={}",
        det_p, det_n, rmw_p, rmw_n
    );

    assert_eq!(det_p, 0, "control determinism packed");
    assert_eq!(det_n, 0, "control determinism norms");
    assert_eq!(rmw_p, 0, "control rmw packed");
    assert_eq!(rmw_n, 0, "control rmw norms");
}
