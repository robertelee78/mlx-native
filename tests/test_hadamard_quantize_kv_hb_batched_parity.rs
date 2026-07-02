//! ADR-040 M-SPEED-LC Stage 2/3 (codex CHANGES-REQUIRED follow-up) — bit-
//! parity test for `dispatch_hadamard_quantize_kv_hb_batched`.
//!
//! `batched_body.rs`'s FullTq KV-encode arm calls this batched encoder TWICE
//! per layer (once for K, once for V) instead of the scalar production
//! path's single fused `dispatch_hadamard_quantize_kv_hb_dual` dispatch. The
//! comment there previously claimed this was proven equivalent by
//! `test_hadamard_quantize_kv_hb_dual_byte_identity_d256` — that test only
//! covers fused-dual vs two SCALAR single-position dispatches, NOT the
//! batched-multi-query kernel at all. This file supplies the missing proof:
//! `dispatch_hadamard_quantize_kv_hb_batched` (N queries, ONE dispatch,
//! per-query slot/position addressing) must write BIT-IDENTICAL
//! packed/norms bytes to running the SCALAR `dispatch_hadamard_quantize_kv_hb`
//! once per (slot, position) — zero tolerance, `u8`/`f32::to_bits()` equality.
//!
//! Matrix: head_dim {256, 512} x codebook_bits {5, 6, 8}, each combo run at:
//!   - N=8 "ring-mixed": is_sliding=true, shared across the dispatch, with
//!     PER-QUERY DIFFERENT positions such that some wrap (pos >= capacity)
//!     and some don't (pos < capacity) — "ring + linear" in one dispatch,
//!     mirroring how `flash_attn_vec_tq_hb_batched`'s parity test covers
//!     both regimes via mask_type=2 + varying per-query `seq_pos`.
//!   - N=8 "linear": is_sliding=false, distinct in-range positions per query.

use mlx_native::ops::hadamard_quantize_kv::{dispatch_hadamard_quantize_kv_hb, dispatch_hadamard_quantize_kv_hb_batched};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

struct Xor {
    s: u64,
}
impl Xor {
    fn new(seed: u64) -> Self {
        Self { s: seed.wrapping_add(1) }
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.s;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.s = x;
        x
    }
    /// Uniform in [-1, 1).
    fn next_signed(&mut self) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32;
        u * 2.0 - 1.0
    }
}

fn write_f32_buf(device: &MlxDevice, name: &str, data: &[f32], shape: Vec<usize>) -> MlxBuffer {
    let mut buf = device
        .alloc_buffer(data.len() * 4, DType::F32, shape)
        .unwrap_or_else(|e| panic!("alloc {name}: {e}"));
    buf.as_mut_slice::<f32>()
        .unwrap_or_else(|e| panic!("write {name}: {e}"))
        .copy_from_slice(data);
    buf
}

fn write_u32_buf(device: &MlxDevice, name: &str, data: &[u32], shape: Vec<usize>) -> MlxBuffer {
    let mut buf = device
        .alloc_buffer(data.len() * 4, DType::U32, shape)
        .unwrap_or_else(|e| panic!("alloc {name}: {e}"));
    buf.as_mut_slice::<u32>()
        .unwrap_or_else(|e| panic!("write {name}: {e}"))
        .copy_from_slice(data);
    buf
}

const NKV: u32 = 2;
const CAP: u32 = 64;
const N_SLOTS: usize = 8;
// Non-trivial permutation of physical slots — exercises the batched kernel's
// per-query `slot * (nkv*cap*hd)` base-offset arithmetic (not the degenerate
// identity mapping).
const SLOT_PERM: [u32; N_SLOTS] = [3, 7, 0, 5, 1, 6, 2, 4];

fn npp_for(hd: u32) -> usize {
    if hd == 512 { 2 } else { 1 }
}

/// Run the batched encoder once for `n_q` queries into fresh full multi-seq
/// `[N_SLOTS, nkv, cap, hd]` / `[N_SLOTS, nkv, cap, npp]` buffers. Returns
/// (packed_bytes, norms).
#[allow(clippy::too_many_arguments)]
fn run_batched(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    hd: u32,
    cbits: u32,
    is_sliding: bool,
    slot_ids: &[u32],
    seq_pos: &[u32],
    src_data: &[f32], // [n_q, nkv*hd]
) -> (Vec<u8>, Vec<f32>) {
    let n_q = slot_ids.len() as u32;
    let npp = npp_for(hd);
    let packed_elems = N_SLOTS * NKV as usize * CAP as usize * hd as usize;
    let norms_elems = N_SLOTS * NKV as usize * CAP as usize * npp;

    let src_buf = write_f32_buf(device, "src", src_data, vec![n_q as usize, (NKV * hd) as usize]);
    let mut packed_buf = device
        .alloc_buffer(packed_elems, DType::U8, vec![N_SLOTS, NKV as usize, CAP as usize, hd as usize])
        .expect("alloc packed_batched");
    let mut norms_buf = device
        .alloc_buffer(norms_elems * 4, DType::F32, vec![N_SLOTS, NKV as usize, CAP as usize, npp])
        .expect("alloc norms_batched");
    let slot_id_buf = write_u32_buf(device, "slot_id_arr", slot_ids, vec![n_q as usize]);
    let seq_pos_buf = write_u32_buf(device, "seq_pos_arr", seq_pos, vec![n_q as usize]);

    let mut encoder = device.command_encoder().expect("enc");
    dispatch_hadamard_quantize_kv_hb_batched(
        &mut encoder, registry, device.metal_device(),
        &src_buf, &packed_buf, &norms_buf, &slot_id_buf, &seq_pos_buf,
        n_q, NKV, hd, CAP, is_sliding, 1.0, cbits,
    ).expect("dispatch batched");
    encoder.commit_and_wait().expect("commit batched");

    let packed: Vec<u8> = packed_buf.as_mut_slice::<u8>().unwrap().to_vec();
    let norms: Vec<f32> = norms_buf.as_mut_slice::<f32>().unwrap().to_vec();
    (packed, norms)
}

/// Run the SCALAR encoder once per (slot, position) into fresh full multi-seq
/// buffers of the SAME shape as `run_batched`, via a `slice_view` at each
/// query's slot offset — so the two outputs are directly byte-comparable.
#[allow(clippy::too_many_arguments)]
fn run_scalar_per_query(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    hd: u32,
    cbits: u32,
    is_sliding: bool,
    slot_ids: &[u32],
    seq_pos: &[u32],
    src_data: &[f32], // [n_q, nkv*hd]
) -> (Vec<u8>, Vec<f32>) {
    let n_q = slot_ids.len();
    let npp = npp_for(hd);
    let packed_elems = N_SLOTS * NKV as usize * CAP as usize * hd as usize;
    let norms_elems = N_SLOTS * NKV as usize * CAP as usize * npp;

    let mut packed_buf = device
        .alloc_buffer(packed_elems, DType::U8, vec![N_SLOTS, NKV as usize, CAP as usize, hd as usize])
        .expect("alloc packed_scalar");
    let mut norms_buf = device
        .alloc_buffer(norms_elems * 4, DType::F32, vec![N_SLOTS, NKV as usize, CAP as usize, npp])
        .expect("alloc norms_scalar");

    let per_slot_packed = NKV as usize * CAP as usize * hd as usize;
    let per_slot_norms = NKV as usize * CAP as usize * npp;

    for i in 0..n_q {
        let slot = slot_ids[i] as usize;
        let src_row = &src_data[i * (NKV * hd) as usize..(i + 1) * (NKV * hd) as usize];
        let src_buf = write_f32_buf(device, "src_i", src_row, vec![NKV as usize, hd as usize]);
        let packed_view = packed_buf
            .slice_view((slot * per_slot_packed) as u64, per_slot_packed)
            .with_shape(vec![NKV as usize, CAP as usize, hd as usize])
            .expect("packed slice_view");
        let norms_view = norms_buf
            .slice_view((slot * per_slot_norms) as u64 * 4, per_slot_norms)
            .with_shape(if npp == 1 { vec![NKV as usize, CAP as usize] } else { vec![NKV as usize, CAP as usize, npp] })
            .expect("norms slice_view");

        let mut encoder = device.command_encoder().expect("enc");
        dispatch_hadamard_quantize_kv_hb(
            &mut encoder, registry, device.metal_device(),
            &src_buf, &packed_view, &norms_view,
            NKV, hd, CAP, seq_pos[i], is_sliding, 1.0, cbits,
        ).expect("dispatch scalar");
        encoder.commit_and_wait().expect("commit scalar");
    }

    let packed: Vec<u8> = packed_buf.as_mut_slice::<u8>().unwrap().to_vec();
    let norms: Vec<f32> = norms_buf.as_mut_slice::<f32>().unwrap().to_vec();
    (packed, norms)
}

fn assert_bit_equal_u8(label: &str, a: &[u8], b: &[u8]) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(x, y, "{label}: packed byte {i} differs (batched={x} vs scalar={y})");
    }
}

fn assert_bit_equal_f32(label: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(), y.to_bits(),
            "{label}: norm {i} not bit-equal (batched={x:?} vs scalar={y:?})",
        );
    }
}

fn run_combo(device: &MlxDevice, registry: &mut KernelRegistry, hd: u32, cbits: u32) {
    let combo_tag = format!("dk{hd}_cbits{cbits}");
    let mut rng = Xor::new(0xA5A5_0000 ^ (hd as u64) ^ (cbits as u64) << 8);
    let gen_src = |rng: &mut Xor, n_q: usize| -> Vec<f32> {
        (0..(n_q * (NKV * hd) as usize)).map(|_| rng.next_signed()).collect()
    };

    // ---- N=8 "ring-mixed": is_sliding=true, some queries wrap (pos>=CAP),
    // some don't (pos<CAP), full slot permutation, distinct content. ----
    {
        let is_sliding = true;
        let slot_ids: Vec<u32> = SLOT_PERM.to_vec();
        let seq_pos: [u32; N_SLOTS] = [10, CAP + 5, 20, CAP + 30, 40, CAP + 2, 55, CAP + 50];
        let mut saw_wrap = false;
        let mut saw_linear = false;
        for &p in &seq_pos {
            if p >= CAP { saw_wrap = true; } else { saw_linear = true; }
        }
        assert!(saw_wrap && saw_linear, "ring-mixed case must contain both regimes");
        let src = gen_src(&mut rng, N_SLOTS);

        let (packed_b, norms_b) = run_batched(device, registry, hd, cbits, is_sliding, &slot_ids, &seq_pos, &src);
        let (packed_s, norms_s) = run_scalar_per_query(device, registry, hd, cbits, is_sliding, &slot_ids, &seq_pos, &src);

        assert_bit_equal_u8(&format!("{combo_tag} ring-mixed packed"), &packed_b, &packed_s);
        assert_bit_equal_f32(&format!("{combo_tag} ring-mixed norms"), &norms_b, &norms_s);
    }

    // ---- N=8 "linear": is_sliding=false, distinct in-range positions. ----
    {
        let is_sliding = false;
        let slot_ids: Vec<u32> = SLOT_PERM.to_vec();
        let seq_pos: [u32; N_SLOTS] = [1, 5, 12, 20, 30, 40, 50, 63];
        let src = gen_src(&mut rng, N_SLOTS);

        let (packed_b, norms_b) = run_batched(device, registry, hd, cbits, is_sliding, &slot_ids, &seq_pos, &src);
        let (packed_s, norms_s) = run_scalar_per_query(device, registry, hd, cbits, is_sliding, &slot_ids, &seq_pos, &src);

        assert_bit_equal_u8(&format!("{combo_tag} linear packed"), &packed_b, &packed_s);
        assert_bit_equal_f32(&format!("{combo_tag} linear norms"), &norms_b, &norms_s);
    }
}

#[test]
fn hadamard_quantize_kv_hb_batched_bit_parity_matrix() {
    let device = MlxDevice::new().expect("MlxDevice");
    let mut registry = KernelRegistry::new();

    let mut n_run = 0usize;
    for &hd in &[256u32, 512u32] {
        for &cbits in &[5u32, 6u32, 8u32] {
            run_combo(&device, &mut registry, hd, cbits);
            n_run += 1;
        }
    }
    assert_eq!(n_run, 6);
    println!("hadamard_quantize_kv_hb_batched_bit_parity_matrix: {n_run} combos x 2 sub-cases (ring-mixed, linear), all bit-exact");
}
