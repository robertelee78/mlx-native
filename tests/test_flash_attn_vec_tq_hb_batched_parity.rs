//! ADR-040 M-SPEED-LC — bit-parity test for the BATCHED TQ-HB decode kernel.
//!
//! Verifies that `flash_attn_vec_tq_hb_batched` (N queries in ONE dispatch)
//! produces a BIT-IDENTICAL (`f32::to_bits()` equal, zero tolerance) per-query
//! output to running the scalar production hot path
//! (`flash_attn_vec_tq_hb_with_fused_undo`, one dispatch per query) against
//! the same physical-slot content.
//!
//! Content is synthetic uniform-random bytes/floats (NOT run through
//! `hadamard_quantize_kv`) — bit-parity between the batched and single-seq
//! dispatch paths doesn't depend on the packed bytes being a valid
//! Hadamard-quantized encoding, only on both paths reading the identical
//! bytes and running the identical math.
//!
//! Matrix: head_dim {256, 512} × codebook_bits {5, 6, 8} × nsg {1, 4}, each
//! combo run at:
//!   - N=1 "linear"          (mask_type=2, ring never wraps ⇒ rs_b=0)
//!   - N=1 "wrap"            (mask_type=2, ring wraps ⇒ rs_b≠0)
//!   - N=1 "no-ring"         (mask_type=0 sanity check)
//!   - N=8 "mixed"           (mask_type=2, alternating linear/wrapped per
//!                            query, DIFFERENT per-slot content + DIFFERENT
//!                            kv_seq_len via a non-trivial slot permutation)
//!   - N=1 "sliding-inside"  (mask_type=2, sliding_window>0 but
//!                            kv_seq_len ≤ sliding_window ⇒ window_start=0,
//!                            no clipping)
//!   - N=1 "sliding-clipped" (mask_type=2, sliding_window>0 and
//!                            kv_seq_len > sliding_window ⇒
//!                            window_start_logical > 0, clipping active)
//!   - N=1 "sliding+wrap"    (mask_type=2, ring wraps AND sliding-window
//!                            clipping both active simultaneously)
//!   - N=8 "sliding-mixed"   (mask_type=2, shared sliding_window, PER-QUERY
//!                            DIFFERENT kv_seq_len so some queries clip and
//!                            some don't in the SAME batched dispatch, plus
//!                            one query that wraps AND clips)
//!
//! NWG is left at its DEFAULT (kv_capacity=128 ≤ 512 ⇒ compute_nwg returns
//! 16) for every case here. `compute_nwg`'s `HF2Q_TQ_NWG` override is cached
//! in a function-local `AtomicI32` that latches on first read for the life
//! of the process, so nwg=1 cannot be safely exercised in the same binary as
//! the default-nwg matrix — see the sibling
//! `test_flash_attn_vec_tq_hb_batched_parity_nwg1.rs` (separate process) for
//! the forced-NWG=1 branch (SDPA writes directly to `output`, then
//! `fwht_sign_undo_f32` across `n_q*num_heads` rows — the `else` branch of
//! `flash_attn_vec_tq_hb_batched`).

use mlx_native::ops::flash_attn_vec_tq_hb::{self, FlashAttnVecTqHbParams};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

// Deterministic xorshift PRNG (same recipe as
// test_flash_attn_vec_reduce_tq_hb_undo_parity.rs) so the test needs no
// external `rand` dependency.
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
    fn next_u8(&mut self) -> u8 {
        (self.next_u64() >> 32) as u8
    }
    /// Uniform in [0.25, 2.0) — avoids degenerate zero norms.
    fn next_norm(&mut self) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32;
        0.25 + u * 1.75
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

fn write_u8_buf(device: &MlxDevice, name: &str, data: &[u8], shape: Vec<usize>) -> MlxBuffer {
    let mut buf = device
        .alloc_buffer(data.len(), DType::U8, shape)
        .unwrap_or_else(|e| panic!("alloc {name}: {e}"));
    buf.as_mut_slice::<u8>()
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

/// Fixed shape used by every combo in this matrix.
const NH: u32 = 8; // query heads
const NKV: u32 = 2; // KV heads (heads_per_kv = 4)
const CAP: u32 = 128; // kv_capacity — kept ≤512 so compute_nwg always picks 16
                       // (no HF2Q_TQ_NWG override needed — see module doc).
const N_SLOTS: usize = 8;
// Non-trivial permutation of physical slots — exercises slot_id_arr base-
// offset arithmetic (not just the degenerate identity mapping).
const SLOT_PERM: [u32; N_SLOTS] = [3, 7, 0, 5, 1, 6, 2, 4];

/// Derive (kv_seq_len, ring_start) exactly as `flash_attn_vec_tq_hb_batched_impl`
/// derives (ksl_b, rs_b) from (seq_pos, mask_type, kv_capacity).
fn derive_ksl_rs(sp: u32, mask_type: u32, cap: u32) -> (u32, u32) {
    let is_ring = mask_type == 2;
    let ksl = if is_ring { (sp + 1).min(cap) } else { sp + 1 };
    let rs = if is_ring && ksl >= cap { (sp + 1) % cap } else { 0 };
    (ksl, rs)
}

struct SlotData {
    k_packed: Vec<u8>,   // [nkv, cap, hd]
    k_norms: Vec<f32>,   // [nkv, cap, npp]
    v_packed: Vec<u8>,   // [nkv, cap, hd]
    v_norms: Vec<f32>,   // [nkv, cap, npp]
}

fn gen_slot(seed: u64, nkv: u32, cap: u32, hd: u32, npp: u32) -> SlotData {
    let mut rng = Xor::new(seed);
    let kv_elems = (nkv * cap * hd) as usize;
    let norm_elems = (nkv * cap * npp) as usize;
    SlotData {
        k_packed: (0..kv_elems).map(|_| rng.next_u8()).collect(),
        k_norms: (0..norm_elems).map(|_| rng.next_norm()).collect(),
        v_packed: (0..kv_elems).map(|_| rng.next_u8()).collect(),
        v_norms: (0..norm_elems).map(|_| rng.next_norm()).collect(),
    }
}

/// Run the batched dispatch for `n_q` queries and return the flat
/// `[n_q*NH, hd]` output.
#[allow(clippy::too_many_arguments)]
fn run_batched(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    slots: &[SlotData],
    hd: u32,
    npp: u32,
    cbits: u32,
    nsg: u32,
    mask_type: u32,
    sliding_window: u32,
    slot_ids: &[u32],
    seq_pos: &[u32],
    q_data: &[f32], // [n_q, NH, hd]
) -> Vec<f32> {
    let n_q = slot_ids.len() as u32;
    assert_eq!(seq_pos.len() as u32, n_q);

    // Concatenate all N_SLOTS physical slots into the "full multi-seq" buffers.
    let mut k_packed_big = Vec::new();
    let mut k_norms_big = Vec::new();
    let mut v_packed_big = Vec::new();
    let mut v_norms_big = Vec::new();
    for s in slots {
        k_packed_big.extend_from_slice(&s.k_packed);
        k_norms_big.extend_from_slice(&s.k_norms);
        v_packed_big.extend_from_slice(&s.v_packed);
        v_norms_big.extend_from_slice(&s.v_norms);
    }

    let k_packed_buf = write_u8_buf(device, "K_packed_big", &k_packed_big,
        vec![N_SLOTS, NKV as usize, CAP as usize, hd as usize]);
    let k_norms_buf = write_f32_buf(device, "K_norms_big", &k_norms_big,
        vec![N_SLOTS, NKV as usize, CAP as usize, npp as usize]);
    let v_packed_buf = write_u8_buf(device, "V_packed_big", &v_packed_big,
        vec![N_SLOTS, NKV as usize, CAP as usize, hd as usize]);
    let v_norms_buf = write_f32_buf(device, "V_norms_big", &v_norms_big,
        vec![N_SLOTS, NKV as usize, CAP as usize, npp as usize]);

    let q_buf = write_f32_buf(device, "Q", q_data, vec![n_q as usize, NH as usize, hd as usize]);
    let slot_id_buf = write_u32_buf(device, "slot_id_arr", slot_ids, vec![n_q as usize]);
    let seq_pos_buf = write_u32_buf(device, "seq_pos_arr", seq_pos, vec![n_q as usize]);

    let output_buf = device
        .alloc_buffer((n_q * NH * hd * 4) as usize, DType::F32, vec![(n_q * NH) as usize, hd as usize])
        .expect("alloc output");
    let tmp_bytes = flash_attn_vec_tq_hb::tmp_buffer_bytes(n_q * NH, hd);
    let tmp_buf = device
        .alloc_buffer(tmp_bytes, DType::F32, vec![tmp_bytes / 4])
        .expect("alloc tmp");

    // kv_seq_len passed in params is only used for NWG-bucket selection —
    // must be the MAX ksl_b over the queries actually dispatched.
    let max_ksl = seq_pos
        .iter()
        .map(|&sp| derive_ksl_rs(sp, mask_type, CAP).0)
        .max()
        .unwrap();

    let params = FlashAttnVecTqHbParams {
        num_heads: NH,
        num_kv_heads: NKV,
        head_dim: hd,
        kv_seq_len: max_ksl,
        kv_capacity: CAP,
        scale: 1.0 / (hd as f32).sqrt(),
        mask_type,
        sliding_window,
        softcap: 0.0,
        ring_start: 0, // unused by the batched kernel (derived per-query from seq_pos_arr)
        scale_factor_d512: 1.0,
        codebook_bits: cbits,
        fuse_fwht_pre: 1,
        nsg,
    };

    let mut encoder = device.command_encoder().expect("enc");
    flash_attn_vec_tq_hb::flash_attn_vec_tq_hb_batched(
        &mut encoder, registry, device,
        n_q,
        &q_buf, &k_packed_buf, &k_norms_buf, &v_packed_buf, &v_norms_buf,
        &output_buf, &tmp_buf,
        &slot_id_buf, &seq_pos_buf,
        &params,
    ).expect("dispatch batched");
    encoder.commit_and_wait().expect("commit batched");

    output_buf.as_slice::<f32>().unwrap().to_vec()
}

/// Run the scalar production hot path for ONE query against one slot's
/// content and return its `[NH, hd]` output.
#[allow(clippy::too_many_arguments)]
fn run_single_seq_reference(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    slot: &SlotData,
    hd: u32,
    cbits: u32,
    nsg: u32,
    kv_seq_len: u32,
    ring_start: u32,
    mask_type: u32,
    sliding_window: u32,
    q_row: &[f32], // [NH, hd]
) -> Vec<f32> {
    let npp = if hd == 512 { 2usize } else { 1usize };
    let k_packed_buf = write_u8_buf(device, "K_packed", &slot.k_packed,
        vec![NKV as usize, CAP as usize, hd as usize]);
    let k_norms_buf = write_f32_buf(device, "K_norms", &slot.k_norms,
        vec![NKV as usize, CAP as usize, npp]);
    let v_packed_buf = write_u8_buf(device, "V_packed", &slot.v_packed,
        vec![NKV as usize, CAP as usize, hd as usize]);
    let v_norms_buf = write_f32_buf(device, "V_norms", &slot.v_norms,
        vec![NKV as usize, CAP as usize, npp]);
    let q_buf = write_f32_buf(device, "Q", q_row, vec![NH as usize, 1, hd as usize]);

    let output_buf = device
        .alloc_buffer((NH * hd * 4) as usize, DType::F32, vec![NH as usize, hd as usize])
        .expect("alloc output");
    let tmp_bytes = flash_attn_vec_tq_hb::tmp_buffer_bytes(NH, hd);
    let tmp_buf = device
        .alloc_buffer(tmp_bytes, DType::F32, vec![tmp_bytes / 4])
        .expect("alloc tmp");

    let params = FlashAttnVecTqHbParams {
        num_heads: NH,
        num_kv_heads: NKV,
        head_dim: hd,
        kv_seq_len,
        kv_capacity: CAP,
        scale: 1.0 / (hd as f32).sqrt(),
        // mask_type/sliding_window must mirror the batched dispatch's shared
        // values exactly: the kernel's sliding-window branch
        // (`window_start_logical`) reads `params.mask_type`/`sliding_window`
        // directly, not anything derived from ring_start/kv_seq_len — so a
        // mismatch here would silently skip clipping in the reference while
        // the batched kernel still applies it (or vice versa).
        mask_type,
        sliding_window,
        softcap: 0.0,
        ring_start,
        scale_factor_d512: 1.0,
        codebook_bits: cbits,
        fuse_fwht_pre: 1,
        nsg,
    };

    let mut encoder = device.command_encoder().expect("enc");
    flash_attn_vec_tq_hb::flash_attn_vec_tq_hb_with_fused_undo(
        &mut encoder, registry, device,
        &q_buf, &k_packed_buf, &k_norms_buf, &v_packed_buf, &v_norms_buf,
        &output_buf, &tmp_buf,
        &params,
    ).expect("dispatch reference");
    encoder.commit_and_wait().expect("commit reference");

    output_buf.as_slice::<f32>().unwrap().to_vec()
}

fn assert_bit_equal(label: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(), y.to_bits(),
            "{label}: element {i} not bit-equal (batched={x:?} [{:#010x}] vs reference={y:?} [{:#010x}])",
            x.to_bits(), y.to_bits(),
        );
    }
}

/// One matrix point: head_dim × codebook_bits × nsg. Exercises all four
/// sub-cases (N=1 linear/wrap/no-ring, N=8 mixed) against this combo.
fn run_combo(device: &MlxDevice, registry: &mut KernelRegistry, hd: u32, cbits: u32, nsg: u32) {
    let npp = if hd == 512 { 2 } else { 1 };
    let combo_tag = format!("dk{hd}_cbits{cbits}_nsg{nsg}");

    // Fresh synthetic content for all 8 physical slots (reused across the
    // sub-cases below).
    let slots: Vec<SlotData> = (0..N_SLOTS)
        .map(|i| gen_slot(0x9e3779b9_0000_0000 ^ (i as u64) ^ ((hd as u64) << 32) ^ (cbits as u64), NKV, CAP, hd, npp))
        .collect();

    let mut qrng = Xor::new(0xC0FFEE ^ (hd as u64) ^ (cbits as u64) << 8 ^ (nsg as u64) << 16);
    let gen_q = |rng: &mut Xor, n_q: usize| -> Vec<f32> {
        (0..(n_q * NH as usize * hd as usize)).map(|_| rng.next_signed()).collect()
    };

    // ---- N=1 "linear" (mask_type=2, rs_b=0) ----
    {
        let slot = SLOT_PERM[0];
        let sp = 47u32;
        let mask_type = 2u32;
        let (ksl, rs) = derive_ksl_rs(sp, mask_type, CAP);
        assert_eq!(rs, 0, "linear case must not wrap");
        let q = gen_q(&mut qrng, 1);

        let batched = run_batched(
            device, registry, &slots, hd, npp, cbits, nsg, mask_type, 0,
            &[slot], &[sp], &q,
        );
        let reference = run_single_seq_reference(
            device, registry, &slots[slot as usize], hd, cbits, nsg, ksl, rs, mask_type, 0, &q,
        );
        assert_bit_equal(&format!("{combo_tag} N=1 linear"), &batched, &reference);
    }

    // ---- N=1 "wrap" (mask_type=2, rs_b != 0) ----
    {
        let slot = SLOT_PERM[1];
        let sp = (CAP + 10) as u32; // ksl=CAP, rs = (sp+1) % CAP != 0
        let mask_type = 2u32;
        let (ksl, rs) = derive_ksl_rs(sp, mask_type, CAP);
        assert_ne!(rs, 0, "wrap case must actually wrap");
        let q = gen_q(&mut qrng, 1);

        let batched = run_batched(
            device, registry, &slots, hd, npp, cbits, nsg, mask_type, 0,
            &[slot], &[sp], &q,
        );
        let reference = run_single_seq_reference(
            device, registry, &slots[slot as usize], hd, cbits, nsg, ksl, rs, mask_type, 0, &q,
        );
        assert_bit_equal(&format!("{combo_tag} N=1 wrap"), &batched, &reference);
    }

    // ---- N=1 "no-ring" sanity (mask_type=0) ----
    {
        let slot = SLOT_PERM[2];
        let sp = 47u32;
        let mask_type = 0u32;
        let (ksl, rs) = derive_ksl_rs(sp, mask_type, CAP);
        assert_eq!(rs, 0, "non-ring mask_type must never wrap");
        let q = gen_q(&mut qrng, 1);

        let batched = run_batched(
            device, registry, &slots, hd, npp, cbits, nsg, mask_type, 0,
            &[slot], &[sp], &q,
        );
        let reference = run_single_seq_reference(
            device, registry, &slots[slot as usize], hd, cbits, nsg, ksl, rs, mask_type, 0, &q,
        );
        assert_bit_equal(&format!("{combo_tag} N=1 no-ring"), &batched, &reference);
    }

    // ---- N=8 "mixed" (mask_type=2, alternating linear/wrapped per query,
    // full slot permutation, DIFFERENT content + DIFFERENT positions) ----
    {
        let mask_type = 2u32;
        let seq_pos: [u32; N_SLOTS] = [20, CAP + 5, 40, CAP + 30, 60, CAP + 2, 90, CAP + 50];
        let slot_ids: Vec<u32> = SLOT_PERM.to_vec();
        let q = gen_q(&mut qrng, N_SLOTS);

        // Sanity: the mix actually contains both linear (rs=0) and wrapped
        // (rs!=0) queries, as required by the test matrix.
        let mut saw_linear = false;
        let mut saw_wrap = false;
        for &sp in &seq_pos {
            let (_, rs) = derive_ksl_rs(sp, mask_type, CAP);
            if rs == 0 { saw_linear = true; } else { saw_wrap = true; }
        }
        assert!(saw_linear && saw_wrap, "N=8 mixed case must contain both regimes");

        let batched = run_batched(
            device, registry, &slots, hd, npp, cbits, nsg, mask_type, 0,
            &slot_ids, &seq_pos, &q,
        );

        for (i, (&slot, &sp)) in slot_ids.iter().zip(seq_pos.iter()).enumerate() {
            let (ksl, rs) = derive_ksl_rs(sp, mask_type, CAP);
            let q_row = &q[i * NH as usize * hd as usize..(i + 1) * NH as usize * hd as usize];
            let reference = run_single_seq_reference(
                device, registry, &slots[slot as usize], hd, cbits, nsg, ksl, rs, mask_type, 0, q_row,
            );
            let batched_row = &batched[i * NH as usize * hd as usize..(i + 1) * NH as usize * hd as usize];
            assert_bit_equal(&format!("{combo_tag} N=8 mixed query={i} slot={slot}"), batched_row, &reference);
        }
    }

    // Shared sliding_window for all sliding-window sub-cases below.
    // CAP=128 so W=32 leaves plenty of room for both "inside" (kv_seq_len≤W)
    // and "clipped" (kv_seq_len>W) regimes, including combined with wrap.
    const SLIDING_W: u32 = 32;

    // ---- N=1 "sliding-inside" (mask_type=2, sliding_window>0 but
    // kv_seq_len ≤ sliding_window ⇒ window_start_logical=0, no clipping) ----
    {
        let slot = SLOT_PERM[3];
        let sp = 20u32; // ksl=21 ≤ SLIDING_W=32 ⇒ no clip
        let mask_type = 2u32;
        let (ksl, rs) = derive_ksl_rs(sp, mask_type, CAP);
        assert!(ksl <= SLIDING_W, "sliding-inside case must not exceed the window");
        let q = gen_q(&mut qrng, 1);

        let batched = run_batched(
            device, registry, &slots, hd, npp, cbits, nsg, mask_type, SLIDING_W,
            &[slot], &[sp], &q,
        );
        let reference = run_single_seq_reference(
            device, registry, &slots[slot as usize], hd, cbits, nsg, ksl, rs, mask_type, SLIDING_W, &q,
        );
        assert_bit_equal(&format!("{combo_tag} N=1 sliding-inside"), &batched, &reference);
    }

    // ---- N=1 "sliding-clipped" (mask_type=2, kv_seq_len > sliding_window ⇒
    // window_start_logical > 0, clipping active) ----
    {
        let slot = SLOT_PERM[4];
        let sp = 79u32; // ksl=80 > SLIDING_W=32 ⇒ window_start_logical=48
        let mask_type = 2u32;
        let (ksl, rs) = derive_ksl_rs(sp, mask_type, CAP);
        assert!(ksl > SLIDING_W, "sliding-clipped case must exceed the window");
        let q = gen_q(&mut qrng, 1);

        let batched = run_batched(
            device, registry, &slots, hd, npp, cbits, nsg, mask_type, SLIDING_W,
            &[slot], &[sp], &q,
        );
        let reference = run_single_seq_reference(
            device, registry, &slots[slot as usize], hd, cbits, nsg, ksl, rs, mask_type, SLIDING_W, &q,
        );
        assert_bit_equal(&format!("{combo_tag} N=1 sliding-clipped"), &batched, &reference);
    }

    // ---- N=1 "sliding+wrap" (mask_type=2, ring wraps AND sliding-window
    // clipping both active simultaneously) ----
    {
        let slot = SLOT_PERM[5];
        let sp = (CAP + 15) as u32; // ksl=CAP=128 (wrap, rs=(sp+1)%CAP≠0), 128 > SLIDING_W ⇒ clip too
        let mask_type = 2u32;
        let (ksl, rs) = derive_ksl_rs(sp, mask_type, CAP);
        assert_eq!(ksl, CAP, "sliding+wrap case must reach full capacity");
        assert_ne!(rs, 0, "sliding+wrap case must actually wrap");
        assert!(ksl > SLIDING_W, "sliding+wrap case must also clip");
        let q = gen_q(&mut qrng, 1);

        let batched = run_batched(
            device, registry, &slots, hd, npp, cbits, nsg, mask_type, SLIDING_W,
            &[slot], &[sp], &q,
        );
        let reference = run_single_seq_reference(
            device, registry, &slots[slot as usize], hd, cbits, nsg, ksl, rs, mask_type, SLIDING_W, &q,
        );
        assert_bit_equal(&format!("{combo_tag} N=1 sliding+wrap"), &batched, &reference);
    }

    // ---- N=8 "sliding-mixed" (mask_type=2, shared sliding_window, PER-QUERY
    // DIFFERENT kv_seq_len so some queries clip and some don't in the SAME
    // batched dispatch, plus one query that both wraps AND clips) ----
    {
        let mask_type = 2u32;
        // sp -> ksl: 10->11(no clip) 60->61(clip) 25->26(no clip)
        //            100->101(clip)  31->32(boundary: kv_seq_len==W, NOT clipped)
        //            90->91(clip)    15->16(no clip)  (CAP+20)->CAP=128(wrap+clip)
        let seq_pos: [u32; N_SLOTS] = [10, 60, 25, 100, 31, 90, 15, CAP + 20];
        let slot_ids: Vec<u32> = SLOT_PERM.to_vec();
        let q = gen_q(&mut qrng, N_SLOTS);

        let mut saw_no_clip = false;
        let mut saw_clip = false;
        let mut saw_clip_and_wrap = false;
        for &sp in &seq_pos {
            let (ksl, rs) = derive_ksl_rs(sp, mask_type, CAP);
            if ksl > SLIDING_W { saw_clip = true; } else { saw_no_clip = true; }
            if ksl > SLIDING_W && rs != 0 { saw_clip_and_wrap = true; }
        }
        assert!(saw_no_clip && saw_clip, "N=8 sliding-mixed must contain both clip regimes");
        assert!(saw_clip_and_wrap, "N=8 sliding-mixed must contain a combined wrap+clip query");

        let batched = run_batched(
            device, registry, &slots, hd, npp, cbits, nsg, mask_type, SLIDING_W,
            &slot_ids, &seq_pos, &q,
        );

        for (i, (&slot, &sp)) in slot_ids.iter().zip(seq_pos.iter()).enumerate() {
            let (ksl, rs) = derive_ksl_rs(sp, mask_type, CAP);
            let q_row = &q[i * NH as usize * hd as usize..(i + 1) * NH as usize * hd as usize];
            let reference = run_single_seq_reference(
                device, registry, &slots[slot as usize], hd, cbits, nsg, ksl, rs, mask_type, SLIDING_W, q_row,
            );
            let batched_row = &batched[i * NH as usize * hd as usize..(i + 1) * NH as usize * hd as usize];
            assert_bit_equal(&format!("{combo_tag} N=8 sliding-mixed query={i} slot={slot}"), batched_row, &reference);
        }
    }
}

#[test]
fn tq_hb_batched_bit_parity_matrix() {
    let device = MlxDevice::new().expect("MlxDevice");
    let mut registry = KernelRegistry::new();

    // HF2Q_TQ_NWG must stay unset for this binary: compute_nwg's override
    // cache latches on first read for the process lifetime (see module doc).
    std::env::remove_var("HF2Q_TQ_NWG");

    let mut n_run = 0usize;
    for &hd in &[256u32, 512u32] {
        for &cbits in &[5u32, 6u32, 8u32] {
            for &nsg in &[1u32, 4u32] {
                run_combo(&device, &mut registry, hd, cbits, nsg);
                n_run += 1;
            }
        }
    }
    // 2 head_dims × 3 codebook widths × 2 nsg values = 12 combos, each
    // exercising 8 sub-cases (N=1 linear/wrap/no-ring/sliding-inside/
    // sliding-clipped/sliding+wrap + N=8 mixed/sliding-mixed).
    assert_eq!(n_run, 12);
    println!("tq_hb_batched_bit_parity_matrix: {n_run} combos x 8 sub-cases, all bit-exact");
}
