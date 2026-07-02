//! ADR-040 M-SPEED-LC — bit-parity test for the BATCHED TQ-HB decode kernel,
//! FORCED NWG=1 branch.
//!
//! Companion to `test_flash_attn_vec_tq_hb_batched_parity.rs` (which covers
//! the default nwg∈{16,32} / fused-reduce-undo branch). This file is a
//! SEPARATE process specifically so it can force `HF2Q_TQ_NWG=1` as the very
//! first action: `compute_nwg`'s override is cached in a function-local
//! `AtomicI32` that latches on first read for the life of the process, so
//! nwg=1 and nwg>1 cannot both be safely exercised in one binary.
//!
//! At NWG=1, `flash_attn_vec_tq_hb_batched` takes its `else` branch — the
//! SDPA kernel writes the final ROTATED-domain output directly to `output`
//! (skipping the tmp/reduce round-trip), and the dispatcher applies
//! `fwht_sign_undo_f32` across all `n_q * num_heads` rows in one call. This
//! test proves that branch is bit-identical to the scalar
//! `flash_attn_vec_tq_hb_with_fused_undo` reference (whose own NWG=1 branch
//! does the same single-row `fwht_sign_undo_f32` per query).
//!
//! Smaller matrix than the default-nwg file (one D=256 and one D=512 point)
//! since the NWG axis itself is what's under test here, not the full
//! cbits/nsg cross-product (already covered by the sibling file). Each combo
//! runs N=1 linear/wrap, N=8 mixed, and N=8 "sliding-mixed" (mask_type=2,
//! shared sliding_window>0, per-query kv_seq_len straddling the window so
//! some queries clip and some don't, plus one combined wrap+clip query) —
//! covering the sliding-window branch (`window_start_logical`) at NWG=1 too.

use mlx_native::ops::flash_attn_vec_tq_hb::{self, FlashAttnVecTqHbParams};
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
    fn next_u8(&mut self) -> u8 {
        (self.next_u64() >> 32) as u8
    }
    fn next_norm(&mut self) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32;
        0.25 + u * 1.75
    }
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

const NH: u32 = 8;
const NKV: u32 = 2;
const CAP: u32 = 64; // small — kv_seq_len irrelevant to nwg since it's forced
const N_SLOTS: usize = 8;
const SLOT_PERM: [u32; N_SLOTS] = [5, 1, 6, 2, 7, 3, 0, 4];

fn derive_ksl_rs(sp: u32, mask_type: u32, cap: u32) -> (u32, u32) {
    let is_ring = mask_type == 2;
    let ksl = if is_ring { (sp + 1).min(cap) } else { sp + 1 };
    let rs = if is_ring && ksl >= cap { (sp + 1) % cap } else { 0 };
    (ksl, rs)
}

struct SlotData {
    k_packed: Vec<u8>,
    k_norms: Vec<f32>,
    v_packed: Vec<u8>,
    v_norms: Vec<f32>,
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
    q_data: &[f32],
) -> Vec<f32> {
    let n_q = slot_ids.len() as u32;
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
        ring_start: 0,
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
    q_row: &[f32],
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

fn run_combo(device: &MlxDevice, registry: &mut KernelRegistry, hd: u32, cbits: u32, nsg: u32) {
    let npp = if hd == 512 { 2 } else { 1 };
    let combo_tag = format!("NWG1_dk{hd}_cbits{cbits}_nsg{nsg}");

    let slots: Vec<SlotData> = (0..N_SLOTS)
        .map(|i| gen_slot(0x1234_5678_0000 ^ (i as u64) ^ ((hd as u64) << 32) ^ (cbits as u64), NKV, CAP, hd, npp))
        .collect();

    let mut qrng = Xor::new(0xFEED ^ (hd as u64) ^ (cbits as u64) << 8 ^ (nsg as u64) << 16);
    let gen_q = |rng: &mut Xor, n_q: usize| -> Vec<f32> {
        (0..(n_q * NH as usize * hd as usize)).map(|_| rng.next_signed()).collect()
    };

    // N=1 linear + wrap.
    for (label, sp, slot) in [("linear", 20u32, SLOT_PERM[0]), ("wrap", CAP + 7, SLOT_PERM[1])] {
        let mask_type = 2u32;
        let (ksl, rs) = derive_ksl_rs(sp, mask_type, CAP);
        let q = gen_q(&mut qrng, 1);
        let batched = run_batched(
            device, registry, &slots, hd, npp as u32, cbits, nsg, mask_type, 0,
            &[slot], &[sp], &q,
        );
        let reference = run_single_seq_reference(
            device, registry, &slots[slot as usize], hd, cbits, nsg, ksl, rs, mask_type, 0, &q,
        );
        assert_bit_equal(&format!("{combo_tag} N=1 {label}"), &batched, &reference);
    }

    // N=8 mixed.
    {
        let mask_type = 2u32;
        let seq_pos: [u32; N_SLOTS] = [10, CAP + 3, 20, CAP + 15, 30, CAP + 1, 45, CAP + 25];
        let slot_ids: Vec<u32> = SLOT_PERM.to_vec();
        let q = gen_q(&mut qrng, N_SLOTS);

        let batched = run_batched(
            device, registry, &slots, hd, npp as u32, cbits, nsg, mask_type, 0,
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

    // ADR-040 codex review follow-up: sliding-window branch (mask_type=2,
    // sliding_window>0) was previously untested at NWG=1 too. Covers "inside"
    // (no clip), "clipped", and "clipped+wrap" in one N=8 dispatch, run at
    // both this file's D=256 and D=512 combos.
    {
        const SLIDING_W: u32 = 16; // small — CAP=64 here (see module consts)
        let mask_type = 2u32;
        // sp -> ksl: 5->6(no clip) 30->31(clip) 10->11(no clip)
        //            50->51(clip)  15->16(boundary: kv_seq_len==W, not clipped)
        //            40->41(clip)  8->9(no clip)  (CAP+9)->CAP=64(wrap+clip)
        let seq_pos: [u32; N_SLOTS] = [5, 30, 10, 50, 15, 40, 8, CAP + 9];
        let slot_ids: Vec<u32> = SLOT_PERM.to_vec();
        let q = gen_q(&mut qrng, N_SLOTS);

        let mut saw_no_clip = false;
        let mut saw_clip_and_wrap = false;
        for &sp in &seq_pos {
            let (ksl, rs) = derive_ksl_rs(sp, mask_type, CAP);
            if ksl <= SLIDING_W { saw_no_clip = true; }
            if ksl > SLIDING_W && rs != 0 { saw_clip_and_wrap = true; }
        }
        assert!(saw_no_clip, "sliding sub-case must contain an unclipped query");
        assert!(saw_clip_and_wrap, "sliding sub-case must contain a combined wrap+clip query");

        let batched = run_batched(
            device, registry, &slots, hd, npp as u32, cbits, nsg, mask_type, SLIDING_W,
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
fn tq_hb_batched_bit_parity_nwg1() {
    // MUST be the first thing this process does that could reach
    // `compute_nwg` — its override cache latches for the process lifetime.
    std::env::set_var("HF2Q_TQ_NWG", "1");

    let device = MlxDevice::new().expect("MlxDevice");
    let mut registry = KernelRegistry::new();

    run_combo(&device, &mut registry, 256, 8, 1);
    run_combo(&device, &mut registry, 512, 5, 4);

    println!("tq_hb_batched_bit_parity_nwg1: 2 combos x (N=1 linear/wrap + N=8 mixed + N=8 sliding-mixed), all bit-exact, NWG=1 forced");
}
