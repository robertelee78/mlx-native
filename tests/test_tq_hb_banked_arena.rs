use mlx_native::ops::flash_attn_vec_tq_hb::{
    self, flash_attn_vec_tq_hb_batched_banked, FlashAttnVecTqHbParams,
};
use mlx_native::ops::hadamard_quantize_kv::{
    dispatch_hadamard_quantize_kv_hb, dispatch_hadamard_quantize_kv_hb_banked,
};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

const D: u32 = 256;
const NH: u32 = 4;
const NKV: u32 = 2;
const CAPS: [u32; 2] = [16, 64];
const BASES: [u32; 2] = [3, 40];
const SEQ_POS: [u32; 2] = [20, 51];
const ARENA_TOKEN_ROWS: u32 = 173;
const PACKED_CANARY: u8 = 0xcd;
const NORM_CANARY_BITS: u32 = 0x4b41_2345;

fn write_u8(device: &MlxDevice, values: &[u8]) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(values.len(), DType::U8, vec![values.len()])
        .expect("allocate u8 buffer");
    buffer.as_mut_slice::<u8>().unwrap().copy_from_slice(values);
    buffer
}

fn write_u32(device: &MlxDevice, values: &[u32]) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(values.len() * 4, DType::U32, vec![values.len()])
        .expect("allocate u32 buffer");
    buffer
        .as_mut_slice::<u32>()
        .unwrap()
        .copy_from_slice(values);
    buffer
}

fn write_f32(device: &MlxDevice, values: &[f32]) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(values.len() * 4, DType::F32, vec![values.len()])
        .expect("allocate f32 buffer");
    buffer
        .as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(values);
    buffer
}

fn slot_rows(slot: usize) -> std::ops::Range<usize> {
    let start = BASES[slot] as usize;
    start..start + NKV as usize * CAPS[slot] as usize
}

fn fill_arena(seed: u32) -> (Vec<u8>, Vec<f32>) {
    let mut packed = vec![PACKED_CANARY; ARENA_TOKEN_ROWS as usize * D as usize];
    let mut norms = vec![f32::from_bits(NORM_CANARY_BITS); ARENA_TOKEN_ROWS as usize];
    for slot in 0..2 {
        let cap = CAPS[slot] as usize;
        let base = BASES[slot] as usize;
        for head in 0..NKV as usize {
            for pos in 0..cap {
                let row = base + head * cap + pos;
                norms[row] = 0.25 + ((seed as usize + row * 13) % 97) as f32 / 31.0;
                for coord in 0..D as usize {
                    packed[row * D as usize + coord] =
                        (seed as usize + slot * 71 + head * 29 + pos * 17 + coord * 3) as u8;
                }
            }
        }
    }
    (packed, norms)
}

fn assert_arena_guards(packed: &[u8], norms: &[f32]) {
    for row in 0..ARENA_TOKEN_ROWS as usize {
        let owned = slot_rows(0).contains(&row) || slot_rows(1).contains(&row);
        if !owned {
            assert!(
                packed[row * D as usize..(row + 1) * D as usize]
                    .iter()
                    .all(|&byte| byte == PACKED_CANARY),
                "packed guard row {row} was modified"
            );
            assert_eq!(
                norms[row].to_bits(),
                NORM_CANARY_BITS,
                "norm guard row {row} was modified"
            );
        }
    }
}

fn derive_len_start(seq_pos: u32, cap: u32) -> (u32, u32) {
    let len = (seq_pos + 1).min(cap);
    let start = if len == cap { (seq_pos + 1) % cap } else { 0 };
    (len, start)
}

fn params(cap: u32, seq_len: u32, ring_start: u32) -> FlashAttnVecTqHbParams {
    FlashAttnVecTqHbParams {
        num_heads: NH,
        num_kv_heads: NKV,
        head_dim: D,
        kv_seq_len: seq_len,
        kv_capacity: cap,
        scale: 1.0 / (D as f32).sqrt(),
        mask_type: 2,
        sliding_window: cap,
        softcap: 0.0,
        ring_start,
        scale_factor_d512: 1.0,
        codebook_bits: 8,
        fuse_fwht_pre: 1,
        nsg: 1,
    }
}

fn scalar_attention(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    q: &[f32],
    packed_k: &MlxBuffer,
    norms_k: &MlxBuffer,
    packed_v: &MlxBuffer,
    norms_v: &MlxBuffer,
    slot: usize,
) -> Vec<f32> {
    let cap = CAPS[slot] as usize;
    let base = BASES[slot] as usize;
    let packed_len = NKV as usize * cap * D as usize;
    let norm_len = NKV as usize * cap;
    let packed_offset = base * D as usize;
    let k_view = packed_k.slice_view(packed_offset as u64, packed_len);
    let kn_view = norms_k.slice_view((base * 4) as u64, norm_len);
    let v_view = packed_v.slice_view(packed_offset as u64, packed_len);
    let vn_view = norms_v.slice_view((base * 4) as u64, norm_len);
    let q_buffer = write_f32(device, q);
    let output = device
        .alloc_buffer((NH * D * 4) as usize, DType::F32, vec![(NH * D) as usize])
        .unwrap();
    let tmp_bytes = flash_attn_vec_tq_hb::tmp_buffer_bytes(NH, D);
    let tmp = device
        .alloc_buffer(tmp_bytes, DType::F32, vec![tmp_bytes / 4])
        .unwrap();
    let (seq_len, ring_start) = derive_len_start(SEQ_POS[slot], CAPS[slot]);
    let mut encoder = device.command_encoder().unwrap();
    flash_attn_vec_tq_hb::flash_attn_vec_tq_hb_with_fused_undo(
        &mut encoder,
        registry,
        device,
        &q_buffer,
        &k_view,
        &kn_view,
        &v_view,
        &vn_view,
        &output,
        &tmp,
        &params(CAPS[slot], seq_len, ring_start),
    )
    .unwrap();
    encoder.commit_and_wait().unwrap();
    output.as_slice::<f32>().unwrap().to_vec()
}

#[test]
fn unequal_capacity_banked_writer_and_attention_match_scalar_without_leakage() {
    std::env::remove_var("HF2Q_TQ_NWG");
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let (packed_k_initial, norms_k_initial) = fill_arena(11);
    let (packed_v_initial, norms_v_initial) = fill_arena(173);
    let mut packed_k = write_u8(&device, &packed_k_initial);
    let mut norms_k = write_f32(&device, &norms_k_initial);
    let mut packed_v = write_u8(&device, &packed_v_initial);
    let mut norms_v = write_f32(&device, &norms_v_initial);
    let bases = write_u32(&device, &BASES);
    let capacities = write_u32(&device, &CAPS);
    let seq_pos = write_u32(&device, &SEQ_POS);

    let src: Vec<f32> = (0..2 * NKV as usize * D as usize)
        .map(|index| ((index * 37 % 211) as f32 - 105.0) / 41.0)
        .collect();
    let src_buffer = write_f32(&device, &src);
    let mut writer = device.command_encoder().unwrap();
    dispatch_hadamard_quantize_kv_hb_banked(
        &mut writer,
        &mut registry,
        device.metal_device(),
        &src_buffer,
        &packed_k,
        &norms_k,
        &bases,
        &capacities,
        &seq_pos,
        2,
        NKV,
        D,
        ARENA_TOKEN_ROWS,
        true,
        1.0,
        8,
    )
    .unwrap();
    writer.commit_and_wait().unwrap();

    let packed_after = packed_k.as_mut_slice::<u8>().unwrap().to_vec();
    let norms_after = norms_k.as_mut_slice::<f32>().unwrap().to_vec();
    assert_arena_guards(&packed_after, &norms_after);

    for slot in 0..2 {
        let cap = CAPS[slot] as usize;
        let packed_len = NKV as usize * cap * D as usize;
        let norm_len = NKV as usize * cap;
        let src_row =
            &src[slot * NKV as usize * D as usize..(slot + 1) * NKV as usize * D as usize];
        let src_scalar = write_f32(&device, src_row);
        let mut packed_scalar = write_u8(
            &device,
            &packed_k_initial
                [BASES[slot] as usize * D as usize..BASES[slot] as usize * D as usize + packed_len],
        );
        let mut norms_scalar = write_f32(
            &device,
            &norms_k_initial[BASES[slot] as usize..BASES[slot] as usize + norm_len],
        );
        let mut encoder = device.command_encoder().unwrap();
        dispatch_hadamard_quantize_kv_hb(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &src_scalar,
            &packed_scalar,
            &norms_scalar,
            NKV,
            D,
            CAPS[slot],
            SEQ_POS[slot],
            true,
            1.0,
            8,
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();
        assert_eq!(
            &packed_after
                [BASES[slot] as usize * D as usize..BASES[slot] as usize * D as usize + packed_len],
            packed_scalar.as_mut_slice::<u8>().unwrap(),
            "slot {slot} packed writer mismatch"
        );
        for (got, expected) in norms_after[BASES[slot] as usize..BASES[slot] as usize + norm_len]
            .iter()
            .zip(norms_scalar.as_mut_slice::<f32>().unwrap())
        {
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "slot {slot} norm mismatch"
            );
        }
    }

    let q: Vec<f32> = (0..2 * NH as usize * D as usize)
        .map(|index| ((index * 19 % 157) as f32 - 78.0) / 29.0)
        .collect();
    let q_buffer = write_f32(&device, &q);
    let output = device
        .alloc_buffer(
            (2 * NH * D * 4) as usize,
            DType::F32,
            vec![(2 * NH * D) as usize],
        )
        .unwrap();
    let tmp_bytes = flash_attn_vec_tq_hb::tmp_buffer_bytes(2 * NH, D);
    let tmp = device
        .alloc_buffer(tmp_bytes, DType::F32, vec![tmp_bytes / 4])
        .unwrap();
    let mut batch_params = params(64, 52, 0);
    batch_params.sliding_window = 64;
    let mut encoder = device.command_encoder().unwrap();
    flash_attn_vec_tq_hb_batched_banked(
        &mut encoder,
        &mut registry,
        &device,
        2,
        &q_buffer,
        &packed_k,
        &norms_k,
        &packed_v,
        &norms_v,
        &output,
        &tmp,
        &bases,
        &capacities,
        &seq_pos,
        ARENA_TOKEN_ROWS,
        &batch_params,
    )
    .unwrap();
    encoder.commit_and_wait().unwrap();
    let batched = output.as_slice::<f32>().unwrap();
    for slot in 0..2 {
        let q_row = &q[slot * (NH * D) as usize..(slot + 1) * (NH * D) as usize];
        let scalar = scalar_attention(
            &device,
            &mut registry,
            q_row,
            &packed_k,
            &norms_k,
            &packed_v,
            &norms_v,
            slot,
        );
        for (index, (&got, &expected)) in batched
            [slot * (NH * D) as usize..(slot + 1) * (NH * D) as usize]
            .iter()
            .zip(&scalar)
            .enumerate()
        {
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "slot {slot} attention output {index} differs"
            );
        }
    }

    let slot0_before = batched[..(NH * D) as usize].to_vec();
    for row in slot_rows(1) {
        packed_v.as_mut_slice::<u8>().unwrap()[row * D as usize..(row + 1) * D as usize].fill(0);
        norms_v.as_mut_slice::<f32>().unwrap()[row] = 19.0;
    }
    let output_after = device
        .alloc_buffer(
            (2 * NH * D * 4) as usize,
            DType::F32,
            vec![(2 * NH * D) as usize],
        )
        .unwrap();
    let tmp_after = device
        .alloc_buffer(tmp_bytes, DType::F32, vec![tmp_bytes / 4])
        .unwrap();
    let mut encoder = device.command_encoder().unwrap();
    flash_attn_vec_tq_hb_batched_banked(
        &mut encoder,
        &mut registry,
        &device,
        2,
        &q_buffer,
        &packed_k,
        &norms_k,
        &packed_v,
        &norms_v,
        &output_after,
        &tmp_after,
        &bases,
        &capacities,
        &seq_pos,
        ARENA_TOKEN_ROWS,
        &batch_params,
    )
    .unwrap();
    encoder.commit_and_wait().unwrap();
    let after = output_after.as_slice::<f32>().unwrap();
    assert!(slot0_before
        .iter()
        .zip(after)
        .all(|(&a, &b)| a.to_bits() == b.to_bits()));
    assert!(batched[(NH * D) as usize..]
        .iter()
        .zip(&after[(NH * D) as usize..])
        .any(|(&a, &b)| a.to_bits() != b.to_bits()));
}

#[test]
fn invalid_banked_regions_fail_closed_without_touching_arena() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let (packed_initial, norms_initial) = fill_arena(29);
    let mut packed = write_u8(&device, &packed_initial);
    let mut norms = write_f32(&device, &norms_initial);
    let src = write_f32(&device, &vec![0.25; (NKV * D) as usize]);
    let invalid_base = write_u32(&device, &[ARENA_TOKEN_ROWS - 1]);
    let capacity = write_u32(&device, &[64]);
    let seq_pos = write_u32(&device, &[0]);
    let mut encoder = device.command_encoder().unwrap();
    dispatch_hadamard_quantize_kv_hb_banked(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &src,
        &packed,
        &norms,
        &invalid_base,
        &capacity,
        &seq_pos,
        1,
        NKV,
        D,
        ARENA_TOKEN_ROWS,
        false,
        1.0,
        8,
    )
    .unwrap();
    encoder.commit_and_wait().unwrap();
    assert_eq!(packed.as_mut_slice::<u8>().unwrap(), &packed_initial);
    assert!(norms
        .as_mut_slice::<f32>()
        .unwrap()
        .iter()
        .zip(&norms_initial)
        .all(|(a, b)| a.to_bits() == b.to_bits()));

    let q = write_f32(&device, &vec![0.125; (NH * D) as usize]);
    let mut output = write_f32(&device, &vec![17.0; (NH * D) as usize]);
    let tmp_bytes = flash_attn_vec_tq_hb::tmp_buffer_bytes(NH, D);
    let tmp = device
        .alloc_buffer(tmp_bytes, DType::F32, vec![tmp_bytes / 4])
        .unwrap();
    let mut attention = device.command_encoder().unwrap();
    flash_attn_vec_tq_hb_batched_banked(
        &mut attention,
        &mut registry,
        &device,
        1,
        &q,
        &packed,
        &norms,
        &packed,
        &norms,
        &output,
        &tmp,
        &invalid_base,
        &capacity,
        &seq_pos,
        ARENA_TOKEN_ROWS,
        &params(64, 1, 0),
    )
    .unwrap();
    attention.commit_and_wait().unwrap();
    assert!(output
        .as_mut_slice::<f32>()
        .unwrap()
        .iter()
        .all(|value| *value == 0.0));
}
