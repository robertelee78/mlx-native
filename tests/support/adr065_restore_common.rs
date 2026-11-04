// Shared bit-pattern helpers for the ADR-065 DeltaNet capture-restore
// qualification tests (E6 probe 4 and P4a). Included with `include!` after
// `support/adr065_e6_common.rs`.

/// One representative of every special u32 class an f32 copy must preserve.
const SPECIALS: [u32; 10] = [
    0x0000_0000, // +0
    0x8000_0000, // -0
    0x0000_0001, // smallest subnormal
    0x807f_ffff, // largest negative subnormal
    0x7f80_0000, // +inf
    0xff80_0000, // -inf
    0x7fc0_0000, // canonical quiet NaN
    0x7fc1_2345, // quiet NaN payload
    0x7f80_0001, // signaling NaN payload
    0xffa5_a5a5, // negative signaling NaN payload
];

/// Random u32 payload with a special class at every seventh element.
fn payload(rng: &mut E6Rng, len: usize) -> Vec<u32> {
    (0..len)
        .map(|index| {
            if index % 7 == 3 {
                SPECIALS[(rng.next_u64() % SPECIALS.len() as u64) as usize]
            } else {
                rng.next_u64() as u32
            }
        })
        .collect()
}

fn is_nan_bits(bits: u32) -> bool {
    (bits & 0x7f80_0000) == 0x7f80_0000 && (bits & 0x007f_ffff) != 0
}

fn bits_buffer(device: &MlxDevice, bits: &[u32]) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(bits.len() * 4, DType::F32, vec![bits.len()])
        .expect("alloc");
    buffer
        .as_mut_slice::<u32>()
        .expect("bits slice")
        .copy_from_slice(bits);
    buffer
}

fn read_bits(buffer: &MlxBuffer) -> Vec<u32> {
    buffer.as_slice::<u32>().expect("read bits").to_vec()
}
