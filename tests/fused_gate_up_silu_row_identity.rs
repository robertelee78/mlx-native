//! Width-invariance gate for every admitted dense fused gate/up/SiLU codec.
//!
//! A multi-row fused dispatch must produce the same F32 bits per physical row
//! as independent `m=1` fused dispatches over those distinct input rows.

#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used, clippy::panic)]

use mlx_native::ops::fused_gate_up_silu_iq4_nl::{
    dispatch_fused_gate_up_silu_iq4_nl, FusedGateUpSiluIq4NlArgs,
};
use mlx_native::ops::fused_gate_up_silu_q4_K::{
    dispatch_fused_gate_up_silu_q4_K, FusedGateUpSiluQ4_KArgs,
};
use mlx_native::ops::fused_gate_up_silu_q6_K::{
    dispatch_fused_gate_up_silu_q6_K, FusedGateUpSiluQ6_KArgs,
};
use mlx_native::ops::fused_gate_up_silu_q8_0::{
    dispatch_fused_gate_up_silu_q8_0, FusedGateUpSiluQ8_0Args,
};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

const N: usize = 64;
const K: usize = 512;

#[derive(Clone, Copy, Debug)]
enum Codec {
    Q8_0,
    Q4K,
    Q6K,
    Iq4Nl,
}

impl Codec {
    const ALL: [Self; 4] = [Self::Q8_0, Self::Q4K, Self::Q6K, Self::Iq4Nl];

    fn block_shape(self) -> (usize, usize) {
        match self {
            Self::Q8_0 => (32, 34),
            Self::Q4K => (256, 144),
            Self::Q6K => (256, 210),
            Self::Iq4Nl => (32, 18),
        }
    }
}

fn next_u64(state: &mut u64) -> u64 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    state.wrapping_mul(0x2545_f491_4f6c_dd1d)
}

fn valid_weight_bytes(codec: Codec, seed: u64) -> Vec<u8> {
    let (qk, block_bytes) = codec.block_shape();
    let block_count = N * (K / qk);
    let mut state = seed;
    let mut bytes = vec![0_u8; block_count * block_bytes];
    for (block_index, block) in bytes.chunks_exact_mut(block_bytes).enumerate() {
        for byte in block.iter_mut() {
            *byte = next_u64(&mut state) as u8;
        }
        let d = half::f16::from_f32(0.001 + (block_index % 31) as f32 * 0.000_031_25)
            .to_bits()
            .to_le_bytes();
        match codec {
            Codec::Q8_0 | Codec::Iq4Nl => block[..2].copy_from_slice(&d),
            Codec::Q4K => {
                block[..2].copy_from_slice(&d);
                let dmin = half::f16::from_f32(0.000_5 + (block_index % 17) as f32 * 0.000_015_625)
                    .to_bits()
                    .to_le_bytes();
                block[2..4].copy_from_slice(&dmin);
            }
            Codec::Q6K => block[208..210].copy_from_slice(&d),
        }
    }
    bytes
}

fn distinct_inputs(m: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut values = Vec::with_capacity(m * K);
    for row in 0..m {
        for column in 0..K {
            let unit = (next_u64(&mut state) >> 40) as f32 / (1_u32 << 24) as f32;
            values.push(unit - 0.5 + row as f32 * 0.031_25 + column as f32 * 0.000_000_1);
        }
    }
    for row in 1..m {
        assert_ne!(&values[..K], &values[row * K..(row + 1) * K]);
    }
    values
}

#[allow(clippy::too_many_arguments)]
fn dispatch(
    codec: Codec,
    encoder: &mut mlx_native::CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    gate: &MlxBuffer,
    up: &MlxBuffer,
    input: &MlxBuffer,
    output: &MlxBuffer,
    m: usize,
) {
    match codec {
        Codec::Q8_0 => dispatch_fused_gate_up_silu_q8_0(
            encoder,
            registry,
            device,
            gate,
            up,
            input,
            output,
            FusedGateUpSiluQ8_0Args {
                m: m as u32,
                intermediate_size: N as u32,
                hidden_size: K as u32,
            },
        ),
        Codec::Q4K => dispatch_fused_gate_up_silu_q4_K(
            encoder,
            registry,
            device,
            gate,
            up,
            input,
            output,
            FusedGateUpSiluQ4_KArgs {
                m: m as u32,
                intermediate_size: N as u32,
                hidden_size: K as u32,
            },
        ),
        Codec::Q6K => dispatch_fused_gate_up_silu_q6_K(
            encoder,
            registry,
            device,
            gate,
            up,
            input,
            output,
            FusedGateUpSiluQ6_KArgs {
                m: m as u32,
                intermediate_size: N as u32,
                hidden_size: K as u32,
            },
        ),
        Codec::Iq4Nl => dispatch_fused_gate_up_silu_iq4_nl(
            encoder,
            registry,
            device,
            gate,
            up,
            input,
            output,
            FusedGateUpSiluIq4NlArgs {
                m: m as u32,
                intermediate_size: N as u32,
                hidden_size: K as u32,
            },
        ),
    }
    .expect("fused gate/up/SiLU dispatch");
}

fn assert_codec_width(codec: Codec, m: usize) {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let gate_bytes = valid_weight_bytes(codec, 0x6a74_6500_0000_0000 ^ codec as u64);
    let up_bytes = valid_weight_bytes(codec, 0x7570_0000_0000_0000 ^ codec as u64);
    let input_values = distinct_inputs(m, 0x696e_7075_7400_0000 ^ m as u64 ^ codec as u64);

    let mut gate = device
        .alloc_buffer(gate_bytes.len(), DType::U8, vec![gate_bytes.len()])
        .expect("gate weight");
    gate.as_mut_slice::<u8>()
        .expect("gate bytes")
        .copy_from_slice(&gate_bytes);
    let mut up = device
        .alloc_buffer(up_bytes.len(), DType::U8, vec![up_bytes.len()])
        .expect("up weight");
    up.as_mut_slice::<u8>()
        .expect("up bytes")
        .copy_from_slice(&up_bytes);

    let mut expected = Vec::with_capacity(m * N);
    for row in 0..m {
        let mut input = device
            .alloc_buffer(K * 4, DType::F32, vec![1, K])
            .expect("scalar input");
        input
            .as_mut_slice::<f32>()
            .expect("scalar input values")
            .copy_from_slice(&input_values[row * K..(row + 1) * K]);
        let output = device
            .alloc_buffer(N * 4, DType::F32, vec![1, N])
            .expect("scalar output");
        let mut encoder = device.command_encoder().expect("scalar encoder");
        dispatch(
            codec,
            &mut encoder,
            &mut registry,
            &device,
            &gate,
            &up,
            &input,
            &output,
            1,
        );
        encoder.commit_and_wait().expect("scalar GPU execution");
        expected.extend_from_slice(output.as_slice::<f32>().expect("scalar result"));
    }

    let mut input = device
        .alloc_buffer(m * K * 4, DType::F32, vec![m, K])
        .expect("width input");
    input
        .as_mut_slice::<f32>()
        .expect("width input values")
        .copy_from_slice(&input_values);
    let output = device
        .alloc_buffer(m * N * 4, DType::F32, vec![m, N])
        .expect("width output");
    let mut encoder = device.command_encoder().expect("width encoder");
    dispatch(
        codec,
        &mut encoder,
        &mut registry,
        &device,
        &gate,
        &up,
        &input,
        &output,
        m,
    );
    encoder.commit_and_wait().expect("width GPU execution");

    let actual = output.as_slice::<f32>().expect("width result");
    for (index, (&want, &got)) in expected.iter().zip(actual).enumerate() {
        assert!(
            want.is_finite() && got.is_finite(),
            "fused row-identity produced non-finite output: codec={codec:?}, m={m}, row={}, column={}, scalar={want:?}, width={got:?}",
            index / N,
            index % N,
        );
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "fused row-identity mismatch: codec={codec:?}, m={m}, row={}, column={}, scalar={want:?}, width={got:?}",
            index / N,
            index % N,
        );
    }
}

#[test]
fn every_admitted_fused_codec_is_row_bit_identical_at_widths_two_through_eight() {
    for codec in Codec::ALL {
        for m in 2..=8 {
            assert_codec_width(codec, m);
        }
    }
}
