//! ADR-066 N-IQ2XXS: IQ2_XXS (GGML type 16) registration and bit-exact CPU
//! dequant against the peer's `dequantize_row_iq2_xxs`.
//!
//! `tests/fixtures/iq2_xxs_dequant_v1.bin` was produced by
//! `tests/fixtures/gen_iq2_xxs_dequant.c`, which links only the pinned peer
//! checkout's dequantizer. The test reads the checked-in fixture; no peer
//! code runs here.

use mlx_native::gguf::{
    test_only_compute_byte_len, test_only_dequantize, test_only_ggml_type_from_u32,
    test_only_raw_tensor_dtype,
};
use mlx_native::quant::iq_tables::{IQ2XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
use mlx_native::{DType, GgmlType};

const QK_K: usize = 256;
const BLOCK_BYTES: usize = 66;

/// Peer checkout pinned by the ADR-066 IQ2_XXS design (§3.2).
const PEER_COMMIT: &str = "e86f5fd1e9f584b0fcec71d7da9a92de44e8bcbf";
const GGML_COMMON_H_SHA256: &str =
    "0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c";
const GGML_QUANTS_C_SHA256: &str =
    "5574a2dccf7c07e75b143733e04a5412d3d8c819e7945f5217a7b83a2b2ff8ab";

struct Fixture {
    header: Vec<(String, String)>,
    blocks: Vec<u8>,
    expected: Vec<u32>,
}

impl Fixture {
    fn header_value(&self, key: &str) -> &str {
        self.header
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
            .unwrap_or_else(|| panic!("fixture header lacks {key}"))
    }

    fn block_count(&self) -> usize {
        self.blocks.len() / BLOCK_BYTES
    }
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("u32"))
}

fn load_fixture() -> Fixture {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/iq2_xxs_dequant_v1.bin");
    let bytes = std::fs::read(&path).expect("read IQ2_XXS fixture");
    assert_eq!(&bytes[..8], b"IQ2XXSD1", "fixture magic");
    let header_len = read_u32(&bytes, 8) as usize;
    let header_text = std::str::from_utf8(&bytes[12..12 + header_len]).expect("UTF-8 header");
    let header = header_text
        .lines()
        .map(|line| {
            let (key, value) = line.split_once('=').expect("key=value");
            (key.to_string(), value.to_string())
        })
        .collect();
    let mut offset = 12 + header_len;
    let count = read_u32(&bytes, offset) as usize;
    offset += 4;
    let blocks = bytes[offset..offset + count * BLOCK_BYTES].to_vec();
    offset += count * BLOCK_BYTES;
    let expected: Vec<u32> = bytes[offset..]
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().expect("u32")))
        .collect();
    assert_eq!(expected.len(), count * QK_K, "fixture value count");
    Fixture {
        header,
        blocks,
        expected,
    }
}

/// Independent hand port of the C loop (`ggml-quants.c`
/// `dequantize_row_iq2_xxs`), reading the block through `aux32[2]` and
/// `aux8` exactly as the C code does.
fn hand_port(blocks: &[u8]) -> Vec<f32> {
    let mut y = Vec::with_capacity(blocks.len() / BLOCK_BYTES * QK_K);
    for block in blocks.chunks_exact(BLOCK_BYTES) {
        let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        for ib32 in 0..QK_K / 32 {
            let base = 2 + 8 * ib32;
            let aux32 = [
                u32::from_le_bytes(block[base..base + 4].try_into().unwrap()),
                u32::from_le_bytes(block[base + 4..base + 8].try_into().unwrap()),
            ];
            let aux8 = aux32[0].to_le_bytes();
            let db = d * (0.5f32 + (aux32[1] >> 28) as f32) * 0.25f32;
            for l in 0..4 {
                let grid = IQ2XXS_GRID[aux8[l] as usize].to_le_bytes();
                let signs = KSIGNS_IQ2XS[((aux32[1] >> (7 * l)) & 127) as usize];
                for j in 0..8 {
                    let factor = if signs & KMASK_IQ2XS[j] != 0 {
                        -1.0f32
                    } else {
                        1.0f32
                    };
                    y.push(db * grid[j] as f32 * factor);
                }
            }
        }
    }
    y
}

fn assert_matches_peer(label: &str, actual: &[f32], expected: &[u32]) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    for (index, (&got, &want_bits)) in actual.iter().zip(expected).enumerate() {
        let want = f32::from_bits(want_bits);
        if want.is_nan() {
            // The NaN sign bit depends on whether a compiler lowers the ±1
            // factor to a negation or a multiply; compare NaN by class.
            assert!(
                got.is_nan(),
                "{label}: block {} value {}: got {got:?}, want NaN",
                index / QK_K,
                index % QK_K
            );
        } else {
            assert_eq!(
                got.to_bits(),
                want_bits,
                "{label}: block {} value {}: got {got:?} ({:#010x}), want {want:?} ({want_bits:#010x})",
                index / QK_K,
                index % QK_K,
                got.to_bits()
            );
        }
    }
}

#[test]
fn fixture_records_the_pinned_peer_sources() {
    let fixture = load_fixture();
    assert_eq!(fixture.header_value("peer_commit"), PEER_COMMIT);
    assert_eq!(
        fixture.header_value("ggml_common_h_sha256"),
        GGML_COMMON_H_SHA256
    );
    assert_eq!(
        fixture.header_value("ggml_quants_c_sha256"),
        GGML_QUANTS_C_SHA256
    );
    assert_eq!(fixture.header_value("random_blocks"), "4096");
    assert_eq!(
        fixture.header_value("blocks").parse::<usize>().unwrap(),
        fixture.block_count()
    );
}

#[test]
fn fixture_covers_scales_indices_and_edge_scales() {
    let fixture = load_fixture();
    let mut scales = [false; 16];
    let mut grid_at_l = [[false; 256]; 4];
    let mut signs_at_l = [[false; 128]; 4];
    let mut scale_bits = std::collections::BTreeSet::new();
    for block in fixture.blocks.chunks_exact(BLOCK_BYTES) {
        scale_bits.insert(u16::from_le_bytes([block[0], block[1]]));
        for ib32 in 0..8 {
            let base = 2 + 8 * ib32;
            let a0 = u32::from_le_bytes(block[base..base + 4].try_into().unwrap());
            let a1 = u32::from_le_bytes(block[base + 4..base + 8].try_into().unwrap());
            scales[(a1 >> 28) as usize] = true;
            for l in 0..4 {
                grid_at_l[l][((a0 >> (8 * l)) & 255) as usize] = true;
                signs_at_l[l][((a1 >> (7 * l)) & 127) as usize] = true;
            }
        }
    }
    assert!(scales.iter().all(|&seen| seen), "every scale nibble");
    assert!(
        grid_at_l.iter().flatten().all(|&seen| seen),
        "every grid index at every l"
    );
    assert!(
        signs_at_l.iter().flatten().all(|&seen| seen),
        "every sign index at every l"
    );
    // The generator's 12 edge scales: ±0, ±smallest subnormal, 1.0,
    // ±largest finite, ±inf and three NaN encodings.
    for edge in [
        0x0000u16, 0x8000, 0x0001, 0x8001, 0x3c00, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7e00, 0xfe00,
        0x7c01,
    ] {
        assert!(scale_bits.contains(&edge), "edge scale {edge:#06x}");
    }
    let values: Vec<f32> = fixture
        .expected
        .iter()
        .map(|&bits| f32::from_bits(bits))
        .collect();
    assert!(values.iter().any(|v| v.is_nan()));
    assert!(values.iter().any(|v| v.is_infinite() && *v > 0.0));
    assert!(values.iter().any(|v| v.is_infinite() && *v < 0.0));
    assert!(
        values.iter().any(|v| v.to_bits() == 0x8000_0000),
        "negative zero"
    );
    assert!(values
        .iter()
        .any(|v| v.is_subnormal() || (v.abs() > 0.0 && v.abs() < 1e-6)));
}

#[test]
fn iq2_xxs_dequant_is_bit_exact_against_the_peer() {
    let fixture = load_fixture();
    let mut actual = vec![0.0f32; fixture.block_count() * QK_K];
    test_only_dequantize(&fixture.blocks, GgmlType::IQ2_XXS, &mut actual).expect("IQ2_XXS dequant");
    assert_matches_peer("library dequant", &actual, &fixture.expected);
}

#[test]
fn hand_ported_loop_is_bit_exact_against_the_peer() {
    let fixture = load_fixture();
    let actual = hand_port(&fixture.blocks);
    assert_matches_peer("hand port", &actual, &fixture.expected);
}

#[test]
fn iq2_xxs_dequant_rejects_malformed_buffers() {
    let fixture = load_fixture();
    let two_blocks = &fixture.blocks[..2 * BLOCK_BYTES];
    let mut output = vec![0.0f32; 2 * QK_K];
    assert!(test_only_dequantize(
        &two_blocks[..BLOCK_BYTES + 1],
        GgmlType::IQ2_XXS,
        &mut output
    )
    .is_err());
    assert!(
        test_only_dequantize(two_blocks, GgmlType::IQ2_XXS, &mut output[..2 * QK_K - 1]).is_err()
    );
    assert!(test_only_dequantize(two_blocks, GgmlType::IQ2_XXS, &mut output).is_ok());
}

#[test]
fn gguf_type_16_registers_iq2_xxs_geometry_and_storage() {
    let ggml_type = test_only_ggml_type_from_u32(16).expect("GGML type 16");
    assert_eq!(ggml_type, GgmlType::IQ2_XXS);
    assert_eq!(ggml_type.block_values(), 256);
    assert_eq!(ggml_type.block_bytes(), 66);
    assert_eq!(test_only_raw_tensor_dtype(ggml_type), DType::U8);
    // DeepSeek-V4-Flash gate/up experts: [256 experts, 2048 rows, 4096].
    assert_eq!(
        test_only_compute_byte_len(&[256, 2048, 4096], ggml_type).expect("expert stack bytes"),
        256 * 2048 * 16 * 66
    );
    assert!(test_only_compute_byte_len(&[2048, 4000], ggml_type).is_err());
}

#[test]
fn iq2_xxs_serde_name_is_frozen() {
    assert_eq!(
        serde_json::to_string(&GgmlType::IQ2_XXS).unwrap(),
        "\"iq2_xxs\""
    );
    let parsed: GgmlType = serde_json::from_str("\"iq2_xxs\"").unwrap();
    assert_eq!(parsed, GgmlType::IQ2_XXS);
}
