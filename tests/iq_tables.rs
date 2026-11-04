//! ADR-066 N-IQ2XXS: one Rust source for every IQ2_XXS table copy.
//!
//! Pins the sha256 of the little-endian table bytes to the values measured
//! from the peer checkout's `ggml-common.h`, and asserts every generated
//! Metal block equals the rendering of `mlx_native::quant::iq_tables`.
//! Run with `MLX_BLESS_IQ_TABLES=1` to rewrite the shader blocks from the
//! Rust source.

use mlx_native::quant::iq_tables::{
    render_metal_iq2xxs_tables, IQ2XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS, METAL_IQ2XXS_BEGIN,
    METAL_IQ2XXS_END,
};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

const IQ2XXS_GRID_SHA256: &str = "05826b5d3e472a3a78f196be62ac78acf81df0f909626e12ab9fa2a5d490dd54";
const KSIGNS_IQ2XS_SHA256: &str =
    "a76742a603f8beca5212ecce0f1f02f11a4887fd2f7c8b15aca0ea3eb3380c31";
const KMASK_IQ2XS_SHA256: &str = "5ac9831b2e30eb285ef34f8501620f878432d5c04331ad1ae47f977a83ba41a5";

/// Every shader that carries the IQ2_XXS tables.
const IQ2XXS_SHADERS: [&str; 3] = [
    "src/shaders/quantized_matmul_id_ggml.metal",
    "src/shaders/quantized_matmul_id_mm.metal",
    "src/shaders/quantized_matmul_id_mm_tensor.metal",
];

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

#[test]
fn iq2xxs_tables_match_pinned_ggml_digests() {
    let grid_bytes: Vec<u8> = IQ2XXS_GRID.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(sha256_hex(&grid_bytes), IQ2XXS_GRID_SHA256, "iq2xxs_grid");
    assert_eq!(
        sha256_hex(&KSIGNS_IQ2XS),
        KSIGNS_IQ2XS_SHA256,
        "ksigns_iq2xs"
    );
    assert_eq!(sha256_hex(&KMASK_IQ2XS), KMASK_IQ2XS_SHA256, "kmask_iq2xs");
}

/// Byte range of the generated block (BEGIN line through the END line's
/// newline). Panics unless exactly one block exists.
fn generated_block_range(text: &str, path: &str) -> std::ops::Range<usize> {
    assert_eq!(
        text.matches(METAL_IQ2XXS_BEGIN).count(),
        1,
        "{path}: exactly one {METAL_IQ2XXS_BEGIN}"
    );
    assert_eq!(
        text.matches(METAL_IQ2XXS_END).count(),
        1,
        "{path}: exactly one {METAL_IQ2XXS_END}"
    );
    let start = text.find(METAL_IQ2XXS_BEGIN).unwrap();
    let end_marker = text.find(METAL_IQ2XXS_END).unwrap();
    assert!(start < end_marker, "{path}: markers out of order");
    let mut end = end_marker + METAL_IQ2XXS_END.len();
    if text[end..].starts_with('\n') {
        end += 1;
    }
    start..end
}

#[test]
fn shader_iq2xxs_blocks_equal_the_rust_rendering() {
    let rendered = render_metal_iq2xxs_tables();
    let bless = std::env::var("MLX_BLESS_IQ_TABLES").as_deref() == Ok("1");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for relative in IQ2XXS_SHADERS {
        let path = root.join(relative);
        let text = std::fs::read_to_string(&path).expect("read shader");
        let range = generated_block_range(&text, relative);
        if bless {
            let mut updated = String::with_capacity(text.len());
            updated.push_str(&text[..range.start]);
            updated.push_str(&rendered);
            updated.push_str(&text[range.end..]);
            if updated != text {
                std::fs::write(&path, updated).expect("bless shader");
            }
        } else {
            assert!(
                text[range.clone()] == rendered,
                "{relative}: generated iq2xxs block differs from src/quant/iq_tables.rs; \
                 rerun with MLX_BLESS_IQ_TABLES=1"
            );
        }
    }
}
