//! IQ codebook tables: the single source for every CPU and Metal copy.
//!
//! ADR-066 N-IQ2XXS. The values equal the ggml `iq2xxs_grid`,
//! `ksigns_iq2xs` and `kmask_iq2xs` tables; `tests/iq_tables.rs` pins the
//! sha256 of their little-endian bytes. Metal shaders are standalone files,
//! so each shader that needs the tables carries a generated block between
//! `// BEGIN GENERATED iq2xxs` and `// END GENERATED iq2xxs`, rendered by
//! [`render_metal_iq2xxs_tables`]. The same test asserts every shader block
//! is byte-identical to that rendering and rewrites the blocks when run with
//! `MLX_BLESS_IQ_TABLES=1`.
//!
//! ADR-066 N3 adds the other IQ grids beside these.

use std::fmt::Write as _;

/// IQ2_XXS grid: 256 entries of eight magnitude bytes (each 8, 25 or 43),
/// little-endian byte `j` is value `j` of the 8-value group.
#[rustfmt::skip]
pub const IQ2XXS_GRID: [u64; 256] = [
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
];

/// Bit masks selecting the sign bit of value `j` in a sign byte.
pub const KMASK_IQ2XS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

/// Even-parity sign table: the low seven bits carry seven signs and bit 7
/// makes the popcount even, so the 128 seven-bit indices address the 128
/// even-parity sign bytes.
pub const KSIGNS_IQ2XS: [u8; 128] = ksigns_iq2xs();

const fn ksigns_iq2xs() -> [u8; 128] {
    let mut table = [0u8; 128];
    let mut index = 0;
    while index < 128 {
        let parity = (index as u32).count_ones() & 1;
        table[index] = index as u8 | (parity as u8) << 7;
        index += 1;
    }
    table
}

/// Opening marker of the generated IQ2_XXS table block in a Metal shader.
pub const METAL_IQ2XXS_BEGIN: &str = "// BEGIN GENERATED iq2xxs";
/// Closing marker of the generated IQ2_XXS table block in a Metal shader.
pub const METAL_IQ2XXS_END: &str = "// END GENERATED iq2xxs";

/// Render the Metal `constant` tables, markers included, with a trailing
/// newline after the closing marker.
pub fn render_metal_iq2xxs_tables() -> String {
    let mut out = String::with_capacity(8 * 1024);
    out.push_str(METAL_IQ2XXS_BEGIN);
    out.push_str(
        "\n// Generated from src/quant/iq_tables.rs by tests/iq_tables.rs\n\
         // (MLX_BLESS_IQ_TABLES=1). Do not edit by hand.\n",
    );
    out.push_str("constant uint64_t iq2xxs_grid[256] = {\n");
    for row in IQ2XXS_GRID.chunks(4) {
        out.push_str("   ");
        for value in row {
            let _ = write!(out, " 0x{value:016x},");
        }
        out.push('\n');
    }
    out.push_str("};\n");
    out.push_str("constant uint8_t ksigns_iq2xs[128] = {\n");
    for row in KSIGNS_IQ2XS.chunks(16) {
        out.push_str("   ");
        for value in row {
            let _ = write!(out, " {value:3},");
        }
        out.push('\n');
    }
    out.push_str("};\n");
    out.push_str("constant uint8_t kmask_iq2xs[8] = {\n   ");
    for value in KMASK_IQ2XS {
        let _ = write!(out, " {value},");
    }
    out.push_str("\n};\n");
    out.push_str(METAL_IQ2XXS_END);
    out.push('\n');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_bytes_are_iq2_xxs_magnitudes() {
        for entry in IQ2XXS_GRID {
            for byte in entry.to_le_bytes() {
                assert!(matches!(byte, 8 | 25 | 43), "grid byte {byte}");
            }
        }
    }

    #[test]
    fn signs_have_even_parity_and_keep_low_bits() {
        for (index, &signs) in KSIGNS_IQ2XS.iter().enumerate() {
            assert_eq!(signs & 0x7f, index as u8);
            assert_eq!(signs.count_ones() % 2, 0);
        }
    }

    #[test]
    fn rendered_block_is_delimited_once() {
        let block = render_metal_iq2xxs_tables();
        assert!(block.starts_with(METAL_IQ2XXS_BEGIN));
        assert!(block.ends_with(&format!("{METAL_IQ2XXS_END}\n")));
        assert_eq!(block.matches(METAL_IQ2XXS_BEGIN).count(), 1);
        assert_eq!(block.matches("0x").count(), 256);
    }
}
