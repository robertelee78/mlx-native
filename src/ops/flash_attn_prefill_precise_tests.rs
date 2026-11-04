use super::*;

const DEQUANT_SOURCE: &str = include_str!("../shaders/tq_dequantize_kv.metal");
const D256_SOURCE: &str = include_str!("../shaders/flash_attn_prefill.metal");
const D512_SOURCE: &str = include_str!("../shaders/flash_attn_prefill_d512.metal");

/// Parse the float initializer of `constant float <name>[N] = { ... };`.
fn codebook(source: &str, name: &str) -> Vec<f32> {
    let start = source
        .find(&format!("constant float {name}["))
        .unwrap_or_else(|| panic!("{name} not declared"));
    let body = &source[start..];
    let open = body.find('{').expect("initializer");
    let close = body.find("};").expect("initializer end");
    body[open + 1..close]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.trim_end_matches('f').parse::<f32>().expect("float literal"))
        .collect()
}

#[test]
fn precise_params_match_the_msl_layout() {
    assert_eq!(std::mem::size_of::<AttnPreciseParamsGpu>(), 40);
    assert_eq!(std::mem::offset_of!(AttnPreciseParamsGpu, scale_factor_d512), 8);
    assert_eq!(std::mem::offset_of!(AttnPreciseParamsGpu, n_strides), 16);
    for source in [D256_SOURCE, D512_SOURCE] {
        let start = source.find("struct AttnPreciseParams {").expect("struct");
        let fields: Vec<&str> = source[start..]
            .lines()
            .skip(1)
            .take_while(|line| !line.starts_with("};"))
            .map(|line| line.split(';').next().unwrap_or("").trim())
            .collect();
        assert_eq!(
            fields,
            [
                "int window",
                "int codebook_bits",
                "float scale_factor_d512",
                "int norms_per_pos",
                "int64_t N_strides[3]"
            ],
        );
    }
}

#[test]
fn attention_codebooks_are_the_dequant_codebooks() {
    for (dequant, prefix_bits) in [
        ("CODEBOOK_5BIT_DQ", ("5BIT", 32)),
        ("CODEBOOK_6BIT_DQ", ("6BIT", 64)),
        ("CODEBOOK_8BIT_DQ", ("8BIT", 256)),
    ] {
        let reference = codebook(DEQUANT_SOURCE, dequant);
        assert_eq!(reference.len(), prefix_bits.1);
        let d256 = codebook(D256_SOURCE, &format!("PFA_CODEBOOK_{}", prefix_bits.0));
        let d512 = codebook(D512_SOURCE, &format!("D512_CODEBOOK_{}", prefix_bits.0));
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&d256), bits(&reference), "d256 copy of {dequant}");
        assert_eq!(bits(&d512), bits(&reference), "d512 copy of {dequant}");
    }
}

#[test]
fn every_precise_name_is_instantiated_in_its_registered_source() {
    for &name in PRECISE_D256_KERNEL_NAMES {
        assert!(D256_SOURCE.contains(&format!("instantiate_kernel(\"{name}\"")), "{name}");
        assert!(!D512_SOURCE.contains(&format!("\"{name}\"")), "{name}");
    }
    for &name in PRECISE_D512_KERNEL_NAMES {
        assert!(D512_SOURCE.contains(&format!("instantiate_d512_precise(\"{name}\"")), "{name}");
        assert!(!D256_SOURCE.contains(&format!("\"{name}\"")), "{name}");
    }
}

#[test]
fn d256_visibility_selects_exactly_one_mask_mechanism() {
    let get = |kind| {
        let c = d256_constants(true, false, kind);
        let v = |i: usize| c.iter().find(|(idx, _)| *idx == i).map(|(_, b)| *b).unwrap();
        (v(200), v(201), v(300), v(301), v(303), v(310))
    };
    assert_eq!(
        get(PreciseD256MaskKind::External { blk: true }),
        (true, false, true, false, true, false)
    );
    assert_eq!(
        get(PreciseD256MaskKind::External { blk: false }),
        (true, false, true, false, false, false)
    );
    assert_eq!(get(PreciseD256MaskKind::Window), (true, false, false, true, false, true));
    assert_eq!(get(PreciseD256MaskKind::Causal), (true, false, false, true, false, false));
}

#[test]
fn pipeline_labels_use_the_registry_cache_key_format() {
    let label = pipeline_label(K_PRECISE_D512_VCODES, &d512_constants(false, true, true));
    assert_eq!(
        label,
        "flash_attn_prefill_precise_d512_vcodes|200:b0|201:b1|300:b0|301:b1|303:b0"
    );
}

#[test]
fn blk_params_follow_the_precise_tile() {
    let p = precise_d256_blk_params(56, 1079);
    assert_eq!((p.bq, p.bk), (PRECISE_D256_BQ, PRECISE_D256_BK));
    // ceil(56/32) * ceil(1079/16) = 2 * 68 = 136 -> padded to 160.
    assert_eq!(blk_buffer_byte_len(&p).unwrap(), 160);
}
