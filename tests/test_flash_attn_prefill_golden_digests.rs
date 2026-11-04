//! P7 (hf2q ADR-063 Q4-GEMMA-KERNEL design §5.1): every pre-existing
//! `flash_attn_prefill*` entry point produces bitwise the output it produced
//! before the precise kernels were added.
//!
//! The goldens in `tests/fixtures/flash_attn_prefill_golden_digests.json`
//! are SHA-256 digests of the full O buffers (pre-filled with a sentinel, so
//! stray or missing writes change the digest), recorded by this same file
//! on mlx-native 0.15.16 (`20540a5`, before the change) on both metallib
//! paths:
//!
//! ```text
//! MLX_FLASH_ATTN_GOLDEN_RECORD=1 cargo test --release \
//!   --test test_flash_attn_prefill_golden_digests -- --ignored --test-threads=1
//! MLX_PRECOMPILED_METALLIB=0 MLX_FLASH_ATTN_GOLDEN_RECORD=1 cargo test ...
//! ```
//!
//! The digests are bound to the recording GPU and compilers, so the test is
//! ignored by default and run on that host (the fixture names the device).
//! The cases dispatch the kernels directly so this file compiles unchanged
//! against the pre-change crate.

#![cfg(target_os = "macos")]

use std::collections::BTreeMap;
use std::path::PathBuf;

use half::{bf16, f16};
use mlx_native::ops::flash_attn_prefill::{AttnMaskParamsGpu, AttnParamsGpu};
use mlx_native::KernelPipelineOrigin;
use mlx_native::{DType, KernelArg, KernelRegistry, MlxBuffer, MlxDevice};
use sha2::{Digest, Sha256};

const FIXTURE: &str = "tests/fixtures/flash_attn_prefill_golden_digests.json";
const SENTINEL: u8 = 0xA5;

struct Rng(u64);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn bytes_of(dtype: DType, values: &[f32]) -> Vec<u8> {
    match dtype {
        DType::F16 => values
            .iter()
            .flat_map(|&v| f16::from_f32(v).to_le_bytes())
            .collect(),
        DType::BF16 => values
            .iter()
            .flat_map(|&v| bf16::from_f32(v).to_le_bytes())
            .collect(),
        DType::F32 => values.iter().flat_map(|&v| v.to_le_bytes()).collect(),
        other => panic!("unsupported dtype {other}"),
    }
}

fn upload(device: &MlxDevice, dtype: DType, bytes: &[u8], shape: Vec<usize>) -> MlxBuffer {
    let mut buf = device
        .alloc_buffer(bytes.len(), dtype, shape)
        .expect("alloc");
    buf.as_mut_slice::<u8>()
        .expect("map")
        .copy_from_slice(bytes);
    buf
}

fn random(device: &MlxDevice, rng: &mut Rng, dtype: DType, n: usize, amp: f32) -> MlxBuffer {
    let values: Vec<f32> = (0..n).map(|_| rng.next_f32() * amp).collect();
    upload(device, dtype, &bytes_of(dtype, &values), vec![n])
}

/// Sliding-window visibility used for the mask cases: key k visible to row
/// i iff k <= i + off and i + off - k < 24.
fn visible(i: usize, k: usize, off: usize) -> bool {
    k <= i + off && i + off - k < 24
}

/// Mask (additive T: 0 / -inf, or bool) and the matching blk bytes for the
/// (bq, bk) tile grid.
fn mask_and_blk(
    device: &MlxDevice,
    dtype: Option<DType>,
    ql: usize,
    kl: usize,
    off: usize,
    bq: usize,
    bk: usize,
) -> (MlxBuffer, MlxBuffer) {
    let mask = match dtype {
        Some(dtype) => {
            let values: Vec<f32> = (0..ql * kl)
                .map(|e| {
                    if visible(e / kl, e % kl, off) {
                        0.0
                    } else {
                        f32::NEG_INFINITY
                    }
                })
                .collect();
            upload(device, dtype, &bytes_of(dtype, &values), vec![ql, kl])
        }
        None => {
            let values: Vec<u8> = (0..ql * kl)
                .map(|e| u8::from(visible(e / kl, e % kl, off)))
                .collect();
            upload(device, DType::U8, &values, vec![ql, kl])
        }
    };
    let (nq, nk) = (ql.div_ceil(bq), kl.div_ceil(bk));
    let mut blk = vec![0u8; (nq * nk).div_ceil(32) * 32];
    for t in 0..nq {
        for c in 0..nk {
            let (mut any, mut all) = (false, true);
            for i in t * bq..((t + 1) * bq).min(ql) {
                for k in c * bk..((c + 1) * bk).min(kl) {
                    let v = visible(i, k, off);
                    any |= v;
                    all &= v;
                }
            }
            blk[t * nk + c] = if !any {
                0
            } else if all {
                2
            } else {
                1
            };
        }
    }
    let blk = upload(device, DType::U8, &blk, vec![nq, nk]);
    (mask, blk)
}

#[derive(Clone, Copy)]
enum Feature {
    Plain,
    Causal,
    MaskBlk,
    Sinks,
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    name: &str,
    dtype: DType,
    bool_mask: bool,
    d: usize,
    tile: (usize, usize, u64),
    nsg: Option<i32>,
    feature: Feature,
    seed: u64,
) -> (String, String) {
    let (h, h_kv, batch) = (4usize, 2usize, 1usize);
    let (ql, kl, cap, off) = match feature {
        Feature::Causal => (33usize, 101usize, 128usize, 68usize),
        Feature::MaskBlk => (40, 72, 72, 32),
        _ => (37, 53, 53, 0),
    };
    let (bq, bk, threads_y) = tile;
    let mut rng = Rng(seed);
    let q = random(device, &mut rng, dtype, batch * h * ql * d, 1.0);
    let k = random(device, &mut rng, dtype, batch * h_kv * cap * d, 1.0);
    let v = random(device, &mut rng, dtype, batch * h_kv * cap * d, 2.0);
    let o_bytes = vec![SENTINEL; batch * h * ql * d * dtype.size_of()];
    let o = upload(device, dtype, &o_bytes, vec![batch * h * ql * d]);
    let sinks = random(device, &mut rng, DType::F32, h, 1.0);

    let has_mask = matches!(feature, Feature::MaskBlk);
    let has_sinks = matches!(feature, Feature::Sinks);
    let do_causal = matches!(feature, Feature::Causal);
    let mut bools = vec![
        (200usize, ql % bq == 0),
        (201, kl % bk == 0),
        (300, has_mask),
        (301, do_causal),
        (303, has_mask),
    ];
    let mut ints = Vec::new();
    if let Some(nsg) = nsg {
        bools.push((304, has_sinks));
        bools.push((305, false));
        ints.push((322usize, nsg));
    }
    let mut label = name.to_string();
    for &(i, b) in &bools {
        label.push_str(&format!("|{i}:{}", if b { "b1" } else { "b0" }));
    }
    for &(i, v) in &ints {
        label.push_str(&format!("|{i}:i{v}"));
    }

    let mask_dtype = if bool_mask { None } else { Some(dtype) };
    let (mask, blk) = mask_and_blk(device, mask_dtype, ql, kl, off, bq, bk);
    let q_strides = [(h * ql * d) as i64, (ql * d) as i64, d as i64];
    let kv_strides = [(h_kv * cap * d) as i64, (cap * d) as i64, d as i64];
    let params = AttnParamsGpu {
        b: batch as i32,
        h: h as i32,
        d: d as i32,
        ql: ql as i32,
        kl: kl as i32,
        gqa_factor: (h / h_kv) as i32,
        scale: 1.0 / (d as f32).sqrt(),
        softcapping: 1.0,
        nq: ql.div_ceil(bq) as i32,
        nk: kl.div_ceil(bk) as i32,
        nq_aligned: (ql / bq) as i32,
        nk_aligned: (kl / bk) as i32,
        ql_rem: (ql % bq) as i32,
        kl_rem: (kl % bk) as i32,
        ql_off: if do_causal { off as i32 } else { 0 },
        _pad: 0,
        q_strides,
        k_strides: kv_strides,
        v_strides: kv_strides,
        o_strides: q_strides,
    };
    let mask_params = AttnMaskParamsGpu {
        m_strides: [0, 0, kl as i64],
    };

    let pipeline = registry
        .get_pipeline_with_constants(name, device.metal_device(), &bools, &ints)
        .expect("pipeline");
    let mut args = vec![
        (0u64, KernelArg::Buffer(&q)),
        (1, KernelArg::Buffer(&k)),
        (2, KernelArg::Buffer(&v)),
        (3, KernelArg::Buffer(&o)),
        (4, KernelArg::Bytes(bytemuck::bytes_of(&params))),
    ];
    if has_mask {
        args.push((5, KernelArg::Bytes(bytemuck::bytes_of(&mask_params))));
        args.push((6, KernelArg::Buffer(&mask)));
        args.push((7, KernelArg::Buffer(&blk)));
    }
    if has_sinks {
        args.push((8, KernelArg::Buffer(&sinks)));
    }
    let grid = metal::MTLSize::new(ql.div_ceil(bq) as u64, h as u64, batch as u64);
    let mut encoder = device.command_encoder().expect("encoder");
    if nsg.is_some() {
        encoder.encode_threadgroups_with_args_and_shared(
            pipeline,
            &args,
            &[(0, 28_672)],
            grid,
            metal::MTLSize::new(32, threads_y, 1),
        );
    } else {
        encoder.encode_threadgroups_with_args(
            pipeline,
            &args,
            grid,
            metal::MTLSize::new(32, threads_y, 1),
        );
    }
    encoder.commit_and_wait().expect("commit");
    let digest = hex::encode(Sha256::digest(o.as_slice::<u8>().expect("read")));
    (label, digest)
}

fn all_cases(device: &MlxDevice, registry: &mut KernelRegistry) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    // flash_attn_prefill.metal: (D, BQ, BK, WM) families × dtype × mask kind.
    for (d, tile) in [(64usize, (32, 16, 4)), (256, (32, 16, 4)), (512, (8, 8, 1))] {
        for (tag, dtype) in [("bf16", DType::BF16), ("f16", DType::F16)] {
            for bool_mask in [false, true] {
                let suffix = if bool_mask { "_boolmask" } else { "" };
                let name = format!("flash_attn_prefill_{tag}_d{d}{suffix}");
                for feature in [Feature::Plain, Feature::Causal, Feature::MaskBlk] {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let (label, digest) = run_case(
                        device, registry, &name, dtype, bool_mask, d, tile, None, feature, seed,
                    );
                    out.insert(label, digest);
                }
            }
        }
    }
    // flash_attn_prefill_d512.metal: NSG 4 and 8.
    for (tag, dtype) in [("bf16", DType::BF16), ("f16", DType::F16)] {
        for bool_mask in [false, true] {
            let suffix = if bool_mask { "_boolmask" } else { "" };
            let name = format!("flash_attn_prefill_d512_{tag}{suffix}");
            for nsg in [4, 8] {
                for feature in [
                    Feature::Plain,
                    Feature::Causal,
                    Feature::MaskBlk,
                    Feature::Sinks,
                ] {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let tile = (8, 64, nsg as u64);
                    let (label, digest) = run_case(
                        device,
                        registry,
                        &name,
                        dtype,
                        bool_mask,
                        512,
                        tile,
                        Some(nsg),
                        feature,
                        seed,
                    );
                    out.insert(label, digest);
                }
            }
        }
    }
    out
}

#[test]
#[ignore = "GPU goldens bound to the recording host; run with --ignored (P7)"]
fn existing_flash_attn_prefill_entry_points_match_their_pre_change_digests() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    // `register()` after `KernelRegistry::new()` marks these names as runtime
    // source overrides (they are not in the constructor's source map), which
    // would pin both passes to the runtime compiler.  Register them only for
    // the runtime-source pass; the precompiled pass resolves them from the
    // embedded metallib.
    let runtime = matches!(
        std::env::var("MLX_PRECOMPILED_METALLIB").as_deref(),
        Ok("0") | Ok("false") | Ok("off")
    );
    if runtime {
        mlx_native::ops::flash_attn_prefill::register(&mut registry);
        mlx_native::ops::flash_attn_prefill_d512::register(&mut registry);
    }
    let digests = all_cases(&device, &mut registry);
    let origins: std::collections::BTreeSet<String> = digests
        .keys()
        .map(|label| {
            match registry.pipeline_identity(label).expect("identity").origin {
                KernelPipelineOrigin::PrecompiledMetallib => "precompiled_metallib",
                KernelPipelineOrigin::RuntimeSource => "runtime_source",
            }
            .to_string()
        })
        .collect();
    assert_eq!(origins.len(), 1, "mixed pipeline origins {origins:?}");
    let origin = origins.into_iter().next().unwrap();

    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(FIXTURE);
    let mut fixture: serde_json::Value = std::fs::read_to_string(&path)
        .ok()
        .map(|s| serde_json::from_str(&s).expect("fixture json"))
        .unwrap_or_else(|| serde_json::json!({ "device": device.name(), "paths": {} }));
    if std::env::var("MLX_FLASH_ATTN_GOLDEN_RECORD").as_deref() == Ok("1") {
        assert_eq!(
            fixture["device"],
            serde_json::json!(device.name()),
            "recording device"
        );
        fixture["paths"][&origin] = serde_json::to_value(&digests).unwrap();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(
            &path,
            serde_json::to_string_pretty(&fixture).unwrap() + "\n",
        )
        .unwrap();
        eprintln!(
            "recorded {} {origin} digests on {}",
            digests.len(),
            device.name()
        );
        return;
    }
    assert_eq!(
        fixture["device"],
        serde_json::json!(device.name()),
        "goldens were recorded on another GPU"
    );
    let golden: BTreeMap<String, String> =
        serde_json::from_value(fixture["paths"][&origin].clone())
            .unwrap_or_else(|_| panic!("no {origin} goldens in {FIXTURE}"));
    assert_eq!(golden.len(), digests.len(), "case set changed");
    let mismatches: Vec<&String> = digests
        .iter()
        .filter(|(label, d)| golden.get(*label) != Some(*d))
        .map(|(l, _)| l)
        .collect();
    assert!(
        mismatches.is_empty(),
        "{} {origin} entry points changed: {mismatches:?}",
        mismatches.len()
    );
    eprintln!(
        "{} {origin} digests bitwise equal to the pre-change goldens",
        digests.len()
    );
}
