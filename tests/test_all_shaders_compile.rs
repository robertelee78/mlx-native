//! Build-time guarantee that ALL Metal shaders in `src/shaders/` compile.
//!
//! ADR-022 iter-68 root cause was a one-character typo in
//! `quantized_matmul_id_mm_tensor.metal:447` (Q5_K instantiation referenced
//! `GgmlMatmulIdMm_TensorParams` instead of `GgmlMatmulIdMmTensor_MmParams`).
//! The Metal compile error caused the entire .metal source to fail to build,
//! which made the runtime probe (`probe_tensor_mm_id`) FAIL and silently fall
//! back to the slower simdgroup MMA variant. Result: 3+ weeks of degraded
//! prefill performance on Gemma-4 batched mode that nobody noticed because
//! the fallback path produces correct (just slower) output.
//!
//! This test closes that gap by compiling every `.metal` file in `src/shaders/`
//! at test time via `xcrun -sdk macosx metal -c`. Any compile error fails the
//! test loud; any future iter-68-style typo is caught before it ships. The one
//! capability exception is an SDK that does not provide `<metal_tensor>`: only
//! sources that explicitly include that header may report that exact missing-
//! header error. Those optional accelerators are runtime-probed and have
//! non-tensor fallbacks on older Apple GPUs.
//!
//! Warnings (unused variables, etc.) are tolerated — they do not affect
//! pipeline registration. Only actual compile errors (non-zero exit code +
//! `error:` line in stderr) fail the test.
//!
//! Environment hygiene (2026-08-02 RCA, mirrors build.rs): `metal` derives its
//! default `-std` from `MACOSX_DEPLOYMENT_TARGET`. A low target (e.g. 11.0 in
//! ~/.cargo/config.toml, meant for the CPU binary) drops the default below
//! metal3.1 and 42 shaders fail on `bfloat` — a false negative that made this
//! gate red for reasons unrelated to shader correctness. The Rust binary's
//! deployment target has no bearing on GPU shader requirements, so we strip
//! it here exactly as build.rs does: both then compile with the host
//! toolchain's latest std, which is the configuration that actually ships.
//!
//! Apple-only: skipped on non-Apple targets via `cfg(target_vendor = "apple")`.

#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::fs;
use std::path::Path;
use std::process::Command;

fn is_unavailable_metal_tensor_capability(source: &str, stderr: &str) -> bool {
    source.contains("#include <metal_tensor>")
        && stderr.contains("fatal error: 'metal_tensor' file not found")
}

#[test]
fn all_metal_shaders_compile_via_xcrun() {
    let shader_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/shaders");
    assert!(
        shader_dir.is_dir(),
        "src/shaders/ not found at {}",
        shader_dir.display()
    );

    let mut compiled = 0;
    let mut errors: Vec<String> = Vec::new();
    let mut unavailable_tensor_shaders: Vec<String> = Vec::new();

    for entry in fs::read_dir(&shader_dir).expect("read src/shaders/") {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|s| s.to_str()) != Some("metal") {
            continue;
        }
        let name = path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "<unknown>".into());

        let output = Command::new("xcrun")
            .args(["-sdk", "macosx", "metal", "-c"])
            .arg(&path)
            .arg("-o")
            .arg("/dev/null")
            // See header: keep the CPU-binary deployment target out of
            // shader language-version resolution (same as build.rs).
            .env_remove("MACOSX_DEPLOYMENT_TARGET")
            .output()
            .expect("xcrun metal -c invocation");

        compiled += 1;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let source = fs::read_to_string(&path).expect("read Metal source after compile error");
            if is_unavailable_metal_tensor_capability(&source, &stderr) {
                unavailable_tensor_shaders.push(name);
            } else {
                errors.push(format!("=== COMPILE ERROR: {name} ===\n{stderr}"));
            }
        }
    }

    assert!(
        compiled > 0,
        "no .metal files found in {}",
        shader_dir.display()
    );

    if !errors.is_empty() {
        panic!(
            "{} of {} Metal shaders failed to compile:\n\n{}",
            errors.len(),
            compiled,
            errors.join("\n")
        );
    }

    eprintln!(
        "all_metal_shaders_compile_via_xcrun: {} shaders checked; {} tensor shaders unavailable on this SDK",
        compiled,
        unavailable_tensor_shaders.len(),
    );
    if !unavailable_tensor_shaders.is_empty() {
        eprintln!(
            "optional <metal_tensor> capability unavailable for: {}",
            unavailable_tensor_shaders.join(", ")
        );
    }
}

#[test]
fn capability_exception_is_exact_and_fail_closed() {
    let tensor_source = "#include <metal_stdlib>\n#include <metal_tensor>\nkernel void k() {}";
    let ordinary_source = "#include <metal_stdlib>\nkernel void k() {}";
    let missing_header = "fatal error: 'metal_tensor' file not found";

    assert!(is_unavailable_metal_tensor_capability(
        tensor_source,
        missing_header
    ));
    assert!(!is_unavailable_metal_tensor_capability(
        ordinary_source,
        missing_header
    ));
    assert!(!is_unavailable_metal_tensor_capability(
        tensor_source,
        "error: use of undeclared identifier 'broken_kernel'"
    ));
}
