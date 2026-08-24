//! ADR-029 iter-175 Step 1l — precompiled `.metallib` build.
//!
//! Compiles every `.metal` file under `src/shaders/` with `xcrun metal -O3`,
//! links them into a single `default.metallib` in `OUT_DIR`, so the crate's
//! `KernelRegistry` can load it via `device.new_library_with_data` and skip
//! Apple's runtime source-compile path (which produces ~3% larger AIR with
//! occasional 70 µs jitter vs the precompiled ~19 µs steady — see iter 1k
//! empirical test).
//!
//! Behavior:
//! - On macOS / aarch64, probes the Metal Toolchain and hard-errors if
//!   absent — a missing toolchain silently undoes the precompiled-metallib
//!   perf win (ADR-022 iter-68 burn class). The intentional-skip path
//!   (`MLX_NATIVE_SKIP_METALLIB` or non-macOS target) writes an empty file
//!   so `include_bytes!` still compiles.
//! - Compiles ALL shaders; a single ordinary-shader failure no longer loses
//!   the whole metallib — survivors are linked, failures fall back to runtime
//!   source compile per-shader. The Q4 tensor shader has one narrower hosted
//!   downgrade: the SDK exactly reports a missing `<metal_tensor>` header.
//!   Every other failure in that release-qualified shader is fatal.
//! - Re-runs only when a .metal file under src/shaders/ changes.
//!
//! Environment hygiene (2026-08-02 RCA): the `metal` frontend derives its
//! default `-std` from `MACOSX_DEPLOYMENT_TARGET`. A low deployment target
//! (e.g. 11.0, set machine-wide in ~/.cargo/config.toml for the CPU binary)
//! silently drops the default below metal3.1, so `bfloat` and every newer
//! feature fail to parse — and this build script used to swallow that into
//! "write empty metallib, fall back to runtime compile", a silent perf
//! regression of the exact class ADR-022 iter-68 burned on. The deployment
//! target of the Rust CPU binary has NO bearing on GPU shader language
//! requirements (shaders need the toolchain's best: bfloat, mpp::tensor_ops),
//! so we strip the variable from the compiler's environment. With it gone,
//! `metal` defaults to the host toolchain's latest std, which self-consistently
//! tracks the shaders' requirements on every machine. Older toolchains that
//! genuinely lack the `<metal_tensor>` header retain the portable partial
//! artifact, but shader regressions still fail closed.

#[path = "build_support/metal_tensor.rs"]
mod metal_tensor;

use std::env;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(mlx_native_has_metal_tensor_artifact)");
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR set by cargo");
    let out_dir = PathBuf::from(out_dir);
    let metallib_path = out_dir.join("default.metallib");
    let require_tensor_artifact = env::var_os("MLX_NATIVE_REQUIRE_METAL_TENSOR_ARTIFACT").is_some();

    // Tell cargo to re-run if any shader changes.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=build_support/metal_tensor.rs");
    println!("cargo:rerun-if-changed=src/shaders");
    println!("cargo:rerun-if-env-changed=MLX_NATIVE_SKIP_METALLIB");
    println!("cargo:rerun-if-env-changed=MLX_NATIVE_REQUIRE_METAL_TENSOR_ARTIFACT");

    // Allow skipping (e.g. for docs builds or cross-compiled targets).
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let skip = env::var("MLX_NATIVE_SKIP_METALLIB").is_ok() || target_os != "macos";

    if skip {
        if require_tensor_artifact {
            panic!(
                "mlx-native build.rs: the release gate requires a linked Metal tensor artifact, \
                 but metallib construction was skipped (target_os={target_os}, \
                 MLX_NATIVE_SKIP_METALLIB={})",
                env::var_os("MLX_NATIVE_SKIP_METALLIB").is_some(),
            );
        }
        // Write an empty file so include_bytes! still works.
        fs::write(&metallib_path, b"").expect("write empty metallib placeholder");
        println!(
            "cargo:warning=mlx-native: skipping metallib build (target_os={}, skip={})",
            target_os, skip
        );
        return;
    }

    // Locate all .metal files under src/shaders/.
    let shader_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("shaders");

    let mut metal_files: Vec<PathBuf> = match fs::read_dir(&shader_dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("metal"))
            .collect(),
        Err(e) => {
            if require_tensor_artifact {
                panic!(
                    "mlx-native build.rs: the release gate requires a linked Metal tensor \
                     artifact, but the shader directory {} could not be read: {e}",
                    shader_dir.display(),
                );
            }
            println!(
                "cargo:warning=mlx-native: cannot read shader dir {}: {}; writing empty metallib",
                shader_dir.display(),
                e
            );
            fs::write(&metallib_path, b"").expect("write empty metallib placeholder");
            return;
        }
    };
    metal_files.sort();

    if metal_files.is_empty() {
        if require_tensor_artifact {
            panic!(
                "mlx-native build.rs: the release gate requires a linked Metal tensor \
                 artifact, but no .metal files were found in {}",
                shader_dir.display(),
            );
        }
        fs::write(&metallib_path, b"").expect("write empty metallib placeholder");
        println!("cargo:warning=mlx-native: no .metal files found");
        return;
    }

    // Probe the Metal Toolchain ONCE before iterating. On macOS, an absent
    // toolchain is a HARD ERROR — the silent empty-metallib fallback masks a
    // real perf regression (ADR-022 iter-68 burn class: 3% larger AIR, 70µs
    // jitter, one-time source-compile tax on every process start). The
    // intentional-skip path (MLX_NATIVE_SKIP_METALLIB / non-macOS) is handled
    // above and stays graceful.
    //
    // Auto-install: if the probe fails, attempt `xcodebuild -downloadComponent
    // MetalToolchain` so a fresh clone reaches a working state without manual
    // Xcode fiddling. Set `MLX_NATIVE_NO_AUTOINSTALL_METAL=1` to opt out (CI
    // that wants to control the toolchain itself). If install fails or is
    // declined, hard-error with the manual command — never silently degrade.
    if !metal_toolchain_present() {
        if env::var("MLX_NATIVE_NO_AUTOINSTALL_METAL").is_ok() {
            panic!(
                "mlx-native build.rs: Metal Toolchain not found and auto-install disabled \
                 (MLX_NATIVE_NO_AUTOINSTALL_METAL=1). Run manually: \
                 xcodebuild -downloadComponent MetalToolchain",
            );
        }
        println!("cargo:warning=mlx-native: Metal Toolchain not found — attempting auto-install via `xcodebuild -downloadComponent MetalToolchain` (~700 MB download)...");
        match install_metal_toolchain() {
            Ok(()) => println!("cargo:warning=mlx-native: Metal Toolchain installed; re-probing."),
            Err(e) => panic!(
                "mlx-native build.rs: Metal Toolchain auto-install failed ({}). \
                 Run manually: xcodebuild -downloadComponent MetalToolchain",
                e,
            ),
        }
        if !metal_toolchain_present() {
            panic!(
                "mlx-native build.rs: Metal Toolchain still not found after auto-install. \
                 Run manually: xcodebuild -downloadComponent MetalToolchain, then rebuild.",
            );
        }
    }

    // Build .air files under OUT_DIR/air/.
    let air_dir = out_dir.join("air");
    fs::create_dir_all(&air_dir).expect("create air dir");

    // Compile ALL shaders. A single failure no longer loses the whole metallib
    // — survivors are linked into the metallib, failures fall back to runtime
    // source-compile per-shader at load time. If 0 succeed (systemic failure),
    // we hard-error below.
    let mut air_files: Vec<PathBuf> = Vec::with_capacity(metal_files.len());
    let mut failures: Vec<(PathBuf, String)> = Vec::new();
    let mut tensor_air_compiled = false;
    for metal_path in &metal_files {
        let stem = metal_path
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let air_path = air_dir.join(format!("{stem}.air"));
        remove_previous_output(&air_path, "AIR");
        let mut command = Command::new("xcrun");
        command.args(["-sdk", "macosx", "metal", "-O3"]);
        // RoPE sees angles near 1e6 radians at the trained context limit.
        // Default Metal FP32 transcendentals are the `fast` variants and
        // diverge materially between Apple GPU generations at that range.
        // Keep this tiny, latency-insensitive kernel on precise FP32 math so
        // long-context position encoding remains source-reference stable.
        if stem == "deepseek_tail_rope" {
            command.args([
                "-fno-fast-math",
                "-fmetal-math-mode=safe",
                "-fmetal-math-fp32-functions=precise",
            ]);
        }
        let output = command
            .arg("-c")
            .arg(metal_path)
            .arg("-o")
            .arg(&air_path)
            // See header: keep the CPU-binary deployment target out of
            // shader language-version resolution.
            .env_remove("MACOSX_DEPLOYMENT_TARGET")
            .output();

        match output {
            Ok(o) if o.status.success() => {
                if stem == metal_tensor::Q4_TENSOR_SHADER_STEM {
                    let air_bytes = fs::metadata(&air_path).map(|metadata| metadata.len());
                    match air_bytes {
                        Ok(bytes) if bytes > 0 => tensor_air_compiled = true,
                        Ok(_) => panic!(
                            "mlx-native build.rs: the Q4 tensor shader compiler reported success \
                             but produced an empty AIR artifact: {}",
                            air_path.display(),
                        ),
                        Err(error) => panic!(
                            "mlx-native build.rs: the Q4 tensor shader compiler reported success \
                             but its AIR artifact {} cannot be inspected: {error}",
                            air_path.display(),
                        ),
                    }
                }
                air_files.push(air_path);
            }
            Ok(o) => {
                let stderr = String::from_utf8_lossy(&o.stderr).to_string();
                let first_line = stderr.lines().next().unwrap_or(&stderr);
                if stem == metal_tensor::Q4_TENSOR_SHADER_STEM
                    && !metal_tensor::is_exact_missing_metal_tensor_capability(&stem, &stderr)
                {
                    panic!(
                        "mlx-native build.rs: Q4 tensor shader regression while compiling {} \
                         (status={}): {}",
                        metal_path.display(),
                        o.status,
                        first_line,
                    );
                }
                failures.push((metal_path.clone(), first_line.to_string()));
                if stem == metal_tensor::Q4_TENSOR_SHADER_STEM {
                    println!(
                        "cargo:warning=mlx-native: SDK lacks the exact <metal_tensor> header; \
                         the Q4 tensor shader is absent from this hosted artifact: {}",
                        first_line,
                    );
                } else {
                    println!(
                        "cargo:warning=mlx-native: xcrun metal -O3 failed on {} (status={}); \
                         this shader will be runtime-source-compiled at load time: {}",
                        metal_path.display(),
                        o.status,
                        first_line,
                    );
                }
            }
            Err(e) => {
                if stem == metal_tensor::Q4_TENSOR_SHADER_STEM {
                    panic!(
                        "mlx-native build.rs: cannot invoke xcrun metal for the release-qualified \
                         Q4 tensor shader {}: {e}",
                        metal_path.display(),
                    );
                }
                failures.push((metal_path.clone(), e.to_string()));
                println!(
                    "cargo:warning=mlx-native: cannot invoke xcrun metal on {} ({}); \
                     this shader will be runtime-source-compiled at load time",
                    metal_path.display(),
                    e,
                );
            }
        }
    }

    if air_files.is_empty() {
        let last = failures
            .last()
            .map(|(p, e)| format!("{}: {e}", p.display()))
            .unwrap_or_else(|| "unknown".to_string());
        panic!(
            "mlx-native build.rs: all {} metal shader(s) failed to compile — 0 .air files \
             produced. This is a systemic failure, not a single-shader bug. \
             Last failure: {last}. \
             Run: xcodebuild -downloadComponent MetalToolchain",
            metal_files.len(),
        );
    }

    if !failures.is_empty() {
        println!(
            "cargo:warning=mlx-native: {failures} shader(s) failed, {ok} succeeded — \
             linking partial metallib ({ok} shaders precompiled, {failures} excluded; the \
             runtime registry determines per-shader availability)",
            failures = failures.len(),
            ok = air_files.len(),
        );
    }

    if require_tensor_artifact && !tensor_air_compiled {
        panic!(
            "mlx-native build.rs: the release gate requires the Q4 tensor shader, but the \
             current SDK lacks its exact <metal_tensor> capability"
        );
    }

    // Link all .air into a single .metallib.
    let mut metallib_cmd = Command::new("xcrun");
    metallib_cmd.args(["-sdk", "macosx", "metallib"]);
    for air in &air_files {
        metallib_cmd.arg(air);
    }
    metallib_cmd.arg("-o").arg(&metallib_path);
    remove_previous_output(&metallib_path, "metallib");

    let linked_nonempty_artifact = match metallib_cmd.status() {
        Ok(s) if s.success() => {
            let bytes = fs::metadata(&metallib_path)
                .map(|metadata| metadata.len())
                .unwrap_or(0);
            if bytes == 0 {
                println!(
                    "cargo:warning=mlx-native: xcrun metallib reported success but produced an \
                     empty artifact"
                );
                false
            } else {
                // Informational, not a warning — use plain println! (visible with
                // `cargo build -v/-vv`); don't route success through cargo:warning=.
                println!(
                    "mlx-native: built default.metallib ({} shaders, {} bytes)",
                    air_files.len(),
                    bytes,
                );
                true
            }
        }
        Ok(s) => {
            println!(
                "cargo:warning=mlx-native: xcrun metallib link failed (status={}); writing empty",
                s
            );
            fs::write(&metallib_path, b"").expect("write empty metallib placeholder");
            false
        }
        Err(e) => {
            println!(
                "cargo:warning=mlx-native: xcrun metallib failed ({}); writing empty",
                e
            );
            fs::write(&metallib_path, b"").expect("write empty metallib placeholder");
            false
        }
    };

    if !linked_nonempty_artifact {
        if require_tensor_artifact {
            panic!(
                "mlx-native build.rs: the release gate requires a successfully linked, \
                 nonempty Metal tensor artifact"
            );
        }
        return;
    }

    // This cfg is an artifact receipt, not an SDK capability guess: the exact
    // Q4 tensor AIR was included in a successful, nonempty metallib link.
    if tensor_air_compiled {
        println!("cargo:rustc-cfg=mlx_native_has_metal_tensor_artifact");
    }
}

fn remove_previous_output(path: &Path, label: &str) {
    match fs::remove_file(path) {
        Ok(()) => {}
        Err(error) if error.kind() == ErrorKind::NotFound => {}
        Err(error) => panic!(
            "mlx-native build.rs: cannot remove previous {label} output {}: {error}",
            path.display(),
        ),
    }
}

/// Probe whether the Metal Toolchain is installed and `xcrun metal` can run.
/// Returns true on success, false if the toolchain is missing or `xcrun` itself
/// is unavailable. Used to gate the hard-error / auto-install path.
fn metal_toolchain_present() -> bool {
    match Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "--version"])
        .output()
    {
        Ok(o) => o.status.success(),
        Err(_) => false,
    }
}

/// Attempt to install the Metal Toolchain via `xcodebuild -downloadComponent`.
/// Returns Ok(()) on success, Err with a diagnostic on any failure. Idempotent
/// — if the component is already present, xcodebuild reports success quickly.
fn install_metal_toolchain() -> Result<(), String> {
    let output = Command::new("xcodebuild")
        .args(["-downloadComponent", "MetalToolchain"])
        .output()
        .map_err(|e| format!("cannot invoke xcodebuild: {e}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(format!(
            "xcodebuild -downloadComponent MetalToolchain exited {} — stderr: {} stdout: {}",
            output.status,
            stderr.trim(),
            stdout.trim(),
        ));
    }
    Ok(())
}
