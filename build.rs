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
//! - Compiles ALL shaders; a single failure no longer loses the whole
//!   metallib — survivors are linked, failures fall back to runtime source
//!   compile per-shader. If 0 shaders succeed (systemic failure), hard-errors.
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
//! genuinely lack a feature still hit the graceful empty-metallib fallback.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR set by cargo");
    let out_dir = PathBuf::from(out_dir);
    let metallib_path = out_dir.join("default.metallib");

    // Tell cargo to re-run if any shader changes.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/shaders");
    println!("cargo:rerun-if-env-changed=MLX_NATIVE_SKIP_METALLIB");

    // Allow skipping (e.g. for docs builds or cross-compiled targets).
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let skip = env::var("MLX_NATIVE_SKIP_METALLIB").is_ok() || target_os != "macos";

    if skip {
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
    for metal_path in &metal_files {
        let stem = metal_path.file_stem().unwrap().to_string_lossy().to_string();
        let air_path = air_dir.join(format!("{stem}.air"));
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
                air_files.push(air_path);
            }
            Ok(o) => {
                let stderr = String::from_utf8_lossy(&o.stderr).to_string();
                let first_line = stderr.lines().next().unwrap_or(&stderr);
                failures.push((metal_path.clone(), first_line.to_string()));
                println!(
                    "cargo:warning=mlx-native: xcrun metal -O3 failed on {} (status={}); \
                     this shader will be runtime-source-compiled at load time: {}",
                    metal_path.display(),
                    o.status,
                    first_line,
                );
            }
            Err(e) => {
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
             linking partial metallib ({ok} shaders precompiled, {failures} fall back to \
             runtime source compile)",
            failures = failures.len(),
            ok = air_files.len(),
        );
    }

    // Link all .air into a single .metallib.
    let mut metallib_cmd = Command::new("xcrun");
    metallib_cmd.args(["-sdk", "macosx", "metallib"]);
    for air in &air_files {
        metallib_cmd.arg(air);
    }
    metallib_cmd.arg("-o").arg(&metallib_path);

    match metallib_cmd.status() {
        Ok(s) if s.success() => {
            // Informational, not a warning — use plain println! (visible with
            // `cargo build -v/-vv`); don't route success through cargo:warning=.
            println!(
                "mlx-native: built default.metallib ({} shaders, {} bytes)",
                air_files.len(),
                fs::metadata(&metallib_path).map(|m| m.len()).unwrap_or(0)
            );
        }
        Ok(s) => {
            println!(
                "cargo:warning=mlx-native: xcrun metallib link failed (status={}); writing empty",
                s
            );
            fs::write(&metallib_path, b"").expect("write empty metallib placeholder");
        }
        Err(e) => {
            println!(
                "cargo:warning=mlx-native: xcrun metallib failed ({}); writing empty",
                e
            );
            fs::write(&metallib_path, b"").expect("write empty metallib placeholder");
        }
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
