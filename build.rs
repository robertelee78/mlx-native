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
//! - On macOS / aarch64 with xcrun available, builds the .metallib.
//! - On other targets OR if xcrun is missing, writes an empty file so
//!   `include_bytes!` still compiles; the kernel_registry path treats an
//!   empty embedded metallib as "not present" and falls back to runtime
//!   source compile.
//! - Re-runs only when a .metal file under src/shaders/ changes.

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

    // Build .air files under OUT_DIR/air/.
    let air_dir = out_dir.join("air");
    fs::create_dir_all(&air_dir).expect("create air dir");

    let mut air_files: Vec<PathBuf> = Vec::with_capacity(metal_files.len());
    for metal_path in &metal_files {
        let stem = metal_path.file_stem().unwrap().to_string_lossy().to_string();
        let air_path = air_dir.join(format!("{stem}.air"));
        let status = Command::new("xcrun")
            .args(["-sdk", "macosx", "metal", "-O3", "-c"])
            .arg(metal_path)
            .arg("-o")
            .arg(&air_path)
            .status();

        match status {
            Ok(s) if s.success() => {
                air_files.push(air_path);
            }
            Ok(s) => {
                // Compile failure on a single shader: skip building the metallib;
                // hf2q will fall back to runtime source-compile at load time.
                println!(
                    "cargo:warning=mlx-native: xcrun metal -O3 failed on {} (status={}); \
                     writing empty metallib, will fall back to runtime source compile",
                    metal_path.display(),
                    s
                );
                fs::write(&metallib_path, b"").expect("write empty metallib placeholder");
                return;
            }
            Err(e) => {
                println!(
                    "cargo:warning=mlx-native: xcrun not available ({}); \
                     writing empty metallib, will fall back to runtime source compile",
                    e
                );
                fs::write(&metallib_path, b"").expect("write empty metallib placeholder");
                return;
            }
        }
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
            println!(
                "cargo:warning=mlx-native: built default.metallib ({} shaders, {} bytes)",
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
