// Shared helpers for the ADR-065 E6 verify-window probes.
//
// Included with `include!` so every probe stays one test binary (one
// process), which keeps the environment-latched kernel-selection caches
// (`MLX_PRECOMPILED_METALLIB`, `HF2Q_TQ_NWG`, `HF2Q_TQ_NSG`) consistent
// for the whole binary.

use mlx_native::{DType, KernelPipelineOrigin, KernelRegistry, MlxBuffer, MlxDevice};

struct E6Rng(u64);

impl E6Rng {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9e37_79b9_7f4a_7c15)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn next_u8(&mut self) -> u8 {
        (self.next_u64() >> 32) as u8
    }
    /// Uniform in [0, 1).
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }
    /// Uniform in [-scale, scale).
    fn signed(&mut self, scale: f32) -> f32 {
        (self.unit() * 2.0 - 1.0) * scale
    }
}

fn e6_disabled(name: &str) -> bool {
    matches!(
        std::env::var(name).as_deref(),
        Ok("0") | Ok("false") | Ok("off")
    )
}

/// Metallib path this process selected: `precompiled` or `runtime`.
fn e6_metallib_path() -> &'static str {
    if e6_disabled("MLX_PRECOMPILED_METALLIB") {
        "runtime"
    } else {
        "precompiled"
    }
}

/// Assert the pipeline `label` was built from the selected metallib path.
///
/// `MLX_PRECOMPILED_METALLIB=0` must always yield runtime-source pipelines.
/// On the default path a kernel is precompiled when the embedded metallib
/// carries it; function-constant pipelines additionally honor
/// `MLX_PRECOMPILED_METALLIB_FCV`. The observed origin is returned so it is
/// part of the qualification record either way.
fn e6_pipeline_origin(registry: &KernelRegistry, label: &str, function_constants: bool) -> String {
    let identity = registry
        .pipeline_identity(label)
        .unwrap_or_else(|error| panic!("pipeline identity for {label}: {error}"));
    let runtime_selected = e6_disabled("MLX_PRECOMPILED_METALLIB")
        || (function_constants && e6_disabled("MLX_PRECOMPILED_METALLIB_FCV"));
    if runtime_selected {
        assert_eq!(
            identity.origin,
            KernelPipelineOrigin::RuntimeSource,
            "{label}: runtime metallib path selected but pipeline is {:?}",
            identity.origin
        );
    } else {
        assert_eq!(
            identity.origin,
            KernelPipelineOrigin::PrecompiledMetallib,
            "{label}: precompiled metallib path selected but pipeline is {:?}",
            identity.origin
        );
    }
    format!("{:?}", identity.origin)
}

/// Bitwise comparison summary between an authority and a candidate.
struct E6Compare {
    elements: usize,
    mismatched: usize,
    max_abs_diff: f32,
    first_mismatch: Option<(usize, u32, u32)>,
}

impl E6Compare {
    fn new(authority: &[f32], candidate: &[f32]) -> Self {
        assert_eq!(authority.len(), candidate.len(), "compare length mismatch");
        let mut mismatched = 0usize;
        let mut max_abs_diff = 0.0f32;
        let mut first_mismatch = None;
        for (index, (&want, &got)) in authority.iter().zip(candidate).enumerate() {
            if want.to_bits() != got.to_bits() {
                mismatched += 1;
                let diff = (want - got).abs();
                if diff.is_nan() || diff > max_abs_diff {
                    max_abs_diff = if diff.is_nan() { f32::INFINITY } else { diff };
                }
                if first_mismatch.is_none() {
                    first_mismatch = Some((index, want.to_bits(), got.to_bits()));
                }
            }
        }
        Self {
            elements: authority.len(),
            mismatched,
            max_abs_diff,
            first_mismatch,
        }
    }
    fn bitwise(&self) -> bool {
        self.mismatched == 0
    }
}

/// Emit one qualification-record line (`E6_RECORD ...`) on stderr and, when
/// `E6_RECORD_PATH` is set, append it to that file.
fn e6_record(probe: &str, case: &str, fields: &[(&str, String)]) {
    let mut line = format!(
        "E6_RECORD probe={probe} metallib={} case={case}",
        e6_metallib_path()
    );
    for (key, value) in fields {
        line.push(' ');
        line.push_str(key);
        line.push('=');
        line.push_str(value);
    }
    eprintln!("{line}");
    if let Ok(path) = std::env::var("E6_RECORD_PATH") {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .unwrap_or_else(|error| panic!("open {path}: {error}"));
        writeln!(file, "{line}").unwrap_or_else(|error| panic!("write {path}: {error}"));
    }
}

fn e6_compare_fields(compare: &E6Compare) -> Vec<(&'static str, String)> {
    let mut fields = vec![
        ("bitwise", compare.bitwise().to_string()),
        ("elements", compare.elements.to_string()),
        ("mismatched", compare.mismatched.to_string()),
        ("max_abs_diff", format!("{:e}", compare.max_abs_diff)),
    ];
    if let Some((index, want, got)) = compare.first_mismatch {
        fields.push(("first_mismatch", format!("{index}:{want:#010x}/{got:#010x}")));
    }
    fields
}

fn e6_f32_buffer(device: &MlxDevice, data: &[f32]) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(data.len() * 4, DType::F32, vec![data.len()])
        .expect("alloc f32 buffer");
    buffer
        .as_mut_slice::<f32>()
        .expect("f32 buffer slice")
        .copy_from_slice(data);
    buffer
}

fn e6_u32_buffer(device: &MlxDevice, data: &[u32]) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(data.len() * 4, DType::U32, vec![data.len()])
        .expect("alloc u32 buffer");
    buffer
        .as_mut_slice::<u32>()
        .expect("u32 buffer slice")
        .copy_from_slice(data);
    buffer
}

fn e6_u8_buffer(device: &MlxDevice, data: &[u8]) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(data.len(), DType::U8, vec![data.len()])
        .expect("alloc u8 buffer");
    buffer
        .as_mut_slice::<u8>()
        .expect("u8 buffer slice")
        .copy_from_slice(data);
    buffer
}

fn e6_read_f32(buffer: &MlxBuffer) -> Vec<f32> {
    buffer.as_slice::<f32>().expect("read f32").to_vec()
}
