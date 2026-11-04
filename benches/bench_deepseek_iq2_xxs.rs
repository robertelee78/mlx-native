//! ADR-066 N-IQ2XXS microbenchmarks at DeepSeek-V4-Flash expert shapes
//! (256 experts). Recorded, not gated:
//! - IQ2_XXS gate/up `mv_id` (K = 4096, N = 2048) at 6, 12 and 24 routed
//!   rows (one to four decode slots, top-6);
//! - the paired IQ2_XXS gate+up `mm_id` at 512 and 4096 tokens;
//! - Q2_K `mv_id` at the same gate/up shape, the byte-normalized baseline
//!   from hf2q's current artifact;
//! - Q2_K down `mv_id` (K = 2048, N = 4096, `top_k = 1` per routed row).
//!
//! Each sample encodes 16 dispatches into one command buffer and reports
//! host wall time per dispatch, so commit-and-wait latency (about 300 µs
//! per command buffer) does not mask kernel time. Each line reports the
//! median, p10 and p90 of 20 samples (after 5 warmups), and effective
//! GB/s over the weight bytes the routed experts touch.

use mlx_native::{
    quantized_matmul_id_ggml_mv_with_policy, quantized_matmul_id_ggml_pooled_pair_with_policy,
    DType, GgmlQuantizedMatmulIdParams, GgmlRoutingPolicy, GgmlType, IdMmScratch, KernelRegistry,
    MlxBuffer, MlxDevice,
};

const EXPERTS: u32 = 256;
const TOP_K: u32 = 6;
const WARMUP: usize = 5;
const SAMPLES: usize = 20;
const DISPATCHES_PER_SAMPLE: usize = 16;

struct Shape {
    ggml_type: GgmlType,
    n: u32,
    k: u32,
}

impl Shape {
    fn expert_bytes(&self) -> usize {
        self.n as usize
            * (self.k / self.ggml_type.block_values()) as usize
            * self.ggml_type.block_bytes() as usize
    }

    fn params(&self, n_tokens: u32, top_k: u32) -> GgmlQuantizedMatmulIdParams {
        GgmlQuantizedMatmulIdParams {
            n_tokens,
            top_k,
            n: self.n,
            k: self.k,
            n_experts: EXPERTS,
            expert_stride: self.expert_bytes() as u64,
            ggml_type: self.ggml_type,
        }
    }
}

fn buffer(device: &MlxDevice, bytes: usize, dtype: DType) -> MlxBuffer {
    device
        .alloc_buffer(bytes, dtype, vec![bytes / dtype.size_of()])
        .expect("buffer")
}

/// Distinct experts per token, spread over all 256 experts.
fn ids(device: &MlxDevice, tokens: u32, top_k: u32) -> (MlxBuffer, usize) {
    let mut buf = buffer(device, (tokens * top_k) as usize * 4, DType::U32);
    let values = buf.as_mut_slice::<u32>().expect("ids");
    for token in 0..tokens as usize {
        for slot in 0..top_k as usize {
            values[token * top_k as usize + slot] =
                ((token * 37 + slot * 41) % EXPERTS as usize) as u32;
        }
    }
    let mut distinct = values.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    (buf, distinct.len())
}

fn report(label: &str, mut samples: Vec<f64>, bytes: usize) {
    samples.sort_by(f64::total_cmp);
    let median = samples[samples.len() / 2];
    let p10 = samples[samples.len() / 10];
    let p90 = samples[samples.len() * 9 / 10];
    println!(
        "{label}: median={median:.2} us p10={p10:.2} us p90={p90:.2} us \
         weight={:.2} MB effective={:.2} GB/s",
        bytes as f64 / 1e6,
        bytes as f64 / (median * 1e-6) / 1e9
    );
}

fn time(device: &MlxDevice, mut encode: impl FnMut(&mut mlx_native::CommandEncoder)) -> Vec<f64> {
    for _ in 0..WARMUP {
        let mut encoder = device.command_encoder().expect("encoder");
        encode(&mut encoder);
        encoder.commit_and_wait().expect("warmup");
    }
    (0..SAMPLES)
        .map(|_| {
            let mut encoder = device.command_encoder().expect("encoder");
            let started = std::time::Instant::now();
            for _ in 0..DISPATCHES_PER_SAMPLE {
                encode(&mut encoder);
            }
            encoder.commit_and_wait().expect("sample");
            started.elapsed().as_secs_f64() * 1e6 / DISPATCHES_PER_SAMPLE as f64
        })
        .collect()
}

fn bench_mv(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    label: &str,
    shape: &Shape,
    n_tokens: u32,
    top_k: u32,
) {
    let weight = buffer(device, EXPERTS as usize * shape.expert_bytes(), DType::U8);
    let input = buffer(device, (n_tokens * shape.k) as usize * 4, DType::F32);
    let output = buffer(
        device,
        (n_tokens * top_k * shape.n) as usize * 4,
        DType::F32,
    );
    let (ids, distinct) = ids(device, n_tokens, top_k);
    let params = shape.params(n_tokens, top_k);
    let policy = GgmlRoutingPolicy::default();
    let samples = time(device, |encoder| {
        quantized_matmul_id_ggml_mv_with_policy(
            encoder, registry, device, &input, &weight, &ids, &output, &params, &policy,
        )
        .expect("mv_id dispatch");
    });
    report(
        &format!(
            "{label} mv_id {}x{} routed={} N={} K={}",
            n_tokens,
            top_k,
            n_tokens * top_k,
            shape.n,
            shape.k
        ),
        samples,
        distinct * shape.expert_bytes(),
    );
}

fn bench_pair(device: &MlxDevice, registry: &mut KernelRegistry, shape: &Shape, tokens: u32) {
    let gate = buffer(device, EXPERTS as usize * shape.expert_bytes(), DType::U8);
    let up = buffer(device, EXPERTS as usize * shape.expert_bytes(), DType::U8);
    let input = buffer(device, (tokens * shape.k) as usize * 4, DType::F32);
    let len = (tokens * TOP_K * shape.n) as usize * 4;
    let gate_out = buffer(device, len, DType::F32);
    let up_out = buffer(device, len, DType::F32);
    let (ids, distinct) = ids(device, tokens, TOP_K);
    let params = shape.params(tokens, TOP_K);
    let policy = GgmlRoutingPolicy::default();
    let mut scratch = IdMmScratch::alloc(device, EXPERTS, tokens).expect("scratch");
    let samples = time(device, |encoder| {
        quantized_matmul_id_ggml_pooled_pair_with_policy(
            encoder,
            registry,
            device,
            &input,
            &gate,
            &up,
            &ids,
            &gate_out,
            &up_out,
            &mut scratch,
            &params,
            &policy,
        )
        .expect("pair mm_id dispatch");
    });
    report(
        &format!(
            "IQ2_XXS pair mm_id {tokens}x{TOP_K} N={} K={}",
            shape.n, shape.k
        ),
        samples,
        2 * distinct * shape.expert_bytes(),
    );
}

fn main() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let iq2_gate_up = Shape {
        ggml_type: GgmlType::IQ2_XXS,
        n: 2048,
        k: 4096,
    };
    let q2_gate_up = Shape {
        ggml_type: GgmlType::Q2_K,
        n: 2048,
        k: 4096,
    };
    let q2_down = Shape {
        ggml_type: GgmlType::Q2_K,
        n: 4096,
        k: 2048,
    };
    for slots in [1u32, 2, 4] {
        bench_mv(
            &device,
            &mut registry,
            "IQ2_XXS gate/up",
            &iq2_gate_up,
            slots,
            TOP_K,
        );
        bench_mv(
            &device,
            &mut registry,
            "Q2_K gate/up",
            &q2_gate_up,
            slots,
            TOP_K,
        );
        bench_mv(
            &device,
            &mut registry,
            "Q2_K down",
            &q2_down,
            slots * TOP_K,
            1,
        );
    }
    for tokens in [512u32, 4096] {
        bench_pair(&device, &mut registry, &iq2_gate_up, tokens);
    }
}
