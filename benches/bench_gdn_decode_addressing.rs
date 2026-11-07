//! `gated_delta_net_decode_f32_4`: pre-rewrite (`legacy`, per-token `uint`
//! index rebuild) versus rewritten (`new`, running per-token pointers) at
//! Qwen3.8 DeltaNet heads (16 k, 48 v, D_k = D_v = 128, q scale folded in).
//!
//! Cases (one "step" each):
//!
//! * `prefill_n512_x1` — one dispatch, n_tokens 512, n_seqs 1;
//! * `decode_n1_x4_rows` — four n_seqs = 1 dispatches, one per row with its
//!   own inputs and state, no barrier between rows (hf2q's per-row decode
//!   on its concurrent encoder);
//! * `decode_n1_x4_seqs` — the same four rows as one n_seqs = 4 dispatch.
//!
//! Cache modes:
//!
//! * `warm` — every step reuses the same buffers (the peer's perf method);
//! * `cold` — steps rotate through `COLD_ROTATION` distinct buffer sets
//!   (inputs, state_in, state_out, output), so recurrent state is not cache
//!   resident between steps.
//!
//! Clock: `STEPS_PER_CB` (100) steps per command buffer with a memory
//! barrier between steps, `MTLCommandBuffer` GPUStartTime/GPUEndTime
//! divided by the step count; `WARMUP` warm-up buffers, then
//! `GDN_ADDR_ITERS` (default 20) measured buffers. Prints median, p10, p90
//! in µs per step. Needs the GPU and a quiet host:
//!
//! ```text
//! cargo bench --bench bench_gdn_decode_addressing
//! ```

use mlx_native::ops::gated_delta_net::{
    build_gated_delta_net_params_with_q_scale, GatedDeltaNetParams,
};
use mlx_native::ops::gated_delta_net_decode::{
    dispatch_gated_delta_net_decode, dispatch_gated_delta_net_decode_legacy,
};
use mlx_native::{CommandEncoder, DType, KernelRegistry, MlxBuffer, MlxDevice};

const D: u32 = 128;
const N_K_HEADS: u32 = 16;
const N_V_HEADS: u32 = 48;
const STEPS_PER_CB: usize = 100;
const WARMUP: usize = 5;
const DEFAULT_ITERS: usize = 20;
const COLD_ROTATION: usize = 32;

type Dispatch = fn(
    &mut CommandEncoder,
    &mut KernelRegistry,
    &metal::DeviceRef,
    &MlxBuffer,
    &MlxBuffer,
    &MlxBuffer,
    &MlxBuffer,
    &MlxBuffer,
    &MlxBuffer,
    &MlxBuffer,
    &MlxBuffer,
    &MlxBuffer,
    GatedDeltaNetParams,
) -> mlx_native::Result<()>;

struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn signed(&mut self, scale: f32) -> f32 {
        let unit = (self.next() >> 40) as f32 / (1u64 << 24) as f32;
        (2.0 * unit - 1.0) * scale
    }
}

fn f32_buffer(
    device: &MlxDevice,
    n: usize,
    rng: &mut XorShift,
    scale: f32,
    offset: f32,
) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(n * 4, DType::F32, vec![n])
        .expect("f32 buffer");
    for x in buffer.as_mut_slice::<f32>().expect("f32 slice") {
        *x = offset + rng.signed(scale);
    }
    buffer
}

/// One dispatch's buffers (inputs, state, output, params).
struct Set {
    p: GatedDeltaNetParams,
    q: MlxBuffer,
    k: MlxBuffer,
    v: MlxBuffer,
    g: MlxBuffer,
    beta: MlxBuffer,
    state_in: MlxBuffer,
    output: MlxBuffer,
    state_out: MlxBuffer,
    params: MlxBuffer,
}

fn set(device: &MlxDevice, n_tokens: u32, n_seqs: u32, rng: &mut XorShift) -> Set {
    let p = GatedDeltaNetParams {
        d_k: D,
        d_v: D,
        n_k_heads: N_K_HEADS,
        n_v_heads: N_V_HEADS,
        n_tokens,
        n_seqs,
    };
    let tokens = (n_tokens * n_seqs) as usize;
    let qk = (D * N_K_HEADS) as usize * tokens;
    let v = (D * N_V_HEADS) as usize * tokens;
    let scalars = N_V_HEADS as usize * tokens;
    let state = (D * D * N_V_HEADS * n_seqs) as usize;
    let kq_scale = 3.0 / (D as f32).sqrt();
    Set {
        p,
        q: f32_buffer(device, qk, rng, kq_scale, 0.0),
        k: f32_buffer(device, qk, rng, kq_scale, 0.0),
        v: f32_buffer(device, v, rng, 1.0, 0.0),
        g: f32_buffer(device, scalars, rng, 0.25, 0.25),
        beta: f32_buffer(device, scalars, rng, 0.4, 0.5),
        state_in: f32_buffer(device, state, rng, 0.05, 0.0),
        output: f32_buffer(device, v, rng, 0.0, 0.0),
        state_out: f32_buffer(device, state, rng, 0.0, 0.0),
        params: build_gated_delta_net_params_with_q_scale(device, p, 1.0 / (D as f32).sqrt())
            .expect("params"),
    }
}

fn encode(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    dispatch: Dispatch,
    s: &Set,
) {
    dispatch(
        encoder,
        registry,
        device.metal_device(),
        &s.q,
        &s.k,
        &s.v,
        &s.g,
        &s.beta,
        &s.state_in,
        &s.output,
        &s.state_out,
        &s.params,
        s.p,
    )
    .expect("GDN dispatch");
}

struct Stats {
    median_us: f64,
    p10_us: f64,
    p90_us: f64,
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * fraction).round() as usize]
}

/// `steps[i % steps.len()]` is the list of dispatch sets of step `i`; the
/// sets of one step are encoded without a barrier between them.
fn time_steps(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    dispatch: Dispatch,
    steps: &[Vec<&Set>],
    iters: usize,
) -> Stats {
    let sample = |registry: &mut KernelRegistry| {
        let mut encoder = device.command_encoder().expect("command encoder");
        for i in 0..STEPS_PER_CB {
            if i > 0 {
                encoder.memory_barrier();
            }
            for s in &steps[i % steps.len()] {
                encode(&mut encoder, registry, device, dispatch, s);
            }
        }
        let (start, end) = encoder
            .commit_wait_with_gpu_time()
            .expect("measured command buffer");
        (end - start) * 1e6 / STEPS_PER_CB as f64
    };
    for _ in 0..WARMUP {
        sample(registry);
    }
    let mut samples: Vec<f64> = (0..iters).map(|_| sample(registry)).collect();
    samples.sort_by(|a, b| a.partial_cmp(b).expect("finite GPU time"));
    Stats {
        median_us: percentile(&samples, 0.5),
        p10_us: percentile(&samples, 0.1),
        p90_us: percentile(&samples, 0.9),
    }
}

/// Build the per-step dispatch lists for `rotation` distinct step buffers.
fn case_sets(device: &MlxDevice, case: &str, rotation: usize, rng: &mut XorShift) -> Vec<Vec<Set>> {
    (0..rotation)
        .map(|_| match case {
            "prefill_n512_x1" => vec![set(device, 512, 1, rng)],
            "decode_n1_x4_rows" => (0..4).map(|_| set(device, 1, 1, rng)).collect(),
            "decode_n1_x4_seqs" => vec![set(device, 1, 4, rng)],
            other => panic!("unknown case {other}"),
        })
        .collect()
}

fn main() {
    let iters = std::env::var("GDN_ADDR_ITERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_ITERS);
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let mut rng = XorShift(0x5eed_9d17_0000_0001);
    let kernels: [(&str, Dispatch); 2] = [
        ("legacy", dispatch_gated_delta_net_decode_legacy),
        ("new", dispatch_gated_delta_net_decode),
    ];

    println!(
        "# gated_delta_net_decode_f32_4 heads 16k/48v D128; {STEPS_PER_CB} steps/CB with barriers; \
         {iters} CBs after {WARMUP} warm-up; µs per step"
    );
    println!(
        "{:<20} {:<5} {:<7} {:>10} {:>10} {:>10}",
        "case", "cache", "kernel", "median_us", "p10_us", "p90_us"
    );
    for case in ["prefill_n512_x1", "decode_n1_x4_rows", "decode_n1_x4_seqs"] {
        for (cache, rotation) in [("warm", 1usize), ("cold", COLD_ROTATION)] {
            let sets = case_sets(&device, case, rotation, &mut rng);
            let steps: Vec<Vec<&Set>> = sets.iter().map(|step| step.iter().collect()).collect();
            for &(name, dispatch) in &kernels {
                let stats = time_steps(&device, &mut registry, dispatch, &steps, iters);
                println!(
                    "{case:<20} {cache:<5} {name:<7} {:>10.2} {:>10.2} {:>10.2}",
                    stats.median_us, stats.p10_us, stats.p90_us
                );
            }
        }
    }
}
