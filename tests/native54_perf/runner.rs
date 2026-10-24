use super::{
    fixture::*,
    observer::{self, Observation},
};
use mlx_native::ops::deepseek_indexer::{
    dispatch_deepseek_indexer_into_mma, dispatch_deepseek_indexer_into_mma_tiled,
    DEEPSEEK_INDEXER_SCORE_MMA_KERNEL, DEEPSEEK_INDEXER_TOPK_BLOCK_KERNEL,
    DEEPSEEK_INDEXER_TOPK_FINALIZE_KERNEL, DEEPSEEK_INDEXER_TOPK_MERGE_KERNEL,
};
use mlx_native::{KernelRegistry, MlxDevice};
use std::time::Instant;
#[derive(Clone, Copy)]
struct Timing {
    wall: f64,
    start: f64,
    end: f64,
    block: usize,
    position: usize,
    arm: usize,
}
impl Timing {
    const EMPTY: Self = Self {
        wall: 0.0,
        start: 0.0,
        end: 0.0,
        block: 0,
        position: 0,
        arm: 0,
    };
    fn gpu(self) -> f64 {
        self.end - self.start
    }
    fn valid(self) -> bool {
        self.wall.is_finite()
            && self.wall > 0.0
            && self.start.is_finite()
            && self.end.is_finite()
            && self.start > 0.0
            && self.end > self.start
    }
}
fn trial(
    packet: &mut Packet,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    arm: usize,
    block: usize,
    position: usize,
) -> Timing {
    packet.poison(arm);
    let mut encoder = device.command_encoder().unwrap();
    let params = params();
    let begin = Instant::now();
    let dispatch = if arm == 0 {
        dispatch_deepseek_indexer_into_mma
    } else {
        dispatch_deepseek_indexer_into_mma_tiled
    };
    dispatch(
        &mut encoder,
        registry,
        device,
        &packet.q.data,
        &packet.kv.data,
        &packet.weights.data,
        &packet.scores[arm].data,
        &packet.topk[arm].data,
        &packet.outputs[arm].data,
        STRIDE,
        COLUMN,
        &params,
    )
    .unwrap();
    let result = encoder.commit_wait_with_gpu_time();
    let wall = begin.elapsed().as_secs_f64();
    // The checked wait has returned before any readback or owner teardown,
    // including its error path. Timestamp fields are absolute seconds.
    let (start, end) = result.unwrap();
    let record = Timing {
        wall,
        start,
        end,
        block,
        position,
        arm,
    };
    packet.check_arm(arm);
    record
}
fn stats(mut values: [f64; 10]) -> (f64, f64, f64) {
    values.sort_unstable_by(f64::total_cmp);
    ((values[4] + values[5]) / 2.0, values[0], values[9])
}
pub fn run() {
    let epoch = Instant::now();
    let mut observations = [Observation::EMPTY; 14];
    observations[0] = observer::sample(epoch); // before device/root initialization
    let device = MlxDevice::new().unwrap();
    let mut registry = KernelRegistry::new();
    let mut packet = Packet::new(&device);
    observations[1] = observer::sample(epoch); // after fixture/oracle initialization
    observations[2] = observer::sample(epoch); // before fixed warmups
    for (position, arm) in [0, 1, 0, 1].into_iter().enumerate() {
        let _ = trial(
            &mut packet,
            &device,
            &mut registry,
            arm,
            usize::MAX,
            position,
        );
    }
    packet.check_pair();
    // Bind identities to the SAME registry/pipelines actually warmed by both
    // public dispatchers. Collect outside timing, print only after all blocks.
    let pipeline_identities = [
        DEEPSEEK_INDEXER_SCORE_MMA_KERNEL,
        DEEPSEEK_INDEXER_TOPK_BLOCK_KERNEL,
        DEEPSEEK_INDEXER_TOPK_MERGE_KERNEL,
        DEEPSEEK_INDEXER_TOPK_FINALIZE_KERNEL,
    ]
    .map(|kernel| {
        let label = registry
            .get_pipeline(kernel, device.metal_device())
            .unwrap()
            .label()
            .to_owned();
        let identity = registry.pipeline_identity(&label).unwrap();
        assert_eq!(identity.pipeline_label, label);
        assert_eq!(identity.kernel_name, kernel);
        (kernel, identity)
    });
    observations[3] = observer::sample(epoch); // after both warmups per arm
    let mut timings = [Timing::EMPTY; 20];
    for block in 0..5 {
        observations[4 + block * 2] = observer::sample(epoch);
        for (position, arm) in [0, 1, 1, 0].into_iter().enumerate() {
            timings[block * 4 + position] =
                trial(&mut packet, &device, &mut registry, arm, block, position);
        }
        observations[5 + block * 2] = observer::sample(epoch);
        packet.check_pair();
    }
    // No formatting/printing, CPU oracle, thermal sampling, fixture/root creation
    // or encoder creation was inside any recorded dispatch→checked-wait window.
    // Public dispatch's own plan/view Vec allocation and pipeline cache lookup WERE.
    println!("native54_perf_packet q={Q} keys={K} start=251904 stride={STRIDE} column={COLUMN} offset=128 roots=9 root_bytes={ROOT_BYTES} host_numeric_bound={HOST_BYTES} q_seed={Q_SEED:016x} kv_seed={KV_SEED:016x} warmups_per_arm=2 blocks=5 samples_per_arm=10 capture=false model_loaded=false");
    for (kernel, identity) in &pipeline_identities {
        println!("native54_perf_pipeline kernel={kernel} identity={identity:?}");
    }
    println!("native54_perf_oracle cpu_rows=21 exact_full_pair=true rewritten_all_rows=true guards=true invalid_bf16_negative=true altered_expected_negative=true");
    for (index, observation) in observations.iter().enumerate() {
        println!("native54_perf_condition index={index} elapsed_s={:.9} thermal_raw={} low_power={} available={} nominal_lowpower_off={}",observation.elapsed,observation.thermal,observation.low_power,observation.available,observation.qualified());
    }
    for timing in &timings {
        println!("native54_perf_sample block={} position={} arm={} wall_s={:.9} gpu_start_s={:.9} gpu_end_s={:.9} gpu_interval_s={:.9} valid={}",timing.block,timing.position,timing.arm,timing.wall,timing.start,timing.end,timing.gpu(),timing.valid());
    }
    let mut wall = [[0.0; 10]; 2];
    let mut gpu = [[0.0; 10]; 2];
    let mut count = [0usize; 2];
    for t in &timings {
        wall[t.arm][count[t.arm]] = t.wall;
        gpu[t.arm][count[t.arm]] = t.gpu();
        count[t.arm] += 1;
    }
    assert_eq!(count, [10, 10]);
    let valid_timestamps = timings.iter().all(|t| t.valid());
    for arm in 0..2 {
        let (median, min, max) = stats(wall[arm]);
        println!("native54_perf_summary arm={arm} wall_median_s={median:.9} wall_min_s={min:.9} wall_max_s={max:.9} wall_range_s={:.9}",max-min);
        if valid_timestamps {
            let (median, min, max) = stats(gpu[arm]);
            println!("native54_perf_gpu_summary arm={arm} gpu_median_s={median:.9} gpu_min_s={min:.9} gpu_max_s={max:.9} gpu_range_s={:.9}",max-min);
        }
    }
    for block in 0..5 {
        let b = &timings[block * 4..block * 4 + 4];
        let a_wall = (b[0].wall + b[3].wall) / 2.0;
        let b_wall = (b[1].wall + b[2].wall) / 2.0;
        if valid_timestamps {
            println!("native54_perf_block block={block} tiled_over_untiled_wall={:.9} tiled_over_untiled_gpu={:.9}",b_wall/a_wall,(b[1].gpu()+b[2].gpu())/(b[0].gpu()+b[3].gpu()));
        } else {
            println!("native54_perf_block block={block} tiled_over_untiled_wall={:.9} gpu_ratio=unqualified",b_wall/a_wall);
        }
    }
    let wall_ratio = stats(wall[1]).0 / stats(wall[0]).0;
    if valid_timestamps {
        println!("native54_perf_overall tiled_over_untiled_wall={wall_ratio:.9} tiled_over_untiled_gpu={:.9}",stats(gpu[1]).0/stats(gpu[0]).0);
    } else {
        println!(
            "native54_perf_overall tiled_over_untiled_wall={wall_ratio:.9} gpu_ratio=unqualified"
        );
    }
    let nominal = observations.iter().all(|o| o.qualified());
    let measurement = valid_timestamps && nominal;
    println!("native54_perf_complete operator_parity=true timestamps_valid={valid_timestamps} passive_conditions_nominal={nominal} measurement_qualified={measurement} runtime_admission=false performance_model_qualified=false continuous_thermal_proof=false");
    assert!(measurement,"operator parity completed but timing inconclusive: timestamps or passive thermal/power conditions");
}
