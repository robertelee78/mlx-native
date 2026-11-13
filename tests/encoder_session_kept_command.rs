//! `EncoderSession::reset_for_next_stage_keeping`: a stage committed without
//! a host wait leaves the session as a `SubmittedCommand` that reports its
//! own completion and status, at no extra command buffer.
//!
//! Needs `HF2Q_ENCODER_SESSION=1` (the session is absent otherwise); the
//! test prints `SKIP` and passes without it.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use mlx_native::{cmd_buf_count, DType, EncoderSession, KernelRegistry, MlxBuffer, MlxDevice};

fn buffer(device: &MlxDevice, values: &[f32]) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(std::mem::size_of_val(values), DType::F32, vec![values.len()])
        .expect("buffer");
    buffer.as_mut_slice::<f32>().unwrap().copy_from_slice(values);
    buffer
}

#[test]
fn kept_commands_report_their_completion_and_cost_no_command_buffer() {
    if !EncoderSession::env_enabled() {
        eprintln!("[encoder_session_kept_command] SKIP: HF2Q_ENCODER_SESSION is not 1");
        return;
    }
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();
    let one = buffer(&device, &[1.0, 1.0, 1.0, 1.0]);
    // A chain: every stage adds one to the previous stage's output.
    let stages = 6usize;
    let values: Vec<MlxBuffer> = (0..=stages)
        .map(|_| buffer(&device, &[0.0, 10.0, 20.0, 30.0]))
        .collect();

    let mut run = |keep: bool, fenced: bool| -> (u64, Vec<f32>, usize) {
        let mut session = device
            .encoder_session()
            .expect("encoder_session")
            .expect("session enabled");
        let before = cmd_buf_count();
        let mut kept = Vec::new();
        for stage in 0..stages {
            mlx_native::ops::elementwise::elementwise_add(
                session.encoder(),
                &mut registry,
                device.metal_device(),
                &values[stage],
                &one,
                &values[stage + 1],
                4,
                DType::F32,
            )
            .expect("elementwise_add");
            session.begin_stage("kept.stage");
            if stage + 1 == stages {
                // The last stage is committed without a wait as well; the
                // session keeps its command buffer until it is dropped.
                session.commit_stage().expect("commit last stage");
            } else {
                if fenced {
                    session.fence_stage(Some("kept.stage")).expect("fence");
                } else {
                    session.commit_stage().expect("commit");
                }
                if keep {
                    let command = session
                        .reset_for_next_stage_keeping()
                        .expect("rotate")
                        .expect("a committed stage leaves a command");
                    kept.push(command);
                } else {
                    session.reset_for_next_stage().expect("rotate");
                }
            }
        }
        // Not drained: nothing to hand over, and no rotation.
        let command_buffers = cmd_buf_count() - before;
        for command in &kept {
            command.wait_until_completed().expect("a kept command completed");
        }
        session.encoder().wait_until_completed().expect("last stage");
        let result = values[stages].as_slice::<f32>().unwrap().to_vec();
        (command_buffers, result, kept.len())
    };

    let expected: Vec<f32> = [0.0f32, 10.0, 20.0, 30.0]
        .iter()
        .map(|value| value + stages as f32)
        .collect();
    for fenced in [false, true] {
        let (plain_buffers, plain, _) = run(false, fenced);
        let (kept_buffers, kept, handles) = run(true, fenced);
        assert_eq!(plain, expected, "fenced {fenced}: releasing rotation");
        assert_eq!(kept, expected, "fenced {fenced}: keeping rotation");
        assert_eq!(handles, stages - 1, "fenced {fenced}: one command per rotated stage");
        assert_eq!(
            kept_buffers, plain_buffers,
            "fenced {fenced}: keeping a command must not create a command buffer"
        );
    }

    // A session that committed nothing hands nothing over.
    let mut idle = device
        .encoder_session()
        .expect("encoder_session")
        .expect("session enabled");
    assert!(idle.reset_for_next_stage_keeping().expect("no-op").is_none());
}
